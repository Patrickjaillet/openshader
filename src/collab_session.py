"""
collab_session.py
-----------------
Co-édition temps réel pour OpenShader  (v5.0)

Implémentation WebSocket maison sur stdlib pure (asyncio + socket).
Aucune dépendance externe : fonctionne sans websockets, FastAPI, etc.

Architecture
============

  CollabServer  — asyncio server (thread daemon), port LAN configurable
    ├── gère N clients connectés
    ├── broadcast des messages JSON à tous sauf l'expéditeur
    └── conserve un snapshot de l'état courant pour late-join

  CollabClient  — asyncio client (thread daemon)
    ├── se connecte à un CollabServer (LAN ou relay)
    └── reçoit / envoie des messages JSON

  CollabSession (QObject)  — orchestrateur Qt-safe
    ├── expose start_server() / join_server()
    ├── emet des pyqtSignal vers MainWindow
    └── gère la sérialisation / désérialisation des ops

Protocole de messages (JSON)
============================

  { "type": "hello",   "peer_id": str, "name": str, "color": str }
  { "type": "bye",     "peer_id": str }
  { "type": "cursor",  "peer_id": str, "name": str, "color": str,
                        "time": float, "track_id": int|null }
  { "type": "shader",  "peer_id": str, "pass": str, "code": str }
  { "type": "uniform", "peer_id": str, "name": str, "value": ... }
  { "type": "timeline","peer_id": str, "data": {...} }
  { "type": "lock",    "peer_id": str, "name": str,
                        "track_id": int, "locked": bool }
  { "type": "chat",    "peer_id": str, "name": str,
                        "color": str, "text": str, "ts": float }
  { "type": "snapshot","peer_id": str,
                        "shaders": {pass: code, ...},
                        "timeline": {...},
                        "peers": [{peer_id,name,color}, ...] }
  { "type": "ping" }   — keepalive
  { "type": "pong" }

WebSocket framing minimal (RFC 6455 subset)
===========================================

Seuls les frames TEXT non-fragmentés sont utilisés.
Pas de masquage côté serveur. Masquage côté client (obligatoire RFC 6455).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
import re
import socket
import struct
import threading
import time
import uuid
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QScrollArea, QFrame, QTabWidget,
    QDialog, QSplitter, QComboBox, QCheckBox, QSpinBox,
    QListWidget, QListWidgetItem, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from .logger import get_logger

log = get_logger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────────

DEFAULT_PORT    = 9876
WS_MAGIC        = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_PAYLOAD     = 2 * 1024 * 1024   # 2 MB par frame
PING_INTERVAL_S = 15
RECONNECT_DELAY = 3.0
PEER_COLORS     = ["#4e7fff", "#50e0a0", "#e07850", "#e0d050",
                   "#c050e0", "#50c8e0", "#e05080", "#80e050"]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers WebSocket RFC 6455
# ══════════════════════════════════════════════════════════════════════════════

def _ws_handshake_response(key: str) -> bytes:
    """Génère la réponse HTTP 101 pour le handshake WS."""
    accept = base64.b64encode(
        hashlib.sha1((key + WS_MAGIC).encode()).digest()
    ).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode()


def _ws_parse_handshake(data: bytes) -> Optional[str]:
    """Extrait la clé Sec-WebSocket-Key du handshake HTTP entrant."""
    try:
        text = data.decode("utf-8", errors="replace")
        m = re.search(r"Sec-WebSocket-Key:\s*(\S+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def _ws_encode_frame(payload: str) -> bytes:
    """Encode un frame TEXT WebSocket (non masqué, côté serveur)."""
    data = payload.encode("utf-8")
    length = len(data)
    if length <= 125:
        header = bytes([0x81, length])
    elif length <= 65535:
        header = struct.pack(">BBH", 0x81, 126, length)
    else:
        header = struct.pack(">BBQ", 0x81, 127, length)
    return header + data


def _ws_encode_frame_masked(payload: str) -> bytes:
    """Encode un frame TEXT WebSocket masqué (côté client, obligatoire RFC 6455)."""
    data  = payload.encode("utf-8")
    mask  = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    length = len(data)
    if length <= 125:
        header = bytes([0x81, 0x80 | length]) + mask
    elif length <= 65535:
        header = struct.pack(">BBH", 0x81, 0x80 | 126, length) + mask
    else:
        header = struct.pack(">BBQ", 0x81, 0x80 | 127, length) + mask
    return header + masked


async def _ws_read_frame(reader: asyncio.StreamReader) -> Optional[str]:
    """
    Lit un frame WebSocket depuis un StreamReader.
    Retourne le payload TEXT décodé, ou None si connexion fermée / frame non-TEXT.
    Gère les frames masqués (client→serveur) et non masqués (serveur→client).
    """
    try:
        header = await reader.readexactly(2)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None

    fin  = bool(header[0] & 0x80)
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F

    if opcode == 0x8:   # close
        return None
    if opcode == 0x9:   # ping
        return "__ping__"
    if opcode == 0xA:   # pong
        return "__pong__"

    if length == 126:
        ext = await reader.readexactly(2)
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = await reader.readexactly(8)
        length = struct.unpack(">Q", ext)[0]

    if length > MAX_PAYLOAD:
        log.warning("WS frame too large: %d bytes — dropping", length)
        await reader.readexactly(length + (4 if masked else 0))
        return None

    if masked:
        mask_key = await reader.readexactly(4)
        payload  = await reader.readexactly(length)
        payload  = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    else:
        payload = await reader.readexactly(length)

    if opcode not in (0x1, 0x0):   # TEXT or continuation only
        return None

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CollabServer  —  serveur WebSocket asyncio
# ══════════════════════════════════════════════════════════════════════════════

class _ServerPeer:
    def __init__(self, peer_id: str, writer: asyncio.StreamWriter):
        self.peer_id = peer_id
        self.name    = peer_id[:8]
        self.color   = "#ffffff"
        self.writer  = writer
        self.alive   = True

    async def send(self, msg: dict):
        try:
            frame = _ws_encode_frame(json.dumps(msg, ensure_ascii=False))
            self.writer.write(frame)
            await self.writer.drain()
        except Exception as e:
            log.debug("ServerPeer.send error (%s): %s", self.peer_id, e)
            self.alive = False


class CollabServer:
    """
    Serveur WebSocket asyncio.
    Tourne dans un thread daemon séparé.
    """

    def __init__(self):
        self._peers:    dict[str, _ServerPeer] = {}
        self._snapshot: dict = {}          # état snapshot pour late-join
        self._lock      = asyncio.Lock()   # créé dans la loop du thread
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._server    = None
        self._thread:   Optional[threading.Thread] = None
        self.host:      str  = "0.0.0.0"
        self.port:      int  = DEFAULT_PORT
        self.running:   bool = False
        self._on_message: Optional[Callable[[dict], None]] = None  # callback → Qt

    def set_snapshot(self, snapshot: dict):
        """Met à jour le snapshot (état courant) transmis aux nouveaux pairs."""
        self._snapshot = snapshot

    def start(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT,
              on_message: Optional[Callable[[dict], None]] = None):
        self.host = host
        self.port = port
        self._on_message = on_message
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="CollabServer")
        self._thread.start()
        time.sleep(0.3)   # laisser la loop démarrer

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.running = False

    def broadcast_from_host(self, msg: dict, exclude_id: Optional[str] = None):
        """Envoie un message à tous les clients depuis le thread Qt."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(msg, exclude_id), self._loop)

    # ── Boucle asyncio ────────────────────────────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            log.error("CollabServer loop error: %s", e)
        finally:
            self.running = False
            log.info("CollabServer stopped")

    async def _main(self):
        self._lock = asyncio.Lock()
        server = await asyncio.start_server(
            self._handle_client, self.host, self.port,
            reuse_address=True,
        )
        self._server = server
        self.running = True
        log.info("CollabServer listening on %s:%d", self.host, self.port)
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter):
        """Gère un client : handshake → boucle de lecture → déconnexion."""
        # ── Handshake HTTP → WebSocket ─────────────────────────────────────
        try:
            raw = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        except asyncio.TimeoutError:
            writer.close()
            return

        key = _ws_parse_handshake(raw)
        if not key:
            writer.close()
            return

        writer.write(_ws_handshake_response(key))
        await writer.drain()

        peer_id = str(uuid.uuid4())[:8]
        peer    = _ServerPeer(peer_id, writer)

        async with self._lock:
            self._peers[peer_id] = peer

        log.info("CollabServer: peer connected %s", peer_id)

        # ── Envoi du snapshot ─────────────────────────────────────────────
        if self._snapshot:
            snap = dict(self._snapshot)
            snap["type"]    = "snapshot"
            snap["peer_id"] = "__server__"
            snap["peers"]   = [
                {"peer_id": p.peer_id, "name": p.name, "color": p.color}
                for p in self._peers.values() if p.peer_id != peer_id
            ]
            await peer.send(snap)

        # ── Boucle de lecture ─────────────────────────────────────────────
        try:
            while peer.alive:
                text = await asyncio.wait_for(_ws_read_frame(reader), timeout=PING_INTERVAL_S * 2)

                if text is None:
                    break

                if text == "__ping__":
                    await peer.send({"type": "pong"})
                    continue
                if text == "__pong__":
                    continue

                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue

                # Mise à jour des métadonnées du pair
                if msg.get("type") == "hello":
                    peer.name  = msg.get("name",  peer.peer_id)
                    peer.color = msg.get("color", "#ffffff")
                    msg["peer_id"] = peer_id

                # Callback vers Qt (host reçoit ses propres pairs)
                if self._on_message:
                    self._on_message(msg)

                # Broadcast à tous sauf l'expéditeur
                await self._broadcast(msg, exclude_id=peer_id)

        except asyncio.TimeoutError:
            log.debug("CollabServer: peer %s timed out", peer_id)
        except Exception as e:
            log.debug("CollabServer: peer %s error: %s", peer_id, e)
        finally:
            async with self._lock:
                self._peers.pop(peer_id, None)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            bye = {"type": "bye", "peer_id": peer_id}
            await self._broadcast(bye)
            if self._on_message:
                self._on_message(bye)
            log.info("CollabServer: peer disconnected %s", peer_id)

    async def _broadcast(self, msg: dict, exclude_id: Optional[str] = None):
        async with self._lock:
            targets = [p for pid, p in self._peers.items()
                       if pid != exclude_id and p.alive]
        for peer in targets:
            await peer.send(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  CollabClient  —  client WebSocket asyncio
# ══════════════════════════════════════════════════════════════════════════════

class CollabClient:
    """Client WebSocket asyncio — tourne dans un thread daemon."""

    def __init__(self):
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._thread: Optional[threading.Thread] = None
        self.connected:  bool = False
        self.peer_id:    str  = str(uuid.uuid4())[:8]
        self._on_message: Optional[Callable[[dict], None]] = None
        self._on_status:  Optional[Callable[[str], None]]  = None
        self._host = ""
        self._port = DEFAULT_PORT
        self._stop_event: Optional[asyncio.Event] = None

    def connect(self, host: str, port: int,
                on_message: Callable[[dict], None],
                on_status:  Callable[[str], None]):
        self._host       = host
        self._port       = port
        self._on_message = on_message
        self._on_status  = on_status
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="CollabClient")
        self._thread.start()

    def disconnect(self):
        self.connected = False
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def send(self, msg: dict):
        """Envoie un message depuis le thread Qt (thread-safe)."""
        if self._loop and self.connected and self._writer:
            asyncio.run_coroutine_threadsafe(
                self._async_send(msg), self._loop)

    # ── Boucle asyncio ────────────────────────────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            log.error("CollabClient loop: %s", e)
        finally:
            self.connected = False

    async def _connect_loop(self):
        while not self._stop_event.is_set():
            try:
                await self._run_session()
            except Exception as e:
                log.debug("CollabClient session ended: %s", e)
            if self._stop_event.is_set():
                break
            if self._on_status:
                self._on_status(f"Reconnexion dans {RECONNECT_DELAY:.0f}s…")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=RECONNECT_DELAY)
            except asyncio.TimeoutError:
                pass

    async def _run_session(self):
        try:
            reader, writer = await asyncio.open_connection(self._host, self._port)
        except (ConnectionRefusedError, OSError) as e:
            if self._on_status:
                self._on_status(f"Connexion refusée ({self._host}:{self._port})")
            raise

        # ── Handshake HTTP ───────────────────────────────────────────────────
        nonce = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {nonce}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
        writer.write(handshake)
        await writer.drain()

        # Lit la réponse HTTP 101
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            if not chunk:
                raise ConnectionError("Handshake: connexion fermée")
            resp += chunk

        if b"101" not in resp:
            raise ConnectionError(f"Handshake échoué : {resp[:200]}")

        self._writer = writer
        self.connected = True
        if self._on_status:
            self._on_status(f"Connecté à {self._host}:{self._port}")
        log.info("CollabClient connected to %s:%d", self._host, self._port)

        # ── Ping périodique ────────────────────────────────────────────────
        async def _ping_loop():
            while self.connected and not self._stop_event.is_set():
                try:
                    writer.write(bytes([0x89, 0x80]) + os.urandom(4))  # ping masqué
                    await writer.drain()
                except Exception:
                    break
                await asyncio.sleep(PING_INTERVAL_S)

        asyncio.ensure_future(_ping_loop())

        # ── Boucle de lecture ──────────────────────────────────────────────
        try:
            while self.connected and not self._stop_event.is_set():
                text = await asyncio.wait_for(
                    _ws_read_frame(reader), timeout=PING_INTERVAL_S * 3)

                if text is None:
                    break
                if text == "__ping__":
                    # pong frame masqué
                    mask = os.urandom(4)
                    writer.write(bytes([0x8A, 0x80]) + mask)
                    await writer.drain()
                    continue
                if text == "__pong__":
                    continue

                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if self._on_message:
                    self._on_message(msg)

        except asyncio.TimeoutError:
            log.debug("CollabClient: read timeout")
        finally:
            self.connected = False
            self._writer = None
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            if self._on_status:
                self._on_status("Déconnecté")

    async def _async_send(self, msg: dict):
        if self._writer:
            try:
                frame = _ws_encode_frame_masked(json.dumps(msg, ensure_ascii=False))
                self._writer.write(frame)
                await self._writer.drain()
            except Exception as e:
                log.debug("CollabClient.send error: %s", e)
                self.connected = False


# ══════════════════════════════════════════════════════════════════════════════
#  CollabSession  —  orchestrateur Qt-safe
# ══════════════════════════════════════════════════════════════════════════════

class CollabSession(QObject):
    """
    Point d'entrée unique pour la co-édition.

    Signaux Qt (émis depuis threads asyncio via appel thread-safe) :
      peer_joined(str, str, str)       — (peer_id, name, color)
      peer_left(str)                   — peer_id
      cursor_moved(str, str, str, float, object)  — (peer_id, name, color, time, track_id|None)
      shader_changed(str, str, str)    — (peer_id, pass_name, code)
      uniform_changed(str, str, object)— (peer_id, uniform_name, value)
      timeline_changed(str, dict)      — (peer_id, timeline_data)
      track_locked(str, str, int, bool)— (peer_id, name, track_id, locked)
      chat_received(str, str, str, str)— (peer_id, name, color, text)
      status_changed(str)              — message d'état lisible
      snapshot_received(dict)          — état initial reçu à la connexion
    """

    peer_joined      = pyqtSignal(str, str, str)
    peer_left        = pyqtSignal(str)
    cursor_moved     = pyqtSignal(str, str, str, float, object)
    shader_changed   = pyqtSignal(str, str, str)
    uniform_changed  = pyqtSignal(str, str, object)
    timeline_changed = pyqtSignal(str, dict)
    track_locked     = pyqtSignal(str, str, int, bool)
    chat_received    = pyqtSignal(str, str, str, str)
    status_changed   = pyqtSignal(str)
    snapshot_received = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._server:  Optional[CollabServer] = None
        self._client:  Optional[CollabClient] = None
        self.peer_id:  str  = str(uuid.uuid4())[:8]
        self.name:     str  = f"user-{self.peer_id}"
        self.color:    str  = random.choice(PEER_COLORS)
        self.is_host:  bool = False
        self.active:   bool = False
        # Verrous par piste : track_id → peer_id
        self._locks:   dict[int, str] = {}
        # Pairs connus : peer_id → {name, color}
        self._peers:   dict[str, dict] = {}

        # Timer de throttle curseur (ne pas spammer)
        self._cursor_timer  = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(50)  # 20 fps max
        self._pending_cursor: Optional[dict] = None
        self._cursor_timer.timeout.connect(self._flush_cursor)

    # ── API publique ──────────────────────────────────────────────────────────

    @property
    def local_addr(self) -> str:
        """Retourne l'IP LAN locale (pour afficher l'adresse du serveur)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start_server(self, port: int = DEFAULT_PORT,
                     name: str = "", relay: bool = False) -> bool:
        """Lance un serveur (rôle hôte). Retourne True si démarrage OK."""
        if self.active:
            self.stop()

        if name:
            self.name = name

        self._server = CollabServer()
        self._server.start(
            host="0.0.0.0", port=port,
            on_message=self._qt_dispatch,
        )
        time.sleep(0.4)

        if not self._server.running:
            self._server = None
            self.status_changed.emit("Erreur : impossible de démarrer le serveur")
            return False

        # L'hôte se connecte aussi comme client à son propre serveur
        self._client = CollabClient()
        self._client.peer_id = self.peer_id
        self._client.connect(
            host="127.0.0.1", port=port,
            on_message=self._qt_dispatch,
            on_status=lambda s: self.status_changed.emit(s),
        )
        time.sleep(0.3)

        self.is_host = True
        self.active  = True
        ip = self.local_addr
        self.status_changed.emit(
            f"Session hébergée sur {ip}:{port}")
        log.info("CollabSession: server started on port %d", port)

        self._say_hello()
        return True

    def join_server(self, host: str, port: int = DEFAULT_PORT,
                    name: str = "") -> bool:
        """Rejoint un serveur existant. Retourne True si connexion initiée."""
        if self.active:
            self.stop()

        if name:
            self.name = name

        self._client = CollabClient()
        self._client.peer_id = self.peer_id
        self._client.connect(
            host=host, port=port,
            on_message=self._qt_dispatch,
            on_status=lambda s: self.status_changed.emit(s),
        )

        self.is_host = False
        self.active  = True
        self.status_changed.emit(f"Connexion à {host}:{port}…")
        log.info("CollabSession: connecting to %s:%d", host, port)

        QTimer.singleShot(400, self._say_hello)
        return True

    def stop(self):
        """Déconnecte et arrête tout."""
        self.active = False
        if self._client:
            self._client.disconnect()
            self._client = None
        if self._server:
            self._server.stop()
            self._server = None
        self._peers.clear()
        self._locks.clear()
        self.status_changed.emit("Session terminée")
        log.info("CollabSession stopped")

    def update_snapshot(self, shaders: dict, timeline_data: dict):
        """Met à jour le snapshot envoyé aux nouveaux arrivants (hôte seulement)."""
        if self._server:
            self._server.set_snapshot({
                "shaders":  shaders,
                "timeline": timeline_data,
            })

    # ── Envoi d'ops ───────────────────────────────────────────────────────────

    def send_cursor(self, time_: float, track_id: Optional[int] = None):
        """Envoie la position du curseur (throttlé à 20 fps)."""
        self._pending_cursor = {
            "type":     "cursor",
            "peer_id":  self.peer_id,
            "name":     self.name,
            "color":    self.color,
            "time":     round(time_, 4),
            "track_id": track_id,
        }
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _flush_cursor(self):
        if self._pending_cursor:
            self._send(self._pending_cursor)
            self._pending_cursor = None

    def send_shader(self, pass_name: str, code: str):
        self._send({
            "type":    "shader",
            "peer_id": self.peer_id,
            "pass":    pass_name,
            "code":    code,
        })

    def send_uniform(self, name: str, value):
        self._send({
            "type":    "uniform",
            "peer_id": self.peer_id,
            "name":    name,
            "value":   value,
        })

    def send_timeline(self, timeline_data: dict):
        self._send({
            "type":     "timeline",
            "peer_id":  self.peer_id,
            "data":     timeline_data,
        })

    def send_lock(self, track_id: int, locked: bool):
        self._send({
            "type":     "lock",
            "peer_id":  self.peer_id,
            "name":     self.name,
            "track_id": track_id,
            "locked":   locked,
        })

    def send_chat(self, text: str):
        self._send({
            "type":    "chat",
            "peer_id": self.peer_id,
            "name":    self.name,
            "color":   self.color,
            "text":    text,
            "ts":      time.time(),
        })

    def is_track_locked_by_other(self, track_id: int) -> bool:
        locker = self._locks.get(track_id)
        return locker is not None and locker != self.peer_id

    def get_peers(self) -> list[dict]:
        return list(self._peers.values())

    # ── Dispatch interne ──────────────────────────────────────────────────────

    def _send(self, msg: dict):
        if self._client and self._client.connected:
            self._client.send(msg)

    def _qt_dispatch(self, msg: dict):
        """Appelé depuis un thread asyncio → dispatch thread-safe via QTimer.singleShot."""
        QTimer.singleShot(0, lambda m=msg: self._handle_message(m))

    def _handle_message(self, msg: dict):
        """Traite un message reçu (dans le thread Qt)."""
        t       = msg.get("type", "")
        peer_id = msg.get("peer_id", "")

        # Ignore ses propres messages retournés par le serveur
        if peer_id == self.peer_id and t not in ("snapshot", "bye"):
            return

        if t == "hello":
            name  = msg.get("name",  peer_id)
            color = msg.get("color", "#ffffff")
            self._peers[peer_id] = {"peer_id": peer_id, "name": name, "color": color}
            self.peer_joined.emit(peer_id, name, color)

        elif t == "bye":
            self._peers.pop(peer_id, None)
            # Libère les verrous de ce pair
            unlocked = [tid for tid, pid in self._locks.items() if pid == peer_id]
            for tid in unlocked:
                del self._locks[tid]
            self.peer_left.emit(peer_id)

        elif t == "cursor":
            name     = msg.get("name",     peer_id)
            color    = msg.get("color",    "#ffffff")
            time_    = float(msg.get("time", 0.0))
            track_id = msg.get("track_id")
            self.cursor_moved.emit(peer_id, name, color, time_, track_id)

        elif t == "shader":
            self.shader_changed.emit(peer_id, msg.get("pass", "Image"), msg.get("code", ""))

        elif t == "uniform":
            self.uniform_changed.emit(peer_id, msg.get("name", ""), msg.get("value"))

        elif t == "timeline":
            self.timeline_changed.emit(peer_id, msg.get("data", {}))

        elif t == "lock":
            track_id = msg.get("track_id", -1)
            locked   = bool(msg.get("locked", False))
            name     = msg.get("name", peer_id)
            if locked:
                self._locks[track_id] = peer_id
            else:
                self._locks.pop(track_id, None)
            self.track_locked.emit(peer_id, name, track_id, locked)

        elif t == "chat":
            self.chat_received.emit(
                peer_id,
                msg.get("name",  peer_id),
                msg.get("color", "#ffffff"),
                msg.get("text",  ""),
            )

        elif t == "snapshot":
            # Met à jour les pairs connus
            for p in msg.get("peers", []):
                pid = p.get("peer_id", "")
                self._peers[pid] = p
            self.snapshot_received.emit(msg)

        elif t in ("ping", "pong"):
            pass

    def _say_hello(self):
        self._send({
            "type":    "hello",
            "peer_id": self.peer_id,
            "name":    self.name,
            "color":   self.color,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  CollabPanel  —  widget Qt intégrable
# ══════════════════════════════════════════════════════════════════════════════

_BG    = "#0d0f16"
_SURF  = "#12141e"
_SURF2 = "#181a26"
_BORD  = "#1e2235"
_TEXT  = "#c0c8e0"
_MUTED = "#5a6080"
_ACC   = "#4e7fff"
_ACC2  = "#50e0a0"
_WARN  = "#e07850"
_SANS  = "Segoe UI, Arial, sans-serif"
_MONO  = "Cascadia Code, Consolas, monospace"

_SS = f"""
    QWidget           {{ background:{_BG}; color:{_TEXT}; font:9px '{_SANS}'; }}
    QLineEdit         {{ background:{_SURF}; color:{_TEXT}; border:1px solid {_BORD};
                         border-radius:3px; padding:3px 7px; }}
    QLineEdit:focus   {{ border-color:{_ACC}88; }}
    QTextEdit         {{ background:{_SURF2}; color:{_TEXT}; border:1px solid {_BORD};
                         border-radius:3px; padding:4px; font:9px '{_SANS}'; }}
    QSpinBox          {{ background:{_SURF}; color:{_TEXT}; border:1px solid {_BORD};
                         border-radius:3px; padding:2px 6px; }}
    QScrollBar:vertical   {{ background:{_SURF}; width:6px; border:none; }}
    QScrollBar::handle:vertical {{ background:{_BORD}; border-radius:3px; min-height:16px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""


def _btn(label: str, color: str = _ACC, small: bool = False) -> QPushButton:
    fs, pad = ("8px", "2px 8px") if small else ("9px", "4px 12px")
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{_SURF};color:{color};border:1px solid {color}44;"
        f"border-radius:3px;padding:{pad};font:{fs} '{_SANS}';}}"
        f"QPushButton:hover{{background:{color}22;border-color:{color}88;}}"
        f"QPushButton:pressed{{background:{color}33;}}"
        f"QPushButton:disabled{{color:{_MUTED};border-color:{_BORD};}}"
    )
    return b


def _lbl(text: str, color: str = _TEXT, size: str = "9px",
         bold: bool = False, wrap: bool = False) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet(f"color:{color};font:{'bold ' if bold else ''}{size} '{_SANS}';")
    if wrap:
        w.setWordWrap(True)
    return w


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:{_BORD};max-height:1px;")
    return f


class _PeerDot(QLabel):
    """Petit cercle coloré + nom du pair."""
    def __init__(self, name: str, color: str, parent=None):
        super().__init__(parent)
        self._name  = name
        self._color = color
        self._refresh()

    def _refresh(self):
        self.setText(f"● {self._name}")
        self.setStyleSheet(
            f"color:{self._color};font:bold 9px '{_SANS}';"
            f"background:{self._color}18;border-radius:3px;padding:2px 6px;")

    def update_info(self, name: str, color: str):
        self._name  = name
        self._color = color
        self._refresh()


class CollabPanel(QWidget):
    """
    Panneau de co-édition : connexion, liste des pairs, chat.
    S'intègre comme QDockWidget ou onglet dans MainWindow.
    """

    def __init__(self, session: CollabSession, parent=None):
        super().__init__(parent)
        self._session = session
        self._peer_dots: dict[str, _PeerDot] = {}
        self.setStyleSheet(_SS)
        self._build_ui()
        self._connect_signals()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── En-tête identité ──────────────────────────────────────────────────
        id_row = QHBoxLayout()
        id_row.addWidget(_lbl("Pseudo :", _MUTED, "8px"))
        self._name_edit = QLineEdit(self._session.name)
        self._name_edit.setFixedWidth(120)
        self._name_edit.textChanged.connect(self._on_name_changed)
        id_row.addWidget(self._name_edit)
        self._color_dot = QLabel("●")
        self._color_dot.setStyleSheet(
            f"color:{self._session.color};font:16px;padding:0 4px;")
        id_row.addWidget(self._color_dot)
        id_row.addStretch()
        root.addLayout(id_row)

        root.addWidget(_sep())

        # ── Tabs Héberger / Rejoindre ─────────────────────────────────────────
        self._conn_tabs = QTabWidget()
        self._conn_tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {_BORD};background:{_SURF};}}"
            f"QTabBar::tab{{background:{_SURF};color:{_MUTED};padding:4px 12px;"
            f"border:1px solid {_BORD};margin-right:2px;font:8px '{_SANS}';}}"
            f"QTabBar::tab:selected{{color:{_TEXT};border-bottom:2px solid {_ACC};}}"
        )

        # Tab héberger
        host_tab = QWidget()
        ht_lay   = QVBoxLayout(host_tab)
        ht_lay.setContentsMargins(8, 8, 8, 8)
        ht_lay.setSpacing(6)

        port_row = QHBoxLayout()
        port_row.addWidget(_lbl("Port LAN :", _MUTED, "8px"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(DEFAULT_PORT)
        self._port_spin.setFixedWidth(70)
        port_row.addWidget(self._port_spin)
        port_row.addStretch()
        ht_lay.addLayout(port_row)

        self._host_btn = _btn("📡 Héberger la session", _ACC2)
        self._host_btn.clicked.connect(self._on_host_click)
        ht_lay.addWidget(self._host_btn)

        self._addr_lbl = _lbl("", _MUTED, "8px")
        self._addr_lbl.setWordWrap(True)
        ht_lay.addWidget(self._addr_lbl)
        self._conn_tabs.addTab(host_tab, "📡 Héberger")

        # Tab rejoindre
        join_tab = QWidget()
        jt_lay   = QVBoxLayout(join_tab)
        jt_lay.setContentsMargins(8, 8, 8, 8)
        jt_lay.setSpacing(6)

        host_row = QHBoxLayout()
        host_row.addWidget(_lbl("Adresse :", _MUTED, "8px"))
        self._host_edit = QLineEdit("192.168.1.")
        host_row.addWidget(self._host_edit, 1)
        host_row.addWidget(_lbl(":", _MUTED, "8px"))
        self._jport_spin = QSpinBox()
        self._jport_spin.setRange(1024, 65535)
        self._jport_spin.setValue(DEFAULT_PORT)
        self._jport_spin.setFixedWidth(70)
        host_row.addWidget(self._jport_spin)
        jt_lay.addLayout(host_row)

        self._join_btn = _btn("🔗 Rejoindre", _ACC)
        self._join_btn.clicked.connect(self._on_join_click)
        jt_lay.addWidget(self._join_btn)
        self._conn_tabs.addTab(join_tab, "🔗 Rejoindre")

        root.addWidget(self._conn_tabs)

        # Bouton déconnexion
        self._stop_btn = _btn("✕ Quitter la session", _WARN, small=True)
        self._stop_btn.hide()
        self._stop_btn.clicked.connect(self._on_stop_click)
        root.addWidget(self._stop_btn)

        # ── Statut ────────────────────────────────────────────────────────────
        self._status_lbl = _lbl("Hors ligne", _MUTED, "8px")
        root.addWidget(self._status_lbl)

        root.addWidget(_sep())

        # ── Participants ──────────────────────────────────────────────────────
        root.addWidget(_lbl("Participants", _TEXT, "9px", bold=True))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(80)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_SURF};}}")
        self._peers_widget = QWidget()
        self._peers_widget.setStyleSheet(f"background:{_SURF};")
        self._peers_lay = QVBoxLayout(self._peers_widget)
        self._peers_lay.setContentsMargins(4, 4, 4, 4)
        self._peers_lay.setSpacing(3)
        self._peers_lay.addStretch()
        scroll.setWidget(self._peers_widget)
        root.addWidget(scroll)

        # Soi-même
        self._self_dot = _PeerDot(
            f"{self._session.name} (vous)", self._session.color)
        self._peers_lay.insertWidget(0, self._self_dot)

        root.addWidget(_sep())

        # ── Chat ──────────────────────────────────────────────────────────────
        root.addWidget(_lbl("Chat", _TEXT, "9px", bold=True))

        self._chat_view = QTextEdit()
        self._chat_view.setReadOnly(True)
        self._chat_view.setFixedHeight(140)
        root.addWidget(self._chat_view)

        msg_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Message… (Entrée pour envoyer)")
        self._chat_input.returnPressed.connect(self._on_chat_send)
        self._chat_input.setEnabled(False)
        msg_row.addWidget(self._chat_input, 1)
        send_btn = _btn("→", _ACC, small=True)
        send_btn.setFixedWidth(28)
        send_btn.clicked.connect(self._on_chat_send)
        msg_row.addWidget(send_btn)
        root.addLayout(msg_row)

        root.addStretch()

    def _connect_signals(self):
        s = self._session
        s.peer_joined.connect(self._on_peer_joined)
        s.peer_left.connect(self._on_peer_left)
        s.chat_received.connect(self._on_chat_received)
        s.status_changed.connect(self._on_status_changed)

    # ── Connexion ─────────────────────────────────────────────────────────────

    def _on_name_changed(self, name: str):
        self._session.name = name.strip() or f"user-{self._session.peer_id}"
        self._self_dot.update_info(
            f"{self._session.name} (vous)", self._session.color)

    def _on_host_click(self):
        port = self._port_spin.value()
        ok   = self._session.start_server(port=port, name=self._name_edit.text().strip())
        if ok:
            ip = self._session.local_addr
            self._addr_lbl.setText(
                f"Partagez cette adresse :\n{ip}:{port}")
            self._addr_lbl.setStyleSheet(f"color:{_ACC2};font:bold 8px '{_SANS}';")
            self._host_btn.setEnabled(False)
            self._join_btn.setEnabled(False)
            self._stop_btn.show()
            self._chat_input.setEnabled(True)

    def _on_join_click(self):
        host = self._host_edit.text().strip()
        port = self._jport_spin.value()
        if not host:
            return
        self._session.join_server(host=host, port=port,
                                   name=self._name_edit.text().strip())
        self._host_btn.setEnabled(False)
        self._join_btn.setEnabled(False)
        self._stop_btn.show()
        self._chat_input.setEnabled(True)

    def _on_stop_click(self):
        self._session.stop()
        self._host_btn.setEnabled(True)
        self._join_btn.setEnabled(True)
        self._stop_btn.hide()
        self._chat_input.setEnabled(False)
        self._addr_lbl.setText("")
        # Supprime les pairs
        for dot in list(self._peer_dots.values()):
            self._peers_lay.removeWidget(dot)
            dot.deleteLater()
        self._peer_dots.clear()

    # ── Chat ──────────────────────────────────────────────────────────────────

    def _on_chat_send(self):
        text = self._chat_input.text().strip()
        if not text:
            return
        self._chat_input.clear()
        self._session.send_chat(text)
        # Affiche localement
        self._append_chat(
            self._session.name, self._session.color, text, is_self=True)

    def _on_chat_received(self, peer_id: str, name: str, color: str, text: str):
        self._append_chat(name, color, text, is_self=False)

    def _append_chat(self, name: str, color: str, text: str, is_self: bool = False):
        align = "right" if is_self else "left"
        bg    = f"{color}22"
        self._chat_view.append(
            f'<div style="text-align:{align};margin:3px 0;">'
            f'<span style="color:{color};font-weight:bold;">{name}</span> '
            f'<span style="color:{_MUTED};font-size:8px;">'
            f'{time.strftime("%H:%M")}</span><br>'
            f'<span style="background:{bg};border-radius:4px;'
            f'padding:2px 6px;color:{_TEXT};">{text}</span></div>'
        )

    # ── Pairs ─────────────────────────────────────────────────────────────────

    def _on_peer_joined(self, peer_id: str, name: str, color: str):
        if peer_id not in self._peer_dots:
            dot = _PeerDot(name, color)
            self._peers_lay.insertWidget(
                self._peers_lay.count() - 1, dot)
            self._peer_dots[peer_id] = dot
        self._append_chat("système", _MUTED,
                          f"→ {name} a rejoint la session", is_self=False)

    def _on_peer_left(self, peer_id: str):
        dot = self._peer_dots.pop(peer_id, None)
        if dot:
            self._peers_lay.removeWidget(dot)
            dot.deleteLater()

    def _on_status_changed(self, msg: str):
        self._status_lbl.setText(msg)
        color = _ACC2 if "Connecté" in msg or "hébergée" in msg else _MUTED
        self._status_lbl.setStyleSheet(f"color:{color};font:8px '{_SANS}';")


# ══════════════════════════════════════════════════════════════════════════════
#  CollabCursorOverlay  —  curseurs nommés dans la timeline
# ══════════════════════════════════════════════════════════════════════════════

class CollabCursorOverlay:
    """
    Gère les curseurs des pairs dans la TimelineWidget.
    Appelé par MainWindow pour dessiner les curseurs distants sur le canvas.

    Utilisation (dans TimelineCanvas.paintEvent) :
        overlay.draw(painter, pixels_per_second, scroll_x, canvas_height)
    """

    def __init__(self):
        self._cursors: dict[str, dict] = {}   # peer_id → {name, color, time, track_id}

    def update_cursor(self, peer_id: str, name: str, color: str,
                      time_: float, track_id):
        self._cursors[peer_id] = {
            "name": name, "color": color,
            "time": time_, "track_id": track_id,
        }

    def remove_cursor(self, peer_id: str):
        self._cursors.pop(peer_id, None)

    def draw(self, painter, pixels_per_second: float, scroll_x: float,
             canvas_height: int):
        """Dessine les curseurs sur le painter de la timeline canvas."""
        from PyQt6.QtGui import QPen, QBrush, QPainter, QColor, QFont
        from PyQt6.QtCore import Qt, QRect

        for peer_id, cur in self._cursors.items():
            x = int(cur["time"] * pixels_per_second - scroll_x)
            if x < 0 or x > 4000:
                continue

            color = QColor(cur["color"])
            pen   = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(x, 0, x, canvas_height)

            # Badge nom
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            name_str = cur["name"][:12]
            fm       = painter.fontMetrics()
            tw       = fm.horizontalAdvance(name_str) + 8
            badge    = QRect(x, 2, tw, 16)
            painter.drawRoundedRect(badge, 3, 3)
            painter.setPen(QPen(QColor("#000000")))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, name_str)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_collab_session(parent=None) -> tuple[CollabSession, CollabPanel, CollabCursorOverlay]:
    """Crée et retourne (session, panel, cursor_overlay)."""
    session = CollabSession(parent)
    panel   = CollabPanel(session, parent)
    overlay = CollabCursorOverlay()
    return session, panel, overlay
