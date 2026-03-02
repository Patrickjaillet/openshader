"""
osc_engine.py
-------------
v1.0 — Support OSC (Open Sound Control) pour OpenShader / DemoMaker.

Compatible TouchOSC, Resolume Arena, Pure Data (via mrpeach/udpreceive),
ainsi que tout équipement supportant OSC/UDP standard.

Architecture :
  - OscMapping    : association adresse OSC → uniform GLSL + scaling
  - OscEngine     : serveur UDP dans un thread, émission vers équipements externes
  - Signaux PyQt6 : uniform_changed(name, value), message_received(address, args)

Usage minimal :
    engine = OscEngine()
    engine.add_mapping('/1/fader1', 'uBrightness', lo=0.0, hi=1.0)
    engine.uniform_changed.connect(lambda name, val: shader.set_uniform(name, val))
    engine.start(in_port=9000)

Protocole entrant :
  - Adresse OSC  → valeur float normalisée [lo, hi]
  - Valeur brute peut être float [0..1], int [0..127] ou bool (0/1)

Émission OSC (sortie) :
    engine.send('/dmx/ch1', 0.75, host='192.168.1.10', port=7000)

Présets de configuration par scène :
    engine.to_dict()   → dict JSON-sérialisable
    engine.from_dict() → restauration complète
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)

# ── Import conditionnel de python-osc ────────────────────────────────────────

try:
    from pythonosc.dispatcher import Dispatcher as _Dispatcher        # type: ignore
    from pythonosc.osc_server import BlockingOSCUDPServer             # type: ignore
    from pythonosc.udp_client import SimpleUDPClient                  # type: ignore
    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _Dispatcher = None                                                 # type: ignore
    BlockingOSCUDPServer = None                                        # type: ignore
    SimpleUDPClient = None                                             # type: ignore
    _PYTHONOSC_AVAILABLE = False
    log.info("python-osc non installé — OSC désactivé (pip install python-osc)")


# ── Modèle de données ────────────────────────────────────────────────────────

@dataclass
class OscMapping:
    """Association adresse OSC → uniform GLSL."""
    address:  str                      # ex: '/1/fader1' ou '/resolume/layer1/clip1'
    uniform:  str                      # nom du uniform cible dans le shader
    lo:       float       = 0.0        # valeur correspondant à l'entrée OSC 0.0
    hi:       float       = 1.0        # valeur correspondant à l'entrée OSC 1.0
    curve:    str         = 'linear'   # 'linear' | 'log' | 'exp'
    arg_index: int        = 0          # index de l'argument OSC à utiliser (défaut 0)

    def scale(self, raw: float) -> float:
        """Convertit une valeur OSC brute [0..1] vers [lo..hi]."""
        t = float(raw)
        # Normalise si la valeur ressemble à un entier MIDI-style [0..127]
        if t > 1.0:
            t = t / 127.0
        t = max(0.0, min(1.0, t))

        if self.curve == 'log':
            import math
            t = math.log1p(t * (math.e - 1))
        elif self.curve == 'exp':
            t = t * t

        return self.lo + t * (self.hi - self.lo)


# ── Moteur OSC ───────────────────────────────────────────────────────────────

class OscEngine(QObject):
    """
    Serveur OSC UDP en écoute + client pour l'émission.

    Signaux
    -------
    uniform_changed  (str, float)       — (uniform_name, value)
    message_received (str, list)        — (address, [args...])  brut, pour debug/learn
    server_started   (str, int)         — (host, port)
    server_stopped   ()
    """

    uniform_changed  = pyqtSignal(str, float)
    message_received = pyqtSignal(str, list)   # adresse + liste d'args bruts
    server_started   = pyqtSignal(str, int)
    server_stopped   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mappings:   list[OscMapping]       = []
        self._clients:    dict[tuple, Any]        = {}   # (host, port) → SimpleUDPClient
        self._server                              = None
        self._thread:     threading.Thread | None = None
        self._running:    bool                    = False
        self._in_host:    str                     = '0.0.0.0'
        self._in_port:    int                     = 9000
        self._learn_mode: bool                    = False

    # ── Disponibilité ────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return _PYTHONOSC_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def in_port(self) -> int:
        return self._in_port

    @property
    def in_host(self) -> str:
        return self._in_host

    # ── Mappings ─────────────────────────────────────────────────────────────

    def add_mapping(self, address: str, uniform: str,
                    lo: float = 0.0, hi: float = 1.0,
                    curve: str = 'linear', arg_index: int = 0) -> OscMapping:
        m = OscMapping(address=address, uniform=uniform,
                       lo=lo, hi=hi, curve=curve, arg_index=arg_index)
        self._mappings.append(m)
        log.debug("OSC mapping ajouté : %s → %s [%.2f..%.2f]", address, uniform, lo, hi)
        if self._running and _PYTHONOSC_AVAILABLE and self._server:
            self._register_handler(m)
        return m

    def remove_mapping(self, mapping: OscMapping):
        if mapping in self._mappings:
            self._mappings.remove(mapping)

    def clear_mappings(self):
        self._mappings.clear()

    def get_mappings(self) -> list[OscMapping]:
        return list(self._mappings)

    def set_mappings(self, mappings: list[OscMapping]):
        self._mappings = list(mappings)

    # ── Sérialisation (compatible avec le système de presets DemoMaker) ──────

    def to_dict(self) -> dict:
        return {
            "in_host": self._in_host,
            "in_port": self._in_port,
            "mappings": [
                {
                    "address":   m.address,
                    "uniform":   m.uniform,
                    "lo":        m.lo,
                    "hi":        m.hi,
                    "curve":     m.curve,
                    "arg_index": m.arg_index,
                }
                for m in self._mappings
            ],
        }

    def from_dict(self, data: dict):
        self._in_host = data.get("in_host", "0.0.0.0")
        self._in_port = int(data.get("in_port", 9000))
        self._mappings = [
            OscMapping(
                address=d.get("address", "/"),
                uniform=d.get("uniform", ""),
                lo=float(d.get("lo", 0.0)),
                hi=float(d.get("hi", 1.0)),
                curve=d.get("curve", "linear"),
                arg_index=int(d.get("arg_index", 0)),
            )
            for d in data.get("mappings", [])
        ]

    # ── Démarrage / Arrêt ────────────────────────────────────────────────────

    def start(self, host: str = '0.0.0.0', port: int = 9000):
        """Lance le serveur OSC UDP dans un thread dédié."""
        if not _PYTHONOSC_AVAILABLE:
            log.warning("python-osc non disponible — impossible de démarrer l'OSC.")
            return
        self.stop()
        self._in_host = host
        self._in_port = port
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, args=(host, port),
            daemon=True, name="OscEngine"
        )
        self._thread.start()

    def stop(self):
        """Arrête proprement le serveur OSC."""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.server_stopped.emit()

    # ── Émission OSC (sortie) ─────────────────────────────────────────────────

    def send(self, address: str, *args, host: str = '127.0.0.1', port: int = 7000):
        """
        Envoie un message OSC vers un équipement externe.

        Cas d'usage typiques :
          - Ponts DMX (OLA, LX Protocol)
          - Consoles d'éclairage (ETC Eos, grandMA)
          - Resolume en mode pilotage
          - Pure Data, Max/MSP

        engine.send('/dmx/ch1', 0.75, host='192.168.1.10', port=7000)
        engine.send('/resolume/layer/1/clip/1/connect', 1, host='192.168.1.20', port=7000)
        """
        if not _PYTHONOSC_AVAILABLE:
            log.warning("python-osc non disponible — émission OSC impossible.")
            return
        key = (host, port)
        if key not in self._clients:
            self._clients[key] = SimpleUDPClient(host, port)
        try:
            self._clients[key].send_message(address, list(args) if len(args) > 1 else args[0])
            log.debug("OSC envoyé → %s:%d  %s  %s", host, port, address, args)
        except (OSError, Exception) as e:
            log.error("Erreur émission OSC vers %s:%d : %s", host, port, e)

    # ── Mode OSC Learn ────────────────────────────────────────────────────────

    def start_learn(self):
        """Active le mode OSC Learn : le prochain message reçu est capturé."""
        self._learn_mode = True
        log.debug("OSC Learn activé — en attente du prochain message...")

    def stop_learn(self):
        self._learn_mode = False

    # ── Thread serveur ────────────────────────────────────────────────────────

    def _register_handler(self, mapping: OscMapping):
        """Enregistre un handler pour une adresse OSC dans le dispatcher actif."""
        if self._server is None:
            return
        disp = self._server.dispatcher
        disp.map(mapping.address, self._make_handler(mapping))

    def _make_handler(self, mapping: OscMapping):
        """Crée une closure PyQt-safe pour un mapping donné."""
        def _handler(address, *args):
            try:
                raw = args[mapping.arg_index] if args else 0.0
                value = mapping.scale(raw)
                self.uniform_changed.emit(mapping.uniform, value)
                log.debug("OSC → uniform '%s' = %.4f (raw=%s)", mapping.uniform, value, raw)
            except (IndexError, TypeError, ValueError) as e:
                log.warning("OSC handler erreur pour %s : %s", address, e)
        return _handler

    def _make_wildcard_handler(self):
        """Handler générique pour les adresses non mappées (learn + debug)."""
        def _handler(address, *args):
            self.message_received.emit(address, list(args))
            if self._learn_mode:
                log.debug("OSC Learn capturé : %s  %s", address, args)
                self._learn_mode = False
        return _handler

    def _run(self, host: str, port: int):
        """Thread principal du serveur OSC."""
        try:
            disp = _Dispatcher()

            # Enregistre tous les mappings existants
            for m in self._mappings:
                disp.map(m.address, self._make_handler(m))

            # Handler générique pour OSC Learn et debug
            disp.set_default_handler(self._make_wildcard_handler())

            server = BlockingOSCUDPServer((host, port), disp)
            self._server = server
            self.server_started.emit(host, port)
            log.info("Serveur OSC démarré sur %s:%d", host, port)

            while self._running:
                server.handle_request()  # timeout géré par select() interne

        except OSError as e:
            log.error("Impossible de démarrer le serveur OSC sur %s:%d — %s", host, port, e)
        except Exception as e:
            log.error("Erreur serveur OSC : %s", e)
        finally:
            self._running = False
            self.server_stopped.emit()
            log.info("Serveur OSC arrêté.")
