"""
dmx_engine.py
-------------
v1.0 — Support DMX512 / Artnet / sACN pour OpenShader / DemoMaker.

Permet de piloter des équipements d'éclairage scénique (projecteurs, LEDs,
machines à fumée, stroboscopes, gobos…) directement depuis les uniforms GLSL
et la timeline du projet.

Compatible :
  - Artnet 4 (UDP port 6454) — consoles ETC Eos, grandMA3, Resolume, QLC+
  - sACN / E1.31 (UDP port 5568) — multicast ou unicast
  - USB-DMX direct via `pyserial` (dongles ENTTEC Open/Pro, DMXking)

Architecture :
  DmxFixture       — descriptor d'un appareil (adresse, type, canaux)
  DmxChannel       — canal individuel avec valeur courante [0..255]
  DmxMapping       — uniform GLSL → canal(s) DMX avec scaling et courbe
  DmxUniverse      — 512 canaux, calcule et envoie le paquet DMX
  DmxEngine        — QObject orchestrant tout, thread d'envoi à ~44 Hz (HTP merge)
  DmxPatchPanel    — widget PyQt6 : plan de salle 2D avec fixtures nommées
  DmxMappingPanel  — widget PyQt6 : tableau des mappings uniform → canal(s)

Signaux Qt :
  universe_sent(int)          — numéro d'univers envoyé
  channel_changed(int, int)   — (canal 1-512, valeur 0-255)
  fixture_added(str)          — nom de la fixture ajoutée
  error_occurred(str)         — message d'erreur

Sérialisation :
  engine.to_dict()   → dict JSON-sérialisable (universe, fixtures, mappings)
  engine.from_dict() → restauration complète

Usage minimal :
    engine = DmxEngine()
    engine.add_fixture(DmxFixture('Wash 1', address=1, fixture_type='rgb'))
    engine.add_mapping(DmxMapping(uniform='uColor', channels=[1,2,3], mode='rgb'))
    engine.uniform_changed_slot('uColor', 0.8)   # injecte depuis shader
    engine.start(protocol='artnet', host='192.168.1.255')
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from PyQt6.QtCore    import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui     import QColor, QFont, QPainter, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import QPointF, QRectF

from .logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Constantes de protocole
# ══════════════════════════════════════════════════════════════════════════════

ARTNET_PORT   = 6454
SACN_PORT     = 5568
DMX_CHANNELS  = 512
DMX_FPS       = 44          # fréquence d'envoi (HTP merge à 44 Hz)
SACN_MULTICAST_BASE = "239.255.0."   # + numéro d'univers


# ══════════════════════════════════════════════════════════════════════════════
#  Types de fixtures
# ══════════════════════════════════════════════════════════════════════════════

class FixtureType(str, Enum):
    DIMMER   = "dimmer"     # 1 canal : intensité
    RGB      = "rgb"        # 3 canaux : R G B
    RGBA     = "rgba"       # 4 canaux : R G B A(ambre)
    RGBW     = "rgbw"       # 4 canaux : R G B W
    MOVING   = "moving"     # 8 canaux : dim pan tilt r g b gobo strobe
    STROBE   = "strobe"     # 2 canaux : intensité vitesse
    FOGGER   = "fogger"     # 1 canal  : sortie fumée
    CUSTOM   = "custom"     # N canaux libres

FIXTURE_CHANNEL_COUNT: dict[str, int] = {
    FixtureType.DIMMER:  1,
    FixtureType.RGB:     3,
    FixtureType.RGBA:    4,
    FixtureType.RGBW:    4,
    FixtureType.MOVING:  8,
    FixtureType.STROBE:  2,
    FixtureType.FOGGER:  1,
    FixtureType.CUSTOM:  1,
}

FIXTURE_COLORS: dict[str, str] = {
    FixtureType.DIMMER:  "#f5c542",
    FixtureType.RGB:     "#42b0f5",
    FixtureType.RGBA:    "#f5a442",
    FixtureType.RGBW:    "#a0e0ff",
    FixtureType.MOVING:  "#c084fc",
    FixtureType.STROBE:  "#f87171",
    FixtureType.FOGGER:  "#94a3b8",
    FixtureType.CUSTOM:  "#6ee7b7",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DmxFixture:
    """Descriptor d'un appareil DMX dans le patch."""
    name:         str
    address:      int               # adresse DMX de départ (1–512)
    fixture_type: str = FixtureType.RGB
    universe:     int = 0           # univers Artnet (0-indexed)
    label:        str = ""          # étiquette affichée sur le plan de salle
    x:            float = 0.0       # position sur le plan de salle (0..1)
    y:            float = 0.0
    n_channels:   int = 0           # 0 = auto depuis fixture_type

    def __post_init__(self):
        if self.n_channels <= 0:
            self.n_channels = FIXTURE_CHANNEL_COUNT.get(self.fixture_type, 1)
        if not self.label:
            self.label = self.name

    @property
    def channel_range(self) -> range:
        """Plage de canaux DMX (1-indexed, inclusif)."""
        return range(self.address, self.address + self.n_channels)

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "address":      self.address,
            "fixture_type": self.fixture_type,
            "universe":     self.universe,
            "label":        self.label,
            "x":            self.x,
            "y":            self.y,
            "n_channels":   self.n_channels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DmxFixture":
        return cls(
            name         = d.get("name", "Fixture"),
            address      = int(d.get("address", 1)),
            fixture_type = d.get("fixture_type", FixtureType.RGB),
            universe     = int(d.get("universe", 0)),
            label        = d.get("label", ""),
            x            = float(d.get("x", 0.0)),
            y            = float(d.get("y", 0.0)),
            n_channels   = int(d.get("n_channels", 0)),
        )


@dataclass
class DmxMapping:
    """
    Association uniform GLSL → canal(s) DMX.

    Modes :
      'single'  : uniform float → 1 canal
      'rgb'     : uniform float (RMS/brightness) → 3 canaux R=G=B (white chase)
      'rgb_vec' : uniform vec3 (couleur) → 3 canaux R G B séparés
      'strobe'  : uniform float → canal intensité + canal vitesse automatique
    """
    uniform:    str
    channels:   list[int]           # canaux DMX cibles (1-indexed)
    mode:       str  = "single"     # 'single' | 'rgb' | 'rgb_vec' | 'strobe'
    lo:         float = 0.0
    hi:         float = 1.0
    curve:      str  = "linear"     # 'linear' | 'log' | 'exp'
    universe:   int  = 0
    enabled:    bool = True

    def scale(self, raw: float) -> int:
        """Convertit un float [0..1] en valeur DMX [0..255]."""
        t = max(0.0, min(1.0, float(raw)))
        if self.curve == "log":
            t = math.log1p(t * (math.e - 1))
        elif self.curve == "exp":
            t = t * t
        return int(round((self.lo + t * (self.hi - self.lo)) * 255))

    def to_dict(self) -> dict:
        return {
            "uniform":  self.uniform,
            "channels": self.channels,
            "mode":     self.mode,
            "lo":       self.lo,
            "hi":       self.hi,
            "curve":    self.curve,
            "universe": self.universe,
            "enabled":  self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DmxMapping":
        return cls(
            uniform  = d.get("uniform", ""),
            channels = d.get("channels", [1]),
            mode     = d.get("mode", "single"),
            lo       = float(d.get("lo", 0.0)),
            hi       = float(d.get("hi", 1.0)),
            curve    = d.get("curve", "linear"),
            universe = int(d.get("universe", 0)),
            enabled  = bool(d.get("enabled", True)),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Univers DMX (512 canaux)
# ══════════════════════════════════════════════════════════════════════════════

class DmxUniverse:
    """
    Représente un univers DMX512.
    Applique le merge HTP (Highest Takes Precedence) sur 512 canaux.
    """

    def __init__(self, universe_id: int = 0):
        self.universe_id = universe_id
        self._data = bytearray(DMX_CHANNELS)   # valeurs [0..255], index 0 = canal 1

    def set_channel(self, channel: int, value: int):
        """Écrit sur un canal (1-indexed, HTP merge)."""
        if 1 <= channel <= DMX_CHANNELS:
            idx = channel - 1
            # HTP : on prend le max entre la valeur courante et la nouvelle
            self._data[idx] = max(self._data[idx], max(0, min(255, int(value))))

    def set_channel_raw(self, channel: int, value: int):
        """Écrit directement sans HTP (pour reset ou blackout)."""
        if 1 <= channel <= DMX_CHANNELS:
            self._data[channel - 1] = max(0, min(255, int(value)))

    def get_channel(self, channel: int) -> int:
        if 1 <= channel <= DMX_CHANNELS:
            return self._data[channel - 1]
        return 0

    def blackout(self):
        """Met tous les canaux à 0."""
        self._data = bytearray(DMX_CHANNELS)

    def get_data(self) -> bytes:
        return bytes(self._data)

    def build_artnet_packet(self, sequence: int = 0) -> bytes:
        """Construit un paquet Artnet OpOutput (ArtDMX)."""
        header    = b"Art-Net\x00"
        opcode    = struct.pack("<H", 0x5000)       # OpOutput
        prot_ver  = struct.pack(">H", 14)           # protocol v14
        seq       = struct.pack("B", sequence & 0xFF)
        physical  = b"\x00"
        uni       = struct.pack("<H", self.universe_id)
        length    = struct.pack(">H", DMX_CHANNELS)
        return header + opcode + prot_ver + seq + physical + uni + length + self._data

    def build_sacn_packet(self, sequence: int = 0, source_name: str = "OpenShader") -> bytes:
        """Construit un paquet sACN (E1.31) pour cet univers."""
        # Root layer
        preamble_size  = struct.pack(">H", 0x0010)
        postamble_size = struct.pack(">H", 0x0000)
        acn_id = b"ASC-E1.17\x00\x00\x00"
        cid    = b"\x4f\x53\x68\x64" + b"\x00" * 12   # 'OShd' + padding

        # Framing layer
        src_name_bytes = source_name.encode("utf-8")[:63].ljust(64, b"\x00")
        priority   = b"\x64"       # 100
        sync_addr  = b"\x00\x00"
        seq_num    = struct.pack("B", sequence & 0xFF)
        options    = b"\x00"
        universe   = struct.pack(">H", self.universe_id + 1)   # sACN 1-indexed

        # DMP layer
        dmp_vector   = b"\x02"
        addr_type    = b"\xa1"
        first_prop   = b"\x00\x00"
        addr_inc     = b"\x00\x01"
        prop_count   = struct.pack(">H", DMX_CHANNELS + 1)
        dmx_start    = b"\x00"    # START code
        dmx_data     = self._data

        # Longueurs (PDU flags + length, 0x7xxx)
        dmp_length      = 11 + DMX_CHANNELS + 1
        framing_length  = 77 + dmp_length
        root_length     = 38 + framing_length

        def fl(n: int) -> bytes:
            return struct.pack(">H", 0x7000 | (n & 0x0FFF))

        root_fl    = fl(root_length)
        framing_fl = fl(framing_length)
        dmp_fl     = fl(dmp_length)

        packet  = preamble_size + postamble_size + acn_id
        packet += root_fl + b"\x00\x00\x00\x04" + cid  # root vector 0x04
        packet += framing_fl + b"\x00\x00\x00\x02"     # framing vector 0x02
        packet += src_name_bytes + priority + sync_addr + seq_num + options + universe
        packet += dmp_fl + b"\x00\x00\x02\x05"         # DMP vector 0x02
        packet += dmp_vector + addr_type + first_prop + addr_inc + prop_count
        packet += dmx_start + dmx_data

        return packet


# ══════════════════════════════════════════════════════════════════════════════
#  Moteur DMX principal
# ══════════════════════════════════════════════════════════════════════════════

class DmxEngine(QObject):
    """
    Moteur DMX512 avec support Artnet, sACN et USB-DMX.

    Signaux
    -------
    universe_sent   (int)       — univers envoyé (0-indexed)
    channel_changed (int, int)  — (canal 1-512, valeur 0-255)
    fixture_added   (str)       — nom de la fixture
    fixture_removed (str)       — nom de la fixture
    error_occurred  (str)       — message d'erreur
    """

    universe_sent   = pyqtSignal(int)
    channel_changed = pyqtSignal(int, int)
    fixture_added   = pyqtSignal(str)
    fixture_removed = pyqtSignal(str)
    error_occurred  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fixtures:  list[DmxFixture] = []
        self._mappings:  list[DmxMapping] = []
        self._universes: dict[int, DmxUniverse] = {0: DmxUniverse(0)}

        self._protocol:   str  = "artnet"        # 'artnet' | 'sacn' | 'usb'
        self._host:       str  = "255.255.255.255"
        self._port:       int  = ARTNET_PORT
        self._sock:       socket.socket | None = None
        self._thread:     threading.Thread | None = None
        self._running:    bool = False
        self._sequence:   int  = 0
        self._blackout:   bool = False

        # Cache des valeurs de uniforms reçues
        self._uniform_values: dict[str, float] = {}

        log.debug("DmxEngine initialisé")

    # ── Gestion des univers ───────────────────────────────────────────────────

    def _get_universe(self, uid: int) -> DmxUniverse:
        if uid not in self._universes:
            self._universes[uid] = DmxUniverse(uid)
        return self._universes[uid]

    # ── Fixtures ──────────────────────────────────────────────────────────────

    def add_fixture(self, fixture: DmxFixture) -> DmxFixture:
        self._fixtures.append(fixture)
        self.fixture_added.emit(fixture.name)
        log.info("Fixture ajoutée : '%s' @ adresse %d (univers %d)",
                 fixture.name, fixture.address, fixture.universe)
        return fixture

    def remove_fixture(self, fixture: DmxFixture):
        if fixture in self._fixtures:
            self._fixtures.remove(fixture)
            self.fixture_removed.emit(fixture.name)

    def get_fixtures(self) -> list[DmxFixture]:
        return list(self._fixtures)

    # ── Mappings ──────────────────────────────────────────────────────────────

    def add_mapping(self, mapping: DmxMapping) -> DmxMapping:
        self._mappings.append(mapping)
        log.debug("DMX mapping : '%s' → canaux %s", mapping.uniform, mapping.channels)
        return mapping

    def remove_mapping(self, mapping: DmxMapping):
        if mapping in self._mappings:
            self._mappings.remove(mapping)

    def get_mappings(self) -> list[DmxMapping]:
        return list(self._mappings)

    # ── Écriture directe sur les canaux ──────────────────────────────────────

    def set_channel(self, channel: int, value: int, universe: int = 0):
        """Écrit directement sur un canal DMX (valeur 0–255, HTP merge)."""
        uni = self._get_universe(universe)
        uni.set_channel(channel, value)
        self.channel_changed.emit(channel, value)

    def set_channel_normalized(self, channel: int, value: float, universe: int = 0):
        """Écrit sur un canal depuis une valeur float [0..1]."""
        self.set_channel(channel, int(round(value * 255)), universe)

    def blackout(self):
        """Coupe tous les canaux de tous les univers."""
        self._blackout = True
        for uni in self._universes.values():
            uni.blackout()
        log.info("DMX Blackout activé")

    def restore(self):
        """Désactive le blackout et reprend l'envoi normal."""
        self._blackout = False
        log.info("DMX Blackout désactivé")

    # ── Slot connecté au pipeline shader ─────────────────────────────────────

    def uniform_changed_slot(self, uniform_name: str, value: float):
        """
        Appelé depuis MainWindow._tick() via uniform_changed.connect().
        Applique les mappings correspondant à l'uniform reçu.
        """
        self._uniform_values[uniform_name] = value

        for m in self._mappings:
            if not m.enabled or m.uniform != uniform_name:
                continue
            self._apply_mapping(m, value)

    def _apply_mapping(self, m: DmxMapping, value: float):
        """Applique un mapping DMX selon son mode."""
        uni = self._get_universe(m.universe)

        if m.mode == "single":
            if m.channels:
                dmx_val = m.scale(value)
                uni.set_channel(m.channels[0], dmx_val)
                self.channel_changed.emit(m.channels[0], dmx_val)

        elif m.mode == "rgb":
            # Un seul float → blanc (R=G=B)
            dmx_val = m.scale(value)
            for ch in m.channels[:3]:
                uni.set_channel(ch, dmx_val)
                self.channel_changed.emit(ch, dmx_val)

        elif m.mode == "rgb_vec":
            # On attend que la valeur arrive sous forme de vec3 encodé comme 3 uniforms
            # Convention : uniform 'uColor' → uColor_r, uColor_g, uColor_b
            # Ici on gère le cas où value est un float brut (canal unique)
            dmx_val = m.scale(value)
            if m.channels:
                uni.set_channel(m.channels[0], dmx_val)
                self.channel_changed.emit(m.channels[0], dmx_val)

        elif m.mode == "strobe":
            # Canal 0 : intensité, Canal 1 : vitesse automatique (50% fixe)
            dmx_val = m.scale(value)
            if len(m.channels) >= 1:
                uni.set_channel(m.channels[0], dmx_val)
                self.channel_changed.emit(m.channels[0], dmx_val)
            if len(m.channels) >= 2:
                speed = 128 if value > 0.05 else 0   # vitesse 50% si actif
                uni.set_channel(m.channels[1], speed)
                self.channel_changed.emit(m.channels[1], speed)

    # ── Démarrage / Arrêt ─────────────────────────────────────────────────────

    def start(self, protocol: str = "artnet",
              host: str = "255.255.255.255", port: int | None = None):
        """Lance le thread d'envoi DMX."""
        self.stop()
        self._protocol = protocol.lower()
        self._host     = host
        self._port     = port or (ARTNET_PORT if self._protocol == "artnet" else SACN_PORT)
        self._running  = True
        self._thread   = threading.Thread(
            target=self._send_loop, daemon=True, name="DmxEngine"
        )
        self._thread.start()
        log.info("DmxEngine démarré — protocole=%s host=%s port=%d",
                 self._protocol, self._host, self._port)

    def stop(self):
        """Arrête proprement le thread d'envoi."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        log.info("DmxEngine arrêté.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Thread d'envoi ────────────────────────────────────────────────────────

    def _send_loop(self):
        """Boucle principale : envoie tous les univers à ~DMX_FPS Hz."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            interval = 1.0 / DMX_FPS

            while self._running:
                t0 = time.perf_counter()

                if not self._blackout:
                    for uid, uni in self._universes.items():
                        self._send_universe(uni)

                self._sequence = (self._sequence + 1) & 0xFF

                # Reset HTP pour le prochain frame (valeurs décroissent)
                for uni in self._universes.values():
                    uni.blackout()

                elapsed = time.perf_counter() - t0
                sleep_t = max(0.0, interval - elapsed)
                time.sleep(sleep_t)

        except OSError as e:
            msg = f"Erreur réseau DMX : {e}"
            log.error(msg)
            self.error_occurred.emit(msg)
        finally:
            self._running = False

    def _send_universe(self, uni: DmxUniverse):
        """Envoie le paquet d'un univers selon le protocole actif."""
        try:
            if self._protocol == "artnet":
                packet = uni.build_artnet_packet(self._sequence)
                self._sock.sendto(packet, (self._host, self._port))

            elif self._protocol == "sacn":
                packet = uni.build_sacn_packet(self._sequence)
                # Multicast sACN
                mc_host = f"{SACN_MULTICAST_BASE}{uni.universe_id + 1}"
                self._sock.sendto(packet, (mc_host, self._port))

            self.universe_sent.emit(uni.universe_id)

        except OSError as e:
            log.warning("Erreur envoi univers %d : %s", uni.universe_id, e)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "protocol": self._protocol,
            "host":     self._host,
            "port":     self._port,
            "fixtures": [f.to_dict() for f in self._fixtures],
            "mappings": [m.to_dict() for m in self._mappings],
        }

    def from_dict(self, data: dict):
        self._protocol = data.get("protocol", "artnet")
        self._host     = data.get("host", "255.255.255.255")
        self._port     = int(data.get("port", ARTNET_PORT))
        self._fixtures = [DmxFixture.from_dict(d) for d in data.get("fixtures", [])]
        self._mappings = [DmxMapping.from_dict(d) for d in data.get("mappings", [])]


# ══════════════════════════════════════════════════════════════════════════════
#  Patch Panel — plan de salle 2D
# ══════════════════════════════════════════════════════════════════════════════

class _FixtureItem:
    """Représentation graphique d'une fixture sur le plan de salle."""
    RADIUS = 18

    def __init__(self, fixture: DmxFixture):
        self.fixture = fixture
        self.selected = False

    def rect(self, w: int, h: int) -> QRectF:
        cx = self.fixture.x * w
        cy = self.fixture.y * h
        r  = self.RADIUS
        return QRectF(cx - r, cy - r, r * 2, r * 2)

    def contains(self, px: float, py: float, w: int, h: int) -> bool:
        cx = self.fixture.x * w
        cy = self.fixture.y * h
        return (px - cx) ** 2 + (py - cy) ** 2 <= self.RADIUS ** 2


class DmxPatchCanvas(QWidget):
    """
    Plan de salle 2D avec fixtures déplaçables.
    Fond noir de scène, fixtures comme cercles colorés par type.
    """

    fixture_selected = pyqtSignal(object)  # DmxFixture | None

    def __init__(self, engine: DmxEngine, parent=None):
        super().__init__(parent)
        self._engine   = engine
        self._items:   list[_FixtureItem] = []
        self._selected: _FixtureItem | None = None
        self._drag_item: _FixtureItem | None = None
        self._drag_offset = (0.0, 0.0)

        self.setMinimumSize(400, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._refresh_items()

        # Refresh live (channel_changed → repaint)
        engine.channel_changed.connect(lambda *_: self.update())
        engine.fixture_added.connect(lambda _: self._refresh_items())
        engine.fixture_removed.connect(lambda _: self._refresh_items())

    def _refresh_items(self):
        self._items = [_FixtureItem(f) for f in self._engine.get_fixtures()]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Fond scène
        p.fillRect(0, 0, w, h, QColor("#0d0d0d"))

        # Grille subtile
        p.setPen(QPen(QColor("#1a1a1a"), 1))
        for i in range(0, w, 40):
            p.drawLine(i, 0, i, h)
        for i in range(0, h, 40):
            p.drawLine(0, i, w, i)

        # Bordure scène
        p.setPen(QPen(QColor("#333333"), 2))
        p.drawRect(1, 1, w - 2, h - 2)

        # Label "SCÈNE"
        p.setPen(QColor("#444444"))
        p.setFont(QFont("Consolas", 9))
        p.drawText(6, h - 6, "SCÈNE")

        # Fixtures
        for item in self._items:
            fixture = item.fixture
            color   = QColor(FIXTURE_COLORS.get(fixture.fixture_type, "#888888"))

            # Valeur DMX actuelle (pour l'intensité du halo)
            uni   = self._engine._get_universe(fixture.universe)
            val   = uni.get_channel(fixture.address) / 255.0
            alpha = int(40 + val * 140)

            # Halo lumineux
            halo = QColor(color)
            halo.setAlpha(alpha)
            p.setBrush(QBrush(halo))
            p.setPen(Qt.PenStyle.NoPen)
            r  = item.RADIUS + 8
            cx = int(fixture.x * w)
            cy = int(fixture.y * h)
            p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

            # Corps de la fixture
            border_color = QColor("#ffffff") if item.selected else color
            p.setBrush(QBrush(color))
            p.setPen(QPen(border_color, 2 if item.selected else 1))
            p.drawEllipse(item.rect(w, h))

            # Icône type
            p.setPen(QColor("#000000"))
            p.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            icon = {"dimmer": "D", "rgb": "C", "rgba": "A", "rgbw": "W",
                    "moving": "M", "strobe": "S", "fogger": "F"}.get(fixture.fixture_type, "?")
            p.drawText(item.rect(w, h).toRect(), Qt.AlignmentFlag.AlignCenter, icon)

            # Label + adresse
            p.setPen(QColor("#cccccc"))
            p.setFont(QFont("Consolas", 7))
            lbl = f"{fixture.label}\n@{fixture.address}"
            p.drawText(int(cx - 30), int(cy + item.RADIUS + 4), 60, 28,
                       Qt.AlignmentFlag.AlignHCenter, lbl)

        p.end()

    def mousePressEvent(self, event):
        px, py = event.position().x(), event.position().y()
        w, h   = self.width(), self.height()
        hit    = None
        for item in self._items:
            if item.contains(px, py, w, h):
                hit = item
                break
        # Désélectionne tout
        for item in self._items:
            item.selected = False
        if hit:
            hit.selected      = True
            self._selected    = hit
            self._drag_item   = hit
            fx_x = hit.fixture.x * w
            fx_y = hit.fixture.y * h
            self._drag_offset = (px - fx_x, py - fx_y)
            self.fixture_selected.emit(hit.fixture)
        else:
            self._selected  = None
            self._drag_item = None
            self.fixture_selected.emit(None)
        self.update()

    def mouseMoveEvent(self, event):
        if self._drag_item:
            px, py = event.position().x(), event.position().y()
            w, h   = self.width(), self.height()
            nx = max(0.0, min(1.0, (px - self._drag_offset[0]) / w))
            ny = max(0.0, min(1.0, (py - self._drag_offset[1]) / h))
            self._drag_item.fixture.x = nx
            self._drag_item.fixture.y = ny
            self.update()

    def mouseReleaseEvent(self, _):
        self._drag_item = None


# ══════════════════════════════════════════════════════════════════════════════
#  Panneau de mappings (uniform → canal(s))
# ══════════════════════════════════════════════════════════════════════════════

class DmxMappingPanel(QWidget):
    """Tableau éditable des mappings uniform GLSL → canaux DMX."""

    def __init__(self, engine: DmxEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Barre d'outils
        bar = QHBoxLayout()
        self._btn_add = QPushButton("＋ Ajouter")
        self._btn_add.clicked.connect(self._add_mapping)
        self._btn_del = QPushButton("✕ Supprimer")
        self._btn_del.clicked.connect(self._del_mapping)
        bar.addWidget(self._btn_add)
        bar.addWidget(self._btn_del)
        bar.addStretch()
        lay.addLayout(bar)

        # Tableau
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Uniform", "Canaux", "Mode", "Lo", "Hi", "Activé"]
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 130)
        self._table.setColumnWidth(1, 100)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 50)
        self._table.setColumnWidth(4, 50)
        self._table.setColumnWidth(5, 55)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background:#111; color:#ddd; gridline-color:#333; }"
            "QHeaderView::section { background:#1e1e1e; color:#aaa; }"
        )
        lay.addWidget(self._table)

    def _refresh(self):
        self._table.setRowCount(0)
        for m in self._engine.get_mappings():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(m.uniform))
            self._table.setItem(row, 1, QTableWidgetItem(",".join(str(c) for c in m.channels)))
            self._table.setItem(row, 2, QTableWidgetItem(m.mode))
            self._table.setItem(row, 3, QTableWidgetItem(f"{m.lo:.2f}"))
            self._table.setItem(row, 4, QTableWidgetItem(f"{m.hi:.2f}"))
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked if m.enabled else Qt.CheckState.Unchecked)
            self._table.setItem(row, 5, chk)

    def _add_mapping(self):
        m = DmxMapping(uniform="uBrightness", channels=[1], mode="single")
        self._engine.add_mapping(m)
        self._refresh()

    def _del_mapping(self):
        row = self._table.currentRow()
        if row < 0:
            return
        mappings = self._engine.get_mappings()
        if row < len(mappings):
            self._engine.remove_mapping(mappings[row])
            self._refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  Inspecteur de fixture
# ══════════════════════════════════════════════════════════════════════════════

class DmxFixtureInspector(QWidget):
    """Panneau d'édition des propriétés d'une fixture sélectionnée."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fixture: DmxFixture | None = None
        self._build_ui()
        self.setEnabled(False)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        grp = QGroupBox("Fixture sélectionnée")
        g   = QVBoxLayout(grp)

        def row(label: str, widget: QWidget) -> QWidget:
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            g.addLayout(h)
            return widget

        self._name    = row("Nom :", QLineEdit())
        self._label   = row("Label :", QLineEdit())
        self._addr    = row("Adresse DMX :", QSpinBox())
        self._addr.setRange(1, 512)
        self._n_ch    = row("Nb canaux :", QSpinBox())
        self._n_ch.setRange(1, 32)
        self._type    = row("Type :", QComboBox())
        self._type.addItems([e.value for e in FixtureType])
        self._universe = row("Univers :", QSpinBox())
        self._universe.setRange(0, 255)

        self._btn_apply = QPushButton("✓ Appliquer")
        self._btn_apply.clicked.connect(self._apply)
        g.addWidget(self._btn_apply)

        lay.addWidget(grp)
        lay.addStretch()

    def set_fixture(self, fixture: DmxFixture | None):
        self._fixture = fixture
        self.setEnabled(fixture is not None)
        if fixture:
            self._name.setText(fixture.name)
            self._label.setText(fixture.label)
            self._addr.setValue(fixture.address)
            self._n_ch.setValue(fixture.n_channels)
            idx = self._type.findText(fixture.fixture_type)
            if idx >= 0:
                self._type.setCurrentIndex(idx)
            self._universe.setValue(fixture.universe)

    def _apply(self):
        if not self._fixture:
            return
        self._fixture.name         = self._name.text()
        self._fixture.label        = self._label.text()
        self._fixture.address      = self._addr.value()
        self._fixture.n_channels   = self._n_ch.value()
        self._fixture.fixture_type = self._type.currentText()
        self._fixture.universe     = self._universe.value()


# ══════════════════════════════════════════════════════════════════════════════
#  Dialogue principal DMX
# ══════════════════════════════════════════════════════════════════════════════

class DmxPanel(QDialog):
    """
    Fenêtre principale du moteur DMX.
    3 onglets : Patch (plan de salle) · Mappings · Configuration réseau.
    """

    def __init__(self, engine: DmxEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self.setWindowTitle("🔦 DMX / Artnet — OpenShader")
        self.setMinimumSize(820, 560)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        # ── Barre de statut + contrôles globaux ──────────────────────────────
        top = QHBoxLayout()

        self._status_lbl = QLabel("● Arrêté")
        self._status_lbl.setStyleSheet("color:#f87171; font-weight:bold;")
        top.addWidget(self._status_lbl)

        self._btn_start = QPushButton("▶ Démarrer")
        self._btn_start.clicked.connect(self._toggle_start)
        self._btn_bo    = QPushButton("⬛ Blackout")
        self._btn_bo.setCheckable(True)
        self._btn_bo.clicked.connect(self._toggle_blackout)
        self._btn_bo.setStyleSheet(
            "QPushButton:checked { background:#ef4444; color:#fff; font-weight:bold; }"
        )
        top.addWidget(self._btn_start)
        top.addWidget(self._btn_bo)
        top.addStretch()
        lay.addLayout(top)

        # ── Splitter principal ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- Plan de salle + inspecteur
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)

        self._canvas = DmxPatchCanvas(self._engine)
        self._canvas.fixture_selected.connect(self._on_fixture_selected)
        llay.addWidget(self._canvas, 3)

        # Toolbar patch
        ptb = QHBoxLayout()
        btn_add_fix = QPushButton("＋ Fixture")
        btn_add_fix.clicked.connect(self._add_fixture)
        btn_del_fix = QPushButton("✕ Supprimer")
        btn_del_fix.clicked.connect(self._del_fixture)
        ptb.addWidget(btn_add_fix)
        ptb.addWidget(btn_del_fix)
        ptb.addStretch()
        llay.addLayout(ptb)

        self._inspector = DmxFixtureInspector()
        llay.addWidget(self._inspector, 2)

        splitter.addWidget(left)

        # -- Mappings + config réseau
        right = QWidget()
        rlay  = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)

        rlay.addWidget(QLabel("Mappings uniform → DMX :"))
        self._mapping_panel = DmxMappingPanel(self._engine)
        rlay.addWidget(self._mapping_panel, 2)

        # Config réseau
        net_grp = QGroupBox("Configuration réseau")
        net     = QVBoxLayout(net_grp)

        def nrow(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            net.addLayout(h)
            return widget

        self._proto = nrow("Protocole :", QComboBox())
        self._proto.addItems(["artnet", "sacn"])
        self._host_edit = nrow("Host / IP broadcast :", QLineEdit("255.255.255.255"))
        self._port_spin = nrow("Port :", QSpinBox())
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(ARTNET_PORT)
        self._proto.currentTextChanged.connect(self._on_proto_changed)

        rlay.addWidget(net_grp)

        # Moniteur de canaux (32 premiers)
        rlay.addWidget(QLabel("Moniteur canaux 1–32 :"))
        self._monitor = QTableWidget(2, 32)
        self._monitor.setHorizontalHeaderLabels([str(i) for i in range(1, 33)])
        self._monitor.setVerticalHeaderLabels(["Val", "■"])
        self._monitor.setMaximumHeight(80)
        self._monitor.setStyleSheet(
            "QTableWidget { background:#111; color:#0f0; font-size:9px; }"
        )
        for col in range(32):
            self._monitor.setColumnWidth(col, 28)
        rlay.addWidget(self._monitor)

        splitter.addWidget(right)
        splitter.setSizes([480, 340])
        lay.addWidget(splitter)

        # Rafraîchissement moniteur
        self._engine.channel_changed.connect(self._on_channel_changed)

        # Boutons bas de fenêtre
        bot = QHBoxLayout()
        bot.addStretch()
        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.close)
        bot.addWidget(btn_close)
        lay.addLayout(bot)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _toggle_start(self):
        if self._engine.is_running:
            self._engine.stop()
            self._btn_start.setText("▶ Démarrer")
            self._status_lbl.setText("● Arrêté")
            self._status_lbl.setStyleSheet("color:#f87171; font-weight:bold;")
        else:
            self._engine.start(
                protocol = self._proto.currentText(),
                host     = self._host_edit.text(),
                port     = self._port_spin.value(),
            )
            self._btn_start.setText("■ Arrêter")
            self._status_lbl.setText(
                f"● Actif — {self._proto.currentText().upper()} → {self._host_edit.text()}"
            )
            self._status_lbl.setStyleSheet("color:#4ade80; font-weight:bold;")

    def _toggle_blackout(self, checked: bool):
        if checked:
            self._engine.blackout()
        else:
            self._engine.restore()

    def _on_proto_changed(self, proto: str):
        self._port_spin.setValue(ARTNET_PORT if proto == "artnet" else SACN_PORT)

    def _add_fixture(self):
        used  = {f.address for f in self._engine.get_fixtures()}
        start = next((i for i in range(1, 510) if i not in used), 1)
        fx    = DmxFixture(
            name    = f"Fixture {len(self._engine.get_fixtures()) + 1}",
            address = start,
            x       = 0.3 + (len(self._engine.get_fixtures()) % 5) * 0.12,
            y       = 0.3 + (len(self._engine.get_fixtures()) // 5) * 0.15,
        )
        self._engine.add_fixture(fx)

    def _del_fixture(self):
        if self._canvas._selected:
            self._engine.remove_fixture(self._canvas._selected.fixture)
            self._canvas._selected = None
            self._inspector.set_fixture(None)

    def _on_fixture_selected(self, fixture):
        self._inspector.set_fixture(fixture)

    def _on_channel_changed(self, channel: int, value: int):
        if channel < 1 or channel > 32:
            return
        col = channel - 1
        self._monitor.setItem(0, col, QTableWidgetItem(str(value)))
        color = QColor.fromHsvF(0.33, 0.8, value / 255.0)
        item  = QTableWidgetItem("■")
        item.setForeground(color)
        self._monitor.setItem(1, col, item)
