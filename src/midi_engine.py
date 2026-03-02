"""
midi_engine.py
--------------
v2.0 — Mapping MIDI → uniforms en temps réel.

Utilise `mido` si disponible, sinon fonctionne en mode stub silencieux.
Architecture :
  - MidiEngine  : scanne les ports, lit les messages CC/Note dans un thread
  - MidiMapping  : association (port, channel, cc/note) → uniform name + scaling
  - Signal PyQt6 : uniform_changed(name, value) émis dans le thread Qt

Usage minimal :
    engine = MidiEngine()
    engine.add_mapping(cc=1, uniform='uBrightness', lo=0.0, hi=1.0)
    engine.uniform_changed.connect(lambda name, val: shader.set_uniform(name, val))
    engine.start(port_name='...')

Protocole :
  - CC  0–127  → valeur normalisée [lo, hi]
  - Note On    → valeur hi  (trigger)
  - Note Off   → valeur lo  (release)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)

# ── Import conditionnel de mido ──────────────────────────────────────────────

try:
    import mido  # type: ignore
    _MIDO_AVAILABLE = True
except ImportError:
    mido = None  # type: ignore
    _MIDO_AVAILABLE = False
    log.info("mido non installé — MIDI désactivé (pip install mido python-rtmidi)")


# ── Modèle de données ────────────────────────────────────────────────────────

@dataclass
class MidiMapping:
    """Association MIDI CC/Note → uniform GLSL."""
    uniform:  str                     # nom du uniform cible
    cc:       int   | None = None     # numéro de Control Change (0-127)
    note:     int   | None = None     # numéro de note (0-127)
    channel:  int         = -1        # -1 = tous les canaux
    lo:       float       = 0.0       # valeur minimale (midi 0)
    hi:       float       = 1.0       # valeur maximale (midi 127)
    curve:    str         = 'linear'  # 'linear' | 'log' | 'exp'

    def scale(self, raw: int) -> float:
        """Convertit une valeur MIDI brute [0..127] vers [lo..hi]."""
        t = raw / 127.0
        if self.curve == 'log':
            import math
            t = math.log1p(t * (math.e - 1))  # log normalisé [0,1]
        elif self.curve == 'exp':
            t = t * t
        return self.lo + t * (self.hi - self.lo)


# ── Moteur MIDI ──────────────────────────────────────────────────────────────

class MidiEngine(QObject):
    """Thread de lecture MIDI + émission de signaux Qt."""

    uniform_changed  = pyqtSignal(str, float)   # (uniform_name, value)
    port_opened      = pyqtSignal(str)           # nom du port
    port_closed      = pyqtSignal()
    learn_triggered  = pyqtSignal(int, int, int) # (channel, cc_or_note, value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mappings:    list[MidiMapping] = []
        self._port        = None
        self._thread:     threading.Thread | None = None
        self._running:    bool = False
        self._learn_mode: bool = False           # MIDI Learn actif

    # ── Détection des ports ──────────────────────────────────────────────────

    @staticmethod
    def list_ports() -> list[str]:
        """Retourne la liste des ports MIDI d'entrée disponibles."""
        if not _MIDO_AVAILABLE:
            return []
        try:
            return mido.get_input_names()
        except (OSError, RuntimeError) as e:
            log.warning("Impossible de lister les ports MIDI : %s", e)
            return []

    @property
    def is_available(self) -> bool:
        return _MIDO_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Mappings ─────────────────────────────────────────────────────────────

    def add_mapping(self, uniform: str, cc: int | None = None,
                    note: int | None = None, channel: int = -1,
                    lo: float = 0.0, hi: float = 1.0,
                    curve: str = 'linear') -> MidiMapping:
        m = MidiMapping(uniform=uniform, cc=cc, note=note,
                        channel=channel, lo=lo, hi=hi, curve=curve)
        self._mappings.append(m)
        log.debug("MIDI mapping ajouté : cc=%s note=%s → %s [%.2f..%.2f]",
                  cc, note, uniform, lo, hi)
        return m

    def remove_mapping(self, mapping: MidiMapping):
        if mapping in self._mappings:
            self._mappings.remove(mapping)

    def clear_mappings(self):
        self._mappings.clear()

    def get_mappings(self) -> list[MidiMapping]:
        return list(self._mappings)

    def set_mappings(self, mappings: list[MidiMapping]):
        self._mappings = list(mappings)

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> list[dict]:
        return [
            {'uniform': m.uniform, 'cc': m.cc, 'note': m.note,
             'channel': m.channel, 'lo': m.lo, 'hi': m.hi, 'curve': m.curve}
            for m in self._mappings
        ]

    def from_dict(self, data: list[dict]):
        self._mappings = [
            MidiMapping(
                uniform=d.get('uniform', ''),
                cc=d.get('cc'),
                note=d.get('note'),
                channel=d.get('channel', -1),
                lo=float(d.get('lo', 0.0)),
                hi=float(d.get('hi', 1.0)),
                curve=d.get('curve', 'linear'),
            )
            for d in data
        ]

    # ── Start / Stop ─────────────────────────────────────────────────────────

    def start(self, port_name: str | None = None):
        """Ouvre le port MIDI et démarre le thread de lecture."""
        if not _MIDO_AVAILABLE:
            log.warning("mido non disponible — impossible de démarrer le MIDI.")
            return
        self.stop()
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, args=(port_name,),
            daemon=True, name="MidiEngine"
        )
        self._thread.start()

    def stop(self):
        """Arrête proprement le thread MIDI."""
        self._running = False
        if self._port:
            try:
                self._port.close()
            except (OSError, RuntimeError):
                pass
            self._port = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def start_learn(self):
        """Active le mode MIDI Learn : le prochain message CC/Note est émis via learn_triggered."""
        self._learn_mode = True

    def stop_learn(self):
        self._learn_mode = False

    # ── Thread de lecture ────────────────────────────────────────────────────

    def _run(self, port_name: str | None):
        ports = mido.get_input_names()
        if not ports:
            log.warning("Aucun port MIDI d'entrée disponible.")
            self._running = False
            return

        chosen = port_name if (port_name and port_name in ports) else ports[0]
        try:
            with mido.open_input(chosen) as port:
                self._port = port
                self.port_opened.emit(chosen)
                log.info("Port MIDI ouvert : %s", chosen)

                while self._running:
                    for msg in port.iter_pending():
                        self._handle_message(msg)
                    import time; time.sleep(0.002)  # ~500 Hz polling
        except (OSError, RuntimeError) as e:
            log.error("Erreur port MIDI '%s' : %s", chosen, e)
        finally:
            self._running = False
            self.port_closed.emit()
            log.info("Port MIDI fermé.")

    def _handle_message(self, msg):
        """Traite un message MIDI entrant."""
        ch = getattr(msg, 'channel', 0)

        # MIDI Learn : émet le signal et sort
        if self._learn_mode:
            if msg.type == 'control_change':
                self.learn_triggered.emit(ch, msg.control, msg.value)
                self._learn_mode = False
                return
            elif msg.type in ('note_on', 'note_off'):
                self.learn_triggered.emit(ch, 128 + msg.note, msg.velocity)
                self._learn_mode = False
                return

        # Application des mappings
        for m in self._mappings:
            if m.channel != -1 and m.channel != ch:
                continue

            if msg.type == 'control_change' and m.cc is not None:
                if msg.control == m.cc:
                    val = m.scale(msg.value)
                    self.uniform_changed.emit(m.uniform, val)

            elif msg.type == 'note_on' and msg.velocity > 0 and m.note is not None:
                if msg.note == m.note:
                    self.uniform_changed.emit(m.uniform, m.hi)

            elif msg.type in ('note_off', 'note_on') and m.note is not None:
                # note_on velocity=0 aussi traité comme note_off
                is_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
                if is_off and msg.note == m.note:
                    self.uniform_changed.emit(m.uniform, m.lo)
