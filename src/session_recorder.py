"""
session_recorder.py
-------------------
Enregistrement live de toutes les interactions (sliders, MIDI, OSC) vers
une timeline rejouable.

Fonctionnalités :
  - Capture en temps réel de tout changement d'uniform (source : UI, MIDI, OSC)
  - Punch-in / Punch-out : enregistrement sur une plage de temps définie
  - Overdub : superposition sur les pistes existantes (sans tout écraser)
  - Export de la session enregistrée vers le modèle Timeline (.demomaker)
  - Signal `recording_state_changed` pour mise à jour de l'UI

Usage typique (dans main_window.py) :
    self.session_recorder = SessionRecorder(self.timeline)

    # Brancher toutes les sources d'uniforms
    self.left_panel.uniform_value_changed.connect(self.session_recorder.record_event)
    self.midi_engine.uniform_changed.connect(self.session_recorder.record_event)
    self.osc_engine.uniform_changed.connect(self.session_recorder.record_event)

    # Démarrer / arrêter
    self.session_recorder.start(current_time)
    self.session_recorder.stop()

    # Punch-in/out
    self.session_recorder.set_punch(punch_in=2.0, punch_out=8.0)
    self.session_recorder.start(current_time, mode='punch')

    # Overdub
    self.session_recorder.start(current_time, mode='overdub')
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal

from .timeline import Timeline, Track, Keyframe

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mode d'enregistrement
# ─────────────────────────────────────────────────────────────────────────────

RECORD_MODE_NORMAL  = 'normal'   # Enregistrement depuis t_start jusqu'à stop()
RECORD_MODE_PUNCH   = 'punch'    # Enregistrement uniquement dans [punch_in, punch_out]
RECORD_MODE_OVERDUB = 'overdub'  # Superposition sur les keyframes existants


# ─────────────────────────────────────────────────────────────────────────────
# Événement capturé
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecordedEvent:
    """Un changement d'uniform capturé à un instant précis."""
    timeline_time: float      # Temps dans la timeline (secondes)
    uniform_name:  str        # Nom GLSL de l'uniform
    value:         Any        # Valeur au moment de l'événement
    source:        str = ''   # 'ui' | 'midi' | 'osc' | ''


# ─────────────────────────────────────────────────────────────────────────────
# SessionRecorder
# ─────────────────────────────────────────────────────────────────────────────

class SessionRecorder(QObject):
    """
    Capture les interactions live et les convertit en keyframes sur la Timeline.

    Signaux
    -------
    recording_state_changed(bool)
        Émis quand l'enregistrement démarre ou s'arrête.
    event_captured(str, float)
        Émis à chaque événement enregistré : (uniform_name, timeline_time).
    """

    recording_state_changed: pyqtSignal = pyqtSignal(bool)
    event_captured: pyqtSignal = pyqtSignal(str, float)

    # ── Cycle de vie ─────────────────────────────────────────────────────────

    def __init__(self, timeline: Timeline, parent: QObject | None = None):
        super().__init__(parent)

        self._timeline: Timeline = timeline

        # État courant
        self._is_recording:    bool  = False
        self._mode:            str   = RECORD_MODE_NORMAL
        self._wall_start:      float = 0.0   # time.monotonic() au démarrage
        self._timeline_start:  float = 0.0   # position de la tête de lecture au démarrage

        # Punch-in / Punch-out
        self._punch_in:  float | None = None
        self._punch_out: float | None = None

        # Buffer d'événements (flush vers la timeline à la fin)
        self._events: list[RecordedEvent] = []

        # Fournisseur de temps courant (injecté par main_window pour rester
        # synchronisé avec la tête de lecture réelle)
        self._time_provider: Callable[[], float] | None = None

        # Interp mode appliqué aux nouveaux KFs
        self.default_interp: str = 'linear'

    def set_timeline(self, timeline: Timeline):
        """Remplace la timeline cible (ex: lors du chargement d'un projet)."""
        self._timeline = timeline

    def set_time_provider(self, provider: Callable[[], float]):
        """
        Fournit une fonction qui retourne le temps courant de la tête de lecture
        (en secondes).  Si non fourni, le recorder calcule lui-même le temps
        depuis le démarrage.
        """
        self._time_provider = provider

    # ── Configuration Punch ──────────────────────────────────────────────────

    def set_punch(self, punch_in: float, punch_out: float):
        """
        Définit la plage de punch-in / punch-out (en secondes).
        L'enregistrement n'a lieu que dans [punch_in, punch_out] quand
        le mode 'punch' est actif.
        """
        if punch_in >= punch_out:
            raise ValueError("punch_in doit être strictement inférieur à punch_out.")
        self._punch_in  = punch_in
        self._punch_out = punch_out
        log.debug("Punch défini : [%.2f, %.2f]", punch_in, punch_out)

    def clear_punch(self):
        """Désactive le mode punch (plage libre)."""
        self._punch_in  = None
        self._punch_out = None

    # ── Démarrage / Arrêt ────────────────────────────────────────────────────

    def start(self, timeline_position: float = 0.0,
              mode: str = RECORD_MODE_NORMAL):
        """
        Démarre l'enregistrement.

        Parameters
        ----------
        timeline_position : float
            Position courante de la tête de lecture (secondes).
        mode : str
            'normal', 'punch' ou 'overdub'.
        """
        if self._is_recording:
            log.warning("SessionRecorder déjà en cours — arrêt automatique avant redémarrage.")
            self.stop()

        if mode == RECORD_MODE_PUNCH and (self._punch_in is None or self._punch_out is None):
            raise RuntimeError("Mode punch demandé mais punch_in/out non définis. "
                               "Appelez set_punch() d'abord.")

        self._mode           = mode
        self._timeline_start = timeline_position
        self._wall_start     = time.monotonic()
        self._events         = []
        self._is_recording   = True

        log.info("⏺ SessionRecorder démarré — mode=%s  t0=%.2f", mode, timeline_position)
        self.recording_state_changed.emit(True)

    def stop(self) -> list[RecordedEvent]:
        """
        Arrête l'enregistrement et applique les événements capturés
        à la timeline.

        Returns
        -------
        list[RecordedEvent]
            La liste des événements enregistrés (pour debug / export).
        """
        if not self._is_recording:
            return []

        self._is_recording = False
        events = list(self._events)
        self._flush_to_timeline(events)

        log.info("⏹ SessionRecorder arrêté — %d événements enregistrés.", len(events))
        self.recording_state_changed.emit(False)
        return events

    def cancel(self):
        """Annule l'enregistrement sans rien appliquer à la timeline."""
        if not self._is_recording:
            return
        self._is_recording = False
        self._events = []
        log.info("✗ SessionRecorder annulé.")
        self.recording_state_changed.emit(False)

    # ── Capture d'événements ─────────────────────────────────────────────────

    def record_event(self, uniform_name: str, value: Any,
                     source: str = ''):
        """
        Slot à connecter sur uniform_value_changed / midi.uniform_changed /
        osc.uniform_changed.

        Parameters
        ----------
        uniform_name : str
            Nom GLSL de l'uniform.
        value : Any
            Valeur courante.
        source : str
            Optionnel — identifiant de la source ('ui', 'midi', 'osc').
        """
        if not self._is_recording:
            return

        t = self._current_timeline_time()

        # ── Filtre Punch ──────────────────────────────────────────────────
        if self._mode == RECORD_MODE_PUNCH:
            if self._punch_in is None or self._punch_out is None:
                return
            if not (self._punch_in <= t <= self._punch_out):
                return

        event = RecordedEvent(
            timeline_time=t,
            uniform_name=uniform_name,
            value=value,
            source=source,
        )
        self._events.append(event)
        self.event_captured.emit(uniform_name, t)

    # Surcharge pratique : slots typés pour les signaux Qt (str, float) et
    # (str, object) — PyQt exige la correspondance de signature.
    def record_event_float(self, uniform_name: str, value: float):
        self.record_event(uniform_name, value)

    def record_event_any(self, uniform_name: str, value: object):
        self.record_event(uniform_name, value)

    # ── Propriétés ───────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def punch_in(self) -> float | None:
        return self._punch_in

    @property
    def punch_out(self) -> float | None:
        return self._punch_out

    @property
    def events(self) -> list[RecordedEvent]:
        """Événements capturés depuis le dernier start() (lecture seule)."""
        return list(self._events)

    # ── Internals ────────────────────────────────────────────────────────────

    def _current_timeline_time(self) -> float:
        """
        Retourne la position courante de la tête de lecture en secondes.
        Utilise le fournisseur externe si disponible, sinon calcule depuis
        le wall-clock.
        """
        if self._time_provider is not None:
            return self._time_provider()
        return self._timeline_start + (time.monotonic() - self._wall_start)

    def _flush_to_timeline(self, events: list[RecordedEvent]):
        """
        Applique les événements capturés à la timeline.

        - Crée les pistes manquantes automatiquement.
        - En mode OVERDUB, conserve les keyframes existants hors de la
          zone enregistrée.
        - En mode PUNCH, ne modifie que les keyframes dans [punch_in, punch_out].
        """
        if not events:
            return

        # Regrouper par uniform
        by_uniform: dict[str, list[RecordedEvent]] = {}
        for ev in events:
            by_uniform.setdefault(ev.uniform_name, []).append(ev)

        for uname, evs in by_uniform.items():
            track = self._timeline.get_track_by_uniform(uname)

            # ── Créer la piste si elle n'existe pas ───────────────────────
            if track is None:
                # Déduire le value_type depuis la première valeur
                first_val = evs[0].value
                vtype = _infer_value_type(first_val)
                track = self._timeline.add_track(
                    name=uname,
                    uniform_name=uname,
                    value_type=vtype,
                )
                log.debug("Piste créée : %s (%s)", uname, vtype)

            # ── En mode non-overdub : supprimer les KFs dans la zone ──────
            if self._mode in (RECORD_MODE_NORMAL, RECORD_MODE_PUNCH):
                t_min = evs[0].timeline_time
                t_max = evs[-1].timeline_time
                # Punch restreint la plage aux bornes définies
                if self._mode == RECORD_MODE_PUNCH:
                    t_min = max(t_min, self._punch_in or t_min)
                    t_max = min(t_max, self._punch_out or t_max)
                _remove_keyframes_in_range(track, t_min, t_max)

            # ── Écriture des nouveaux keyframes ───────────────────────────
            for ev in evs:
                # En mode punch : double-vérification de la plage
                if self._mode == RECORD_MODE_PUNCH:
                    if self._punch_in is None or self._punch_out is None:
                        continue
                    if not (self._punch_in <= ev.timeline_time <= self._punch_out):
                        continue
                track.add_keyframe(ev.timeline_time, ev.value, self.default_interp)

        log.debug("Flush timeline : %d pistes mises à jour.", len(by_uniform))

    # ── Export ───────────────────────────────────────────────────────────────

    def export_to_dict(self) -> dict:
        """
        Sérialise la session enregistrée en un dict compatible avec
        Timeline.to_dict() / from_dict(), pouvant être sauvegardé en .demomaker.
        """
        return self._timeline.to_dict()

    def export_events_log(self) -> list[dict]:
        """
        Exporte le journal brut des événements capturés (utile pour debug
        ou intégrations externes).
        """
        return [
            {
                'time':    ev.timeline_time,
                'uniform': ev.uniform_name,
                'value':   list(ev.value) if isinstance(ev.value, tuple) else ev.value,
                'source':  ev.source,
            }
            for ev in self._events
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _infer_value_type(value: Any) -> str:
    """Déduit le value_type d'une piste depuis la valeur Python."""
    if isinstance(value, (int, float)):
        return 'float'
    if isinstance(value, (tuple, list)):
        n = len(value)
        return {2: 'vec2', 3: 'vec3', 4: 'vec4'}.get(n, 'float')
    return 'float'


def _remove_keyframes_in_range(track: Track, t_min: float, t_max: float,
                                margin: float = 0.001):
    """
    Supprime tous les keyframes d'une piste dont le temps est dans
    [t_min - margin, t_max + margin].
    """
    to_remove = [
        kf for kf in track.keyframes
        if t_min - margin <= kf.time <= t_max + margin
    ]
    for kf in to_remove:
        track.remove_keyframe(kf.time)
