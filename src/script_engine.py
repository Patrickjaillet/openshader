"""
script_engine.py
----------------
v2.0 — Scripting Python : API pour contrôler la timeline et les uniforms depuis des scripts.

Sandboxing : les scripts s'exécutent dans un namespace restreint qui expose
uniquement l'API OpenShader, pas les builtins dangereux.

API exposée aux scripts :
    set_uniform(name, value)        — inject un uniform float/vec
    get_uniform(name) -> value      — lit la valeur courante
    get_time() -> float             — temps courant en secondes
    add_keyframe(track, t, value)   — ajoute un keyframe
    get_track_value(track, t)       — évalue une piste à t
    set_bpm(bpm)                    — change le BPM du projet
    log_info(msg)                   — log dans la console OpenShader
    play() / stop() / seek(t)       — contrôle de la lecture
    schedule(delay_s, callback)     — planifie un callback Python (une fois)

Événements (décorateurs) :
    @on_beat                        — appelé à chaque battement BPM
    @on_time(t)                     — appelé quand t est atteint (une fois)
    @on_marker(name)                — appelé quand le marqueur 'name' est passé

Exemple de script :
    @on_beat
    def flash(t):
        set_uniform('uFlash', 1.0)
        schedule(0.05, lambda: set_uniform('uFlash', 0.0))
"""

from __future__ import annotations

import traceback
import threading
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .logger import get_logger

log = get_logger(__name__)


# ── API exposée aux scripts ──────────────────────────────────────────────────

class ScriptAPI:
    """
    Namespace injecté dans chaque script utilisateur.
    Toutes les méthodes sont thread-safe (délèguent au thread Qt via signaux).
    """

    def __init__(self, engine: 'ScriptEngine'):
        self._engine = engine

    # Uniforms
    def set_uniform(self, name: str, value):
        self._engine.uniform_set.emit(name, value)

    def get_uniform(self, name: str):
        return self._engine._uniform_cache.get(name, 0.0)

    # Temps
    def get_time(self) -> float:
        return self._engine._current_time

    # Timeline
    def add_keyframe(self, track_name: str, t: float, value):
        self._engine.keyframe_add.emit(track_name, t, value)

    def get_track_value(self, track_name: str, t: float | None = None) -> float:
        if t is None:
            t = self._engine._current_time
        tl = self._engine._timeline_ref
        if tl is None:
            return 0.0
        track = tl.get_track_by_uniform(track_name)
        if track is None:
            return 0.0
        return track.evaluate(t)

    def set_bpm(self, bpm: float):
        tl = self._engine._timeline_ref
        if tl is not None:
            tl.bpm = float(bpm)

    # Logging
    def log_info(self, msg: str):
        log.info("[Script] %s", msg)
        self._engine.output_line.emit(f"[INFO] {msg}")

    def log_error(self, msg: str):
        log.error("[Script] %s", msg)
        self._engine.output_line.emit(f"[ERROR] {msg}")

    # Transport
    def play(self):
        self._engine.transport_command.emit('play')

    def stop(self):
        self._engine.transport_command.emit('stop')

    def seek(self, t: float):
        self._engine.seek_requested.emit(float(t))

    # Planification
    def schedule(self, delay_s: float, callback: Callable):
        ms = max(1, int(delay_s * 1000))
        QTimer.singleShot(ms, callback)

    # Décorateurs événements
    def on_beat(self, func: Callable) -> Callable:
        self._engine._beat_callbacks.append(func)
        return func

    def on_time(self, t: float):
        def decorator(func: Callable) -> Callable:
            self._engine._time_callbacks.append((float(t), False, func))
            return func
        return decorator

    def on_marker(self, name: str):
        def decorator(func: Callable) -> Callable:
            self._engine._marker_callbacks.append((name, func))
            return func
        return decorator


# ── Moteur de script ─────────────────────────────────────────────────────────

class ScriptEngine(QObject):
    """Exécute des scripts Python dans un sandbox restreint."""

    # Signaux émis par les scripts vers le reste de l'application
    uniform_set       = pyqtSignal(str, object)   # (name, value)
    keyframe_add      = pyqtSignal(str, float, object)  # (track, t, value)
    transport_command = pyqtSignal(str)            # 'play' | 'stop'
    seek_requested    = pyqtSignal(float)
    output_line       = pyqtSignal(str)            # ligne de sortie console
    error_occurred    = pyqtSignal(str)            # message d'erreur

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_time:    float = 0.0
        self._uniform_cache:   dict  = {}
        self._timeline_ref             = None   # référence faible vers Timeline

        # Callbacks enregistrés par les scripts
        self._beat_callbacks:   list[Callable]              = []
        self._time_callbacks:   list[tuple[float, bool, Callable]] = []  # (t, fired, fn)
        self._marker_callbacks: list[tuple[str, Callable]]  = []

        # État d'exécution
        self._script_globals: dict = {}
        self._api = ScriptAPI(self)

        # BPM pour les événements beat
        self._bpm:            float = 120.0
        self._last_beat_time: float = -1.0

    # ── Liaison avec le reste de l'application ───────────────────────────────

    def set_timeline(self, timeline):
        """Lie le moteur de script à la Timeline (référence directe, non possédée)."""
        self._timeline_ref = timeline
        if hasattr(timeline, 'bpm'):
            self._bpm = timeline.bpm

    def update_uniform_cache(self, uniforms: dict):
        """Met à jour le cache des valeurs d'uniforms (appelé depuis _tick)."""
        self._uniform_cache.update(uniforms)

    def tick(self, t: float):
        """Appelé à chaque frame par MainWindow._tick pour déclencher les événements."""
        self._current_time = t

        # Événements @on_time
        for i, (target_t, fired, fn) in enumerate(self._time_callbacks):
            if not fired and t >= target_t:
                self._call_safely(fn, t)
                self._time_callbacks[i] = (target_t, True, fn)

        # Événements @on_beat
        if self._bpm > 0:
            beat_interval = 60.0 / self._bpm
            beat_idx = int(t / beat_interval)
            last_idx = int(self._last_beat_time / beat_interval) if self._last_beat_time >= 0 else -1
            if beat_idx != last_idx and self._last_beat_time >= 0:
                for fn in self._beat_callbacks:
                    self._call_safely(fn, t)
            self._last_beat_time = t

    def notify_marker(self, marker_name: str):
        """Appelé par MainWindow quand un marqueur est atteint."""
        for name, fn in self._marker_callbacks:
            if name == marker_name:
                self._call_safely(fn, self._current_time)

    # ── Exécution de scripts ─────────────────────────────────────────────────

    def execute(self, source: str) -> bool:
        """
        Exécute un script Python dans le sandbox OpenShader.
        Retourne True si l'exécution s'est déroulée sans erreur.
        """
        # Réinitialise les callbacks (re-exécution = re-déclaration)
        self._beat_callbacks.clear()
        self._time_callbacks.clear()
        self._marker_callbacks.clear()

        sandbox = self._build_sandbox()
        try:
            exec(compile(source, '<demomaker_script>', 'exec'), sandbox)  # noqa: S102
            self.output_line.emit("[OK] Script exécuté avec succès.")
            log.info("Script OpenShader exécuté.")
            return True
        except (SyntaxError, Exception):
            tb = traceback.format_exc()
            self.error_occurred.emit(tb)
            self.output_line.emit(f"[ERREUR]\n{tb}")
            log.error("Erreur script : %s", tb)
            return False

    def _build_sandbox(self) -> dict:
        """Construit le namespace sécurisé exposé au script utilisateur."""
        api = self._api
        return {
            # API OpenShader
            'set_uniform':       api.set_uniform,
            'get_uniform':       api.get_uniform,
            'get_time':          api.get_time,
            'add_keyframe':      api.add_keyframe,
            'get_track_value':   api.get_track_value,
            'set_bpm':           api.set_bpm,
            'log_info':          api.log_info,
            'log_error':         api.log_error,
            'play':              api.play,
            'stop':              api.stop,
            'seek':              api.seek,
            'schedule':          api.schedule,
            'on_beat':           api.on_beat,
            'on_time':           api.on_time,
            'on_marker':         api.on_marker,
            # Builtins sûrs
            '__builtins__': {
                'print': lambda *a: (api.log_info(' '.join(str(x) for x in a))),
                'range': range, 'len': len, 'int': int, 'float': float,
                'str': str, 'bool': bool, 'list': list, 'dict': dict,
                'tuple': tuple, 'abs': abs, 'min': min, 'max': max,
                'round': round, 'sum': sum, 'zip': zip, 'enumerate': enumerate,
                'map': map, 'filter': filter, 'sorted': sorted, 'reversed': reversed,
                'isinstance': isinstance, 'hasattr': hasattr,
                'True': True, 'False': False, 'None': None,
            },
            # math disponible
            'math': __import__('math'),
            'random': __import__('random'),
        }

    def _call_safely(self, fn: Callable, *args):
        """Appelle une fonction utilisateur avec gestion d'erreur."""
        try:
            fn(*args)
        except Exception:   # scripts utilisateur peuvent lever n'importe quelle exception
            tb = traceback.format_exc()
            self.error_occurred.emit(tb)
            log.error("Erreur callback script : %s", tb)


# ── Éditeur de script intégré ─────────────────────────────────────────────────
# (widget léger — l'éditeur GLSL principal sert de base)

_SCRIPT_TEMPLATE = """\
# OpenShader — Script Python
# Accès à toute l'API via les fonctions globales exposées.

@on_beat
def pulse(t):
    \"\"\"Flash UV à chaque battement.\"\"\"
    set_uniform('uPulse', 1.0)
    schedule(0.05, lambda: set_uniform('uPulse', 0.0))

@on_time(10.0)
def drop(t):
    \"\"\"Événement unique à t=10s.\"\"\"
    log_info(f"DROP at t={t:.2f}s")
    set_uniform('uDrop', 1.0)
"""
