"""
tests/test_session_recorder.py
-------------------------------
Tests unitaires pour session_recorder.py

Lancer avec :  pytest tests/test_session_recorder.py -v
"""

import time
import pytest
from unittest.mock import MagicMock

# Stub minimal PyQt6 pour les tests hors affichage
import sys
import types

# ── Stub PyQt6 ──────────────────────────────────────────────────────────────
_qt_stub = types.ModuleType("PyQt6")
_qtcore  = types.ModuleType("PyQt6.QtCore")
_qtwidg  = types.ModuleType("PyQt6.QtWidgets")

class _FakeSignal:
    def __init__(self, *args): self._cbs = []
    def connect(self, cb): self._cbs.append(cb)
    def emit(self, *args):
        for cb in self._cbs: cb(*args)

class _FakeQObject:
    def __init__(self, parent=None): pass

_qtcore.QObject   = _FakeQObject
_qtcore.pyqtSignal = _FakeSignal

sys.modules["PyQt6"]           = _qt_stub
sys.modules["PyQt6.QtCore"]    = _qtcore
sys.modules["PyQt6.QtWidgets"] = _qtwidg

# Patch les imports dans le module cible
import importlib, os, sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Imports projet ───────────────────────────────────────────────────────────
# On importe directement pour éviter les dépendances Qt dans les tests
from src.timeline import Timeline, Track
from src.session_recorder import (
    SessionRecorder, RecordedEvent,
    RECORD_MODE_NORMAL, RECORD_MODE_PUNCH, RECORD_MODE_OVERDUB,
    _infer_value_type, _remove_keyframes_in_range,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def timeline():
    return Timeline(duration=60.0)

@pytest.fixture
def recorder(timeline):
    rec = SessionRecorder.__new__(SessionRecorder)
    rec._timeline       = timeline
    rec._is_recording   = False
    rec._mode           = RECORD_MODE_NORMAL
    rec._wall_start     = 0.0
    rec._timeline_start = 0.0
    rec._punch_in       = None
    rec._punch_out      = None
    rec._events         = []
    rec._time_provider  = None
    rec.default_interp  = 'linear'
    # Fake signals
    rec.recording_state_changed = _FakeSignal(bool)
    rec.event_captured          = _FakeSignal(str, float)
    return rec


# ── Tests utilitaires ─────────────────────────────────────────────────────────

class TestInferValueType:
    def test_float(self):    assert _infer_value_type(0.5)       == 'float'
    def test_int(self):      assert _infer_value_type(1)         == 'float'
    def test_vec2(self):     assert _infer_value_type((0.1, 0.2))         == 'vec2'
    def test_vec3(self):     assert _infer_value_type((1.0, 0.0, 0.0))   == 'vec3'
    def test_vec4(self):     assert _infer_value_type((1,1,1,1))          == 'vec4'
    def test_unknown(self):  assert _infer_value_type("text")   == 'float'


class TestRemoveKeyframesInRange:
    def test_removes_within_range(self, timeline):
        track = timeline.add_track("A", "uA")
        track.add_keyframe(1.0, 0.5)
        track.add_keyframe(3.0, 0.8)
        track.add_keyframe(5.0, 1.0)
        _remove_keyframes_in_range(track, 2.0, 4.0)
        times = [kf.time for kf in track.keyframes]
        assert 3.0 not in times
        assert 1.0 in times
        assert 5.0 in times

    def test_empty_track(self, timeline):
        track = timeline.add_track("B", "uB")
        _remove_keyframes_in_range(track, 0.0, 10.0)  # ne doit pas lever


# ── Tests SessionRecorder ─────────────────────────────────────────────────────

class TestRecorderBasicFlow:
    def test_initial_state(self, recorder):
        assert not recorder.is_recording
        assert recorder.mode == RECORD_MODE_NORMAL

    def test_start_stop(self, recorder):
        recorder.set_time_provider(lambda: 0.0)
        recorder.start(0.0)
        assert recorder.is_recording
        events = recorder.stop()
        assert not recorder.is_recording
        assert isinstance(events, list)

    def test_cancel_discards_events(self, recorder, timeline):
        recorder.set_time_provider(lambda: 1.0)
        recorder.start(0.0)
        recorder.record_event("uX", 0.5)
        recorder.cancel()
        assert not recorder.is_recording
        # Aucun KF ne doit avoir été ajouté
        track = timeline.get_track_by_uniform("uX")
        assert track is None

    def test_stop_creates_track_and_keyframe(self, recorder, timeline):
        t_val = [0.0]
        recorder.set_time_provider(lambda: t_val[0])
        recorder.start(0.0)
        t_val[0] = 1.5
        recorder.record_event("uIntensity", 0.75)
        t_val[0] = 3.0
        recorder.record_event("uIntensity", 0.90)
        recorder.stop()

        track = timeline.get_track_by_uniform("uIntensity")
        assert track is not None
        assert len(track.keyframes) == 2
        assert track.keyframes[0].time == pytest.approx(1.5)
        assert track.keyframes[0].value == pytest.approx(0.75)
        assert track.keyframes[1].time == pytest.approx(3.0)
        assert track.keyframes[1].value == pytest.approx(0.90)


class TestPunchMode:
    def test_punch_in_out_required(self, recorder):
        with pytest.raises(RuntimeError):
            recorder.start(0.0, mode=RECORD_MODE_PUNCH)

    def test_punch_invalid_range(self, recorder):
        with pytest.raises(ValueError):
            recorder.set_punch(5.0, 2.0)

    def test_events_outside_punch_ignored(self, recorder, timeline):
        recorder.set_punch(2.0, 5.0)
        t_val = [0.0]
        recorder.set_time_provider(lambda: t_val[0])
        recorder.start(0.0, mode=RECORD_MODE_PUNCH)

        t_val[0] = 0.5   # avant punch_in → ignoré
        recorder.record_event("uA", 0.1)
        t_val[0] = 3.0   # dans la plage → capturé
        recorder.record_event("uA", 0.5)
        t_val[0] = 7.0   # après punch_out → ignoré
        recorder.record_event("uA", 0.9)

        recorder.stop()

        track = timeline.get_track_by_uniform("uA")
        assert track is not None
        assert len(track.keyframes) == 1
        assert track.keyframes[0].time == pytest.approx(3.0)
        assert track.keyframes[0].value == pytest.approx(0.5)


class TestOverdubMode:
    def test_overdub_preserves_existing_keyframes(self, recorder, timeline):
        # Pré-remplir la piste avec des KFs
        track = timeline.add_track("Gain", "uGain")
        track.add_keyframe(0.0, 0.2)
        track.add_keyframe(2.0, 0.4)
        track.add_keyframe(8.0, 0.9)

        t_val = [0.0]
        recorder.set_time_provider(lambda: t_val[0])
        recorder.start(0.0, mode=RECORD_MODE_OVERDUB)

        # Enregistre à t=5.0 seulement
        t_val[0] = 5.0
        recorder.record_event("uGain", 0.75)
        recorder.stop()

        track = timeline.get_track_by_uniform("uGain")
        times = [kf.time for kf in track.keyframes]
        # Les KFs existants doivent être préservés
        assert 0.0 in times
        assert 2.0 in times
        assert 8.0 in times
        # Le nouveau KF doit exister
        assert 5.0 in times
        assert len(track.keyframes) == 4


class TestMultiUniform:
    def test_multiple_uniforms_create_multiple_tracks(self, recorder, timeline):
        t_val = [1.0]
        recorder.set_time_provider(lambda: t_val[0])
        recorder.start(0.0)

        recorder.record_event("uColor", (1.0, 0.0, 0.5))
        t_val[0] = 2.0
        recorder.record_event("uSpeed", 2.5)
        recorder.stop()

        assert timeline.get_track_by_uniform("uColor") is not None
        assert timeline.get_track_by_uniform("uSpeed") is not None
        assert timeline.get_track_by_uniform("uColor").value_type == 'vec3'
        assert timeline.get_track_by_uniform("uSpeed").value_type == 'float'


class TestExport:
    def test_export_to_dict_structure(self, recorder, timeline):
        recorder.set_time_provider(lambda: 0.0)
        recorder.start(0.0)
        recorder.stop()
        d = recorder.export_to_dict()
        assert 'tracks' in d
        assert 'duration' in d

    def test_export_events_log(self, recorder):
        t_val = [1.0]
        recorder.set_time_provider(lambda: t_val[0])
        recorder.start(0.0)
        recorder.record_event("uX", 0.5, source='midi')
        events = recorder.export_events_log()
        # La liste d'événements est celle avant le flush (stop non appelé)
        assert len(events) == 1
        assert events[0]['uniform'] == 'uX'
        assert events[0]['source'] == 'midi'
        recorder.cancel()
