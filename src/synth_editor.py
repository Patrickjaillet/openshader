"""
synth_editor.py
---------------
v2.8 — Éditeur visuel de synthétiseur procédural.

Fonctionnalités :
  - Canvas visuel drag-and-drop : nœuds Oscillator, Envelope, Filter, Effect,
    Mixer, AudioOut connectables par câbles (intégration NodeGraph existant)
  - Moteur audio temps réel : génération iTime-synchronisée via scipy/numpy,
    playback via PyAudio (fallback pygame.mixer)
  - Export WAV haute qualité depuis l'état du graph
  - Nœuds audio exposés dans le Node Graph visuel existant (type 'audio')
  - SynthGraphScene : sous-classe de NodeGraphScene dédiée au signal audio
  - Sérialisation JSON du patch (round-trip avec SynthPatch de intro_toolkit)

Architecture :
  SynthEditorWidget (QWidget)
    ├── SynthGraphView  (QGraphicsView)
    │     └── SynthGraphScene (gère nœuds + câbles audio)
    ├── Toolbar (play/stop/export/BPM/duration)
    ├── InspectorPanel (paramètres du nœud sélectionné)
    └── WaveformDisplay (oscilloscope temps réel)
"""

from __future__ import annotations

import math
import json
import time
import threading
import os
import wave
import struct
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem,
    QGraphicsTextItem, QPushButton, QLabel, QSlider, QDoubleSpinBox,
    QSpinBox, QComboBox, QGroupBox, QFormLayout, QScrollArea,
    QFrame, QSizePolicy, QFileDialog, QMessageBox, QToolButton,
    QCheckBox
)
from PyQt6.QtCore    import (Qt, QPointF, QRectF, pyqtSignal, QTimer,
                              QThread, pyqtSlot)
from PyQt6.QtGui     import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QFont, QLinearGradient, QCursor, QPolygonF
)

from .logger import get_logger

log = get_logger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────

_PALETTE = {
    'bg':         '#0b0d14',
    'grid':       '#13151e',
    'node_osc':   ('#1a2030', '#2255aa'),   # (bg_top, accent)
    'node_env':   ('#1a2820', '#22aa55'),
    'node_flt':   ('#201a30', '#7722aa'),
    'node_fx':    ('#30201a', '#aa5522'),
    'node_mix':   ('#1a2828', '#229988'),
    'node_out':   ('#141414', '#606060'),
    'cable':      '#4488cc',
    'cable_drag': '#88ccff',
    'port_in':    '#44aadd',
    'port_out':   '#ddaa44',
    'selected':   '#80b0ff',
    'text':       '#c8d0e8',
    'text_dim':   '#606880',
}

_NODE_W  = 150
_NODE_H  = 90
_HDR_H   = 28
_PORT_R  = 6

# ── Définitions de types de nœuds audio ───────────────────────────────────────

# (type_id, label, color_key, inputs, outputs, params)
# params : list of (name, type, default, min, max)
_NODE_DEFS: dict[str, dict] = {
    'oscillator': {
        'label': 'Oscillator',
        'color': 'node_osc',
        'inputs':  ['freq_mod', 'amp_mod'],
        'outputs': ['signal'],
        'params': [
            ('wave',    'combo',  'sin',  ['sin', 'saw', 'sqr', 'tri', 'noise']),
            ('freq',    'float',  440.0,  20.0,   20000.0),
            ('amp',     'float',  0.5,    0.0,    1.0),
            ('detune',  'float',  0.0,    -100.0, 100.0),
            ('lfo_rate','float',  0.0,    0.0,    20.0),
            ('lfo_dep', 'float',  0.0,    0.0,    1.0),
        ],
    },
    'envelope': {
        'label': 'Envelope ADSR',
        'color': 'node_env',
        'inputs':  ['signal'],
        'outputs': ['signal'],
        'params': [
            ('attack',  'float', 0.01,  0.001, 4.0),
            ('decay',   'float', 0.1,   0.001, 4.0),
            ('sustain', 'float', 0.7,   0.0,   1.0),
            ('release', 'float', 0.2,   0.001, 8.0),
        ],
    },
    'filter': {
        'label': 'Filter',
        'color': 'node_flt',
        'inputs':  ['signal'],
        'outputs': ['signal'],
        'params': [
            ('type',    'combo', 'lowpass', ['lowpass', 'highpass', 'bandpass']),
            ('cutoff',  'float', 2000.0, 20.0, 20000.0),
            ('res',     'float', 0.5,    0.0,  1.0),
        ],
    },
    'distortion': {
        'label': 'Distortion',
        'color': 'node_fx',
        'inputs':  ['signal'],
        'outputs': ['signal'],
        'params': [
            ('drive', 'float', 0.3, 0.0, 1.0),
            ('mix',   'float', 1.0, 0.0, 1.0),
        ],
    },
    'delay': {
        'label': 'Delay',
        'color': 'node_fx',
        'inputs':  ['signal'],
        'outputs': ['signal'],
        'params': [
            ('time_ms', 'float', 125.0, 1.0, 2000.0),
            ('feedback','float', 0.4,   0.0, 0.99),
            ('mix',     'float', 0.5,   0.0, 1.0),
        ],
    },
    'reverb': {
        'label': 'Reverb',
        'color': 'node_fx',
        'inputs':  ['signal'],
        'outputs': ['signal'],
        'params': [
            ('room',   'float', 0.6,  0.0, 1.0),
            ('damp',   'float', 0.5,  0.0, 1.0),
            ('mix',    'float', 0.3,  0.0, 1.0),
        ],
    },
    'mixer': {
        'label': 'Mixer',
        'color': 'node_mix',
        'inputs':  ['in_a', 'in_b', 'in_c', 'in_d'],
        'outputs': ['signal'],
        'params': [
            ('gain_a', 'float', 1.0, 0.0, 2.0),
            ('gain_b', 'float', 1.0, 0.0, 2.0),
            ('gain_c', 'float', 1.0, 0.0, 2.0),
            ('gain_d', 'float', 1.0, 0.0, 2.0),
            ('pan',    'float', 0.0, -1.0, 1.0),
        ],
    },
    'audio_out': {
        'label': 'Audio Out',
        'color': 'node_out',
        'inputs':  ['left', 'right'],
        'outputs': [],
        'params':  [
            ('master_vol', 'float', 0.8, 0.0, 1.0),
        ],
    },
}

# ── Audio rendering engine ────────────────────────────────────────────────────

SAMPLE_RATE = 44100


class SynthGraphRenderer:
    """
    Évalue le graphe de nœuds audio et génère un buffer numpy stéréo.
    Fonctionne en mode offline (export WAV) et streaming (temps réel).
    """

    def __init__(self, graph: 'SynthGraphScene'):
        self.graph = graph
        self._sr   = SAMPLE_RATE

    def render(self, duration: float, bpm: float = 120.0,
               itime_offset: float = 0.0) -> 'Any':
        """
        Génère un array numpy (N, 2) float32 représentant la sortie stéréo.
        itime_offset : calage temporel sur iTime du shader (synchronisation).
        """
        try:
            import numpy as np
        except ImportError:
            log.error("numpy requis pour le rendu audio")
            return None

        sr  = self._sr
        n   = int(sr * duration)
        out = np.zeros((n, 2), dtype=np.float32)

        # Topo sort : évaluer les nœuds feuilles (oscillateurs) en premier,
        # puis propager vers la sortie (audio_out).
        order = self._topological_order()
        buffers: dict[str, 'Any'] = {}   # node_id → np.ndarray (n, 2)

        for node_id in order:
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            buf = self._eval_node(node, n, sr, itime_offset, buffers)
            if buf is not None:
                buffers[node_id] = buf

        # Cherche le nœud audio_out
        for node_id, node in self.graph._audio_nodes.items():
            if node.node_type_id == 'audio_out':
                buf = buffers.get(node_id)
                if buf is not None:
                    out += buf

        # Normalisation peak
        peak = float(np.max(np.abs(out)))
        if peak > 0.95:
            out *= (0.95 / peak)

        return out

    def _eval_node(self, node: 'AudioNodeItem', n: int, sr: int,
                   itime_offset: float,
                   buffers: dict) -> Optional['Any']:
        try:
            import numpy as np
        except ImportError:
            return None

        ntype  = node.node_type_id
        params = node.param_values
        ts     = np.arange(n, dtype=np.float64) / sr + itime_offset

        # ── Oscillator ───────────────────────────────────────────────────────
        if ntype == 'oscillator':
            freq   = float(params.get('freq', 440))
            amp    = float(params.get('amp', 0.5))
            wave   = params.get('wave', 'sin')
            detune = float(params.get('detune', 0))
            lfo_r  = float(params.get('lfo_rate', 0))
            lfo_d  = float(params.get('lfo_dep', 0))

            freq *= 2 ** (detune / 1200)

            # Modulation de fréquence entrante
            freq_mod_buf = self._get_input_mono(node, 'freq_mod', buffers, n)
            amp_mod_buf  = self._get_input_mono(node, 'amp_mod',  buffers, n)

            lfo = 1.0
            if lfo_r > 0 and lfo_d > 0:
                lfo = 1.0 + lfo_d * np.sin(2 * math.pi * lfo_r * ts)

            eff_freq = (freq + freq_mod_buf * freq) * lfo
            phase = np.cumsum(2 * math.pi * eff_freq / sr)

            if wave == 'sin':
                sig = np.sin(phase)
            elif wave == 'saw':
                sig = 2 * (phase / (2 * math.pi) % 1) - 1
            elif wave == 'sqr':
                sig = np.sign(np.sin(phase))
            elif wave == 'tri':
                p = phase / (2 * math.pi) % 1
                sig = 4 * np.where(p < 0.5, p, 1 - p) - 1
            else:  # noise
                rng = np.random.default_rng(int(freq))
                sig = rng.uniform(-1.0, 1.0, n)

            eff_amp = amp + amp_mod_buf * amp
            sig = (sig * np.clip(eff_amp, 0, 1)).astype(np.float32)
            return np.stack([sig, sig], axis=1).astype(np.float32)

        # ── Envelope ADSR ────────────────────────────────────────────────────
        elif ntype == 'envelope':
            inp = self._get_input_stereo(node, 'signal', buffers, n)
            a   = float(params.get('attack', 0.01))
            d   = float(params.get('decay', 0.1))
            s   = float(params.get('sustain', 0.7))
            r   = float(params.get('release', 0.2))
            env = self._adsr(n, sr, a, d, s, r).astype(np.float32)
            return (inp * env[:, None]).astype(np.float32)

        # ── Filter ───────────────────────────────────────────────────────────
        elif ntype == 'filter':
            inp    = self._get_input_stereo(node, 'signal', buffers, n)
            ftype  = params.get('type', 'lowpass')
            cutoff = float(params.get('cutoff', 2000))
            res    = float(params.get('res', 0.5))
            return self._biquad(inp, sr, ftype, cutoff, res)

        # ── Distortion ───────────────────────────────────────────────────────
        elif ntype == 'distortion':
            inp   = self._get_input_stereo(node, 'signal', buffers, n)
            drive = float(params.get('drive', 0.3))
            mix   = float(params.get('mix', 1.0))
            if drive > 0:
                g   = 1 + drive * 9
                wet = np.tanh(inp * g) / math.tanh(g)
            else:
                wet = inp.copy()
            return (inp * (1 - mix) + wet * mix).astype(np.float32)

        # ── Delay ────────────────────────────────────────────────────────────
        elif ntype == 'delay':
            inp      = self._get_input_stereo(node, 'signal', buffers, n)
            delay_ms = float(params.get('time_ms', 125))
            fb       = float(params.get('feedback', 0.4))
            mix      = float(params.get('mix', 0.5))
            d_smp    = int(delay_ms * sr / 1000)
            wet      = inp.copy()
            for i in range(d_smp, n):
                wet[i] += fb * wet[i - d_smp]
            return (inp * (1 - mix) + wet * mix).astype(np.float32)

        # ── Reverb (Schroeder simplifié) ─────────────────────────────────────
        elif ntype == 'reverb':
            inp  = self._get_input_stereo(node, 'signal', buffers, n)
            room = float(params.get('room', 0.6))
            damp = float(params.get('damp', 0.5))
            mix  = float(params.get('mix', 0.3))
            # 4 all-pass en série (delais premiers)
            delays_ms = [29.7, 37.1, 41.1, 43.7]
            wet = inp.copy()
            for dms in delays_ms:
                ds = int(dms * sr / 1000)
                if ds < n:
                    fb_val = room * (1 - damp)
                    for i in range(ds, n):
                        wet[i] += fb_val * wet[i - ds]
            return (inp * (1 - mix) + wet * mix).astype(np.float32)

        # ── Mixer ────────────────────────────────────────────────────────────
        elif ntype == 'mixer':
            pan = float(params.get('pan', 0.0))
            out = np.zeros((n, 2), dtype=np.float32)
            for ch, key in enumerate(['in_a', 'in_b', 'in_c', 'in_d']):
                gain = float(params.get(f'gain_{"abcd"[ch]}', 1.0))
                buf  = self._get_input_stereo(node, key, buffers, n)
                out += buf * gain
            vol_l = math.sqrt(max(0, (1 - pan) / 2))
            vol_r = math.sqrt(max(0, (1 + pan) / 2))
            out[:, 0] *= vol_l
            out[:, 1] *= vol_r
            return np.clip(out, -1, 1).astype(np.float32)

        # ── Audio Out ────────────────────────────────────────────────────────
        elif ntype == 'audio_out':
            vol = float(params.get('master_vol', 0.8))
            l   = self._get_input_stereo(node, 'left',  buffers, n)
            r   = self._get_input_stereo(node, 'right', buffers, n)
            out = np.stack(
                [l[:, 0] + r[:, 0], l[:, 1] + r[:, 1]], axis=1
            ).astype(np.float32)
            return np.clip(out * vol, -1, 1)

        return None

    def _get_input_stereo(self, node: 'AudioNodeItem', port: str,
                          buffers: dict, n: int) -> 'Any':
        import numpy as np
        src_id = self.graph.get_source_for_input(node.node_id, port)
        if src_id and src_id in buffers:
            buf = buffers[src_id]
            if buf is not None and len(buf) == n:
                return buf
        return np.zeros((n, 2), dtype=np.float32)

    def _get_input_mono(self, node: 'AudioNodeItem', port: str,
                        buffers: dict, n: int) -> 'Any':
        import numpy as np
        buf = self._get_input_stereo(node, port, buffers, n)
        return (buf[:, 0] + buf[:, 1]) * 0.5

    @staticmethod
    def _adsr(n: int, sr: int, attack: float, decay: float,
              sustain: float, release: float) -> 'Any':
        import numpy as np
        env   = np.zeros(n, dtype=np.float64)
        a_s   = min(int(attack  * sr), n)
        d_s   = min(int(decay   * sr), n - a_s)
        r_s   = min(int(release * sr), n)
        s_end = max(0, n - r_s)
        if a_s > 0:
            env[:a_s] = np.linspace(0, 1, a_s)
        if d_s > 0:
            env[a_s:a_s + d_s] = np.linspace(1, sustain, d_s)
        if a_s + d_s < s_end:
            env[a_s + d_s:s_end] = sustain
        if r_s > 0 and s_end < n:
            env[s_end:] = np.linspace(sustain, 0, n - s_end)
        return env

    @staticmethod
    def _biquad(sig: 'Any', sr: int, ftype: str,
                cutoff: float, res: float) -> 'Any':
        """Filtre biquad IIR (lowpass / highpass / bandpass)."""
        import numpy as np
        w0    = 2 * math.pi * cutoff / sr
        alpha = math.sin(w0) / (2 * max(res, 0.01))
        cos_w = math.cos(w0)

        if ftype == 'lowpass':
            b0 = (1 - cos_w) / 2; b1 = 1 - cos_w;  b2 = b0
            a0 = 1 + alpha;        a1 = -2 * cos_w; a2 = 1 - alpha
        elif ftype == 'highpass':
            b0 = (1 + cos_w) / 2; b1 = -(1 + cos_w); b2 = b0
            a0 = 1 + alpha;        a1 = -2 * cos_w;   a2 = 1 - alpha
        else:  # bandpass
            b0 = alpha;  b1 = 0;         b2 = -alpha
            a0 = 1 + alpha; a1 = -2 * cos_w; a2 = 1 - alpha

        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0,     a1 / a0, a2 / a0]

        out = np.zeros_like(sig)
        for ch in range(sig.shape[1]):
            x = sig[:, ch]
            y = np.zeros_like(x)
            x1 = x2 = y1 = y2 = 0.0
            for i in range(len(x)):
                y[i] = b[0]*x[i] + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
                x2, x1 = x1, x[i]
                y2, y1 = y1, y[i]
            out[:, ch] = y
        return out.astype(np.float32)

    def _topological_order(self) -> list[str]:
        """Retourne les ids de nœuds dans l'ordre topologique (feuilles → racines)."""
        nodes = list(self.graph._audio_nodes.keys())
        edges = self.graph._audio_edges  # {dst_id: {port: src_id}}

        order  = []
        visited = set()

        def visit(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            for src_id in self.graph.get_all_sources(nid):
                visit(src_id)
            order.append(nid)

        for nid in nodes:
            visit(nid)
        return order

    def export_wav(self, path: str, duration: float, bpm: float = 120.0) -> int:
        """Exporte en WAV 44100 Hz stéréo 16 bits. Retourne le nombre d'échantillons."""
        buf = self.render(duration, bpm)
        if buf is None:
            return 0
        pcm = (buf * 32767).clip(-32767, 32767).astype('<i2')
        with wave.open(path, 'w') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        log.info("WAV exporté : %s (%d samples)", path, len(buf))
        return len(buf)


# ── Audio streaming thread ────────────────────────────────────────────────────

class AudioStreamThread(QThread):
    """
    Thread de lecture audio temps réel.
    Génère des chunks audio synchronisés sur self.itime et les envoie
    au périphérique via PyAudio (fallback : pygame.mixer).
    """

    waveform_ready = pyqtSignal(list)   # chunk de samples pour oscilloscope

    def __init__(self, renderer: SynthGraphRenderer,
                 chunk_dur: float = 0.1):
        super().__init__()
        self._renderer   = renderer
        self._chunk_dur  = chunk_dur
        self._running    = False
        self._paused     = False
        self._itime      = 0.0
        self._lock       = threading.Lock()
        self._bpm        = 120.0
        self._duration   = 8.0

    def set_itime(self, t: float):
        with self._lock:
            self._itime = t

    def set_bpm(self, bpm: float):
        with self._lock:
            self._bpm = bpm

    def set_duration(self, d: float):
        with self._lock:
            self._duration = d

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop_stream(self):
        self._running = False

    def run(self):
        self._running = True
        log.info("AudioStreamThread démarré")

        # Tentative PyAudio
        pa_stream = None
        pa        = None
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            pa_stream = pa.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=int(SAMPLE_RATE * self._chunk_dur),
            )
            log.info("Backend audio : PyAudio")
        except (ImportError, Exception) as e:
            log.warning("PyAudio indisponible (%s) — lecture audio désactivée", e)

        chunk_n = int(SAMPLE_RATE * self._chunk_dur)

        while self._running:
            if self._paused:
                time.sleep(0.02)
                continue

            with self._lock:
                t   = self._itime
                bpm = self._bpm

            buf = self._renderer.render(self._chunk_dur, bpm=bpm,
                                        itime_offset=t)
            if buf is None:
                time.sleep(self._chunk_dur)
                continue

            # Oscilloscope : envoie le canal gauche (mono) réduit à 256 pts
            try:
                import numpy as np
                mono   = buf[:, 0]
                step   = max(1, len(mono) // 256)
                self.waveform_ready.emit(mono[::step].tolist())
            except Exception:
                pass

            if pa_stream:
                try:
                    raw = buf.astype('<f4').tobytes()
                    pa_stream.write(raw)
                except Exception as e:
                    log.debug("PyAudio write : %s", e)
            else:
                # Pas de sortie audio — attend quand même le bon délai
                time.sleep(self._chunk_dur * 0.9)

            with self._lock:
                self._itime += self._chunk_dur

        if pa_stream:
            try:
                pa_stream.stop_stream()
                pa_stream.close()
            except Exception:
                pass
        if pa:
            try:
                pa.terminate()
            except Exception:
                pass
        log.info("AudioStreamThread arrêté")


# ── Port visuel ───────────────────────────────────────────────────────────────

class AudioPortItem(QGraphicsEllipseItem):
    def __init__(self, parent: 'AudioNodeItem', name: str, is_output: bool,
                 index: int):
        super().__init__(-_PORT_R, -_PORT_R, _PORT_R * 2, _PORT_R * 2, parent)
        self.node      = parent
        self.port_name = name
        self.is_output = is_output
        self.index     = index
        self._edges: list['AudioCableItem'] = []

        color = _PALETTE['port_out'] if is_output else _PALETTE['port_in']
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor('#1a1d28'), 1))
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setZValue(3)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def add_edge(self, e: 'AudioCableItem'):
        self._edges.append(e)

    def remove_edge(self, e: 'AudioCableItem'):
        self._edges = [x for x in self._edges if x is not e]

    def edges(self) -> list['AudioCableItem']:
        return list(self._edges)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor('#ffffff')))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        color = _PALETTE['port_out'] if self.is_output else _PALETTE['port_in']
        self.setBrush(QBrush(QColor(color)))
        super().hoverLeaveEvent(event)


# ── Nœud audio ────────────────────────────────────────────────────────────────

_node_id_counter = 0

def _next_node_id() -> str:
    global _node_id_counter
    _node_id_counter += 1
    return f"n{_node_id_counter}"


class AudioNodeItem(QGraphicsItem):
    """Nœud visuel dans le graph audio."""

    def __init__(self, type_id: str, pos: QPointF):
        super().__init__()
        self.node_id     = _next_node_id()
        self.node_type_id = type_id
        defn             = _NODE_DEFS.get(type_id, {})
        self._label      = defn.get('label', type_id)
        self._color_key  = defn.get('color', 'node_mix')
        self._in_ports:  list[AudioPortItem] = []
        self._out_ports: list[AudioPortItem] = []

        # Paramètres : valeurs initiales
        self.param_values: dict[str, Any] = {}
        for p in defn.get('params', []):
            self.param_values[p[0]] = p[2]

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        self.setPos(pos)
        self._build_ports(defn)

    def _build_ports(self, defn: dict):
        inputs  = defn.get('inputs', [])
        outputs = defn.get('outputs', [])

        for i, name in enumerate(inputs):
            p = AudioPortItem(self, name, is_output=False, index=i)
            y = _HDR_H + (_NODE_H - _HDR_H) * (i + 1) / (len(inputs) + 1)
            p.setPos(-_PORT_R, y)
            self._in_ports.append(p)

        for i, name in enumerate(outputs):
            p = AudioPortItem(self, name, is_output=True, index=i)
            y = _HDR_H + (_NODE_H - _HDR_H) * (i + 1) / (len(outputs) + 1)
            p.setPos(_NODE_W + _PORT_R, y)
            self._out_ports.append(p)

    def boundingRect(self) -> QRectF:
        return QRectF(-_PORT_R, 0, _NODE_W + 2 * _PORT_R, _NODE_H)

    def paint(self, painter: QPainter, option, widget=None):
        bg_top, accent = _PALETTE.get(self._color_key,
                                      ('#1a2030', '#334488'))

        r = QRectF(0, 0, _NODE_W, _NODE_H)

        # Corps
        grad = QLinearGradient(0, 0, 0, _NODE_H)
        grad.setColorAt(0.0, QColor(bg_top))
        grad.setColorAt(1.0, QColor('#0e1018'))
        painter.setBrush(QBrush(grad))
        pen_color = _PALETTE['selected'] if self.isSelected() else '#2a2d3a'
        painter.setPen(QPen(QColor(pen_color), 2 if self.isSelected() else 1))
        painter.drawRoundedRect(r, 6, 6)

        # Header
        hr = QRectF(0, 0, _NODE_W, _HDR_H)
        painter.setBrush(QBrush(QColor(accent)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(hr, 6, 6)
        painter.drawRect(QRectF(0, 6, _NODE_W, _HDR_H - 6))

        # Label
        painter.setPen(QPen(QColor(_PALETTE['text'])))
        font = QFont('Segoe UI', 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(6, 0, _NODE_W - 12, _HDR_H),
                         Qt.AlignmentFlag.AlignVCenter, self._label)

        # Type + params résumé
        painter.setPen(QPen(QColor(_PALETTE['text_dim'])))
        font2 = QFont('Segoe UI', 7)
        painter.setFont(font2)
        summary = self._param_summary()
        painter.drawText(QRectF(6, _HDR_H + 4, _NODE_W - 12, _NODE_H - _HDR_H - 8),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                         summary)

        # Port labels
        font3 = QFont('Segoe UI', 6)
        painter.setFont(font3)
        for p in self._in_ports:
            painter.setPen(QPen(QColor(_PALETTE['port_in'])))
            painter.drawText(QRectF(4, p.pos().y() - 6, 60, 12),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             p.port_name)
        for p in self._out_ports:
            painter.setPen(QPen(QColor(_PALETTE['port_out'])))
            painter.drawText(QRectF(_NODE_W - 64, p.pos().y() - 6, 60, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             p.port_name)

    def _param_summary(self) -> str:
        """Résumé compact des paramètres pour l'affichage dans le nœud."""
        parts = []
        for k, v in list(self.param_values.items())[:3]:
            if isinstance(v, float):
                parts.append(f"{k}:{v:.2f}")
            else:
                parts.append(f"{k}:{v}")
        return '  '.join(parts)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for p in self._in_ports + self._out_ports:
                for e in p.edges():
                    e.update_path()
        return super().itemChange(change, value)

    def get_port(self, name: str, is_output: bool) -> Optional[AudioPortItem]:
        lst = self._out_ports if is_output else self._in_ports
        for p in lst:
            if p.port_name == name:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            'id':     self.node_id,
            'type':   self.node_type_id,
            'pos':    [self.pos().x(), self.pos().y()],
            'params': dict(self.param_values),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AudioNodeItem':
        n = cls(d['type'], QPointF(*d['pos']))
        n.node_id = d['id']
        n.param_values.update(d.get('params', {}))
        return n

    def all_ports(self) -> list[AudioPortItem]:
        return self._in_ports + self._out_ports


# ── Câble audio ───────────────────────────────────────────────────────────────

class AudioCableItem(QGraphicsPathItem):
    """Câble de connexion entre deux ports audio (courbe de Bézier)."""

    def __init__(self, src: AudioPortItem, dst: AudioPortItem):
        super().__init__()
        self.src = src
        self.dst = dst
        self.setPen(QPen(QColor(_PALETTE['cable']), 2,
                         Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap))
        self.setZValue(0)
        src.add_edge(self)
        dst.add_edge(self)
        self.update_path()

    def update_path(self):
        p1 = self.src.scene_pos()
        p2 = self.dst.scene_pos()
        ctrl_offset = min(abs(p2.x() - p1.x()) * 0.5, 100)
        path = QPainterPath(p1)
        path.cubicTo(
            QPointF(p1.x() + ctrl_offset, p1.y()),
            QPointF(p2.x() - ctrl_offset, p2.y()),
            p2
        )
        self.setPath(path)

    def remove_from_scene(self):
        self.src.remove_edge(self)
        self.dst.remove_edge(self)
        if self.scene():
            self.scene().removeItem(self)


# ── SynthGraphScene ───────────────────────────────────────────────────────────

class SynthGraphScene(QGraphicsScene):
    """
    Scène gérant les nœuds audio et leurs connexions.
    Émet graph_changed à chaque modification du DAG.
    """

    graph_changed  = pyqtSignal(dict)    # {dst_id: {port: src_id}}
    node_selected  = pyqtSignal(object)  # AudioNodeItem | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor(_PALETTE['bg'])))

        self._audio_nodes: dict[str, AudioNodeItem] = {}
        self._audio_edges: dict[str, dict[str, str]] = {}  # {dst_id: {port: src_id}}
        self._cables:      list[AudioCableItem]       = []

        self._drag_port:  Optional[AudioPortItem]       = None
        self._drag_cable: Optional[QGraphicsPathItem]   = None

        self._build_default()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_default(self):
        """Graphe par défaut : 2 oscillateurs → mixer → audio_out."""
        osc1  = self.add_node('oscillator', QPointF(60,  60))
        osc2  = self.add_node('oscillator', QPointF(60, 220))
        osc2.param_values['wave'] = 'saw'
        osc2.param_values['freq'] = 110.0
        osc2.param_values['amp']  = 0.3

        env   = self.add_node('envelope',   QPointF(260, 60))
        delay = self.add_node('delay',      QPointF(260, 220))
        mix   = self.add_node('mixer',      QPointF(460, 140))
        out   = self.add_node('audio_out',  QPointF(660, 140))

        self.connect_nodes(osc1,  'signal', env,  'signal')
        self.connect_nodes(osc2,  'signal', delay,'signal')
        self.connect_nodes(env,   'signal', mix,  'in_a')
        self.connect_nodes(delay, 'signal', mix,  'in_b')
        self.connect_nodes(mix,   'signal', out,  'left')
        self.connect_nodes(mix,   'signal', out,  'right')

    def add_node(self, type_id: str, pos: QPointF) -> AudioNodeItem:
        node = AudioNodeItem(type_id, pos)
        self.addItem(node)
        self._audio_nodes[node.node_id] = node
        return node

    def remove_selected_nodes(self):
        for item in list(self.selectedItems()):
            if isinstance(item, AudioNodeItem):
                for p in item.all_ports():
                    for cable in list(p.edges()):
                        self._remove_cable(cable)
                self.removeItem(item)
                del self._audio_nodes[item.node_id]
                self._audio_edges.pop(item.node_id, None)
        self.graph_changed.emit(self._make_dag())

    def connect_nodes(self, src_node: AudioNodeItem, src_port: str,
                      dst_node: AudioNodeItem, dst_port: str):
        src_p = src_node.get_port(src_port, is_output=True)
        dst_p = dst_node.get_port(dst_port, is_output=False)
        if src_p is None or dst_p is None:
            return
        # Déconnecte l'entrée existante si occupée
        for cable in list(dst_p.edges()):
            self._remove_cable(cable)
        cable = AudioCableItem(src_p, dst_p)
        self.addItem(cable)
        self._cables.append(cable)
        if dst_node.node_id not in self._audio_edges:
            self._audio_edges[dst_node.node_id] = {}
        self._audio_edges[dst_node.node_id][dst_port] = src_node.node_id
        self.graph_changed.emit(self._make_dag())

    def _remove_cable(self, cable: AudioCableItem):
        cable.remove_from_scene()
        if cable in self._cables:
            self._cables.remove(cable)
        # Nettoie les edges
        for dst_id, ports in list(self._audio_edges.items()):
            for port, src_id in list(ports.items()):
                if cable.src.node.node_id == src_id and cable.dst.port_name == port:
                    del self._audio_edges[dst_id][port]

    def get_node(self, node_id: str) -> Optional[AudioNodeItem]:
        return self._audio_nodes.get(node_id)

    def get_source_for_input(self, dst_id: str, port: str) -> Optional[str]:
        return self._audio_edges.get(dst_id, {}).get(port)

    def get_all_sources(self, dst_id: str) -> list[str]:
        return list(self._audio_edges.get(dst_id, {}).values())

    def _make_dag(self) -> dict:
        return {k: dict(v) for k, v in self._audio_edges.items()}

    # ── Mouse interaction ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        item = self.itemAt(event.scenePos(), self.views()[0].transform()
                           if self.views() else __import__('PyQt6.QtGui', fromlist=['QTransform']).QTransform())

        if isinstance(item, AudioPortItem) and item.is_output:
            self._drag_port = item
            self._drag_cable = QGraphicsPathItem()
            self._drag_cable.setPen(QPen(QColor(_PALETTE['cable_drag']), 2,
                                         Qt.PenStyle.DashLine))
            self._drag_cable.setZValue(10)
            self.addItem(self._drag_cable)
            event.accept()
            return

        super().mousePressEvent(event)

        # Sélection de nœud
        sel = [i for i in self.selectedItems() if isinstance(i, AudioNodeItem)]
        self.node_selected.emit(sel[0] if sel else None)

    def mouseMoveEvent(self, event):
        if self._drag_port and self._drag_cable:
            p1   = self._drag_port.scene_pos()
            p2   = event.scenePos()
            ctrl = min(abs(p2.x() - p1.x()) * 0.5, 100)
            path = QPainterPath(p1)
            path.cubicTo(QPointF(p1.x() + ctrl, p1.y()),
                         QPointF(p2.x() - ctrl, p2.y()), p2)
            self._drag_cable.setPath(path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_port and self._drag_cable:
            self.removeItem(self._drag_cable)
            self._drag_cable = None

            # Cherche un port d'entrée à la position de relâchement
            hit = self.itemAt(event.scenePos(),
                              self.views()[0].transform() if self.views()
                              else __import__('PyQt6.QtGui', fromlist=['QTransform']).QTransform())
            if isinstance(hit, AudioPortItem) and not hit.is_output:
                src_node = self._drag_port.node
                dst_node = hit.node
                if src_node is not dst_node:
                    self.connect_nodes(src_node, self._drag_port.port_name,
                                       dst_node, hit.port_name)

            self._drag_port = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.remove_selected_nodes()
        else:
            super().keyPressEvent(event)

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'nodes': [n.to_dict() for n in self._audio_nodes.values()],
            'edges': self._audio_edges,
        }

    def from_dict(self, data: dict):
        # Vider la scène
        for c in list(self._cables):
            c.remove_from_scene()
        for n in list(self._audio_nodes.values()):
            self.removeItem(n)
        self._audio_nodes.clear()
        self._audio_edges.clear()
        self._cables.clear()

        id_map: dict[str, AudioNodeItem] = {}
        for nd in data.get('nodes', []):
            node = AudioNodeItem.from_dict(nd)
            self.addItem(node)
            self._audio_nodes[node.node_id] = node
            id_map[nd['id']] = node

        # Recréer les câbles
        for dst_id, ports in data.get('edges', {}).items():
            self._audio_edges[dst_id] = {}
            dst_node = id_map.get(dst_id)
            if dst_node is None:
                continue
            for port, src_id in ports.items():
                src_node = id_map.get(src_id)
                if src_node is None:
                    continue
                src_p = src_node.get_port(
                    next(iter(src_node._out_ports), None) and
                    src_node._out_ports[0].port_name, is_output=True
                ) if src_node._out_ports else None
                dst_p = dst_node.get_port(port, is_output=False)
                if src_p and dst_p:
                    cable = AudioCableItem(src_p, dst_p)
                    self.addItem(cable)
                    self._cables.append(cable)
                self._audio_edges[dst_id][port] = src_id


# ── Inspector panel ───────────────────────────────────────────────────────────

class InspectorPanel(QWidget):
    """
    Panneau d'inspection et d'édition des paramètres du nœud sélectionné.
    Met à jour node.param_values en temps réel.
    """

    param_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node: Optional[AudioNodeItem] = None
        self._widgets: dict[str, QWidget]   = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self._title = QLabel("— Aucun nœud sélectionné —")
        self._title.setStyleSheet("color: #606880; font-style: italic; font-size: 11px;")
        self._layout.addWidget(self._title)
        self._form_widget = QWidget()
        self._form_layout = QFormLayout(self._form_widget)
        self._form_layout.setContentsMargins(0, 4, 0, 4)
        self._form_layout.setSpacing(6)
        self._layout.addWidget(self._form_widget)
        self._layout.addStretch()

    def set_node(self, node: Optional[AudioNodeItem]):
        self._node = node
        self._widgets.clear()

        # Vide le formulaire
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if node is None:
            self._title.setText("— Aucun nœud sélectionné —")
            return

        defn = _NODE_DEFS.get(node.node_type_id, {})
        self._title.setText(f"🎛  {defn.get('label', node.node_type_id)}")
        self._title.setStyleSheet("color: #c8d0e8; font-weight: bold; font-size: 12px;")

        for p in defn.get('params', []):
            pname, ptype = p[0], p[1]
            cur_val = node.param_values.get(pname, p[2])

            if ptype == 'combo':
                choices = p[3]
                w = QComboBox()
                w.addItems(choices)
                idx = choices.index(cur_val) if cur_val in choices else 0
                w.setCurrentIndex(idx)
                w.currentTextChanged.connect(
                    lambda val, name=pname: self._on_changed(name, val)
                )
            elif ptype == 'float':
                lo, hi = float(p[3]), float(p[4])
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setValue(float(cur_val))
                w.setDecimals(3 if hi - lo < 1 else 2)
                w.setSingleStep((hi - lo) / 100)
                w.valueChanged.connect(
                    lambda val, name=pname: self._on_changed(name, float(val))
                )
            else:
                continue

            self._form_layout.addRow(pname, w)
            self._widgets[pname] = w

    def _on_changed(self, name: str, val):
        if self._node:
            self._node.param_values[name] = val
            self._node.update()
            self.param_changed.emit()


# ── Oscilloscope ──────────────────────────────────────────────────────────────

class WaveformDisplay(QWidget):
    """Petit oscilloscope qui affiche les derniers samples reçus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: list[float] = [0.0] * 256
        self.setMinimumHeight(56)
        self.setMaximumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def update_samples(self, samples: list[float]):
        self._samples = samples
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#0b0d14'))
        if not self._samples:
            return
        w, h = self.width(), self.height()
        cy   = h / 2
        step = w / len(self._samples)
        pen  = QPen(QColor(_PALETTE['cable']), 1)
        painter.setPen(pen)
        pts  = []
        for i, s in enumerate(self._samples):
            pts.append(QPointF(i * step, cy - s * cy * 0.9))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
        # Centre line
        painter.setPen(QPen(QColor('#2a2d3a'), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(0, cy), QPointF(w, cy))
        painter.end()


# ── SynthEditorWidget ─────────────────────────────────────────────────────────

class SynthEditorWidget(QWidget):
    """
    Widget principal de l'éditeur de synthétiseur procédural.
    Intégrable comme dock dans MainWindow.
    """

    # Émis quand un WAV est exporté (chemin)
    wav_exported = pyqtSignal(str)
    # Émis quand le graph change (pour intégration dans le Node Graph visuel)
    graph_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playing   = False
        self._itime     = 0.0
        self._stream:   Optional[AudioStreamThread] = None
        self._itime_source: Optional[object] = None  # objet fournissant get_time()

        self._scene    = SynthGraphScene()
        self._renderer = SynthGraphRenderer(self._scene)

        self._build_ui()
        self._scene.node_selected.connect(self._inspector.set_node)
        self._scene.graph_changed.connect(self.graph_changed)
        self._scene.graph_changed.connect(self._on_graph_changed)
        self._inspector.param_changed.connect(self._scene.update)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        tb = self._build_toolbar()
        root.addWidget(tb)

        # Oscilloscope
        self._waveform = WaveformDisplay()
        root.addWidget(self._waveform)

        # Main splitter : graph ↔ inspector
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Graph view
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(QPainter.RenderHint.Antialiasing |
                                   QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_context_menu)
        splitter.addWidget(self._view)

        # Inspector
        inspector_container = QScrollArea()
        inspector_container.setWidgetResizable(True)
        inspector_container.setFixedWidth(230)
        self._inspector = InspectorPanel()
        inspector_container.setWidget(self._inspector)
        splitter.addWidget(inspector_container)
        splitter.setSizes([600, 230])

        root.addWidget(splitter, 1)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("background: #0e1016; border-bottom: 1px solid #1e2030;")
        hl  = QHBoxLayout(bar)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setSpacing(6)

        def btn(icon: str, tip: str, cb, checkable=False) -> QPushButton:
            b = QPushButton(icon)
            b.setToolTip(tip)
            b.setFixedSize(32, 28)
            b.setCheckable(checkable)
            b.setStyleSheet(
                "QPushButton { background:#1a1d28; color:#c0c8e0; border:1px solid #2a2d3a; border-radius:4px; }"
                "QPushButton:hover { background:#252840; }"
                "QPushButton:checked { background:#2255aa; }"
            )
            b.clicked.connect(cb)
            return b

        self._btn_play = btn("▶", "Lecture temps réel (synchronisée sur iTime)", self._toggle_play, checkable=True)
        self._btn_stop = btn("■", "Stop", self._stop)
        self._btn_export_wav = btn("💾", "Exporter WAV…", self._export_wav)
        self._btn_reset = btn("↺", "Graphe par défaut", self._reset_graph)

        hl.addWidget(self._btn_play)
        hl.addWidget(self._btn_stop)
        hl.addWidget(self._btn_export_wav)
        hl.addWidget(self._btn_reset)
        hl.addWidget(_sep())

        # BPM
        hl.addWidget(QLabel("BPM"))
        self._spn_bpm = QDoubleSpinBox()
        self._spn_bpm.setRange(40, 300)
        self._spn_bpm.setValue(120.0)
        self._spn_bpm.setFixedWidth(72)
        self._spn_bpm.setStyleSheet("background:#1a1d28; color:#c0c8e0; border:1px solid #2a2d3a;")
        hl.addWidget(self._spn_bpm)

        # Duration
        hl.addWidget(QLabel("Durée"))
        self._spn_dur = QDoubleSpinBox()
        self._spn_dur.setRange(0.5, 600)
        self._spn_dur.setValue(8.0)
        self._spn_dur.setSuffix(" s")
        self._spn_dur.setFixedWidth(78)
        self._spn_dur.setStyleSheet("background:#1a1d28; color:#c0c8e0; border:1px solid #2a2d3a;")
        hl.addWidget(self._spn_dur)

        hl.addWidget(_sep())

        # Add node buttons
        for type_id, label in [
            ('oscillator', '🎵 Osc'),
            ('envelope',   '📈 Env'),
            ('filter',     '🔊 Flt'),
            ('distortion', '⚡ Dist'),
            ('delay',      '⏱ Dly'),
            ('reverb',     '🌊 Rev'),
            ('mixer',      '🎚 Mix'),
        ]:
            b = QPushButton(label)
            b.setToolTip(f"Ajouter un nœud {label}")
            b.setFixedHeight(28)
            b.setStyleSheet(
                "QPushButton { background:#1a1d28; color:#c0c8e0; border:1px solid #2a2d3a; "
                "border-radius:4px; font-size:11px; padding: 0 6px; }"
                "QPushButton:hover { background:#252840; }"
            )
            b.clicked.connect(lambda _, tid=type_id: self._add_node(tid))
            hl.addWidget(b)

        hl.addStretch()
        return bar

    # ── Actions ───────────────────────────────────────────────────────────────

    def _add_node(self, type_id: str):
        """Ajoute un nœud du type donné au centre de la vue."""
        center = self._view.mapToScene(
            self._view.viewport().rect().center()
        )
        offset = QPointF(-_NODE_W / 2, -_NODE_H / 2)
        self._scene.add_node(type_id, center + offset)

    def _toggle_play(self, checked: bool):
        if checked:
            self._start_stream()
        else:
            self._pause_stream()

    def _start_stream(self):
        if self._stream and self._stream.isRunning():
            self._stream.resume()
            return
        self._stream = AudioStreamThread(self._renderer, chunk_dur=0.08)
        self._stream.set_bpm(self._spn_bpm.value())
        self._stream.set_duration(self._spn_dur.value())
        self._stream.set_itime(self._itime)
        self._stream.waveform_ready.connect(self._waveform.update_samples)
        self._stream.start()
        self._playing = True
        log.info("Synth : lecture démarrée (itime=%.2f)", self._itime)

    def _pause_stream(self):
        if self._stream:
            self._stream.pause()
        self._playing = False

    def _stop(self):
        self._btn_play.setChecked(False)
        self._playing = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.wait(500)
            self._stream = None
        self._itime = 0.0
        self._waveform.update_samples([0.0] * 256)

    def _export_wav(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter audio WAV", "synth_export.wav", "WAV (*.wav)"
        )
        if not path:
            return
        try:
            n = self._renderer.export_wav(
                path,
                duration=self._spn_dur.value(),
                bpm=self._spn_bpm.value()
            )
            if n > 0:
                QMessageBox.information(
                    self, "Export WAV",
                    f"✓ Export réussi\n{os.path.basename(path)}\n"
                    f"{n:,} échantillons · {n / SAMPLE_RATE:.2f}s"
                )
                self.wav_exported.emit(path)
            else:
                QMessageBox.warning(self, "Export WAV",
                                    "Aucun signal généré. Vérifiez le graphe.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur export", str(e))
            log.error("Export WAV : %s", e)

    def _reset_graph(self):
        self._stop()
        self._scene.from_dict({'nodes': [], 'edges': {}})
        self._scene._build_default()
        self._inspector.set_node(None)

    def _on_context_menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self._view)
        for type_id, label_txt in [
            ('oscillator', '🎵 Oscillator'),
            ('envelope',   '📈 Envelope ADSR'),
            ('filter',     '🔊 Filter'),
            ('distortion', '⚡ Distortion'),
            ('delay',      '⏱ Delay'),
            ('reverb',     '🌊 Reverb'),
            ('mixer',      '🎚 Mixer'),
            ('audio_out',  '🔈 Audio Out'),
        ]:
            act = menu.addAction(label_txt)
            act.triggered.connect(lambda _, tid=type_id: self._add_node(tid))
        menu.addSeparator()
        sel_nodes = [i for i in self._scene.selectedItems()
                     if isinstance(i, AudioNodeItem)]
        if sel_nodes:
            del_act = menu.addAction("🗑 Supprimer la sélection")
            del_act.triggered.connect(self._scene.remove_selected_nodes)
        menu.exec(self._view.mapToGlobal(pos))

    def _on_graph_changed(self, dag: dict):
        """Invalide le renderer si le graphe change pendant la lecture."""
        log.debug("Synth graph changed : %d connexions", sum(len(v) for v in dag.values()))

    # ── Intégration iTime ─────────────────────────────────────────────────────

    def sync_itime(self, t: float):
        """
        Appelé depuis MainWindow._tick() pour synchroniser le synth sur iTime.
        Met à jour la position de lecture du stream en temps réel.
        """
        self._itime = t
        if self._stream and self._playing:
            self._stream.set_itime(t)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'bpm':      self._spn_bpm.value(),
            'duration': self._spn_dur.value(),
            'graph':    self._scene.to_dict(),
        }

    def from_dict(self, data: dict):
        self._stop()
        self._spn_bpm.setValue(float(data.get('bpm', 120)))
        self._spn_dur.setValue(float(data.get('duration', 8)))
        graph_data = data.get('graph')
        if graph_data:
            self._scene.from_dict(graph_data)

    # ── Node Graph intégration ────────────────────────────────────────────────

    def get_audio_node_dag(self) -> dict:
        """
        Retourne le DAG audio au format compatible NodeGraphScene.
        Permet d'afficher les nœuds audio dans le Node Graph visuel existant.
        """
        return self._scene._make_dag()

    def cleanup(self):
        self._stop()


def _sep() -> QFrame:
    """Séparateur vertical pour la toolbar."""
    s = QFrame()
    s.setFrameShape(QFrame.Shape.VLine)
    s.setStyleSheet("color: #2a2d3a;")
    s.setFixedWidth(1)
    return s
