"""
timeline_widget.py
------------------
Widget visuel de la timeline style Adobe Premiere.
"""

from __future__ import annotations
import math
import os
import wave
import struct
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QScrollArea, QMenu, QInputDialog,
                              QColorDialog, QDoubleSpinBox, QFileDialog,
                              QComboBox, QSizePolicy, QDialog, QFormLayout,
                              QLineEdit, QDialogButtonBox, QSlider, QFrame)
from PyQt6.QtCore  import Qt, QRect, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui   import (QPainter, QColor, QPen, QBrush, QFont,
                            QMouseEvent, QContextMenuEvent, QCursor, QWheelEvent,
                            QKeyEvent, QKeySequence, QActionGroup,
                            QUndoStack, QUndoCommand)
from .timeline import Timeline, Track, Keyframe, BezierHandle
from .marker   import Marker, MarkerTrack


# ── Constantes visuelles — palette exacte tm.html ──────────────────────────
TRACK_H   = 38          # hauteur par défaut (peut être overridée par canvas._track_h)
LABEL_W   = 150
RULER_H   = 24
KF_RADIUS = 5

# Palette exacte du source PyQt / tm.html
COL_BG         = QColor(0x18, 0x1a, 0x20)   # #181a20
COL_TRACK_HDR  = QColor(0x14, 0x16, 0x1c)   # #14161c  (bg-label)
COL_RULER      = QColor(0x12, 0x14, 0x1a)   # #12141a  (bg-ruler)
COL_RULER_TEXT = QColor(0x96, 0x9b, 0xb5)   # #969bb5
COL_PLAYHEAD   = QColor(0xdc, 0x78, 0x38)   # #dc7838
COL_KF         = QColor(0xe6, 0xb4, 0x3c)   # #e6b43c
COL_KF_SEL     = QColor(0xff, 0xdc, 0x64)   # #ffdc64
COL_KF_BORDER  = QColor(0xb4, 0x82, 0x1e)   # #b4821e
COL_GRID       = QColor(0x28, 0x2b, 0x36)   # #282b36
COL_BORDER     = QColor(0x2a, 0x2d, 0x3a)   # #2a2d3a
COL_TRACK_ALT  = QColor(0x16, 0x18, 0x20)   # #161820 (pistes alternées)

# Piste de marqueurs
MARKER_H         = 18
COL_MARKER_BG    = QColor(0x10, 0x12, 0x1a)  # #10121a

# Multi-sélection
COL_RUBBERBAND        = QColor(80, 140, 255, 40)
COL_RUBBERBAND_BORDER = QColor(80, 140, 255, 178)

# Bézier handles
COL_HANDLE      = QColor(0x50, 0x80, 0xc8)       # #5080c8
COL_HANDLE_SEL  = QColor(0x8c, 0xc0, 0xff)       # #8cc0ff
COL_HANDLE_LINE = QColor(80, 140, 255, 115)       # rgba(80,140,255,0.45)
COL_CURVE       = QColor(80, 200, 140, 191)       # rgba(80,200,140,0.75)
HANDLE_RADIUS   = 4

# Loop region — exactement tm.html
COL_LOOP_REGION = QColor(100, 180, 255, 35)       # rgba(100,180,255,0.14)
COL_LOOP_IN     = QColor(60,  220, 140, 229)      # rgba(60,220,140,0.9)
COL_LOOP_OUT    = QColor(220, 100,  60, 229)      # rgba(220,100,60,0.9)
LOOP_HANDLE_W   = 6

# Palette clips shader — exacte tm.html
_CLIP_PALETTE = [
    QColor(0x32, 0x70, 0xc8),   # #3270c8
    QColor(0xb4, 0x46, 0x37),   # #b44637
    QColor(0x32, 0x9b, 0x5f),   # #329b5f
    QColor(0x9b, 0x5a, 0xc3),   # #9b5ac3
    QColor(0xc3, 0x82, 0x2e),   # #c3822e
    QColor(0x32, 0xaa, 0xaa),   # #32aaaa
    QColor(0xc3, 0x41, 0x79),   # #c34179
    QColor(0x5f, 0x9b, 0x37),   # #5f9b37
]
_clip_color_cache: dict[str, QColor] = {}

def _color_for_path(path: str) -> QColor:
    if path not in _clip_color_cache:
        idx = len(_clip_color_cache) % len(_CLIP_PALETTE)
        _clip_color_cache[path] = _CLIP_PALETTE[idx]
    return _clip_color_cache[path]


# ── Cache waveform ─────────────────────────────────────────────────────────
_waveform_cache: dict[str, np.ndarray] = {}

def _load_waveform(path: str, num_samples: int = 4000) -> np.ndarray | None:
    """
    Charge et réduit la forme d'onde d'un fichier WAV en un tableau de num_samples
    valeurs RMS (entre 0.0 et 1.0). Retourne None si non supporté ou erreur.

    Formats WAV supportés :
      - PCM 8-bit  (format tag 1, sampwidth 1)
      - PCM 16-bit (format tag 1, sampwidth 2)
      - PCM 32-bit (format tag 1, sampwidth 4)
      - IEEE float 32-bit (format tag 3, sampwidth 4) — lecture manuelle via struct
      - IEEE float 64-bit (format tag 3, sampwidth 8) — lecture manuelle via struct
    """
    if path in _waveform_cache:
        return _waveform_cache[path]

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext != '.wav':
            # MP3/OGG : pas de lib standard disponible, waveform décorative
            return None

        # ── Tentative via wave.open (PCM standard) ────────────────────────────
        samples = None
        n_channels = 1

        try:
            with wave.open(path, 'rb') as wf:
                n_channels  = wf.getnchannels()
                sampwidth   = wf.getsampwidth()
                n_frames    = wf.getnframes()
                raw         = wf.readframes(n_frames)

            if sampwidth == 1:
                samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            elif sampwidth == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
            # sampwidth inconnu → samples reste None → lecture manuelle ci-dessous

        except wave.Error:
            # Format non-PCM (ex: IEEE float tag 3) — lecture manuelle du RIFF/WAV
            samples = _load_wav_float(path)
            if samples is None:
                return None
            # _load_wav_float retourne un tableau mono float32 déjà aplati
            n_channels = 1

        if samples is None:
            return None

        # Mix multi-canal → mono
        if n_channels > 1:
            try:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            except ValueError:
                pass

        # Réduction par blocs RMS
        total = len(samples)
        if total == 0:
            return None
        block    = max(1, total // num_samples)
        n_blocks = total // block
        trimmed  = samples[:n_blocks * block].reshape(n_blocks, block)
        rms      = np.sqrt((trimmed ** 2).mean(axis=1))
        peak     = rms.max()
        if peak > 0:
            rms = rms / peak
        _waveform_cache[path] = rms
        return rms

    except (OSError, ValueError, ZeroDivisionError, struct.error):
        return None


def _load_wav_float(path: str) -> 'np.ndarray | None':
    """
    Lecture manuelle d'un fichier WAV IEEE float (format tag 3).
    Analyse le header RIFF/WAVE et décode les données brutes.
    Retourne un tableau numpy float32 aplati (tous canaux mélangés), ou None.
    """
    try:
        with open(path, 'rb') as f:
            # RIFF header
            riff_id = f.read(4)
            if riff_id != b'RIFF':
                return None
            f.read(4)  # taille totale
            wave_id = f.read(4)
            if wave_id != b'WAVE':
                return None

            fmt_tag = n_ch = samp_rate = sampwidth = None

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data = f.read(chunk_size)

                if chunk_id == b'fmt ':
                    fmt_tag   = struct.unpack('<H', chunk_data[0:2])[0]
                    n_ch      = struct.unpack('<H', chunk_data[2:4])[0]
                    # samp_rate = struct.unpack('<I', chunk_data[4:8])[0]
                    # block_align = struct.unpack('<H', chunk_data[12:14])[0]
                    sampwidth = struct.unpack('<H', chunk_data[14:16])[0] // 8

                elif chunk_id == b'data':
                    if fmt_tag is None or n_ch is None or sampwidth is None:
                        return None
                    # Format 3 = IEEE float
                    if fmt_tag == 3:
                        if sampwidth == 4:
                            samples = np.frombuffer(chunk_data, dtype=np.float32).copy()
                        elif sampwidth == 8:
                            samples = np.frombuffer(chunk_data, dtype=np.float64).astype(np.float32)
                        else:
                            return None
                    elif fmt_tag == 1:
                        # PCM tombé ici quand wave.open a échoué pour une autre raison
                        if sampwidth == 2:
                            samples = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sampwidth == 4:
                            samples = np.frombuffer(chunk_data, dtype=np.int32).astype(np.float32) / 2**31
                        elif sampwidth == 1:
                            samples = (np.frombuffer(chunk_data, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                        else:
                            return None
                    else:
                        return None

                    # Mix multi-canal → mono
                    if n_ch > 1:
                        try:
                            samples = samples.reshape(-1, n_ch).mean(axis=1)
                        except ValueError:
                            pass
                    return samples.astype(np.float32)

        return None
    except (OSError, struct.error, ValueError):
        return None


# ── Commandes Undo/Redo ────────────────────────────────────────────────────

class AddKeyframeCommand(QUndoCommand):
    def __init__(self, track, time, value, canvas, interp='linear'):
        super().__init__("Ajout Keyframe")
        self.track = track; self.time = time; self.value = value
        self.interp = interp; self.canvas = canvas; self.kf = None

    def redo(self):
        if self.kf is None:
            self.track.add_keyframe(self.time, self.value, self.interp)
            for k in self.track.keyframes:
                if abs(k.time - self.time) < 1e-5:
                    self.kf = k; break
        else:
            if self.kf not in self.track.keyframes:
                self.track.keyframes.append(self.kf)
                self.track.keyframes.sort(key=lambda k: k.time)
        self.canvas.update(); self.canvas.data_changed.emit()

    def undo(self):
        if self.kf and self.kf in self.track.keyframes:
            self.track.keyframes.remove(self.kf)
        self.canvas.update(); self.canvas.data_changed.emit()


class DeleteKeyframeCommand(QUndoCommand):
    def __init__(self, track, kf, canvas):
        super().__init__("Suppression Keyframe")
        self.track = track; self.kf = kf; self.canvas = canvas

    def redo(self):
        if self.kf in self.track.keyframes:
            self.track.keyframes.remove(self.kf)
            if self.canvas._selected_kf and self.canvas._selected_kf[1] is self.kf:
                self.canvas._selected_kf = None
            self.canvas._selected_kfs.discard((self.track, self.kf))
        self.canvas.update(); self.canvas.data_changed.emit()

    def undo(self):
        if self.kf not in self.track.keyframes:
            self.track.keyframes.append(self.kf)
            self.track.keyframes.sort(key=lambda k: k.time)
        self.canvas.update(); self.canvas.data_changed.emit()


class MoveKeyframeCommand(QUndoCommand):
    def __init__(self, track, kf, old_t, new_t, canvas):
        super().__init__("Déplacement Keyframe")
        self.track = track; self.kf = kf
        self.old_t = old_t; self.new_t = new_t; self.canvas = canvas

    def redo(self):
        self.kf.time = self.new_t
        self.track.keyframes.sort(key=lambda k: k.time)
        self.canvas.update(); self.canvas.data_changed.emit()

    def undo(self):
        self.kf.time = self.old_t
        self.track.keyframes.sort(key=lambda k: k.time)
        self.canvas.update(); self.canvas.data_changed.emit()


class ChangeInterpolationCommand(QUndoCommand):
    def __init__(self, track, kf, new_interp, canvas):
        super().__init__("Interpolation")
        self.track = track; self.kf = kf
        self.old_interp = kf.interp; self.new_interp = new_interp; self.canvas = canvas

    def redo(self):
        self.kf.interp = self.new_interp
        self.canvas.update(); self.canvas.data_changed.emit()

    def undo(self):
        self.kf.interp = self.old_interp
        self.canvas.update(); self.canvas.data_changed.emit()


class SplitClipCommand(QUndoCommand):
    """Coupe un clip shader en deux à un instant donné."""
    def __init__(self, track, kf, split_time, canvas):
        super().__init__("Couper le clip")
        self.track      = track
        self.kf         = kf          # keyframe du clip à couper
        self.split_time = split_time
        self.canvas     = canvas
        self.new_kf     = None        # keyframe créé à la coupure

    def redo(self):
        if self.new_kf is None:
            self.new_kf = self.track.add_keyframe(self.split_time, self.kf.value, 'step')
        else:
            if self.new_kf not in self.track.keyframes:
                self.track.keyframes.append(self.new_kf)
                self.track.keyframes.sort(key=lambda k: k.time)
        self.canvas.update()
        self.canvas.data_changed.emit()

    def undo(self):
        if self.new_kf and self.new_kf in self.track.keyframes:
            self.track.keyframes.remove(self.new_kf)
        self.canvas.update()
        self.canvas.data_changed.emit()


class ChangeTrackColorCommand(QUndoCommand):
    def __init__(self, track, new_color, canvas):
        super().__init__("Couleur Piste")
        self.track = track; self.new_color = new_color
        self.old_color = track.color; self.canvas = canvas

    def redo(self):
        self.track.color = self.new_color; self.canvas.update()

    def undo(self):
        self.track.color = self.old_color; self.canvas.update()


# ── Commande Undo : déplacement de plusieurs keyframes ────────────────────

class MoveMultipleKeyframesCommand(QUndoCommand):
    """
    Déplace un groupe de keyframes d'un même Δtime.
    moves = list of (track, kf, old_t, new_t)
    """
    def __init__(self, moves: list, canvas):
        n = len(moves)
        super().__init__(f"Déplacer {n} keyframe{'s' if n > 1 else ''}")
        self._moves  = moves   # [(track, kf, old_t, new_t), ...]
        self._canvas = canvas

    def _apply(self, use_new: bool):
        for track, kf, old_t, new_t in self._moves:
            kf.time = new_t if use_new else old_t
            track.keyframes.sort(key=lambda k: k.time)
        self._canvas.update()
        self._canvas.data_changed.emit()

    def redo(self): self._apply(True)
    def undo(self): self._apply(False)


# ── Commande Undo : déplacement handle Bézier ─────────────────────────────
class MoveHandleCommand(QUndoCommand):
    """Déplace un handle Bézier (in ou out) d'un keyframe."""
    def __init__(self, kf: Keyframe, side: str,
                 old_dt: float, old_dv,
                 new_dt: float, new_dv,
                 canvas):
        super().__init__("Déplacement handle")
        self.kf = kf; self.side = side
        self.old_dt = old_dt; self.old_dv = old_dv
        self.new_dt = new_dt; self.new_dv = new_dv
        self.canvas = canvas

    def _apply(self, dt, dv):
        h = self.kf.handle_in if self.side == 'in' else self.kf.handle_out
        h.dt = dt; h.dv = dv
        # Si handles liés, symétrise l'autre côté
        if self.kf.handles_linked:
            mirror = self.kf.handle_out if self.side == 'in' else self.kf.handle_in
            mirror.dt = -dt
            if isinstance(dv, (int, float)):
                mirror.dv = -dv
            elif isinstance(dv, tuple):
                mirror.dv = tuple(-v for v in dv)
        self.canvas.update(); self.canvas.data_changed.emit()

    def redo(self):  self._apply(self.new_dt, self.new_dv)
    def undo(self):  self._apply(self.old_dt, self.old_dv)


# ── Commande Undo : changement mode handles (lié / brisé) ──────────────────
class ToggleHandlesLinkedCommand(QUndoCommand):
    def __init__(self, kf: Keyframe, canvas):
        super().__init__("Toggle handles liés")
        self.kf = kf; self.canvas = canvas

    def redo(self):
        self.kf.handles_linked = not self.kf.handles_linked
        self.canvas.update(); self.canvas.data_changed.emit()

    def undo(self):
        self.redo()  # toggle est son propre inverse


# ── Commande Undo : coller des keyframes ──────────────────────────────────
class PasteKeyframesCommand(QUndoCommand):
    """
    Colle un ou plusieurs keyframes sur leurs pistes respectives.
    pastes = list of (track, kf)  — les KFs sont déjà construits, prêts à insérer.
    """
    def __init__(self, pastes: list, canvas):
        n = len(pastes)
        super().__init__(f"Coller {n} keyframe{'s' if n > 1 else ''}")
        self._pastes = pastes   # [(track, kf), ...]
        self._canvas = canvas

    def redo(self):
        for track, kf in self._pastes:
            if kf not in track.keyframes:
                track.keyframes.append(kf)
                track.keyframes.sort(key=lambda k: k.time)
        self._canvas.update()
        self._canvas.data_changed.emit()

    def undo(self):
        for track, kf in self._pastes:
            if kf in track.keyframes:
                track.keyframes.remove(kf)
        self._canvas.update()
        self._canvas.data_changed.emit()


# ── TimelineCanvas ─────────────────────────────────────────────────────────

class TimelineCanvas(QWidget):
    time_changed      = pyqtSignal(float)
    keyframe_moved    = pyqtSignal(Track, Keyframe, float, float)
    keyframe_selected = pyqtSignal(Track, Keyframe)
    data_changed      = pyqtSignal()
    zoom_changed      = pyqtSignal(float)   # pixels_per_sec

    def __init__(self, timeline: Timeline, undo_stack: QUndoStack, parent=None):
        super().__init__(parent)
        self._timeline_widget = parent
        self.timeline   = timeline
        self.undo_stack = undo_stack

        self._scroll_offset  = 0.0
        self._current_time   = 0.0
        self._pixels_per_sec = 60.0
        self._track_h        = TRACK_H   # hauteur de piste ajustable

        self._drag_kf:           tuple[Track, Keyframe] | None = None
        self._drag_orig_t:       float = 0.0
        self._drag_start_x:      int   = 0
        self._drag_clip_src_track: 'Track | None' = None   # Bug #2 fix: init manquant
        self._selected_kf:       tuple[Track, Keyframe] | None = None

        # ── Marqueurs ───────────────────────────────────────────────────────
        self._drag_marker:      Marker | None = None
        self._drag_marker_orig: float         = 0.0
        self._drag_marker_x0:   int           = 0
        self._selected_marker:  Marker | None = None

        # ── Multi-sélection ─────────────────────────────────────────────────
        # set of (track, kf) — tous les KFs sélectionnés
        self._selected_kfs:      set = set()
        # Drag groupe : origines t de chaque KF au début du drag
        self._multi_drag_origins: dict = {}   # kf → float (t original)
        # Rubber-band : rectangle de box-select
        self._rubber_band_start: QPoint | None = None
        self._rubber_band_rect:  QRect  | None = None

        # ── État handles Bézier ─────────────────────────────────────────────
        self._drag_handle: tuple[Keyframe, str] | None = None   # (kf, 'in'|'out')
        self._drag_handle_orig_dt: float = 0.0
        self._drag_handle_orig_dv = 0.0
        self._selected_handle: tuple[Keyframe, str] | None = None

        self._resize_clip:   tuple[Track, Keyframe] | None = None
        self._resize_start_x: int = 0
        # Redim gauche : on déplace directement le keyframe du clip
        self._resize_left_clip:  tuple[Track, Keyframe] | None = None
        self._resize_left_orig_t: float = 0.0

        # ── Loop region drag ─────────────────────────────────────────────────
        self._drag_loop: str | None = None   # 'in' | 'out' | 'region'
        self._drag_loop_orig_in:  float = 0.0
        self._drag_loop_orig_out: float = 0.0
        self._drag_loop_x0: int = 0

        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

    # ── Coordonnées ──────────────────────────────────────────────────────────

    def _time_to_x(self, t: float) -> int:
        return LABEL_W + int((t - self._scroll_offset) * self._pixels_per_sec)

    def _x_to_time(self, x: int) -> float:
        return (x - LABEL_W) / self._pixels_per_sec + self._scroll_offset

    # ── Helpers coordonnées pistes ───────────────────────────────────────────

    def _marker_offset(self) -> int:
        """Hauteur de la bande marqueurs (0 si pas de marker_track)."""
        return MARKER_H if getattr(self.timeline, 'marker_track', None) is not None else 0

    def _track_y(self, idx: int) -> int:
        """Retourne la coordonnée Y du début de la piste idx."""
        y = RULER_H + self._marker_offset()
        for i, track in enumerate(self.timeline.tracks):
            if i == idx:
                return y
            y += getattr(track, 'height', self._track_h)
        return y

    def _track_h_for(self, idx: int) -> int:
        """Retourne la hauteur de la piste idx."""
        if 0 <= idx < len(self.timeline.tracks):
            return getattr(self.timeline.tracks[idx], 'height', self._track_h)
        return self._track_h

    def _track_idx_at_y(self, y: int) -> int:
        """Retourne l'index de la piste à la coordonnée Y (ou -1)."""
        cy = RULER_H + self._marker_offset()
        for i, track in enumerate(self.timeline.tracks):
            h = getattr(track, 'height', self._track_h)
            if cy <= y < cy + h:
                return i
            cy += h
        return -1

    def _in_marker_band(self, y: int) -> bool:
        """Retourne True si y est dans la bande marqueurs."""
        mt = getattr(self.timeline, 'marker_track', None)
        return mt is not None and RULER_H <= y < RULER_H + MARKER_H

    # ── Peinture ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        # Fond global
        p.fillRect(0, 0, w, h, COL_BG)

        # ── Règle ─────────────────────────────────────────────────────────────
        p.fillRect(0, 0, w, RULER_H, COL_RULER)
        self._draw_ruler(p, w)

        # ── Bande marqueurs ───────────────────────────────────────────────────
        mt = getattr(self.timeline, 'marker_track', None)
        if mt is not None:
            p.fillRect(0, RULER_H, w, MARKER_H, COL_MARKER_BG)
            # Bordure basse de la bande marqueurs
            p.fillRect(0, RULER_H + MARKER_H - 1, w, 1, COL_BORDER)
            self._draw_marker_band(p, mt, w)

        # ── Pistes ────────────────────────────────────────────────────────────
        for i, track in enumerate(self.timeline.tracks):
            y  = self._track_y(i)
            th = self._track_h_for(i)

            # Fond de piste alterné — exactement tm.html
            track_bg = COL_BG if i % 2 == 0 else COL_TRACK_ALT
            p.fillRect(LABEL_W, y, w - LABEL_W, th, track_bg)

            # Overlay mute
            if not track.enabled:
                p.fillRect(LABEL_W, y, w - LABEL_W, th, QColor(0, 0, 0, 97))

            # Grille
            self._draw_grid(p, y, th, w)

            # Bordure basse de piste
            p.fillRect(0, y + th - 1, w, 1, QColor(0x0f, 0x10, 0x13))

            # ── Label (colonne gauche) ────────────────────────────────────────
            p.fillRect(0, y, LABEL_W, th, COL_TRACK_HDR)

            # Liseré couleur à gauche (3px) — tm.html .tl-color-bar
            bar_col = QColor(track.color)
            p.fillRect(0, y, 3, th, bar_col)

            # Nom de la piste
            if track.group:
                group_prefix = ("▶ " if track.group_folded else "▼ ")
            else:
                group_prefix = ""

            # Nom (haut) — font 8px weight 500
            p.setPen(QColor(0xc8, 0xcd, 0xd7))   # #c8cdd7 tl-name-top
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(12, y, LABEL_W - 52, th // 2 + 2),
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                       group_prefix + track.name)

            # Type/uniform (bas) — font 7px color #5a6480
            p.setPen(QColor(0x5a, 0x64, 0x80))
            p.setFont(QFont("Segoe UI", 7))
            if track.value_type == 'camera':
                type_label = "[🎥 Caméra]"
            elif track.group:
                type_label = f"[{track.group}] {track.uniform_name}"
            else:
                type_label = f"[{track.uniform_name}]"
            p.drawText(QRect(12, y + th // 2, LABEL_W - 52, th // 2),
                       Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                       type_label)

            # Indicateur d'expression ƒ(t)
            has_expr = any(getattr(kf, 'expression', '') for kf in track.keyframes)
            if has_expr:
                p.setPen(QColor(160, 100, 230))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(QRect(LABEL_W - 20, y, 18, th // 2),
                           Qt.AlignmentFlag.AlignCenter, "ƒ(t)")

            # Boutons M / S / R (14×14 px, alignés à droite du label)
            self._draw_track_buttons(p, track, y, th)

            # ── Contenu de la piste ───────────────────────────────────────────
            if track.value_type == 'shader':
                self._draw_shader_clips(p, track, y, th)
            elif track.value_type == 'trans':
                self._draw_trans_clips(p, track, y, th)
            elif track.value_type == 'audio':
                self._draw_audio_track(p, track, y, w, th)
            elif track.value_type == 'camera':
                self._draw_camera_track(p, track, y, w, th)
            else:
                has_bezier = any(kf.interp == 'bezier' for kf in track.keyframes)
                if has_bezier and len(track.keyframes) >= 2:
                    self._draw_bezier_curve_preview(p, track, y, th)
                for kf in track.keyframes:
                    self._draw_keyframe(p, kf, y, track, th)

        # ── Séparateur colonne labels ─────────────────────────────────────────
        p.fillRect(LABEL_W - 1, 0, 1, h, COL_BORDER)

        # ── Rubber-band ───────────────────────────────────────────────────────
        if self._rubber_band_rect:
            p.setPen(QPen(COL_RUBBERBAND_BORDER, 1, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(COL_RUBBERBAND))
            p.drawRect(self._rubber_band_rect)

        # ── Loop region ───────────────────────────────────────────────────────
        self._draw_loop_region(p, w, h)

        # ── Playhead ──────────────────────────────────────────────────────────
        ph_x = self._time_to_x(self._current_time)
        if LABEL_W <= ph_x <= w:
            # Halo orange (glow)
            glow = QColor(220, 120, 56, 30)
            p.setPen(QPen(glow, 7))
            p.drawLine(ph_x, 0, ph_x, h)
            # Ligne principale
            p.setPen(QPen(COL_PLAYHEAD, 1.5))
            p.drawLine(ph_x, 0, ph_x, h)
            # Triangle tête de lecture
            pts = [QPoint(ph_x - 5, 0), QPoint(ph_x + 5, 0), QPoint(ph_x, 10)]
            p.setBrush(QBrush(COL_PLAYHEAD))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(*pts)

        # v5.0 — Curseurs co-édition
        if hasattr(self, '_collab_overlay') and self._collab_overlay is not None:
            pps = self._pixels_per_second if hasattr(self, '_pixels_per_second') else 100.0
            sx  = self._scroll_offset     if hasattr(self, '_scroll_offset')     else 0.0
            self._collab_overlay.draw(p, pps, sx * pps, h)

        p.end()

    def _draw_ruler(self, p: QPainter, w: int):
        """Règle temporelle — rendu identique à tm.html drawRuler()."""
        t_start = max(0.0, self._scroll_offset)
        t_end   = self._x_to_time(w)

        # Adapter le pas selon le zoom (même logique que tm.html)
        pps = self._pixels_per_sec
        step = 1.0
        if pps < 10:   step = 10.0
        elif pps < 20: step = 5.0
        elif pps > 200: step = 0.5
        elif pps > 100: step = 0.25

        # Region de loop en fond de règle
        tl = self.timeline
        if getattr(tl, 'loop_enabled', False):
            lx0 = self._time_to_x(tl.loop_in)
            lx1 = self._time_to_x(tl.loop_out)
            p.fillRect(max(LABEL_W, lx0), 0, max(0, lx1 - lx0), RULER_H,
                       QColor(100, 180, 255, 20))

        import math as _math
        t = _math.floor(t_start / step) * step
        p.setFont(QFont("Segoe UI", 7))

        while t <= t_end + step:
            x = self._time_to_x(t)
            if x < LABEL_W - 20:
                t = round((t + step) * 10000) / 10000
                continue

            is_maj = (abs(t % 5) < 0.001) or (step >= 5 and abs(t % step) < 0.001)

            # Sous-ticks (quarts)
            if step >= 1 and pps > 30:
                for sub in range(1, 4):
                    sx = x + sub * pps * step / 4
                    p.setPen(QPen(QColor(0x1c, 0x20, 0x30), 1))
                    p.drawLine(int(sx), RULER_H - 2, int(sx), RULER_H)

            # Tick principal / secondaire
            tick_h = 10 if is_maj else 5
            p.setPen(QPen(COL_RULER_TEXT, 1))
            p.drawLine(int(x), RULER_H - tick_h, int(x), RULER_H)

            # Libellé
            if is_maj or pps > 40:
                col_text = COL_RULER_TEXT if is_maj else QColor(0x50, 0x54, 0x70)
                p.setPen(col_text)
                lbl = f"{int(t)}s" if t % 1 == 0 else f"{t:.2f}s"
                p.drawText(int(x) + 2, 2, w, RULER_H - 4, 0, lbl)

            t = round((t + step) * 10000) / 10000

        # Grille BPM (si snap activé)
        if getattr(tl, 'snap_to_grid', False) and pps > 8:
            bpm = getattr(tl, 'bpm', 120.0)
            snap_div = getattr(tl, 'snap_division', 4)
            beat_sec = 60.0 / max(1, bpm) / max(1, snap_div)
            bt = _math.floor(t_start / beat_sec) * beat_sec
            while bt <= t_end:
                bx = self._time_to_x(bt)
                p.setPen(QPen(QColor(100, 160, 255, 64), 1))
                p.drawLine(int(bx), RULER_H - 3, int(bx), RULER_H)
                bt += beat_sec

        # Handles loop sur la règle
        if getattr(tl, 'loop_enabled', False):
            for lt, col, lbl, direction in [
                (tl.loop_in,  COL_LOOP_IN,  'IN',  1),
                (tl.loop_out, COL_LOOP_OUT, 'OUT', -1),
            ]:
                lx = self._time_to_x(lt)
                if LABEL_W - 20 <= lx <= w + 20:
                    p.setPen(QPen(col, 2))
                    p.drawLine(int(lx), 0, int(lx), RULER_H)
                    p.setBrush(QBrush(col))
                    p.setPen(Qt.PenStyle.NoPen)
                    if direction > 0:
                        pts = [QPoint(int(lx), RULER_H),
                               QPoint(int(lx) + 8, RULER_H - 8),
                               QPoint(int(lx), RULER_H - 8)]
                    else:
                        pts = [QPoint(int(lx), RULER_H),
                               QPoint(int(lx) - 8, RULER_H - 8),
                               QPoint(int(lx), RULER_H - 8)]
                    p.drawPolygon(*pts)
                    p.setPen(QColor(0, 0, 0, 178))
                    p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                    tx = int(lx) + 3 if direction > 0 else int(lx) - 15
                    p.drawText(tx, RULER_H - 11, lbl)

        # Marqueurs dans la règle
        mt = getattr(self.timeline, 'marker_track', None)
        if mt:
            for m in mt.markers:
                mx = self._time_to_x(m.time)
                if LABEL_W <= mx <= w:
                    p.setPen(QPen(QColor(m.color), 2))
                    p.drawLine(int(mx), 0, int(mx), RULER_H)

        # Le triangle du playhead est dessiné dans paintEvent (pas ici, évite le doublon)

        # Bordure basse de la règle
        p.fillRect(0, RULER_H - 1, w, 1, QColor(0x23, 0x26, 0x35))

    def _draw_loop_region(self, p: QPainter, w: int, h: int):
        """Région de boucle In/Out — rendu identique tm.html."""
        tl = self.timeline
        if not getattr(tl, 'loop_enabled', False):
            return
        x_in  = self._time_to_x(tl.loop_in)
        x_out = self._time_to_x(tl.loop_out)
        # Overlay coloré
        if x_out > x_in:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(COL_LOOP_REGION))
            p.drawRect(max(LABEL_W, x_in), 0,
                       min(w, x_out) - max(LABEL_W, x_in), h)
        # Ligne In (verte) pointillée
        if LABEL_W <= x_in <= w:
            pen = QPen(COL_LOOP_IN, 1.5, Qt.PenStyle.DashLine)
            pen.setDashPattern([4, 4])
            p.setPen(pen)
            p.drawLine(x_in, 0, x_in, h)
        # Ligne Out (rouge-orange) pointillée
        if LABEL_W <= x_out <= w:
            pen = QPen(COL_LOOP_OUT, 1.5, Qt.PenStyle.DashLine)
            pen.setDashPattern([4, 4])
            p.setPen(pen)
            p.drawLine(x_out, 0, x_out, h)

    def _loop_hit_test(self, x: int, y: int) -> str | None:
        """Retourne 'in', 'out', 'region' ou None selon la zone cliquée (sur la règle)."""
        tl = self.timeline
        if not getattr(tl, 'loop_enabled', False):
            return None
        if y > RULER_H:
            return None
        x_in  = self._time_to_x(tl.loop_in)
        x_out = self._time_to_x(tl.loop_out)
        if abs(x - x_in) <= LOOP_HANDLE_W + 2:
            return 'in'
        if abs(x - x_out) <= LOOP_HANDLE_W + 2:
            return 'out'
        if x_in < x < x_out:
            return 'region'
        return None

    def _draw_grid(self, p: QPainter, y: int, h: int, w: int):
        """Grille verticale — rendu identique tm.html."""
        # Grille secondes
        t0 = max(0.0, self._scroll_offset)
        t1 = self._x_to_time(w)
        t = math.floor(t0)
        if t < 0:
            t = 0
        while t <= t1 + 1:
            x = self._time_to_x(t)
            if 0 <= x <= w:
                p.setPen(QPen(COL_GRID, 1))
                p.drawLine(x, y, x, y + h)
            t += 1

        # Grille BPM si snap activé
        tl = self.timeline
        if getattr(tl, 'snap_to_grid', False) and self._pixels_per_sec > 12:
            bpm = getattr(tl, 'bpm', 120.0)
            snap_div = getattr(tl, 'snap_division', 4)
            beat_sec = 60.0 / max(1, bpm) / max(1, snap_div)
            import math as _math
            bt = _math.floor(t0 / beat_sec) * beat_sec
            while bt <= t1:
                bx = self._time_to_x(bt)
                if 0 <= bx <= w:
                    p.setPen(QPen(QColor(100, 160, 255, 30), 1))
                    p.drawLine(bx, y, bx, y + h)
                bt += beat_sec

    def _draw_keyframe(self, p: QPainter, kf: Keyframe, track_y: int, track: Track, th: int = None):
        if th is None:
            th = self._track_h
        x = self._time_to_x(kf.time)
        y = track_y + th // 2
        r = KF_RADIUS
        is_sel = (self._selected_kf and
                  self._selected_kf[0] is track and
                  self._selected_kf[1] is kf)
        # Multi-sélection : halo blanc supplémentaire
        in_multi = any(k is kf for (_, k) in self._selected_kfs)
        if in_multi and not is_sel:
            p.setPen(QPen(QColor(255, 255, 255, 120), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPoint(x, y), r + 3, r + 3)
        pts = [QPoint(x, y - r), QPoint(x + r, y), QPoint(x, y + r), QPoint(x - r, y)]
        p.setBrush(QBrush(COL_KF_SEL if (is_sel or in_multi) else COL_KF))
        p.setPen(QPen(COL_KF_BORDER, 1))
        p.drawPolygon(*pts)

        # ── Handles Bézier (visibles si KF sélectionné et interp == 'bezier') ──
        if kf.interp == 'bezier' and is_sel:
            self._draw_bezier_handles(p, kf, x, y, th)

    def _draw_bezier_handles(self, p: QPainter, kf: Keyframe, kf_x: int, kf_y: int, th: int = None):
        """Dessine les handles tangents in/out d'un keyframe Bézier."""
        if th is None:
            th = self._track_h
        for side in ('in', 'out'):
            h = kf.handle_in if side == 'in' else kf.handle_out
            if h.dt == 0.0:
                continue

            h_t = kf.time + h.dt
            dv = h.dv if isinstance(h.dv, (int, float)) else (h.dv[0] if h.dv else 0.0)
            h_x = self._time_to_x(h_t)
            half_h = max(1, th // 2 - 4)
            h_y = kf_y - int(dv * half_h * 0.5)

            is_h_sel = (self._selected_handle and
                        self._selected_handle[0] is kf and
                        self._selected_handle[1] == side)

            # Ligne handle → KF (pointillée)
            pen = QPen(COL_HANDLE_LINE, 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(kf_x, kf_y, h_x, h_y)

            # Cercle handle
            p.setPen(QPen(COL_HANDLE if not is_h_sel else COL_HANDLE_SEL, 1))
            p.setBrush(QBrush(COL_HANDLE if not is_h_sel else COL_HANDLE_SEL))
            p.drawEllipse(QPoint(h_x, h_y), HANDLE_RADIUS, HANDLE_RADIUS)

    def _draw_bezier_curve_preview(self, p: QPainter, track: Track, track_y: int, th: int = None):
        """
        Trace la courbe Bézier interpolée entre tous les KFs d'une piste float.
        Appelée depuis paintEvent pour les pistes numériques avec ≥2 KFs bézier.
        """
        if th is None:
            th = self._track_h
        kfs = [kf for kf in track.keyframes if isinstance(kf.value, (int, float))]
        if len(kfs) < 2:
            return

        cy     = track_y + th // 2
        half_h = max(1, th // 2 - 4)

        # Récupère les bornes de valeur pour normaliser l'axe Y
        vals  = [kf.value for kf in kfs]
        v_min = min(vals)
        v_max = max(vals)
        v_range = max(1e-6, v_max - v_min)

        pen = QPen(COL_CURVE, 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        w = self.width()
        prev_pt = None

        for px in range(LABEL_W, w, 2):
            t = self._x_to_time(px)
            if t < kfs[0].time or t > kfs[-1].time:
                prev_pt = None
                continue
            val = track.get_value_at(t)
            if val is None:
                prev_pt = None
                continue
            norm = (val - v_min) / v_range   # 0..1
            py   = cy + int((0.5 - norm) * 2 * half_h)
            pt   = QPoint(px, py)
            if prev_pt:
                p.drawLine(prev_pt, pt)
            prev_pt = pt

    def _draw_shader_clips(self, p: QPainter, track: Track, track_y: int, th: int = None):
        """Clips shader — rendu identique à tm.html drawShaderClips()."""
        if th is None:
            th = self._track_h
        kfs = track.keyframes
        for i, kf in enumerate(kfs):
            if not kf.value:
                continue
            next_kf = kfs[i + 1] if i + 1 < len(kfs) else None
            x_start = self._time_to_x(kf.time)
            x_end   = (self._time_to_x(next_kf.time) if next_kf is not None
                       else self._time_to_x(self.timeline.duration))
            clip_w  = max(0, x_end - x_start)
            if clip_w < 2:
                continue
            if x_start > self.width() or x_end < 0:
                continue

            is_sel = (self._selected_kf and
                      self._selected_kf[0] is track and
                      self._selected_kf[1] is kf)

            base    = _color_for_path(str(kf.value))
            alpha   = 209 if is_sel else 171   # 0.82 / 0.67

            clip_rect = QRect(x_start, track_y + 2, clip_w, th - 4)

            # Corps du clip
            p.save()
            p.setClipRect(clip_rect)
            fill = QColor(base)
            fill.setAlpha(alpha)
            p.setBrush(QBrush(fill))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(clip_rect, 3, 3)

            # Barre de titre (haut 9px)
            title_rect = QRect(x_start, track_y + 2, clip_w, 9)
            lighter = base.lighter(160)
            lighter.setAlpha(230)
            p.setBrush(QBrush(lighter))
            p.drawRoundedRect(title_rect, 3, 3)
            # Remplissage bas du title rect (pour enlever les coins ronds du bas)
            p.fillRect(QRect(x_start, track_y + 2 + 4, clip_w, 5), lighter)

            # Liseré gauche (3px) — très clair
            bright = base.lighter(200)
            bright.setAlpha(242)
            p.fillRect(QRect(x_start, track_y + 2, 3, th - 4), bright)

            # Contour
            p.setPen(QPen(QColor(255, 255, 255, 178) if is_sel
                         else QColor(base.red(), base.green(), base.blue(), 128), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(clip_rect.adjusted(0, 0, -1, -1), 3, 3)

            # Nom du shader
            name = os.path.splitext(os.path.basename(str(kf.value)))[0].replace('_', ' ')
            p.setClipRect(QRect(x_start + 4, track_y, clip_w - 8, th))
            p.setPen(QColor(255, 255, 255, 230))
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            p.drawText(x_start + 6, track_y + 3, name)

            p.restore()

            # Poignées de redimensionnement (si clip assez large)
            if clip_w > 16:
                handle_col = QColor(255, 255, 255, 82)
                p.setBrush(QBrush(handle_col))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(QRect(x_end - 6, track_y + 4, 5, th - 8), 2, 2)
                p.drawRoundedRect(QRect(x_start + 1, track_y + 4, 5, th - 8), 2, 2)

    def _draw_trans_clips(self, p: QPainter, track: Track, track_y: int, th: int = None):
        """Clips de transition — rendu identique tm.html drawTransClips()."""
        if th is None:
            th = self._track_h
        kfs = track.keyframes
        for i, kf in enumerate(kfs):
            if not kf.value:
                continue
            next_kf = kfs[i + 1] if i + 1 < len(kfs) else None
            x_start = self._time_to_x(kf.time)
            x_end   = (self._time_to_x(next_kf.time) if next_kf is not None
                       else self._time_to_x(self.timeline.duration))
            clip_w  = max(0, x_end - x_start)
            if clip_w < 2:
                continue

            is_sel = (self._selected_kf and
                      self._selected_kf[0] is track and
                      self._selected_kf[1] is kf)

            clip_rect = QRect(x_start, track_y + 2, clip_w, th - 4)

            p.save()
            p.setClipRect(clip_rect)

            # Corps
            fill = QColor(110, 60, 160, 200 if is_sel else 160)
            p.setBrush(QBrush(fill))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(clip_rect, 3, 3)

            # Croix diagonales
            if clip_w > 20:
                mid_x = x_start + clip_w // 2
                p.setPen(QPen(QColor(255, 200, 255, 178), 1))
                p.drawLine(x_start + 3, track_y + 4, mid_x, track_y + th - 4)
                p.drawLine(mid_x, track_y + 4, x_end - 3, track_y + th - 4)

            # Barre de titre violette
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(150, 90, 210, 220)))
            title_rect = QRect(x_start, track_y + 2, clip_w, 9)
            p.drawRoundedRect(title_rect, 3, 3)
            p.fillRect(QRect(x_start, track_y + 2 + 4, clip_w, 5), QColor(150, 90, 210, 220))

            # Liseré gauche violet vif
            p.fillRect(QRect(x_start, track_y + 2, 3, th - 4), QColor(180, 100, 255, 242))

            # Contour
            p.setPen(QPen(QColor(255, 255, 255, 178) if is_sel
                         else QColor(140, 80, 200, 160), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(clip_rect.adjusted(0, 0, -1, -1), 3, 3)

            # Nom
            if clip_w > 20:
                name = os.path.splitext(os.path.basename(str(kf.value)))[0]
                p.setClipRect(QRect(x_start + 4, track_y, clip_w - 8, th))
                p.setPen(QColor(240, 210, 255, 230))
                p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                p.drawText(x_start + 6, track_y + 3, name)

            p.restore()

            # Poignées
            if clip_w > 16:
                p.setBrush(QBrush(QColor(255, 255, 255, 82)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(QRect(x_end - 6, track_y + 4, 5, th - 8), 2, 2)
                p.drawRoundedRect(QRect(x_start + 1, track_y + 4, 5, th - 8), 2, 2)

    def _draw_audio_track(self, p: QPainter, track, track_y: int, w: int, th: int = None):
        """Dessine la forme d'onde d'une piste audio."""
        if th is None:
            th = self._track_h
        if not track.audio_path:
            p.setPen(QColor(80, 100, 80))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(LABEL_W + 8, track_y, w - LABEL_W - 8, th),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       "Glissez un fichier audio ici…")
            return

        waveform = _load_waveform(track.audio_path)
        if waveform is None:
            p.setPen(QColor(180, 100, 60))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(QRect(LABEL_W + 8, track_y, w - LABEL_W - 8, th),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"⚠ Format non supporté pour la waveform : {os.path.basename(track.audio_path)}")
            return

        tl_widget = self._timeline_widget
        audio_duration = tl_widget.timeline.duration if tl_widget else 60.0
        if hasattr(track, '_audio_duration') and track._audio_duration > 0:
            audio_duration = track._audio_duration

        if audio_duration <= 0:
            try:
                with wave.open(track.audio_path, 'rb') as wf:
                    audio_duration = wf.getnframes() / float(wf.getframerate())
                track._audio_duration = audio_duration
            except (OSError, EOFError, wave.Error):
                pass
        if audio_duration <= 0:
            audio_duration = max(1.0, tl_widget.timeline.duration if tl_widget else 60.0)

        n = len(waveform)
        if n == 0:
            return

        x0     = LABEL_W
        cy     = track_y + th // 2
        half_h = (th - 8) // 2

        wave_col     = QColor(80, 200, 120)
        wave_fill    = QColor(80, 200, 120, 60)
        wave_col_dim = QColor(50, 140, 80)

        p.setClipRect(QRect(x0, track_y, w - x0, th))

        prev_x = x0
        poly_top = []
        poly_bot = []

        for px in range(x0, w):
            t = self._x_to_time(px)
            if t < 0 or t > audio_duration:
                continue
            if audio_duration <= 0:
                continue
            frac = t / audio_duration
            idx  = int(frac * n)
            if idx >= n:
                continue
            amp  = float(waveform[idx])
            dy   = int(amp * half_h)
            poly_top.append(QPoint(px, cy - dy))
            poly_bot.append(QPoint(px, cy + dy))

        if poly_top:
            # Zone de remplissage
            poly = poly_top + list(reversed(poly_bot))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(wave_fill))
            p.drawPolygon(*poly)

            # Contour haut
            p.setPen(QPen(wave_col, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(poly_top) - 1):
                p.drawLine(poly_top[i], poly_top[i + 1])

            # Contour bas
            p.setPen(QPen(wave_col_dim, 1))
            for i in range(len(poly_bot) - 1):
                p.drawLine(poly_bot[i], poly_bot[i + 1])

        p.setClipping(False)

        # Nom du fichier
        p.setPen(QColor(150, 220, 160, 200))
        p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        name = os.path.basename(track.audio_path)
        p.drawText(QRect(LABEL_W + 6, track_y + 1, 300, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, name)

    def _draw_camera_track(self, p: QPainter, track, track_y: int, w: int, th: int = None):
        """Piste caméra 3D — rendu identique tm.html drawCameraTrack()."""
        if th is None:
            th = self._track_h
        cy  = track_y + th // 2
        h_h = max(4, th // 2 - 4)

        # Fond légèrement bleuté
        p.fillRect(LABEL_W, track_y, w - LABEL_W, th, QColor(20, 20, 50, 82))

        if not track.keyframes:
            p.setPen(QColor(80, 96, 176))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(LABEL_W + 8, track_y, w - LABEL_W - 8, th),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       "🎥 Clic droit → Ajouter un keyframe caméra…")
            return

        # Ligne de trajectoire Z (pointillée) — comme tm.html
        if len(track.keyframes) >= 2:
            z_vals = [kf.value[2] if isinstance(kf.value, (tuple, list)) and len(kf.value) > 2
                      else 3.0 for kf in track.keyframes]
            z_min, z_max = min(z_vals), max(z_vals)
            z_range = max(0.1, z_max - z_min)

            pen = QPen(QColor(80, 120, 255, 160), 1, Qt.PenStyle.DashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            pts = []
            for kf in track.keyframes:
                kx = self._time_to_x(kf.time)
                z = kf.value[2] if isinstance(kf.value, (tuple, list)) and len(kf.value) > 2 else 3.0
                ky = track_y + h_h - int((z - z_min) / z_range * (h_h * 2 - 8)) + 4
                pts.append((kx, ky))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

        # Keyframes caméra — cercles bleus/dorés
        for kf in track.keyframes:
            kx = self._time_to_x(kf.time)
            is_sel = (self._selected_kf and self._selected_kf[1] is kf) or \
                     any(k is kf for (_, k) in self._selected_kfs)
            col = QColor(0xff, 0xc8, 0x40) if is_sel else QColor(0x64, 0xa0, 0xff)
            border = QColor(0xb4, 0x80, 0x10) if is_sel else QColor(0x30, 0x40, 0x80)
            p.setPen(QPen(border, 1))
            p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(kx, cy), KF_RADIUS, KF_RADIUS)
            if isinstance(kf.value, (tuple, list)) and len(kf.value) >= 7:
                p.setPen(QColor(160, 200, 255, 204))
                p.setFont(QFont("Segoe UI", 6))
                p.drawText(kx + 7, cy - 2, f"FOV:{kf.value[6]:.0f}°")

    # ── Événements souris ─────────────────────────────────────────────────────

    # ── Helper : hit-test handle Bézier ─────────────────────────────────────

    def _find_handle_at(self, x: int, y: int, tol: int = HANDLE_RADIUS + 3):
        """
        Cherche un handle Bézier (in ou out) à la position (x, y).
        Retourne (kf, 'in'|'out', track) ou None.
        """
        for track_idx, track in enumerate(self.timeline.tracks):
            if track.value_type in ('shader', 'trans', 'audio'):
                continue
            track_y = self._track_y(track_idx)
            th      = self._track_h_for(track_idx)
            cy      = track_y + th // 2
            half_h  = max(1, th // 2 - 4)

            for kf in track.keyframes:
                if kf.interp != 'bezier':
                    continue
                if not (self._selected_kf and
                        self._selected_kf[1] is kf):
                    continue

                for side in ('in', 'out'):
                    h = kf.handle_in if side == 'in' else kf.handle_out
                    if h.dt == 0.0:
                        continue
                    h_x = self._time_to_x(kf.time + h.dt)
                    dv  = h.dv if isinstance(h.dv, (int, float)) else (h.dv[0] if h.dv else 0.0)
                    h_y = cy - int(dv * half_h * 0.5)
                    if abs(x - h_x) <= tol and abs(y - h_y) <= tol:
                        return kf, side, track
        return None

    def mousePressEvent(self, e: QMouseEvent):
        self.setFocus()
        x = int(e.position().x())
        y = int(e.position().y())

        # ── Clic sur un handle Bézier (prioritaire) ───────────────────────
        hit = self._find_handle_at(x, y)
        if hit:
            kf, side, track = hit
            h = kf.handle_in if side == 'in' else kf.handle_out
            self._drag_handle          = (kf, side)
            self._drag_handle_orig_dt  = h.dt
            self._drag_handle_orig_dv  = h.dv
            self._selected_handle      = (kf, side)
            # Alt+clic → brise la tangente
            if e.modifiers() & Qt.KeyboardModifier.AltModifier:
                self.undo_stack.push(ToggleHandlesLinkedCommand(kf, self))
            self.update()
            return

        # Clic sur la bande marqueurs
        if self._in_marker_band(y) and x >= LABEL_W:
            self._handle_marker_press(e)
            return

        # Clic sur la règle → seek (ou drag loop region)
        if y <= RULER_H and x >= LABEL_W:
            loop_hit = self._loop_hit_test(x, y)
            if loop_hit:
                self._drag_loop = loop_hit
                self._drag_loop_orig_in  = self.timeline.loop_in
                self._drag_loop_orig_out = self.timeline.loop_out
                self._drag_loop_x0 = x
                self.update()
                return
            t = max(0.0, self._x_to_time(x))
            self._current_time = t
            self.time_changed.emit(t)
            self.update()
            return

        if x < LABEL_W or y <= RULER_H + self._marker_offset():
            return

        track_idx = self._track_idx_at_y(y)
        if not (0 <= track_idx < len(self.timeline.tracks)):
            return
        track = self.timeline.tracks[track_idx]

        # Piste audio : pas d'interaction (lecture seule)
        if track.value_type == 'audio':
            return

        # ── Piste Shader / Trans ──────────────────────────────────────────────
        if track.value_type in ('shader', 'trans'):
            # Vérifier d'abord si on est sur un bord de redim
            for i, kf in enumerate(track.keyframes):
                # Ignorer les KFs de fin (value == '') pour le hit-test resize droit/gauche
                # sauf pour le bord droit — ils servent justement de borne de fin
                kfs = track.keyframes

                # ── Redim droite : bord droit du clip courant ────────────────
                # = position du KF suivant (borne de fin ou vrai clip suivant)
                # ou = durée de la timeline s'il n'y a pas de KF suivant
                if kf.value:  # seulement pour les vrais clips, pas les bornes de fin
                    if i + 1 < len(kfs):
                        x_end = self._time_to_x(kfs[i + 1].time)
                    else:
                        x_end = self._time_to_x(self.timeline.duration)
                    if abs(x - x_end) <= 8:
                        if i + 1 < len(kfs):
                            # Déplace le KF suivant (borne de fin ou début clip suivant)
                            self._resize_clip    = (track, kfs[i + 1])
                            self._drag_orig_t    = kfs[i + 1].time
                        else:
                            # Pas de KF suivant : crée une borne de fin virtuelle au drag
                            # On stocke un sentinel None pour le cas "durée libre"
                            self._resize_clip    = (track, None)
                            self._drag_orig_t    = self.timeline.duration
                        self._resize_start_x = x
                        return

                # ── Redim gauche : bord gauche du clip courant (TOUS les clips) ──
                if kf.value:  # seulement pour les vrais clips
                    x_start = self._time_to_x(kf.time)
                    if abs(x - x_start) <= 8:
                        self._resize_left_clip   = (track, kf)
                        self._resize_left_orig_t = kf.time
                        self._resize_start_x     = x
                        return

            # Sinon sélection du clip sous le curseur
            kf = self._find_shader_clip_at(track, x)
            if kf:
                self._selected_kf        = (track, kf)
                self._drag_kf            = (track, kf)
                self._drag_orig_t        = kf.time
                self._drag_start_x       = x
                self._drag_clip_src_track = track   # mémorise la piste source
                self.keyframe_selected.emit(track, kf)
            else:
                self._selected_kf         = None
                self._drag_clip_src_track = None
            self.update()
            return

        # ── Piste normale ─────────────────────────────────────────────────────
        kf = self._find_keyframe_at(track, x)
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if kf:
            if ctrl:
                # Ctrl+clic : toggle dans la multi-sélection
                pair = (track, kf)
                if pair in self._selected_kfs:
                    self._selected_kfs.discard(pair)
                else:
                    self._selected_kfs.add(pair)
                self._selected_kf = pair   # aussi le KF "actif" (pour inspect)
            else:
                # Clic simple sur un KF : si déjà dans la multi-sél → drag groupe
                # sinon → sélection solo
                pair = (track, kf)
                if pair not in self._selected_kfs:
                    self._selected_kfs    = {pair}
                    self._selected_handle = None
                self._selected_kf = pair

            # Prépare le drag (solo ou groupe)
            self._drag_kf      = (track, kf)
            self._drag_orig_t  = kf.time
            self._drag_start_x = x
            # Mémorise les origines de tous les KFs sélectionnés pour le drag groupe
            self._multi_drag_origins = {k: k.time for (_, k) in self._selected_kfs}
            self._selected_handle = None
            self.keyframe_selected.emit(track, kf)

        else:
            if ctrl:
                # Ctrl+clic dans le vide : début rubber-band additif
                self._rubber_band_start = QPoint(x, y)
                self._rubber_band_rect  = QRect(x, y, 0, 0)
            else:
                # Clic simple dans le vide : désélectionne tout + begin rubber-band
                self._selected_kfs    = set()
                self._selected_kf     = None
                self._selected_handle = None
                self._rubber_band_start = QPoint(x, y)
                self._rubber_band_rect  = QRect(x, y, 0, 0)
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent):
        x = int(e.position().x())
        y = int(e.position().y())

        # ── Drag loop region ──────────────────────────────────────────────
        if self._drag_loop:
            dt = (x - self._drag_loop_x0) / self._pixels_per_sec
            tl = self.timeline
            dur = tl.duration
            if self._drag_loop == 'in':
                tl.loop_in = max(0.0, min(tl.loop_out - 0.1,
                                          self._snap_time(self._drag_loop_orig_in + dt)))
            elif self._drag_loop == 'out':
                tl.loop_out = max(tl.loop_in + 0.1,
                                  min(dur, self._snap_time(self._drag_loop_orig_out + dt)))
            else:  # 'region'
                span = self._drag_loop_orig_out - self._drag_loop_orig_in
                new_in = max(0.0, min(dur - span,
                                      self._snap_time(self._drag_loop_orig_in + dt)))
                tl.loop_in  = new_in
                tl.loop_out = new_in + span
            self.update()
            self.data_changed.emit()
            return

        # ── Drag d'un marqueur ────────────────────────────────────────────
        if self._drag_marker:
            dt = (x - self._drag_marker_x0) / self._pixels_per_sec
            self._drag_marker.time = max(0.0, self._snap_time(
                self._drag_marker_orig + dt))
            self.update()
            return

        # ── Drag d'un handle Bézier ───────────────────────────────────────
        if self._drag_handle:
            kf, side = self._drag_handle
            track_idx = next(
                (i for i, tr in enumerate(self.timeline.tracks)
                 for k in tr.keyframes if k is kf), 0)
            th     = self._track_h_for(track_idx)
            cy     = self._track_y(track_idx) + th // 2
            half_h = max(1, th // 2 - 4)

            new_t  = self._x_to_time(x)
            new_dt = new_t - kf.time
            # Contrainte : handle_out à droite, handle_in à gauche
            if side == 'out':
                new_dt = max(0.001, new_dt)
            else:
                new_dt = min(-0.001, new_dt)

            # Delta Y → delta valeur
            dy     = cy - y
            new_dv = (dy / max(1, half_h * 0.5)) if half_h > 0 else 0.0

            h = kf.handle_in if side == 'in' else kf.handle_out
            h.dt = new_dt
            h.dv = new_dv

            # Si handles liés : symétrise l'autre côté
            if kf.handles_linked:
                mirror = kf.handle_out if side == 'in' else kf.handle_in
                mirror.dt = -new_dt
                mirror.dv = -new_dv

            self.update()
            return

        if self._resize_clip:
            track_r, next_kf = self._resize_clip
            dt  = (x - self._resize_start_x) / self._pixels_per_sec
            new_t = max(0.0, self._drag_orig_t + dt)
            if next_kf is not None:
                # Contraint : ne pas dépasser le KF suivant du suivant
                idx_n = track_r.keyframes.index(next_kf)
                # Ne pas aller avant le clip auquel cette borne appartient
                clip_kf = track_r.keyframes[idx_n - 1] if idx_n > 0 else None
                min_t = (clip_kf.time + 0.1) if clip_kf else 0.0
                next_kf.time = max(min_t, new_t)
            # Si next_kf est None (pas de KF de fin), on ne fait rien ici
            # (cas sans borne de fin → durée libre, géré par la timeline)
            self.update()
            return

        if self._resize_left_clip:
            track, kf = self._resize_left_clip
            dt = (x - self._resize_start_x) / self._pixels_per_sec
            # Ne pas dépasser le keyframe précédent ni le suivant
            idx = track.keyframes.index(kf)
            prev_kf = track.keyframes[idx - 1] if idx > 0 else None
            next_kf = track.keyframes[idx + 1] if idx + 1 < len(track.keyframes) else None
            # min_t : ne pas aller avant le KF précédent (ou 0)
            min_t = (prev_kf.time + 0.05) if prev_kf and prev_kf.value else 0.0
            # max_t : ne pas dépasser le KF suivant moins 0.1s
            max_t = (next_kf.time - 0.1) if next_kf else float('inf')
            kf.time = max(min_t, min(max_t, self._resize_left_orig_t + dt))
            track.keyframes.sort(key=lambda k: k.time)
            self.update()
            return

        if self._drag_kf:
            dt    = (x - self._drag_start_x) / self._pixels_per_sec
            new_t = max(0.0, self._drag_orig_t + dt)
            # Snap à la seconde entière si proche (<10px)
            snap = round(new_t)
            if abs(new_t - snap) * self._pixels_per_sec < 10.0:
                new_t = float(snap)
                dt = new_t - self._drag_orig_t

            if len(self._selected_kfs) > 1:
                # Déplace tout le groupe du même Δt
                for (tr, k) in self._selected_kfs:
                    orig = self._multi_drag_origins.get(k, k.time)
                    k.time = max(0.0, orig + dt)
                    tr.keyframes.sort(key=lambda kk: kk.time)
            else:
                track_d, kf_d = self._drag_kf
                # ── Bug #1 fix : contrainte anti-croisement pour clips shader/trans ──
                # Sans contrainte, kf_d.time peut dépasser ses voisins → sort() produit
                # un ordre incohérent → paintEvent freeze sur les itérations suivantes.
                if track_d.value_type in ('shader', 'trans') and kf_d.value:
                    idx = track_d.keyframes.index(kf_d)
                    prev_kf = track_d.keyframes[idx - 1] if idx > 0 else None
                    next_kf = track_d.keyframes[idx + 1] if idx + 1 < len(track_d.keyframes) else None
                    # Durée courante du clip (jusqu'à sa borne de fin)
                    clip_dur = (next_kf.time - kf_d.time) if next_kf is not None else 0.0
                    # Borne min : ne pas aller avant le KF précédent
                    min_t = (prev_kf.time + 0.05) if prev_kf else 0.0
                    # Borne max : ne pas faire déborder la fin dans le clip d'après
                    after_next = track_d.keyframes[idx + 2] if idx + 2 < len(track_d.keyframes) else None
                    if after_next is not None:
                        max_t = after_next.time - clip_dur - 0.05
                    else:
                        max_t = max(min_t, self.timeline.duration - clip_dur)
                    new_t = max(min_t, min(max_t, new_t))
                    # Déplace aussi le KF de fin (borne vide) pour préserver la durée du clip
                    if next_kf is not None and not next_kf.value:
                        next_kf.time = new_t + clip_dur
                # ── Bug #4 fix : trier après chaque modification pour éviter
                # un état incohérent qui freeze paintEvent ──
                kf_d.time = new_t
                track_d.keyframes.sort(key=lambda kk: kk.time)
            self.update()
            return

        # Mise à jour rubber-band
        if self._rubber_band_start:
            rx = min(self._rubber_band_start.x(), x)
            ry = min(self._rubber_band_start.y(), y)
            rw = abs(x - self._rubber_band_start.x())
            rh = abs(y - self._rubber_band_start.y())
            self._rubber_band_rect = QRect(rx, ry, rw, rh)
            self.update()
            return

        # Curseur de redim sur les pistes shader
        if y > RULER_H and x >= LABEL_W:
            track_idx = self._track_idx_at_y(y)
            if 0 <= track_idx < len(self.timeline.tracks):
                track = self.timeline.tracks[track_idx]
                if track.value_type in ('shader', 'trans'):
                    kfs = track.keyframes
                    on_edge = False
                    for i, kf in enumerate(kfs):
                        if not kf.value:
                            continue  # ignorer les bornes de fin
                        # Bord droit
                        x_end = (self._time_to_x(kfs[i+1].time) if i+1 < len(kfs)
                                 else self._time_to_x(self.timeline.duration))
                        if abs(x - x_end) <= 8:
                            on_edge = True; break
                        # Bord gauche
                        if abs(x - self._time_to_x(kf.time)) <= 8:
                            on_edge = True; break
                    self.setCursor(Qt.CursorShape.SizeHorCursor if on_edge
                                   else Qt.CursorShape.ArrowCursor)
                    return
        # Curseur sur les handles de loop region (dans la règle)
        if y <= RULER_H and x >= LABEL_W:
            lh = self._loop_hit_test(x, y)
            if lh in ('in', 'out'):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
            if lh == 'region':
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e: QMouseEvent):
        # ── Fin drag loop region ──────────────────────────────────────────
        if self._drag_loop:
            self._drag_loop = None
            self.data_changed.emit()
            self.update()
            return

        # ── Fin drag marqueur ─────────────────────────────────────────────
        if self._drag_marker:
            self._drag_marker = None
            self.data_changed.emit()
            self.update()
            return

        # ── Fin drag handle Bézier → pousse dans l'undo stack ────────────
        if self._drag_handle:
            kf, side = self._drag_handle
            h = kf.handle_in if side == 'in' else kf.handle_out
            if abs(h.dt - self._drag_handle_orig_dt) > 1e-5:
                self.undo_stack.push(MoveHandleCommand(
                    kf, side,
                    self._drag_handle_orig_dt, self._drag_handle_orig_dv,
                    h.dt, h.dv,
                    self,
                ))
            self._drag_handle = None
            self.data_changed.emit()
            self.update()
            return

        if self._resize_clip:
            track_r, next_kf = self._resize_clip
            if next_kf is not None and abs(next_kf.time - self._drag_orig_t) > 0.01:
                self.keyframe_moved.emit(track_r, next_kf, self._drag_orig_t, next_kf.time)
            self._resize_clip = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        if self._resize_left_clip:
            track, kf = self._resize_left_clip
            if abs(kf.time - self._resize_left_orig_t) > 0.01:
                self.keyframe_moved.emit(track, kf, self._resize_left_orig_t, kf.time)
            self._resize_left_clip = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        if self._drag_kf:
            track, kf = self._drag_kf
            x_rel = int(e.position().x())
            y_rel = int(e.position().y())

            if len(self._selected_kfs) > 1:
                # Construit la liste des déplacements pour undo atomique
                moves = []
                for (tr, k) in self._selected_kfs:
                    orig = self._multi_drag_origins.get(k, k.time)
                    if abs(k.time - orig) > 0.01:
                        moves.append((tr, k, orig, k.time))
                if moves:
                    self.undo_stack.push(MoveMultipleKeyframesCommand(moves, self))
            else:
                # ── Déplacement inter-pistes pour les clips shader/trans ──────
                src_track = self._drag_clip_src_track
                if src_track is not None and src_track.value_type in ('shader', 'trans'):
                    dst_idx = self._track_idx_at_y(y_rel)
                    dst_tracks = self.timeline.tracks
                    if 0 <= dst_idx < len(dst_tracks):
                        dst_track = dst_tracks[dst_idx]
                        if (dst_track is not src_track
                                and dst_track.value_type == src_track.value_type):
                            # Déplace le clip (kf + son KF de fin éventuel) vers dst_track
                            idx = src_track.keyframes.index(kf)
                            candidate = (src_track.keyframes[idx + 1]
                                         if idx + 1 < len(src_track.keyframes) else None)
                            # Bug #3 fix : ne prendre le KF suivant que s'il est bien une
                            # borne de fin vide, PAS le début d'un autre clip
                            end_kf = candidate if (candidate is not None and not candidate.value) else None
                            # Retire de la source
                            src_track.keyframes.remove(kf)
                            if end_kf is not None:
                                src_track.keyframes.remove(end_kf)
                            # Insère dans la destination
                            dst_track.keyframes.append(kf)
                            if end_kf is not None:
                                dst_track.keyframes.append(end_kf)
                            dst_track.keyframes.sort(key=lambda k: k.time)
                            self._selected_kf = (dst_track, kf)
                        elif abs(kf.time - self._drag_orig_t) > 0.01:
                            # Même piste : clip shader déplacé en intra-piste.
                            # On n'utilise PAS keyframe_moved (qui ne gère pas la borne de fin)
                            # → la position est déjà correcte depuis mouseMoveEvent (avec borne de fin).
                            # On émet juste data_changed pour mettre à jour la scène.
                            self.data_changed.emit()
                            # Invalide le chemin mémorisé dans main_window pour forcer
                            # le rechargement du bon shader au prochain tick (is_dragging=False)
                            mw = self.window()
                            if hasattr(mw, '_active_scene_a_path'):
                                mw._active_scene_a_path = None
                    else:
                        if abs(kf.time - self._drag_orig_t) > 0.01:
                            if src_track is not None and src_track.value_type in ('shader', 'trans'):
                                # Idem : clip shader déplacé, borne de fin déjà à jour
                                self.data_changed.emit()
                                mw = self.window()
                                if hasattr(mw, '_active_scene_a_path'):
                                    mw._active_scene_a_path = None
                            else:
                                self.keyframe_moved.emit(track, kf, self._drag_orig_t, kf.time)
                else:
                    if abs(kf.time - self._drag_orig_t) > 0.01:
                        self.keyframe_moved.emit(track, kf, self._drag_orig_t, kf.time)

            self._drag_kf             = None
            self._drag_clip_src_track = None
            self._multi_drag_origins  = {}
            self.update()

        # Fin rubber-band : sélectionne les KFs encadrés
        if self._rubber_band_rect and self._rubber_band_rect.width() > 4:
            self._select_kfs_in_rect(self._rubber_band_rect)
        self._rubber_band_start = None
        self._rubber_band_rect  = None
        self.update()

    def keyPressEvent(self, e: QKeyEvent):
        if e.matches(QKeySequence.StandardKey.Copy) and self._selected_kf:
            self._copy_kf(*self._selected_kf); e.accept()
        elif e.matches(QKeySequence.StandardKey.Paste):
            target = self._selected_kf[0] if self._selected_kf else (
                next((tr for (tr, _) in self._selected_kfs), None))
            if target:
                self._paste_kf(target, self._current_time)
            e.accept()
        elif e.key() == Qt.Key.Key_Delete:
            if self._selected_marker:
                self._delete_marker(self._selected_marker)
            elif self._selected_kfs:
                for (tr, k) in list(self._selected_kfs):
                    self.undo_stack.push(DeleteKeyframeCommand(tr, k, self))
                self._selected_kfs = set()
                self._selected_kf  = None
            elif self._selected_kf:
                self._delete_kf(*self._selected_kf)
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            # Escape → tout désélectionner
            self._selected_kfs    = set()
            self._selected_kf     = None
            self._selected_handle = None
            self._selected_marker = None
            self.update(); e.accept()
        elif e.key() == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+A → sélectionne tous les KFs de toutes les pistes
            self._selected_kfs = set()
            for tr in self.timeline.tracks:
                if tr.value_type not in ('shader', 'trans', 'audio'):
                    for k in tr.keyframes:
                        self._selected_kfs.add((tr, k))
            self.update(); e.accept()
        elif e.key() == Qt.Key.Key_S:
            self._split_at_playhead(); e.accept()
        elif e.key() == Qt.Key.Key_M:
            # M → ajouter un marqueur au playhead
            mt = getattr(self.timeline, 'marker_track', None)
            if mt:
                self._add_marker_at(self._current_time)
            e.accept()
        elif (e.key() == Qt.Key.Key_Left
              and e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Ctrl+← → sauter au marqueur précédent
            mt = getattr(self.timeline, 'marker_track', None)
            if mt:
                m = mt.prev(self._current_time)
                if m:
                    self._current_time = m.time
                    self.time_changed.emit(m.time)
                    self.update()
            e.accept()
        elif (e.key() == Qt.Key.Key_Right
              and e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Ctrl+→ → sauter au marqueur suivant
            mt = getattr(self.timeline, 'marker_track', None)
            if mt:
                m = mt.next(self._current_time)
                if m:
                    self._current_time = m.time
                    self.time_changed.emit(m.time)
                    self.update()
            e.accept()
        else:
            super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            path = e.mimeData().urls()[0].toLocalFile()
            if path.lower().endswith(('.st', '.glsl', '.trans', '.wav', '.mp3', '.ogg')):
                e.acceptProposedAction(); return
        e.ignore()

    def dropEvent(self, e):
        path = e.mimeData().urls()[0].toLocalFile()
        pos  = e.position()
        track_idx = self._track_idx_at_y(int(pos.y()))
        if not (0 <= track_idx < len(self.timeline.tracks)):
            # Drop hors piste : si audio, on le signale au widget parent
            if path.lower().endswith(('.wav', '.mp3', '.ogg')):
                self._timeline_widget.audio_file_dropped.emit(path)
                e.accept()
            return
        track = self.timeline.tracks[track_idx]

        # Drop audio sur piste audio
        if path.lower().endswith(('.wav', '.mp3', '.ogg')):
            if track.value_type == 'audio':
                _waveform_cache.pop(track.audio_path, None)
                track.audio_path = path
                if hasattr(track, '_audio_duration'):
                    del track._audio_duration
                self.update()
                self.data_changed.emit()
            # Dans tous les cas, charger l'audio dans le moteur
            self._timeline_widget.audio_file_dropped.emit(path)
            e.accept()
            return

        if track.value_type == 'trans' and path.lower().endswith('.trans'):
            t = max(0.0, self._x_to_time(int(pos.x())))
            cmd = AddKeyframeCommand(track, t, path, self, 'step')
            self.undo_stack.push(cmd)
            # KF de fin automatique à t + 20 s (même logique que les shaders)
            kfs_after = [k for k in track.keyframes if k.time > t + 0.01]
            end_limit = kfs_after[0].time if kfs_after else None
            end_t = min(t + 20.0, end_limit) if end_limit is not None else t + 20.0
            if end_t > t + 0.1:
                self.undo_stack.push(AddKeyframeCommand(track, end_t, '', self, 'step'))
            e.accept()
            return
        if track.value_type not in ('shader', 'trans'):
            return
        t   = max(0.0, self._x_to_time(int(pos.x())))
        cmd = AddKeyframeCommand(track, t, path, self, 'step')
        self.undo_stack.push(cmd)
        # Crée automatiquement un KF de fin à t + 20 s (ou avant le clip suivant)
        # pour que le clip soit immédiatement redimensionnable
        kfs_after = [k for k in track.keyframes if k.time > t + 0.01]
        end_limit = kfs_after[0].time if kfs_after else None
        default_end = t + 20.0
        end_t = min(default_end, end_limit) if end_limit is not None else default_end
        if end_t > t + 0.1:
            self.undo_stack.push(AddKeyframeCommand(track, end_t, '', self, 'step'))
        e.accept()

    def wheelEvent(self, e: QWheelEvent):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.1 if e.angleDelta().y() > 0 else 0.9
            mx     = e.position().x()
            t_mouse = self._x_to_time(mx)
            self._pixels_per_sec = max(5.0, min(2000.0, self._pixels_per_sec * factor))
            self._scroll_offset  = t_mouse - (mx - LABEL_W) / self._pixels_per_sec
            self.zoom_changed.emit(self._pixels_per_sec)
            self.update(); e.accept()
        else:
            super().wheelEvent(e)

    def contextMenuEvent(self, e: QContextMenuEvent):
        x = e.pos().x()
        y = e.pos().y()

        # Menu clic droit dans la bande marqueurs
        if self._in_marker_band(y) and x >= LABEL_W:
            self._context_menu_marker(e, x)
            return

        track_idx = self._track_idx_at_y(y)
        if not (0 <= track_idx < len(self.timeline.tracks)):
            return

        # Clic dans le label → menu piste
        if x < LABEL_W:
            track = self.timeline.tracks[track_idx]
            menu  = QMenu(self); menu.setStyleSheet(_MENU_STYLE)
            menu.addAction("🎨 Changer la couleur…").triggered.connect(
                lambda: self._change_track_color(track))
            # v2.2 — groupes
            menu.addSeparator()
            group_label = f"📁 Groupe : {track.group}" if track.group else "📁 Assigner à un groupe…"
            menu.addAction(group_label).triggered.connect(
                lambda: self._set_track_group(track))
            if track.group:
                fold_label = "▶ Replier le groupe" if not track.group_folded else "▼ Déplier le groupe"
                menu.addAction(fold_label).triggered.connect(
                    lambda: self._toggle_group_fold(track.group))
            menu.addSeparator()
            menu.addAction("🗑 Supprimer la piste").triggered.connect(
                lambda: self._delete_track(track))
            menu.exec(e.globalPos()); return

        track = self.timeline.tracks[track_idx]
        t     = max(0.0, self._x_to_time(x))
        menu  = QMenu(self); menu.setStyleSheet(_MENU_STYLE)
        clipboard = self._timeline_widget._keyframe_clipboard

        # ── Menu piste shader ──────────────────────────────────────────────
        if track.value_type == 'shader':
            kf = self._find_shader_clip_at(track, x)
            if kf:
                menu.addAction("✂ Couper ici").triggered.connect(
                    lambda: self._split_clip(track, kf, t))
                menu.addAction("📋 Copier ce clip").triggered.connect(
                    lambda: self._copy_kf(track, kf))
                menu.addSeparator()
                menu.addAction("⏱ Définir la durée…").triggered.connect(
                    lambda: self._set_clip_duration(track, kf))
                menu.addSeparator()
                menu.addAction("🗑 Supprimer ce clip").triggered.connect(
                    lambda: self._delete_kf(track, kf))
            else:
                menu.addAction("➕ Ajouter un shader ici…").triggered.connect(
                    lambda: self._add_kf(track, t))
                if clipboard and clipboard.get('source_type') in ('shader', 'trans'):
                    menu.addAction("📋 Coller le clip").triggered.connect(
                        lambda: self._paste_kf(track, t))
            menu.exec(e.globalPos()); return

        # ── Menu piste caméra (v2.2) ───────────────────────────────────────
        if track.value_type == 'camera':
            kf = self._find_keyframe_at(track, x)
            if kf:
                menu.addAction("✏ Éditer la caméra…").triggered.connect(
                    lambda: self._edit_camera_kf(track, kf))
                menu.addSeparator()
                menu.addAction("🗑 Supprimer").triggered.connect(
                    lambda: self._delete_kf(track, kf))
            else:
                menu.addAction("➕ Ajouter un keyframe caméra ici").triggered.connect(
                    lambda: self._add_camera_kf(track, t))
            menu.exec(e.globalPos()); return

        # ── Menu piste transition ───────────────────────────────────────────
        if track.value_type == 'trans':
            kf = self._find_shader_clip_at(track, x)
            if kf:
                menu.addAction("✂ Couper ici").triggered.connect(
                    lambda: self._split_clip(track, kf, t))
                menu.addSeparator()
                menu.addAction("⏱ Définir la durée…").triggered.connect(
                    lambda: self._set_clip_duration(track, kf))
                menu.addSeparator()
                menu.addAction("🗑 Supprimer ce clip").triggered.connect(
                    lambda: self._delete_kf(track, kf))
            else:
                menu.addAction("➕ Ajouter une transition ici…").triggered.connect(
                    lambda: self._add_trans_kf(track, t))
            menu.exec(e.globalPos()); return

        # ── Menu piste normale ─────────────────────────────────────────────
        kf = self._find_keyframe_at(track, x)
        if kf:
            menu.addAction("📋 Copier ce keyframe").triggered.connect(
                lambda: self._copy_kf(track, kf))
            interp_menu = menu.addMenu("Interpolation")
            grp = QActionGroup(self)
            for mode in ['linear', 'step', 'smooth', 'bezier']:
                act = interp_menu.addAction(mode.capitalize())
                act.setCheckable(True); act.setChecked(kf.interp == mode)
                act.setActionGroup(grp)
                act.triggered.connect(
                    lambda checked, m=mode: self._change_interp(track, kf, m))
            interp_menu.addSeparator()
            interp_menu.addAction("✨ Auto-tangent (Catmull-Rom)").triggered.connect(
                lambda: self._apply_auto_tangent(track, kf))
            menu.addSeparator()
            # v2.2 — Expression
            expr_label = f"📐 Expression : {kf.expression[:20]}…" if kf.expression else "📐 Ajouter une expression…"
            menu.addAction(expr_label).triggered.connect(
                lambda: self._edit_kf_expression(track, kf))
            if kf.expression:
                menu.addAction("✕ Effacer l'expression").triggered.connect(
                    lambda: self._clear_kf_expression(track, kf))
            menu.addSeparator()
            menu.addAction("🗑 Supprimer").triggered.connect(
                lambda: self._delete_kf(track, kf))
        else:
            menu.addAction("➕ Ajouter un keyframe ici").triggered.connect(
                lambda: self._add_kf(track, t))
            if clipboard and clipboard.get('source_type') == track.value_type:
                menu.addAction("📋 Coller").triggered.connect(
                    lambda: self._paste_kf(track, t))

        # v2.2 — Groupement de pistes
        menu.addSeparator()
        group_label = f"📁 Groupe : {track.group}" if track.group else "📁 Assigner à un groupe…"
        menu.addAction(group_label).triggered.connect(
            lambda: self._set_track_group(track))
        if track.group:
            fold_label = "▶ Replier le groupe" if not track.group_folded else "▼ Déplier le groupe"
            menu.addAction(fold_label).triggered.connect(
                lambda: self._toggle_group_fold(track.group))

        menu.exec(e.globalPos())

    # ── Helpers ──────────────────────────────────────────────────────────────

    # ── Helpers multi-sélection ─────────────────────────────────────────────

    def _select_kfs_in_rect(self, rect: QRect):
        """Sélectionne (ou ajoute à la sélection) tous les KFs dans le rectangle."""
        for i, track in enumerate(self.timeline.tracks):
            if track.value_type in ('shader', 'trans', 'audio'):
                continue
            track_y = self._track_y(i)
            th_i    = self._track_h_for(i)
            for kf in track.keyframes:
                kx = self._time_to_x(kf.time)
                ky = track_y + th_i // 2
                if rect.contains(kx, ky):
                    self._selected_kfs.add((track, kf))
        self.update()

    def _clear_multi_selection(self):
        """Vide la multi-sélection et la sélection simple."""
        self._selected_kfs    = set()
        self._selected_kf     = None
        self._selected_handle = None
        self.update()

    def _find_keyframe_at(self, track: Track, x: int, tol: int = KF_RADIUS + 3) -> Keyframe | None:
        for kf in track.keyframes:
            if abs(self._time_to_x(kf.time) - x) <= tol:
                return kf
        return None

    def _find_shader_clip_at(self, track: Track, x: int) -> Keyframe | None:
        """Retourne le clip (keyframe) dont le corps contient x.
        Les bornes de fin (value == '') ne sont pas des clips cliquables."""
        kfs = track.keyframes
        for i, kf in enumerate(kfs):
            if not kf.value:
                continue  # borne de fin → pas un clip
            x_start = self._time_to_x(kf.time)
            next_kf = kfs[i + 1] if i + 1 < len(kfs) else None
            if next_kf is not None:
                x_end = self._time_to_x(next_kf.time)
            else:
                x_end = self._time_to_x(self.timeline.duration)
            if x_start <= x < x_end:
                return kf
        return None

    def _set_clip_duration(self, track: Track, kf: Keyframe):
        """Dialogue pour définir la durée exacte d'un clip shader."""
        idx = track.keyframes.index(kf)
        next_kf = track.keyframes[idx + 1] if idx + 1 < len(track.keyframes) else None

        current_dur = (next_kf.time - kf.time) if next_kf else 5.0

        from PyQt6.QtWidgets import QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox
        dlg = QDialog(self.parent() if self.parent() else self)
        dlg.setWindowTitle("Durée du clip")
        dlg.setStyleSheet("background:#1c1e24; color:#d0d3de;")
        dlg.setFixedWidth(260)

        form = QFormLayout(dlg)
        form.setContentsMargins(16, 16, 16, 12)

        spin = QDoubleSpinBox()
        spin.setRange(0.1, 3600.0)
        spin.setDecimals(2)
        spin.setSuffix(" s")
        spin.setSingleStep(0.5)
        spin.setValue(round(current_dur, 2))
        spin.setStyleSheet("""
            QDoubleSpinBox { background:#12141a; color:#c8ccd8; border:1px solid #2a2d3a;
                             border-radius:3px; padding:3px 6px; font:11px 'Segoe UI'; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button
                           { background:#1a1c24; border:none; width:14px; }
        """)
        form.addRow("Durée :", spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet("""
            QPushButton { background:#2a2d3a; color:#c8ccd8; border:1px solid #3a3d4d;
                          border-radius:3px; padding:4px 16px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#343748; }
        """)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_dur = spin.value()
        new_end = kf.time + new_dur

        if next_kf:
            old_t = next_kf.time
            next_kf.time = new_end
            track.keyframes.sort(key=lambda k: k.time)
            self.keyframe_moved.emit(track, next_kf, old_t, new_end)
        else:
            # Pas de keyframe suivant : on en crée un (marqueur de fin)
            self.undo_stack.push(AddKeyframeCommand(
                track, new_end, kf.value, self, 'step'))

        self.update()
        self.data_changed.emit()

    def _split_clip(self, track: Track, kf: Keyframe, t: float):
        """Coupe le clip kf au temps t."""
        # t doit être à l'intérieur du clip (après kf.time et avant le suivant)
        idx = track.keyframes.index(kf)
        next_kf = track.keyframes[idx + 1] if idx + 1 < len(track.keyframes) else None
        if next_kf and not (kf.time < t < next_kf.time):
            return
        if not next_kf and t <= kf.time:
            return
        self.undo_stack.push(SplitClipCommand(track, kf, t, self))

    def _split_at_playhead(self):
        """Coupe tous les clips shader sous le playhead sur toutes les pistes."""
        t = self._current_time
        split_done = False
        for track in self.timeline.tracks:
            if track.value_type not in ('shader', 'trans'):
                continue
            kf = self._find_shader_clip_at(track, self._time_to_x(t))
            if kf:
                idx = track.keyframes.index(kf)
                next_kf = track.keyframes[idx + 1] if idx + 1 < len(track.keyframes) else None
                if next_kf and not (kf.time < t < next_kf.time):
                    continue
                if not next_kf and t <= kf.time:
                    continue
                self.undo_stack.push(SplitClipCommand(track, kf, t, self))
                split_done = True

    def _add_trans_kf(self, track: Track, t: float):
        """Ouvre un dialogue de sélection de fichier .trans et l'ajoute à la piste."""
        from PyQt6.QtWidgets import QFileDialog
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(
            self._timeline_widget.__class__.__module__.replace('.', '/') + '.py'
        ))) if False else ''
        # On cherche le dossier shaders/trans relatif au projet
        try:
            import sys
            for path_dir in sys.path:
                candidate = os.path.join(path_dir, '..', 'shaders', 'trans')
                if os.path.isdir(candidate):
                    base = candidate
                    break
        except (OSError, IndexError):
            pass
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un shader de transition", base or "",
            "Transitions (*.trans);;Tous les fichiers (*)")
        if path:
            cmd = AddKeyframeCommand(track, t, path, self, 'step')
            self.undo_stack.push(cmd)

    def _add_kf(self, track: Track, t: float):
        if track.value_type == 'shader':
            path, _ = QFileDialog.getOpenFileName(
                self, "Choisir un shader", "", "Shaders (*.st *.glsl)")
            if not path: return
            value, interp = path, 'step'
        else:
            value, interp = track.get_default_value(), 'linear'
        self.undo_stack.push(AddKeyframeCommand(track, t, value, self, interp))

    def _delete_kf(self, track: Track, kf: Keyframe):
        self.undo_stack.push(DeleteKeyframeCommand(track, kf, self))

    def _copy_kf(self, track: Track, kf: Keyframe):
        import copy
        selected = self._selected_kfs
        if len(selected) >= 2:
            items = sorted(selected, key=lambda p: p[1].time)
            anchor_t = items[0][1].time
            self._timeline_widget._keyframe_clipboard = {
                'multi': True,
                'source_type': track.value_type,
                'entries': [
                    {
                        'track_uniform': tr.uniform_name,
                        'value_type':    tr.value_type,
                        'dt_offset':     k.time - anchor_t,
                        'value':         copy.deepcopy(k.value),
                        'interp':        k.interp,
                        'handle_in':     copy.deepcopy(k.handle_in),
                        'handle_out':    copy.deepcopy(k.handle_out),
                        'handles_linked':k.handles_linked,
                    }
                    for (tr, k) in items
                ]
            }
        else:
            self._timeline_widget._keyframe_clipboard = {
                'multi': False,
                'value':         copy.deepcopy(kf.value),
                'interp':        kf.interp,
                'source_type':   track.value_type,
                'handle_in':     copy.deepcopy(kf.handle_in),
                'handle_out':    copy.deepcopy(kf.handle_out),
                'handles_linked':kf.handles_linked,
            }

    def _paste_kf(self, track: Track, t: float):
        import copy
        cb = self._timeline_widget._keyframe_clipboard
        if not cb:
            return

        def _no_collision(tr, target_t) -> float:
            while any(abs(k.time - target_t) < 0.01 for k in tr.keyframes):
                target_t += 0.05
            return target_t

        if cb.get('multi'):
            pastes = []
            for entry in cb['entries']:
                target_tr = self.timeline.get_track_by_uniform(entry['track_uniform'])
                if target_tr is None or target_tr.value_type != entry['value_type']:
                    target_tr = track
                if target_tr.value_type != entry['value_type']:
                    continue
                paste_t = _no_collision(target_tr, t + entry['dt_offset'])
                from .timeline import Keyframe as _KF, BezierHandle as _BH
                new_kf = _KF(paste_t, copy.deepcopy(entry['value']), entry['interp'],
                             copy.deepcopy(entry['handle_in']),
                             copy.deepcopy(entry['handle_out']),
                             entry['handles_linked'])
                pastes.append((target_tr, new_kf))
            if pastes:
                self.undo_stack.push(PasteKeyframesCommand(pastes, self))
                self._selected_kfs = {(tr, k) for (tr, k) in pastes}
        else:
            if cb.get('source_type') != track.value_type:
                return
            paste_t = _no_collision(track, t)
            from .timeline import Keyframe as _KF, BezierHandle as _BH
            new_kf = _KF(paste_t, copy.deepcopy(cb['value']), cb['interp'],
                         copy.deepcopy(cb['handle_in']),
                         copy.deepcopy(cb['handle_out']),
                         cb['handles_linked'])
            self.undo_stack.push(PasteKeyframesCommand([(track, new_kf)], self))
            self._selected_kfs = {(track, new_kf)}

    def _change_interp(self, track: Track, kf: Keyframe, mode: str):
        if kf.interp != mode:
            self.undo_stack.push(ChangeInterpolationCommand(track, kf, mode, self))
            # Quand on passe en bézier, initialise les handles si nuls
            if mode == 'bezier' and kf.handle_out.dt == 0.0:
                self._apply_auto_tangent(track, kf)

    def _apply_auto_tangent(self, track: Track, kf: Keyframe):
        """Calcule et applique les handles Catmull-Rom sur le keyframe."""
        # Force le mode bézier si ce n'est pas déjà le cas
        if kf.interp != 'bezier':
            self.undo_stack.push(ChangeInterpolationCommand(track, kf, 'bezier', self))
        track.apply_auto_tangents(kf)
        self.update(); self.data_changed.emit()

    def _change_track_color(self, track: Track):
        col = QColorDialog.getColor(QColor(track.color), self, "Couleur de la piste")
        if col.isValid():
            self.undo_stack.push(ChangeTrackColorCommand(track, col.name(), self))

    def _delete_track(self, track: Track):
        self.timeline.remove_track(track)
        self._selected_kf  = None
        self._selected_kfs = {(tr, k) for (tr, k) in self._selected_kfs if tr is not track}
        self.update(); self.data_changed.emit()

    # ── v2.2 — Expressions dans les keyframes ────────────────────────────────

    def _edit_kf_expression(self, track: Track, kf: Keyframe):
        """
        Ouvre une boîte de dialogue pour éditer l'expression Python d'un keyframe.
        Variables : t, beat, bpm, rms, fft[n]
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Expression Python — Keyframe")
        dlg.setMinimumWidth(450)
        dlg.setStyleSheet("background: #1c1e24; color: #d0d3de; font: 11px 'Consolas';")
        lay = QVBoxLayout(dlg)

        lbl_info = QLabel(
            "Expression évaluée à chaque frame.\n"
            "Variables : t (secondes), beat, bpm, rms, fft[n]\n"
            "Fonctions : sin, cos, abs, floor, clamp, mix, smoothstep, …"
        )
        lbl_info.setStyleSheet("color: #7080a0; font: 9px 'Segoe UI';")
        lay.addWidget(lbl_info)

        le_expr = QLineEdit(kf.expression)
        le_expr.setPlaceholderText("ex: sin(t * 2.0) * 0.5 + 0.5")
        le_expr.setStyleSheet(
            "background: #0d0f14; color: #cdd6f4; border: 1px solid #2a3060;"
            "border-radius: 3px; padding: 4px 8px; font: 12px 'Consolas';"
        )
        lay.addWidget(le_expr)

        lbl_preview = QLabel("Résultat : —")
        lbl_preview.setStyleSheet("color: #89b4fa; font: 9px 'Segoe UI';")
        lay.addWidget(lbl_preview)

        from .timeline import _eval_expression as _eval_expr

        def _update_preview():
            expr = le_expr.text().strip()
            if not expr:
                lbl_preview.setText("Résultat : (vide → valeur fixe utilisée)")
                return
            result = _eval_expr(expr, self._timeline_widget.canvas._current_t,
                                self.timeline.bpm)
            if result is None:
                lbl_preview.setText("⚠ Erreur de syntaxe ou variable inconnue")
                lbl_preview.setStyleSheet("color: #f38ba8; font: 9px 'Segoe UI';")
            else:
                lbl_preview.setText(f"Résultat : {result:.6f}")
                lbl_preview.setStyleSheet("color: #89b4fa; font: 9px 'Segoe UI';")

        le_expr.textChanged.connect(_update_preview)
        _update_preview()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            kf.expression = le_expr.text().strip()
            self.update()
            self.data_changed.emit()

    def _clear_kf_expression(self, track: Track, kf: Keyframe):
        """Supprime l'expression d'un keyframe (retour à la valeur fixe)."""
        kf.expression = ""
        self.update()
        self.data_changed.emit()

    # ── v2.2 — Piste de caméra 3D ────────────────────────────────────────────

    def _add_camera_kf(self, track: Track, t: float):
        """Ajoute un keyframe caméra avec des valeurs par défaut."""
        default_val = (0.0, 0.0, 3.0,   # pos XYZ
                       0.0, 0.0, 0.0,   # target XYZ
                       45.0)             # FOV
        kf = track.add_keyframe(t, default_val, 'linear')
        self._edit_camera_kf(track, kf)
        self.update()
        self.data_changed.emit()

    def _edit_camera_kf(self, track: Track, kf: Keyframe):
        """Ouvre un dialog pour éditer les paramètres de caméra d'un keyframe."""
        val = kf.value if isinstance(kf.value, (tuple, list)) and len(kf.value) >= 7 \
              else (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 45.0)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"🎥 Caméra — {track.name} @ t={kf.time:.2f}s")
        dlg.setMinimumWidth(380)
        dlg.setStyleSheet("background: #1c1e24; color: #d0d3de;")
        form = QFormLayout(dlg)

        def _spin(v, lo=-999.0, hi=999.0, step=0.1):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            sb.setValue(float(v))
            sb.setStyleSheet(
                "background: #0d0f14; color: #cdd6f4; border: 1px solid #2a3060;"
                "border-radius: 3px; padding: 2px 4px;"
            )
            return sb

        px_sb = _spin(val[0]); py_sb = _spin(val[1]); pz_sb = _spin(val[2])
        tx_sb = _spin(val[3]); ty_sb = _spin(val[4]); tz_sb = _spin(val[5])
        fov_sb = _spin(val[6], lo=1.0, hi=180.0, step=1.0)

        def _row3(lbl, a, b, c):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w); row_l.setContentsMargins(0, 0, 0, 0); row_l.setSpacing(4)
            for sb, name in ((a, "X"), (b, "Y"), (c, "Z")):
                lbl_c = QLabel(name)
                lbl_c.setStyleSheet("color:#6272a4; font:8px 'Segoe UI';")
                lbl_c.setFixedWidth(10)
                row_l.addWidget(lbl_c); row_l.addWidget(sb)
            form.addRow(lbl, row_w)

        _row3("Position :",  px_sb, py_sb, pz_sb)
        _row3("Cible (target) :", tx_sb, ty_sb, tz_sb)
        form.addRow("FOV (°) :", fov_sb)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            kf.value = (
                px_sb.value(), py_sb.value(), pz_sb.value(),
                tx_sb.value(), ty_sb.value(), tz_sb.value(),
                fov_sb.value(),
            )
            self.update()
            self.data_changed.emit()

    def _set_track_group(self, track: Track):
        """Assigne une piste à un groupe (ou le vide pour la dégrouper)."""
        # Liste des groupes existants
        existing = sorted({t.group for t in self.timeline.tracks if t.group})
        text, ok = QInputDialog.getText(
            self,
            "Groupe de piste",
            "Nom du groupe (vide = aucun groupe) :",
            text=track.group,
        )
        if ok:
            track.group = text.strip()
            self.update()
            self.data_changed.emit()

    def _toggle_group_fold(self, group_name: str):
        """Replie/déplie toutes les pistes d'un groupe."""
        if not group_name:
            return
        # Cherche l'état actuel (basé sur la première piste du groupe)
        tracks_in_group = [t for t in self.timeline.tracks if t.group == group_name]
        if not tracks_in_group:
            return
        new_folded = not tracks_in_group[0].group_folded
        for t in tracks_in_group:
            t.group_folded = new_folded
        self.update()
        self.data_changed.emit()

    def _draw_track_buttons(self, p: QPainter, track: Track, y: int, th: int):
        """Dessine les boutons M/S/R dans la colonne de label — style tm.html .tl-lbtn."""
        btn_size = 14
        btn_gap  = 1
        # Alignés à droite du label, centrés verticalement
        total_w = 3 * btn_size + 2 * btn_gap
        bx = LABEL_W - total_w - 4
        by = y + (th - btn_size) // 2

        labels    = ['M', 'S', 'R']
        is_active = [not track.enabled,
                     getattr(track, 'solo', False),
                     getattr(track, 'armed', False)]
        act_cols  = [QColor(0x50, 0x80, 0xe8),   # mute  — bleu
                     QColor(0xc8, 0x82, 0x1e),   # solo  — doré
                     QColor(0xe0, 0x40, 0x40)]   # rec   — rouge

        for idx, (lbl, active, acol) in enumerate(zip(labels, is_active, act_cols)):
            bxi = bx + idx * (btn_size + btn_gap)
            rect = QRect(bxi, by, btn_size, btn_size)

            # Fond
            if active:
                bg = QColor(acol)
                bg.setAlpha(40)
                p.fillRect(rect, bg)
                p.setPen(QPen(acol.darker(130), 1))
            else:
                p.setPen(QPen(QColor(0x2a, 0x2d, 0x3a), 1))

            p.drawRect(rect)

            # Texte
            p.setPen(acol if active else QColor(0x40, 0x40, 0x60))
            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, lbl)

    def _draw_marker_band(self, p: QPainter, mt, w: int):
        """Bande de marqueurs — rendu identique tm.html drawMarkers()."""
        band_y = RULER_H
        band_h = MARKER_H

        for m in mt.markers:
            mx = self._time_to_x(m.time)
            if mx < LABEL_W or mx > w:
                continue

            col    = QColor(m.color)
            is_sel = (self._selected_marker is m)

            # Ligne verticale
            p.setPen(QPen(col, 2 if is_sel else 1))
            p.drawLine(mx, band_y, mx, band_y + band_h)

            # Drapeau (flag)
            flag_w, flag_h = 6, 7
            pts = [
                QPoint(mx,           band_y),
                QPoint(mx + flag_w,  band_y),
                QPoint(mx + flag_w,  band_y + flag_h - 2),
                QPoint(mx,           band_y + flag_h),
            ]
            flag_col = QColor(col)
            flag_col.setAlpha(230 if is_sel else 191)
            p.setBrush(QBrush(flag_col))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(*pts)

            # Surbrillance sélection
            if is_sel:
                p.setPen(QPen(QColor(255, 255, 255, 160), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(*pts)

            # Label
            if m.label:
                text_x = mx + flag_w + 3
                lbl_col = col.lighter(160)
                p.setPen(lbl_col)
                p.setFont(QFont("Segoe UI", 7))
                p.setClipRect(QRect(text_x, band_y, w - text_x, band_h))
                p.drawText(text_x, band_y + band_h - 4, m.label)
                p.setClipping(False)

        # Playhead dans la bande marqueurs
        ph_x = self._time_to_x(self._current_time)
        if LABEL_W <= ph_x <= w:
            p.setPen(QPen(COL_PLAYHEAD, 1.5))
            p.drawLine(ph_x, band_y, ph_x, band_y + band_h)

    # ── Piste de marqueurs : hit-test ───────────────────────────────────────

    def _find_marker_at(self, x: int, tol: int = 8) -> 'Marker | None':
        """Retourne le marqueur le plus proche de x dans la bande."""
        mt = getattr(self.timeline, 'marker_track', None)
        if not mt:
            return None
        best, best_d = None, tol + 1
        for m in mt.markers:
            d = abs(self._time_to_x(m.time) - x)
            if d < best_d:
                best, best_d = m, d
        return best if best_d <= tol else None

    # ── Piste de marqueurs : interactions ───────────────────────────────────

    def _handle_marker_press(self, e: QMouseEvent):
        """Gère le clic dans la bande marqueurs."""
        x = int(e.position().x())
        mt = getattr(self.timeline, 'marker_track', None)
        if not mt:
            return

        m = self._find_marker_at(x)
        if m:
            self._selected_marker  = m
            self._drag_marker      = m
            self._drag_marker_orig = m.time
            self._drag_marker_x0   = x
        else:
            # Double-clic dans le vide → créer un marqueur
            if (e.type() == e.type().MouseButtonDblClick or
                    getattr(e, '_dbl', False)):
                self._add_marker_at(max(0.0, self._x_to_time(x)))
            else:
                self._selected_marker = None
        self.update()

    def _add_marker_at(self, t: float):
        """Crée un marqueur avec dialogue de label."""
        mt = getattr(self.timeline, 'marker_track', None)
        if not mt:
            return
        label, ok = QInputDialog.getText(
            self, "Nouveau marqueur", "Nom du marqueur :",
            text="")
        if ok:
            m = mt.add(self._snap_time(t), label.strip())
            self._selected_marker = m
            self.update()
            self.data_changed.emit()

    def _rename_marker(self, m: 'Marker'):
        """Dialogue renommage d'un marqueur."""
        new_label, ok = QInputDialog.getText(
            self, "Renommer le marqueur", "Nouveau nom :",
            text=m.label)
        if ok:
            m.label = new_label.strip()
            self.update()
            self.data_changed.emit()

    def _delete_marker(self, m: 'Marker'):
        """Supprime un marqueur."""
        mt = getattr(self.timeline, 'marker_track', None)
        if mt:
            mt.remove(m)
            if self._selected_marker is m:
                self._selected_marker = None
            self.update()
            self.data_changed.emit()

    def _context_menu_marker(self, e: 'QContextMenuEvent', x: int):
        """Menu clic droit dans la bande marqueurs."""
        mt = getattr(self.timeline, 'marker_track', None)
        if not mt:
            return
        t    = max(0.0, self._x_to_time(x))
        m    = self._find_marker_at(x)
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_STYLE)

        if m:
            self._selected_marker = m
            menu.addAction(f"✏ Renommer « {m.label or 'marqueur'} »").triggered.connect(
                lambda: self._rename_marker(m))
            menu.addAction("🎨 Changer la couleur…").triggered.connect(
                lambda: self._change_marker_color(m))
            menu.addSeparator()
            menu.addAction("🗑 Supprimer").triggered.connect(
                lambda: self._delete_marker(m))
        else:
            menu.addAction("➕ Ajouter un marqueur ici").triggered.connect(
                lambda: self._add_marker_at(t))

        menu.exec(e.globalPos())

    def _change_marker_color(self, m: 'Marker'):
        """Dialogue sélection de couleur pour un marqueur."""
        col = QColorDialog.getColor(QColor(m.color), self, "Couleur du marqueur")
        if col.isValid():
            m.color = col.name()
            self.update()
            self.data_changed.emit()

    # ── Resize piste (double-clic label) ─────────────────────────────────────

    def _resize_track_height_at(self, track_idx: int, label_y: int):
        track = self.timeline.tracks[track_idx]
        current = getattr(track, 'height', self._track_h)
        dlg = QDialog(self.window())
        dlg.setWindowTitle("Hauteur de piste")
        dlg.setStyleSheet("background:#1c1e24; color:#d0d3de;")
        dlg.setFixedWidth(220)
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout(dlg)
        form.setContentsMargins(12, 12, 12, 10)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(20, 160)
        slider.setValue(current)
        slider.setStyleSheet(_SLIDER_MINI_STYLE)
        def _apply(v):
            track.height = v
            self.update()
            # Notifie le parent (TimelineWidget) de recalculer la taille
            if hasattr(self._timeline_widget, '_update_canvas_size'):
                self._timeline_widget._update_canvas_size()
        slider.valueChanged.connect(_apply)
        form.addRow(f"Piste « {track.name} » :", slider)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.setStyleSheet("QPushButton{background:#2a2d3a;color:#c8ccd8;"
                           "border:1px solid #3a3d4d;border-radius:3px;"
                           "padding:3px 14px;font:9px 'Segoe UI';}")
        btns.accepted.connect(dlg.accept)
        form.addRow(btns)
        dlg.exec()
        self.data_changed.emit()

    # ── Snap BPM ─────────────────────────────────────────────────────────────

    def _snap_time(self, t: float) -> float:
        snap_fn = getattr(self.timeline, 'snap', None)
        if snap_fn:
            return snap_fn(t)
        snap = round(t)
        if abs(t - snap) * self._pixels_per_sec < 10.0:
            return float(snap)
        return t

    # ── API publique ──────────────────────────────────────────────────────────

    def set_current_time(self, t: float):
        self._current_time = t; self.update()

    @property
    def is_dragging(self) -> bool:
        """Retourne True si un drag de clip est en cours (shader ou redim).
        Utilisé par main_window._tick pour éviter de recharger le shader
        pendant le drag, ce qui provoque un freeze (compilation GLSL bloquante).
        """
        return (self._drag_kf is not None or
                self._resize_clip is not None or
                self._resize_left_clip is not None)

    def set_scroll_offset(self, offset: float):
        self._scroll_offset = offset; self.update()

    def set_zoom(self, pixels_per_sec: float):
        """Définit le zoom horizontal (pixels par seconde)."""
        self._pixels_per_sec = max(5.0, min(2000.0, pixels_per_sec))
        self.updateGeometry()
        self.update()

    def set_track_height(self, h: int):
        """Définit la hauteur de toutes les pistes."""
        self._track_h = max(20, min(120, h))
        self.update()

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        total_h = sum(getattr(tr, 'height', self._track_h)
                      for tr in self.timeline.tracks) or self._track_h
        return QSize(800, RULER_H + self._marker_offset() + total_h)

    def minimumSizeHint(self):
        return self.sizeHint()


# ── TimelineWidget ─────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    time_changed          = pyqtSignal(float)
    timeline_data_changed = pyqtSignal()
    audio_file_dropped    = pyqtSignal(str)   # path du fichier audio droppé

    def __init__(self, timeline: Timeline, parent=None):
        super().__init__(parent)
        self.timeline   = timeline
        self.undo_stack = QUndoStack(self)
        self._keyframe_clipboard = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 3, 8, 3)
        toolbar.setSpacing(6)

        # Label
        lbl = QLabel("TIMELINE")
        lbl.setStyleSheet("color: #4a5070; font: bold 9px 'Segoe UI'; min-width:56px;")
        toolbar.addWidget(lbl)

        # ── Séparateur ────────────────────────────────────────────────────────
        def _vsep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.VLine)
            s.setStyleSheet("color:#2a2d3a; max-width:1px;")
            return s

        toolbar.addWidget(_vsep())

        # ── Zoom horizontal ───────────────────────────────────────────────────
        lbl_zoom = QLabel("🔍")
        lbl_zoom.setStyleSheet("color:#5a6080; font:10px;")
        lbl_zoom.setToolTip("Zoom horizontal (aussi Ctrl+Molette)")
        toolbar.addWidget(lbl_zoom)

        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setFixedSize(20, 20)
        btn_zoom_out.setStyleSheet(_BTN_SMALL_STYLE)
        btn_zoom_out.setToolTip("Dézoomer")
        toolbar.addWidget(btn_zoom_out)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(1, 200)        # 1 = 5px/s … 200 = 2000px/s (log)
        self._zoom_slider.setValue(self._val_to_zoom_slider(60.0))
        self._zoom_slider.setFixedWidth(110)
        self._zoom_slider.setFixedHeight(16)
        self._zoom_slider.setToolTip("Zoom horizontal")
        self._zoom_slider.setStyleSheet(_SLIDER_MINI_STYLE)
        toolbar.addWidget(self._zoom_slider)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedSize(20, 20)
        btn_zoom_in.setStyleSheet(_BTN_SMALL_STYLE)
        btn_zoom_in.setToolTip("Zoomer")
        toolbar.addWidget(btn_zoom_in)

        btn_fit = QPushButton("⊡")
        btn_fit.setFixedSize(22, 20)
        btn_fit.setStyleSheet(_BTN_SMALL_STYLE)
        btn_fit.setToolTip("Ajuster à la fenêtre (fit)")
        toolbar.addWidget(btn_fit)

        # Affichage valeur zoom
        self._lbl_zoom_val = QLabel("60px/s")
        self._lbl_zoom_val.setStyleSheet("color:#505470; font:8px 'Segoe UI'; min-width:42px;")
        toolbar.addWidget(self._lbl_zoom_val)

        toolbar.addWidget(_vsep())

        # ── Hauteur des pistes ────────────────────────────────────────────────
        lbl_h = QLabel("↕")
        lbl_h.setStyleSheet("color:#5a6080; font:12px;")
        lbl_h.setToolTip("Hauteur des pistes")
        toolbar.addWidget(lbl_h)

        self._height_slider = QSlider(Qt.Orientation.Horizontal)
        self._height_slider.setRange(20, 100)
        self._height_slider.setValue(TRACK_H)
        self._height_slider.setFixedWidth(70)
        self._height_slider.setFixedHeight(16)
        self._height_slider.setToolTip("Hauteur des pistes")
        self._height_slider.setStyleSheet(_SLIDER_MINI_STYLE)
        toolbar.addWidget(self._height_slider)

        toolbar.addWidget(_vsep())

        # ── Snap BPM ───────────────────────────────────────────────────────────────
        self._snap_btn = QPushButton('⊞')
        self._snap_btn.setFixedSize(22, 20)
        self._snap_btn.setCheckable(True)
        self._snap_btn.setToolTip('Activer le snap BPM')
        self._snap_btn.setStyleSheet(_BTN_SMALL_STYLE)
        toolbar.addWidget(self._snap_btn)

        lbl_bpm = QLabel('BPM')
        lbl_bpm.setStyleSheet('color:#505470; font:9px \'Segoe UI\';')
        toolbar.addWidget(lbl_bpm)

        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(20.0, 400.0)
        self._bpm_spin.setDecimals(1)
        self._bpm_spin.setValue(getattr(self.timeline, 'bpm', 120.0))
        self._bpm_spin.setFixedWidth(58)
        self._bpm_spin.setFixedHeight(22)
        self._bpm_spin.setStyleSheet("""
            QDoubleSpinBox { background:#12141a; color:#c8ccd8; border:1px solid #2a2d3a;
                             border-radius:3px; padding:1px 3px; font:9px 'Segoe UI'; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button
                           { background:#1a1c24; border:none; width:12px; }
        """)
        toolbar.addWidget(self._bpm_spin)

        self._div_combo = QComboBox()
        self._div_combo.addItems(['1/1', '1/2', '1/4', '1/8'])
        self._div_combo.setCurrentIndex(2)
        self._div_combo.setFixedWidth(44)
        self._div_combo.setFixedHeight(22)
        self._div_combo.setToolTip('Subdivision de la grille')
        self._div_combo.setStyleSheet("""
            QComboBox { background:#12141a; color:#c8ccd8; border:1px solid #2a2d3a;
                        border-radius:3px; padding:1px 4px; font:9px 'Segoe UI'; }
            QComboBox::drop-down { border:none; width:14px; }
            QComboBox QAbstractItemView { background:#1c1e24; color:#c8ccd8;
                                          selection-background-color:#2f3244; }
        """)
        toolbar.addWidget(self._div_combo)

        toolbar.addWidget(_vsep())
        toolbar.addStretch()

        # ── Bouton Loop region ────────────────────────────────────────────────
        self._loop_btn = QPushButton("⟳ Loop")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(getattr(self.timeline, 'loop_enabled', False))
        self._loop_btn.setFixedHeight(22)
        self._loop_btn.setToolTip(
            "Activer/désactiver la boucle In/Out\n"
            "Glissez les handles verts/rouges sur la règle pour définir In et Out"
        )
        self._loop_btn.setStyleSheet("""
            QPushButton {
                background:#1e2030; color:#6080a0;
                border:1px solid #2a2d3a; border-radius:3px;
                padding:2px 8px; font:9px 'Segoe UI';
            }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            QPushButton:checked {
                background:#1a3a28; color:#60e090;
                border:1px solid #2a6040;
            }
            QPushButton:checked:hover { background:#1f4830; }
        """)
        toolbar.addWidget(self._loop_btn)

        # ── Bouton Set In / Set Out ────────────────────────────────────────────
        btn_set_in = QPushButton("[ In")
        btn_set_in.setFixedSize(36, 22)
        btn_set_in.setToolTip("Placer le point In ici (position actuelle du playhead)")
        btn_set_in.setStyleSheet(_BTN_STYLE)
        toolbar.addWidget(btn_set_in)

        btn_set_out = QPushButton("Out ]")
        btn_set_out.setFixedSize(38, 22)
        btn_set_out.setToolTip("Placer le point Out ici (position actuelle du playhead)")
        btn_set_out.setStyleSheet(_BTN_STYLE)
        toolbar.addWidget(btn_set_out)

        toolbar.addWidget(_vsep())

        # ── Bouton Export courbes ─────────────────────────────────────────────
        btn_export = QPushButton("⬇ Export")
        btn_export.setFixedHeight(22)
        btn_export.setToolTip("Exporter les courbes de la timeline en CSV ou JSON")
        btn_export.setStyleSheet(_BTN_STYLE)
        toolbar.addWidget(btn_export)

        toolbar.addWidget(_vsep())
        lbl_dur = QLabel("Durée:")
        lbl_dur.setStyleSheet("color:#505470; font:9px 'Segoe UI';")
        toolbar.addWidget(lbl_dur)

        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setSuffix(" s")
        self.duration_spinbox.setDecimals(1)
        self.duration_spinbox.setRange(1.0, 7200.0)
        self.duration_spinbox.setValue(self.timeline.duration)
        self.duration_spinbox.setFixedWidth(72)
        self.duration_spinbox.setFixedHeight(22)
        self.duration_spinbox.setStyleSheet("""
            QDoubleSpinBox { background:#12141a; color:#c8ccd8; border:1px solid #2a2d3a;
                             border-radius:3px; padding:1px 3px; font:9px 'Segoe UI'; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button
                           { background:#1a1c24; border:none; width:12px; }
        """)
        self.duration_spinbox.valueChanged.connect(self._on_duration_changed)
        toolbar.addWidget(self.duration_spinbox)
        toolbar.addSpacing(6)

        btn_add = QPushButton("＋ Piste")
        btn_add.setFixedHeight(22)
        btn_add.setStyleSheet(_BTN_STYLE)
        btn_add.clicked.connect(self._add_track_dialog)
        toolbar.addWidget(btn_add)

        tb_w = QWidget()
        tb_w.setLayout(toolbar)
        tb_w.setStyleSheet("background: #14161c; border-bottom: 1px solid #2a2d3a;")
        tb_w.setFixedHeight(30)
        root.addWidget(tb_w)

        # ── Canvas scrollable ─────────────────────────────────────────────────
        self.canvas = TimelineCanvas(self.timeline, self.undo_stack, self)
        self.canvas.time_changed.connect(self.time_changed)
        self.canvas.keyframe_moved.connect(self._on_kf_moved)
        self.canvas.data_changed.connect(self.timeline_data_changed.emit)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self.canvas)
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: #181a20; }
            QScrollBar:horizontal {
                background: #0e1014; height: 12px; border-top: 1px solid #2a2d3a;
            }
            QScrollBar::handle:horizontal {
                background: #2a2d3a; border-radius: 4px; min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover { background: #3a3d50; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QScrollBar:vertical {
                background: #0e1014; width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #2a2d3a; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #3a3d50; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        root.addWidget(self._scroll)

        # ── Connexions zoom / hauteur ─────────────────────────────────────────
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        btn_zoom_out.clicked.connect(lambda: self._zoom_slider.setValue(
            max(self._zoom_slider.minimum(), self._zoom_slider.value() - 15)))
        btn_zoom_in.clicked.connect(lambda: self._zoom_slider.setValue(
            min(self._zoom_slider.maximum(), self._zoom_slider.value() + 15)))
        btn_fit.clicked.connect(self._fit_zoom)
        self._height_slider.valueChanged.connect(self._on_height_slider)
        self.canvas.zoom_changed.connect(self.sync_zoom_from_canvas)

        # BPM / snap
        self._snap_btn.toggled.connect(self._on_snap_toggled)
        self._bpm_spin.valueChanged.connect(self._on_bpm_changed)
        self._div_combo.currentIndexChanged.connect(self._on_div_changed)

        # Loop region
        self._loop_btn.toggled.connect(self._on_loop_toggled)
        btn_set_in.clicked.connect(self._on_set_loop_in)
        btn_set_out.clicked.connect(self._on_set_loop_out)

        # Export courbes
        btn_export.clicked.connect(self._on_export_curves)

    # ── Zoom helpers ───────────────────────────────────────────────────────────

    def _val_to_zoom_slider(self, pps: float) -> int:
        """Convertit pixels/sec → valeur slider (échelle log)."""
        import math
        log_min = math.log(5.0)
        log_max = math.log(2000.0)
        log_val = math.log(max(5.0, pps))
        return int((log_val - log_min) / (log_max - log_min) * 199) + 1

    def _zoom_slider_to_pps(self, v: int) -> float:
        """Convertit valeur slider → pixels/sec (échelle log)."""
        import math
        log_min = math.log(5.0)
        log_max = math.log(2000.0)
        return math.exp(log_min + (v - 1) / 199.0 * (log_max - log_min))

    def _on_zoom_slider(self, v: int):
        pps = self._zoom_slider_to_pps(v)
        self.canvas.set_zoom(pps)
        if pps >= 100:
            self._lbl_zoom_val.setText(f"{int(pps)}px/s")
        else:
            self._lbl_zoom_val.setText(f"{pps:.0f}px/s")

    def _fit_zoom(self):
        """Ajuste le zoom pour que toute la durée tienne dans la vue."""
        available = self._scroll.viewport().width() - LABEL_W
        if available <= 0 or self.timeline.duration <= 0:
            return
        pps = available / self.timeline.duration
        pps = max(5.0, min(2000.0, pps))
        self._zoom_slider.setValue(self._val_to_zoom_slider(pps))
        self.canvas._scroll_offset = 0.0
        self.canvas.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_canvas_size()

    def _update_canvas_size(self):
        """Synchronise la taille du canvas avec le contenu réel."""
        hint = self.canvas.sizeHint()
        vp_w = self._scroll.viewport().width()
        w = max(vp_w, hint.width())
        self.canvas.resize(w, hint.height())

    def _on_height_slider(self, v: int):
        self.canvas.set_track_height(v)
        self._update_canvas_size()

    def _on_snap_toggled(self, checked: bool):
        if hasattr(self.timeline, 'snap_to_grid'):
            self.timeline.snap_to_grid = checked
        self.canvas.update()

    def _on_bpm_changed(self, v: float):
        if hasattr(self.timeline, 'bpm'):
            self.timeline.bpm = v
        self.canvas.update()

    def _on_div_changed(self, idx: int):
        divs = [1, 2, 4, 8]
        if hasattr(self.timeline, 'snap_division'):
            self.timeline.snap_division = divs[idx]
        self.canvas.update()

    def sync_bpm_controls(self):
        """Synchronise les contrôles BPM depuis la timeline (ex: chargement projet)."""
        self._bpm_spin.blockSignals(True)
        self._bpm_spin.setValue(getattr(self.timeline, 'bpm', 120.0))
        self._bpm_spin.blockSignals(False)
        div = getattr(self.timeline, 'snap_division', 4)
        idx = {1: 0, 2: 1, 4: 2, 8: 3}.get(div, 2)
        self._div_combo.setCurrentIndex(idx)
        self._snap_btn.setChecked(getattr(self.timeline, 'snap_to_grid', False))

    def sync_zoom_from_canvas(self):
        """Synchronise le slider si le zoom a changé par Ctrl+Molette."""
        pps = self.canvas._pixels_per_sec
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(self._val_to_zoom_slider(pps))
        self._zoom_slider.blockSignals(False)
        if pps >= 100:
            self._lbl_zoom_val.setText(f"{int(pps)}px/s")
        else:
            self._lbl_zoom_val.setText(f"{pps:.0f}px/s")

    # ── Loop region ────────────────────────────────────────────────────────────

    def _on_loop_toggled(self, checked: bool):
        self.timeline.loop_enabled = checked
        self.canvas.update()
        self.timeline_data_changed.emit()

    def _on_set_loop_in(self):
        """Place le point In à la position actuelle du playhead."""
        t = self.canvas._current_time
        self.timeline.loop_in = max(0.0, min(t, self.timeline.loop_out - 0.1))
        self.canvas.update()
        self.timeline_data_changed.emit()

    def _on_set_loop_out(self):
        """Place le point Out à la position actuelle du playhead."""
        t = self.canvas._current_time
        self.timeline.loop_out = max(self.timeline.loop_in + 0.1,
                                     min(t, self.timeline.duration))
        self.canvas.update()
        self.timeline_data_changed.emit()

    def sync_loop_controls(self):
        """Synchronise le bouton loop depuis la timeline (ex: chargement projet)."""
        self._loop_btn.blockSignals(True)
        self._loop_btn.setChecked(getattr(self.timeline, 'loop_enabled', False))
        self._loop_btn.blockSignals(False)

    # ── Export courbes ──────────────────────────────────────────────────────────

    def _on_export_curves(self):
        """Dialogue d'export des courbes de la timeline en CSV ou JSON."""
        import csv

        # Récupère les pistes exportables (float/vec*)
        exportable = [t for t in self.timeline.tracks
                      if t.value_type not in ('shader', 'trans', 'audio') and t.keyframes]
        if not exportable:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export courbes",
                                    "Aucune piste numérique avec des keyframes à exporter.")
            return

        # Dialog de configuration
        dlg = QDialog(self)
        dlg.setWindowTitle("Export courbes")
        dlg.setStyleSheet("background:#1c1e24; color:#d0d3de;")
        dlg.setMinimumWidth(360)
        from PyQt6.QtWidgets import (QVBoxLayout, QGroupBox, QCheckBox,
                                     QRadioButton, QButtonGroup, QHBoxLayout,
                                     QFormLayout, QSpinBox)
        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(14, 12, 14, 10)
        vlay.setSpacing(8)

        # Sélection des pistes
        grp_tracks = QGroupBox("Pistes à exporter")
        grp_tracks.setStyleSheet(
            "QGroupBox{color:#8090b0;font:bold 9px 'Segoe UI';"
            "border:1px solid #2a2d3a;border-radius:4px;padding-top:8px;margin-top:4px;}"
        )
        grp_vlay = QVBoxLayout(grp_tracks)
        track_checks = []
        for t in exportable:
            cb = QCheckBox(f"{t.name}  [{t.uniform_name}]")
            cb.setChecked(True)
            cb.setStyleSheet("QCheckBox{color:#c8ccd8;font:9px 'Segoe UI';}")
            grp_vlay.addWidget(cb)
            track_checks.append((t, cb))
        vlay.addWidget(grp_tracks)

        # Format
        grp_fmt = QGroupBox("Format")
        grp_fmt.setStyleSheet(
            "QGroupBox{color:#8090b0;font:bold 9px 'Segoe UI';"
            "border:1px solid #2a2d3a;border-radius:4px;padding-top:8px;margin-top:4px;}"
        )
        hfmt = QHBoxLayout(grp_fmt)
        rb_csv  = QRadioButton("CSV")
        rb_json = QRadioButton("JSON")
        rb_csv.setChecked(True)
        rb_csv.setStyleSheet("QRadioButton{color:#c8ccd8;font:9px 'Segoe UI';}")
        rb_json.setStyleSheet("QRadioButton{color:#c8ccd8;font:9px 'Segoe UI';}")
        hfmt.addWidget(rb_csv); hfmt.addWidget(rb_json); hfmt.addStretch()
        vlay.addWidget(grp_fmt)

        # Résolution d'échantillonnage
        form_res = QFormLayout()
        form_res.setContentsMargins(0, 0, 0, 0)
        spin_fps = QSpinBox()
        spin_fps.setRange(1, 240)
        spin_fps.setValue(60)
        spin_fps.setSuffix(" échantillons/s")
        spin_fps.setFixedWidth(140)
        spin_fps.setStyleSheet(
            "QSpinBox{background:#12141a;color:#c8ccd8;"
            "border:1px solid #2a2d3a;border-radius:3px;"
            "padding:1px 3px;font:9px 'Segoe UI';}"
        )
        lbl_res = QLabel("Résolution :")
        lbl_res.setStyleSheet("color:#9090a8;font:9px 'Segoe UI';")
        form_res.addRow(lbl_res, spin_fps)
        vlay.addLayout(form_res)

        # Boutons OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet(
            "QPushButton{background:#2a2d3a;color:#c8ccd8;"
            "border:1px solid #3a3d4d;border-radius:3px;"
            "padding:3px 14px;font:9px 'Segoe UI';}"
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vlay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected_tracks = [t for (t, cb) in track_checks if cb.isChecked()]
        if not selected_tracks:
            return

        fps     = spin_fps.value()
        use_csv = rb_csv.isChecked()
        ext     = "csv" if use_csv else "json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les courbes", f"curves.{ext}",
            f"{'CSV' if use_csv else 'JSON'} (*.{ext})"
        )
        if not path:
            return

        # Génère les données échantillonnées
        dur  = self.timeline.duration
        step = 1.0 / max(1, fps)
        times = []
        t = 0.0
        while t <= dur + 1e-9:
            times.append(round(t, 6))
            t += step

        if use_csv:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # En-tête
                header = ['time']
                for tr in selected_tracks:
                    if tr.value_type == 'float':
                        header.append(tr.uniform_name)
                    else:
                        n = {'vec2': 2, 'vec3': 3, 'vec4': 4}.get(tr.value_type, 1)
                        suffix = ['x', 'y', 'z', 'w']
                        header += [f"{tr.uniform_name}.{suffix[i]}" for i in range(n)]
                writer.writerow(header)
                # Lignes
                for ts in times:
                    row = [f"{ts:.6f}"]
                    for tr in selected_tracks:
                        val = tr.get_value_at(ts)
                        if val is None:
                            val = tr.get_default_value()
                        if isinstance(val, (int, float)):
                            row.append(f"{val:.6f}")
                        elif isinstance(val, tuple):
                            row += [f"{v:.6f}" for v in val]
                        else:
                            row.append(str(val))
                    writer.writerow(row)
        else:
            import json as _json
            output = {
                'duration': dur,
                'sample_rate': fps,
                'tracks': []
            }
            for tr in selected_tracks:
                samples = []
                for ts in times:
                    val = tr.get_value_at(ts)
                    if val is None:
                        val = tr.get_default_value()
                    if isinstance(val, tuple):
                        val = list(val)
                    samples.append({'time': ts, 'value': val})
                output['tracks'].append({
                    'name':         tr.name,
                    'uniform_name': tr.uniform_name,
                    'value_type':   tr.value_type,
                    'keyframes': [
                        {'time': kf.time,
                         'value': list(kf.value) if isinstance(kf.value, tuple) else kf.value,
                         'interp': kf.interp}
                        for kf in tr.keyframes
                    ],
                    'samples': samples
                })
            with open(path, 'w', encoding='utf-8') as f:
                _json.dump(output, f, indent=2)

        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Export terminé",
            f"Export réussi !\n{len(selected_tracks)} piste(s) — {len(times)} points\n{path}"
        )

    def _add_track_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Nouvelle piste")
        dlg.setStyleSheet("background: #1c1e24; color: #d0d3de;")
        form = QFormLayout(dlg)
        le_name  = QLineEdit("Intensité")
        le_uni   = QLineEdit("uIntensity")
        le_group = QLineEdit("")
        le_group.setPlaceholderText("(optionnel)")
        cb_type  = QComboBox()
        cb_type.addItems(['float', 'vec2', 'vec3', 'vec4', 'shader', 'trans', 'audio', 'camera'])

        def _on_type_changed(text):
            """Auto-fill pour les pistes spéciales."""
            if text == 'camera':
                le_uni.setText('_camera')
                le_uni.setEnabled(False)
            else:
                le_uni.setEnabled(True)

        cb_type.currentTextChanged.connect(_on_type_changed)

        form.addRow("Nom :",          le_name)
        form.addRow("Uniform GLSL :", le_uni)
        form.addRow("Type :",         cb_type)
        form.addRow("Groupe :",       le_group)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vtype = cb_type.currentText()
            if vtype == 'audio':
                # Piste audio : propose de choisir le fichier tout de suite
                audio_path, _ = QFileDialog.getOpenFileName(
                    self, "Choisir un fichier audio", "",
                    "Audio (*.wav *.mp3 *.ogg)"
                )
                track = self.timeline.add_audio_track(le_name.text(), audio_path or "")
                if audio_path:
                    self.audio_file_dropped.emit(audio_path)
            elif vtype == 'camera':
                track = self.timeline.add_camera_track(le_name.text())
            else:
                track = self.timeline.add_track(le_name.text(), le_uni.text(), vtype)
            track.group = le_group.text().strip()
            self.canvas.update()
            self.timeline_data_changed.emit()

    def _on_kf_moved(self, track, kf, old_t, new_t):
        self.undo_stack.push(MoveKeyframeCommand(track, kf, old_t, new_t, self.canvas))

    def _on_duration_changed(self, value: float):
        self.timeline.duration = value
        self.timeline_data_changed.emit()

    def set_current_time(self, t: float):
        self.canvas.set_current_time(t)

    def set_duration(self, duration: float):
        self.duration_spinbox.blockSignals(True)
        self.duration_spinbox.setValue(duration)
        self.duration_spinbox.blockSignals(False)

    def refresh_audio_waveform(self, audio_path: str, audio_duration: float = 0.0):
        """Met à jour la piste audio existante (ou la première trouvée) avec le nouveau fichier."""
        _waveform_cache.pop(audio_path, None)  # force rechargement
        audio_track = None
        for t in self.timeline.tracks:
            if t.value_type == 'audio':
                audio_track = t
                break
        if audio_track is None:
            # Crée automatiquement une piste audio
            audio_track = self.timeline.add_audio_track("Audio", audio_path)
            # Insère en première position
            self.timeline.tracks.remove(audio_track)
            self.timeline.tracks.insert(0, audio_track)
        audio_track.audio_path = audio_path
        if audio_duration > 0:
            audio_track._audio_duration = audio_duration
        self.canvas.update()


# ── Styles ────────────────────────────────────────────────────────────────────

_BTN_STYLE = """
QPushButton {
    background: #1e2030; color: #8090b0;
    border: 1px solid #2a2d3a; border-radius: 3px;
    padding: 0 8px; font: 9px 'Segoe UI';
}
QPushButton:hover  { background: #2a2d3a; color: #c0c8e0; }
QPushButton:pressed{ background: #303448; }
QPushButton:checked { background: #1e3a28; color: #60e090; border-color: #2a6040; }
"""

_BTN_SMALL_STYLE = """
QPushButton {
    background: #1e2030; color: #8090b0;
    border: 1px solid #2a2d3a; border-radius: 3px;
    font: bold 12px 'Segoe UI'; padding: 0;
}
QPushButton:hover  { background: #2a2d3a; color: #c0c8e0; }
QPushButton:pressed{ background: #303448; }
"""

_SLIDER_MINI_STYLE = """
QSlider::groove:horizontal {
    background: #1e2030; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3a5888; width: 10px; height: 10px;
    margin: -3px 0; border-radius: 5px;
}
QSlider::handle:horizontal:hover { background: #5080c0; }
"""

_MENU_STYLE = """
QMenu {
    background: #1c1e24; color: #c8ccd8;
    border: 1px solid #3a3d4d; border-radius: 4px; padding: 4px;
}
QMenu::item { padding: 5px 20px; border-radius: 3px; }
QMenu::item:selected { background: #2f3244; }
QMenu::separator { height: 1px; background: #2a2d3a; margin: 3px 8px; }
"""
