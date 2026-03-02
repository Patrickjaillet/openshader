"""
arrangement_view.py — Timeline multi-piste avancée : Arrangement View
=======================================================================
v6.0 — Vue d'arrangement style Ableton Live avec :

  ┌─────────────────────────────────────────────────────────────────┐
  │  ArrangementView (QWidget)                                       │
  │    ├── ArrangementToolbar  — transport + contrôles               │
  │    ├── TempoMapEditor      — piste BPM automation (bande dédiée) │
  │    ├── CuePanel            — liste des cue points navigables     │
  │    └── ArrangementCanvas   — vue principale des blocs            │
  │           ├── Règle + BPM ruler                                   │
  │           ├── Piste Cue (bande nommée, fond de grille musical)   │
  │           └── N pistes de blocs (SceneBlock) par type            │
  └─────────────────────────────────────────────────────────────────┘

Modèles de données (sérialisables)
  TempoPoint      — un point d'automation BPM (t, bpm, easing)
  TempoMap        — liste de TempoPoints → bpm_at(t), beat_at(t)
  CuePoint        — marqueur nommé navigable au clavier
  CueTrack        — liste de CuePoint
  ArrangementBlock — un bloc/clip positionné sur une piste
  ArrangementTrack — une piste de l'arrangement (contient des blocs)
  Arrangement     — le modèle complet

Signaux ArrangementView
  time_changed(float)
  arrangement_data_changed()
  cue_activated(CuePoint)         — cue navigué au clavier (live)
  scene_block_activated(str)      — nom de scène à charger dans le moteur
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore  import Qt, QRect, QPoint, QSize, pyqtSignal, QTimer
from PyQt6.QtGui   import (QPainter, QColor, QPen, QBrush, QFont,
                            QLinearGradient, QMouseEvent, QKeyEvent,
                            QWheelEvent, QContextMenuEvent, QFontMetrics,
                            QUndoStack, QUndoCommand)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSplitter, QMenu, QInputDialog, QColorDialog, QMessageBox, QDialog,
    QFormLayout, QDoubleSpinBox, QLineEdit, QComboBox, QDialogButtonBox,
    QFrame, QSizePolicy, QToolButton, QCheckBox, QScrollBar
)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes visuelles
# ──────────────────────────────────────────────────────────────────────────────

LABEL_W        = 140    # largeur des labels de piste (px)
RULER_H        = 22     # hauteur de la règle temporelle (px)
TEMPO_H        = 32     # hauteur de la bande BPM automation
CUE_BAND_H     = 20     # hauteur de la bande cue points (sous la règle)
BLOCK_RADIUS   = 4
MIN_BLOCK_W    = 8      # largeur minimale d'affichage d'un bloc (px)
RESIZE_ZONE    = 8      # zone de sensibilité pour le resize (px)
DEFAULT_PPS    = 80.0   # pixels par seconde au démarrage

# Couleurs
C_BG           = QColor(14, 16, 22)
C_RULER        = QColor(12, 14, 20)
C_RULER_TEXT   = QColor(120, 130, 155)
C_TEMPO_BG     = QColor(10, 12, 18)
C_TEMPO_CURVE  = QColor(80,  200, 130)
C_TEMPO_POINT  = QColor(100, 220, 155)
C_CUE_BG       = QColor(12, 14, 24)
C_CUE_FLAG     = QColor(255, 190, 60)
C_CUE_SEL      = QColor(255, 230, 100)
C_GRID_MEASURE = QColor(35, 38, 50)
C_GRID_BEAT    = QColor(22, 24, 34)
C_PLAYHEAD     = QColor(220, 100, 50)
C_TRACK_HDR    = QColor(18, 20, 28)
C_SEL_OVERLAY  = QColor(80, 140, 255, 40)

# Palette de couleurs pour les blocs (cyclée)
_BLOCK_PALETTE = [
    QColor(48,  108, 195),
    QColor(175,  65,  50),
    QColor(48,  150,  88),
    QColor(148,  85, 190),
    QColor(190, 125,  30),
    QColor(48,  165, 165),
    QColor(190,  60, 120),
    QColor(90,  150,  50),
    QColor(60,  130, 190),
    QColor(200,  90,  50),
]
_block_color_cache: dict[str, QColor] = {}


def _color_for_name(name: str) -> QColor:
    if name not in _block_color_cache:
        idx = len(_block_color_cache) % len(_BLOCK_PALETTE)
        _block_color_cache[name] = _BLOCK_PALETTE[idx]
    return _block_color_cache[name]


# ──────────────────────────────────────────────────────────────────────────────
# Modèles de données
# ──────────────────────────────────────────────────────────────────────────────

EASING_LABELS = {
    "step":    "Instantané",
    "linear":  "Linéaire",
    "smooth":  "Lissé (ease in/out)",
    "ease_in": "Ease In",
    "ease_out":"Ease Out",
}


@dataclass
class TempoPoint:
    """Un point d'automation BPM."""
    time:   float         # secondes
    bpm:    float         # BPM à ce point
    easing: str = "linear"  # "step" | "linear" | "smooth" | "ease_in" | "ease_out"

    def to_dict(self) -> dict:
        return {"time": self.time, "bpm": self.bpm, "easing": self.easing}

    @staticmethod
    def from_dict(d: dict) -> "TempoPoint":
        return TempoPoint(float(d["time"]), float(d["bpm"]),
                          d.get("easing", "linear"))


def _ease(alpha: float, mode: str) -> float:
    """Applique un easing sur alpha ∈ [0, 1]."""
    if mode == "step":     return 0.0
    if mode == "linear":   return alpha
    if mode == "smooth":   return alpha * alpha * (3 - 2 * alpha)
    if mode == "ease_in":  return alpha * alpha
    if mode == "ease_out": return 1 - (1 - alpha) ** 2
    return alpha


class TempoMap:
    """
    Séquence de TempoPoints définissant l'automation du BPM.
    Expose :
      bpm_at(t)   → BPM interpolé à l'instant t (secondes)
      beat_at(t)  → numéro de beat cumulatif à l'instant t
      t_of_beat(b) → temps (secondes) correspondant au beat b
    Le premier point s'applique depuis t=0 même si son time > 0.
    """

    def __init__(self, default_bpm: float = 120.0):
        self.points: list[TempoPoint] = [TempoPoint(0.0, default_bpm, "linear")]

    # ── API ─────────────────────────────────────────────────────────────────

    def set_point(self, t: float, bpm: float, easing: str = "linear") -> TempoPoint:
        for p in self.points:
            if abs(p.time - t) < 0.01:
                p.bpm = bpm; p.easing = easing
                return p
        import bisect
        idx = bisect.bisect_left([p.time for p in self.points], t)
        pt = TempoPoint(t, bpm, easing)
        self.points.insert(idx, pt)
        return pt

    def remove_point(self, pt: TempoPoint):
        if pt in self.points and len(self.points) > 1:
            self.points.remove(pt)

    def bpm_at(self, t: float) -> float:
        """BPM interpolé à l'instant t."""
        pts = self.points
        if not pts:
            return 120.0
        if len(pts) == 1 or t <= pts[0].time:
            return pts[0].bpm
        if t >= pts[-1].time:
            return pts[-1].bpm
        import bisect
        idx  = bisect.bisect_right([p.time for p in pts], t) - 1
        pa, pb = pts[idx], pts[idx + 1]
        span = pb.time - pa.time
        if span <= 0:
            return pa.bpm
        alpha = _ease((t - pa.time) / span, pb.easing)
        return pa.bpm + (pb.bpm - pa.bpm) * alpha

    def beat_at(self, t: float) -> float:
        """Numéro de beat (cumulatif) à l'instant t, calculé par intégration."""
        pts = self.points
        if not pts:
            return t * 120.0 / 60.0
        total_beats = 0.0
        prev_t = 0.0
        for i, pt in enumerate(pts):
            seg_end = pt.time if i < len(pts) - 1 else float("inf")
            if t <= prev_t:
                break
            t_end = min(t, pts[i + 1].time if i + 1 < len(pts) else t)
            # Pour chaque segment on intègre BPM/60
            # approximation Simpson rapide (divise le segment en 8 sous-intervalles)
            dt = t_end - prev_t
            if dt <= 0:
                prev_t = pt.time
                continue
            n = 8
            beats = 0.0
            for k in range(n):
                tt = prev_t + dt * k / n
                beats += self.bpm_at(tt)
            beats = beats / n * dt / 60.0
            total_beats += beats
            prev_t = t_end
            if t_end >= t:
                break
        return total_beats

    def t_of_beat(self, beat: float, max_t: float = 3600.0) -> float:
        """Retourne le temps correspondant au beat b (recherche dichotomique)."""
        lo, hi = 0.0, max_t
        for _ in range(32):
            mid = (lo + hi) * 0.5
            if self.beat_at(mid) < beat:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"points": [p.to_dict() for p in self.points]}

    def from_dict(self, d: dict):
        self.points = [TempoPoint.from_dict(p) for p in d.get("points", [])]
        if not self.points:
            self.points = [TempoPoint(0.0, 120.0)]


@dataclass
class CuePoint:
    """Un cue point nommé, navigable au clavier."""
    cue_id:  str   = field(default_factory=lambda: str(uuid.uuid4())[:6])
    time:    float = 0.0
    label:   str   = "Cue"
    color:   str   = "#F59E0B"
    # shortcut : touche clavier 1-9 ou None
    hotkey:  Optional[str] = None

    def to_dict(self) -> dict:
        return {"cue_id": self.cue_id, "time": self.time,
                "label": self.label, "color": self.color,
                "hotkey": self.hotkey}

    @staticmethod
    def from_dict(d: dict) -> "CuePoint":
        c = CuePoint()
        c.cue_id = d.get("cue_id", str(uuid.uuid4())[:6])
        c.time   = float(d.get("time",  0.0))
        c.label  = d.get("label", "Cue")
        c.color  = d.get("color", "#F59E0B")
        c.hotkey = d.get("hotkey")
        return c


class CueTrack:
    """Liste de CuePoint, triés par temps."""

    def __init__(self):
        self.cues: list[CuePoint] = []

    def add(self, t: float, label: str = "", color: str = "#F59E0B",
            hotkey: Optional[str] = None) -> CuePoint:
        cue = CuePoint(time=t, label=label or f"Cue {len(self.cues)+1}",
                       color=color, hotkey=hotkey)
        import bisect
        idx = bisect.bisect_left([c.time for c in self.cues], t)
        self.cues.insert(idx, cue)
        return cue

    def remove(self, cue: CuePoint):
        if cue in self.cues:
            self.cues.remove(cue)

    def prev(self, t: float) -> Optional[CuePoint]:
        candidates = [c for c in self.cues if c.time < t - 1e-5]
        return candidates[-1] if candidates else None

    def next(self, t: float) -> Optional[CuePoint]:
        candidates = [c for c in self.cues if c.time > t + 1e-5]
        return candidates[0] if candidates else None

    def at_hotkey(self, key: str) -> Optional[CuePoint]:
        for c in self.cues:
            if c.hotkey == key:
                return c
        return None

    def to_dict(self) -> dict:
        return {"cues": [c.to_dict() for c in self.cues]}

    def from_dict(self, d: dict):
        self.cues = [CuePoint.from_dict(c) for c in d.get("cues", [])]


@dataclass
class ArrangementBlock:
    """Un bloc (clip de scène) sur une piste d'arrangement."""
    block_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    scene_name: str   = ""        # nom de la SceneItem à activer
    start:      float = 0.0       # secondes
    duration:   float = 4.0       # secondes
    color:      str   = ""        # vide = auto depuis scene_name
    muted:      bool  = False
    linked_to:  Optional[str] = None   # block_id d'un autre bloc (liaison)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return {
            "block_id":   self.block_id,
            "scene_name": self.scene_name,
            "start":      self.start,
            "duration":   self.duration,
            "color":      self.color,
            "muted":      self.muted,
            "linked_to":  self.linked_to,
        }

    @staticmethod
    def from_dict(d: dict) -> "ArrangementBlock":
        b = ArrangementBlock()
        b.block_id   = d.get("block_id",   str(uuid.uuid4())[:8])
        b.scene_name = d.get("scene_name", "")
        b.start      = float(d.get("start",    0.0))
        b.duration   = float(d.get("duration", 4.0))
        b.color      = d.get("color",      "")
        b.muted      = bool(d.get("muted",  False))
        b.linked_to  = d.get("linked_to")
        return b

    def clone(self) -> "ArrangementBlock":
        c = ArrangementBlock.from_dict(copy.deepcopy(self.to_dict()))
        c.block_id = str(uuid.uuid4())[:8]
        return c


@dataclass
class ArrangementTrack:
    """Une piste de l'arrangement, contenant des blocs."""
    track_id:   str             = field(default_factory=lambda: str(uuid.uuid4())[:6])
    name:       str             = "Piste"
    height:     int             = 48
    color:      str             = "#22283a"
    blocks:     list[ArrangementBlock] = field(default_factory=list)
    solo:       bool            = False
    muted:      bool            = False

    def add_block(self, scene_name: str, start: float,
                  duration: float = 4.0) -> ArrangementBlock:
        b = ArrangementBlock(scene_name=scene_name, start=start, duration=duration)
        self.blocks.append(b)
        self.blocks.sort(key=lambda bl: bl.start)
        return b

    def remove_block(self, block: ArrangementBlock):
        if block in self.blocks:
            self.blocks.remove(block)

    def block_at(self, t: float) -> Optional[ArrangementBlock]:
        for b in self.blocks:
            if b.start <= t < b.end:
                return b
        return None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name":     self.name,
            "height":   self.height,
            "color":    self.color,
            "muted":    self.muted,
            "solo":     self.solo,
            "blocks":   [b.to_dict() for b in self.blocks],
        }

    @staticmethod
    def from_dict(d: dict) -> "ArrangementTrack":
        tr = ArrangementTrack()
        tr.track_id = d.get("track_id", str(uuid.uuid4())[:6])
        tr.name     = d.get("name",   "Piste")
        tr.height   = int(d.get("height", 48))
        tr.color    = d.get("color",  "#22283a")
        tr.muted    = bool(d.get("muted", False))
        tr.solo     = bool(d.get("solo",  False))
        tr.blocks   = [ArrangementBlock.from_dict(b) for b in d.get("blocks", [])]
        return tr


class Arrangement:
    """Modèle complet de l'arrangement."""

    def __init__(self):
        self.duration:   float             = 120.0
        self.tempo_map:  TempoMap          = TempoMap(120.0)
        self.cue_track:  CueTrack          = CueTrack()
        self.tracks:     list[ArrangementTrack] = []

    # ── Pistes ──────────────────────────────────────────────────────────────

    def add_track(self, name: str = "Piste") -> ArrangementTrack:
        tr = ArrangementTrack(name=name)
        self.tracks.append(tr)
        return tr

    def remove_track(self, track: ArrangementTrack):
        if track in self.tracks:
            self.tracks.remove(track)

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version":   "6.0",
            "duration":  self.duration,
            "tempo_map": self.tempo_map.to_dict(),
            "cue_track": self.cue_track.to_dict(),
            "tracks":    [t.to_dict() for t in self.tracks],
        }

    def from_dict(self, d: dict):
        self.duration = float(d.get("duration", 120.0))
        self.tempo_map.from_dict(d.get("tempo_map", {}))
        self.cue_track.from_dict(d.get("cue_track", {}))
        self.tracks = [ArrangementTrack.from_dict(t)
                       for t in d.get("tracks", [])]


# ──────────────────────────────────────────────────────────────────────────────
# Commandes Undo/Redo
# ──────────────────────────────────────────────────────────────────────────────

class MoveBlockCmd(QUndoCommand):
    def __init__(self, block: ArrangementBlock, old_start: float,
                 new_start: float, canvas):
        super().__init__("Déplacer bloc")
        self._b = block; self._old = old_start; self._new = new_start
        self._canvas = canvas

    def redo(self):
        self._b.start = self._new
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._b.start = self._old
        self._canvas.update(); self._canvas.data_changed.emit()


class ResizeBlockCmd(QUndoCommand):
    def __init__(self, block: ArrangementBlock,
                 old_start: float, old_dur: float,
                 new_start: float, new_dur: float, canvas):
        super().__init__("Redimensionner bloc")
        self._b = block
        self._os, self._od = old_start, old_dur
        self._ns, self._nd = new_start, new_dur
        self._canvas = canvas

    def redo(self):
        self._b.start    = self._ns
        self._b.duration = self._nd
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._b.start    = self._os
        self._b.duration = self._od
        self._canvas.update(); self._canvas.data_changed.emit()


class AddBlockCmd(QUndoCommand):
    def __init__(self, track: ArrangementTrack,
                 block: ArrangementBlock, canvas):
        super().__init__("Ajouter bloc")
        self._track = track; self._block = block; self._canvas = canvas

    def redo(self):
        if self._block not in self._track.blocks:
            self._track.blocks.append(self._block)
            self._track.blocks.sort(key=lambda b: b.start)
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._track.remove_block(self._block)
        self._canvas.update(); self._canvas.data_changed.emit()


class DeleteBlockCmd(QUndoCommand):
    def __init__(self, track: ArrangementTrack,
                 block: ArrangementBlock, canvas):
        super().__init__("Supprimer bloc")
        self._track = track; self._block = block; self._canvas = canvas

    def redo(self):
        self._track.remove_block(self._block)
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        if self._block not in self._track.blocks:
            self._track.blocks.append(self._block)
            self._track.blocks.sort(key=lambda b: b.start)
        self._canvas.update(); self._canvas.data_changed.emit()


class SplitBlockCmd(QUndoCommand):
    def __init__(self, track: ArrangementTrack, block: ArrangementBlock,
                 split_t: float, canvas):
        super().__init__("Couper bloc")
        self._track  = track
        self._block  = block
        self._split  = split_t
        self._canvas = canvas
        self._new_block: Optional[ArrangementBlock] = None

    def redo(self):
        if self._new_block is None:
            self._new_block = self._block.clone()
            offset = self._split - self._block.start
            self._new_block.start    = self._split
            self._new_block.duration = self._block.duration - offset
            self._block.duration     = offset
        else:
            self._block.duration = self._split - self._block.start
            if self._new_block not in self._track.blocks:
                self._track.blocks.append(self._new_block)
                self._track.blocks.sort(key=lambda b: b.start)
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._block.duration = self._new_block.end - self._block.start
        self._track.remove_block(self._new_block)
        self._canvas.update(); self._canvas.data_changed.emit()


class DuplicateBlockCmd(QUndoCommand):
    def __init__(self, track: ArrangementTrack,
                 original: ArrangementBlock, canvas):
        super().__init__("Dupliquer bloc")
        self._track = track
        self._orig  = original
        self._clone = original.clone()
        self._clone.start = original.end   # colle juste après
        self._canvas = canvas

    def redo(self):
        if self._clone not in self._track.blocks:
            self._track.blocks.append(self._clone)
            self._track.blocks.sort(key=lambda b: b.start)
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._track.remove_block(self._clone)
        self._canvas.update(); self._canvas.data_changed.emit()


class LinkBlocksCmd(QUndoCommand):
    def __init__(self, a: ArrangementBlock, b: ArrangementBlock, canvas):
        super().__init__("Lier blocs")
        self._a = a; self._b = b; self._canvas = canvas
        self._old_a = a.linked_to; self._old_b = b.linked_to

    def redo(self):
        self._a.linked_to = self._b.block_id
        self._b.linked_to = self._a.block_id
        self._canvas.update(); self._canvas.data_changed.emit()

    def undo(self):
        self._a.linked_to = self._old_a
        self._b.linked_to = self._old_b
        self._canvas.update(); self._canvas.data_changed.emit()


# ──────────────────────────────────────────────────────────────────────────────
# ArrangementCanvas — dessin principal
# ──────────────────────────────────────────────────────────────────────────────

class ArrangementCanvas(QWidget):
    """Widget de dessin de l'arrangement (règle + BPM + cues + blocs)."""

    time_changed   = pyqtSignal(float)
    data_changed   = pyqtSignal()
    cue_jumped     = pyqtSignal(object)   # CuePoint
    block_activated= pyqtSignal(str)      # scene_name
    zoom_changed   = pyqtSignal(float)

    def __init__(self, arrangement: Arrangement,
                 undo_stack: QUndoStack, parent=None):
        super().__init__(parent)
        self._arr        = arrangement
        self._undo       = undo_stack
        self._pps        = DEFAULT_PPS        # pixels per second
        self._scroll_off = 0.0                # offset horizontal (secondes)
        self._current_t  = 0.0               # playhead

        # Interaction state
        self._drag_block:        Optional[ArrangementBlock]  = None
        self._drag_track:        Optional[ArrangementTrack]  = None
        self._drag_block_orig_s: float = 0.0
        self._drag_start_x:      int   = 0
        self._drag_start_y:      int   = 0

        self._resize_block:      Optional[ArrangementBlock]  = None
        self._resize_track:      Optional[ArrangementTrack]  = None
        self._resize_side:       str   = "right"   # "right" | "left"
        self._resize_orig_s:     float = 0.0
        self._resize_orig_dur:   float = 0.0
        self._resize_start_x:    int   = 0

        self._selected_blocks:   set   = set()   # set of ArrangementBlock
        self._selected_cue:      Optional[CuePoint] = None
        self._drag_cue:          Optional[CuePoint] = None
        self._drag_cue_orig_t:   float = 0.0
        self._drag_cue_x0:       int   = 0

        self._drag_tempo:        Optional[TempoPoint] = None
        self._drag_tempo_orig_b: float = 0.0
        self._drag_tempo_x0:     int   = 0
        self._drag_tempo_y0:     int   = 0

        self._rubber_start: Optional[QPoint] = None
        self._rubber_rect:  Optional[QRect]  = None

        self._link_mode:    bool  = False   # En attente du 2e bloc à lier
        self._link_first:   Optional[ArrangementBlock] = None
        self._link_first_tr:Optional[ArrangementTrack] = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    # ── Géométrie ─────────────────────────────────────────────────────────

    def _header_h(self) -> int:
        return RULER_H + TEMPO_H + CUE_BAND_H

    def _track_y(self, idx: int) -> int:
        y = self._header_h()
        for i, tr in enumerate(self._arr.tracks):
            if i == idx:
                return y
            y += tr.height
        return y

    def _track_idx_at_y(self, y: int) -> int:
        cy = self._header_h()
        for i, tr in enumerate(self._arr.tracks):
            if cy <= y < cy + tr.height:
                return i
            cy += tr.height
        return -1

    def _t_to_x(self, t: float) -> int:
        return LABEL_W + int((t - self._scroll_off) * self._pps)

    def _x_to_t(self, x: int) -> float:
        return (x - LABEL_W) / self._pps + self._scroll_off

    def _in_ruler(self, y: int) -> bool:
        return 0 <= y < RULER_H

    def _in_tempo_band(self, y: int) -> bool:
        return RULER_H <= y < RULER_H + TEMPO_H

    def _in_cue_band(self, y: int) -> bool:
        return RULER_H + TEMPO_H <= y < self._header_h()

    def _snap(self, t: float) -> float:
        """Snap à la grille musicale (mesure ou beat)."""
        bpm = self._arr.tempo_map.bpm_at(t)
        beat_dur = 60.0 / max(1.0, bpm)
        measure_dur = beat_dur * 4
        # Cherche le plus proche entre beat et mesure
        snapped_beat    = round(t / beat_dur) * beat_dur
        snapped_measure = round(t / measure_dur) * measure_dur
        candidates = [snapped_beat, snapped_measure, t]
        return min(candidates, key=lambda v: abs(v - t))

    # ── Peinture ─────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, C_BG)

        self._paint_ruler(p, w)
        self._paint_tempo_band(p, w)
        self._paint_cue_band(p, w)
        self._paint_tracks(p, w)
        self._paint_linked_connectors(p)
        self._paint_rubber_band(p)
        self._paint_playhead(p, h)

        p.end()

    def _paint_ruler(self, p: QPainter, w: int):
        p.fillRect(0, 0, w, RULER_H, C_RULER)
        p.setPen(QPen(QColor(30, 35, 48), 1))
        p.drawLine(0, RULER_H - 1, w, RULER_H - 1)

        tm   = self._arr.tempo_map
        t    = max(0.0, self._scroll_off)
        t_end = self._x_to_t(w)

        # On affiche mesures + beats + secondes selon le zoom
        bpm     = tm.bpm_at(t)
        beat_s  = 60.0 / max(1.0, bpm)
        meas_s  = beat_s * 4

        # Choisir la résolution selon le zoom
        if self._pps * meas_s > 50:    tick_dt = meas_s / 4   # beats
        elif self._pps * meas_s > 20:  tick_dt = meas_s       # mesures
        else:                          tick_dt = meas_s * 4   # 4 mesures

        first_tick = math.floor(t / tick_dt) * tick_dt
        tick = first_tick
        p.setFont(QFont("Segoe UI", 7))

        while tick <= t_end + tick_dt:
            x         = self._t_to_x(tick)
            is_measure = abs(tick / meas_s - round(tick / meas_s)) < 0.01
            is_4bar    = abs(tick / (meas_s * 4) - round(tick / (meas_s * 4))) < 0.01

            if LABEL_W <= x <= w:
                tick_h = 10 if is_measure else 5
                p.setPen(QPen(C_RULER_TEXT if is_measure else QColor(60, 65, 80), 1))
                p.drawLine(x, RULER_H - tick_h, x, RULER_H)

                if is_4bar or (is_measure and self._pps * meas_s > 30):
                    beat_num = round(self._arr.tempo_map.beat_at(tick))
                    meas_num = beat_num // 4 + 1
                    label    = f"{meas_num}"
                    p.setPen(C_RULER_TEXT)
                    p.drawText(x + 3, RULER_H - 11, label)

            tick += tick_dt

        # Secondes en petits chiffres gris si on peut
        if self._pps >= 30:
            ts = math.floor(t)
            while ts <= t_end:
                x = self._t_to_x(ts)
                if LABEL_W <= x <= w:
                    p.setPen(QColor(70, 75, 95))
                    p.setFont(QFont("Segoe UI", 6))
                    p.drawText(x + 1, RULER_H - 2, f"{ts}s")
                ts += 1

    def _paint_tempo_band(self, p: QPainter, w: int):
        y0 = RULER_H
        p.fillRect(0, y0, w, TEMPO_H, C_TEMPO_BG)
        p.setPen(QPen(QColor(25, 28, 40), 1))
        p.drawLine(0, y0 + TEMPO_H - 1, w, y0 + TEMPO_H - 1)

        # Label
        p.fillRect(0, y0, LABEL_W, TEMPO_H, C_TRACK_HDR)
        p.setPen(QColor(80, 90, 110))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(QRect(4, y0, LABEL_W - 4, TEMPO_H),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   "⏱ BPM")

        pts = self._arr.tempo_map.points
        if not pts:
            return

        # Trouve les bornes BPM visibles
        all_bpms = [p_.bpm for p_ in pts]
        bpm_min  = max(20.0, min(all_bpms) - 10)
        bpm_max  = min(400.0, max(all_bpms) + 10)
        bpm_rng  = max(1.0, bpm_max - bpm_min)
        cy0      = y0 + 2
        ch       = TEMPO_H - 4

        def _bpm_y(bpm: float) -> int:
            norm = (bpm - bpm_min) / bpm_rng
            return cy0 + int((1.0 - norm) * ch)

        # Courbe interpolée
        t_start = max(0.0, self._scroll_off)
        t_end   = self._x_to_t(w)
        prev_pt = None

        p.setPen(QPen(C_TEMPO_CURVE, 1))
        x = LABEL_W
        while x <= w:
            t  = self._x_to_t(x)
            bv = self._arr.tempo_map.bpm_at(t)
            pt = QPoint(x, _bpm_y(bv))
            if prev_pt:
                p.drawLine(prev_pt, pt)
            prev_pt = pt
            x += 2

        # Points d'automation (cercles déplaçables)
        for tp in pts:
            tx = self._t_to_x(tp.time)
            if LABEL_W <= tx <= w:
                ty     = _bpm_y(tp.bpm)
                is_sel = (tp is self._drag_tempo)
                col    = C_TEMPO_POINT.lighter(140) if is_sel else C_TEMPO_POINT
                p.setPen(QPen(col.darker(130), 1))
                p.setBrush(QBrush(col))
                p.drawEllipse(QPoint(tx, ty), 5 if is_sel else 4, 5 if is_sel else 4)
                p.setPen(QColor(200, 230, 210))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(tx + 7, ty + 4, f"{tp.bpm:.0f}")

    def _paint_cue_band(self, p: QPainter, w: int):
        y0 = RULER_H + TEMPO_H
        p.fillRect(0, y0, w, CUE_BAND_H, C_CUE_BG)
        p.setPen(QPen(QColor(25, 28, 40), 1))
        p.drawLine(0, y0 + CUE_BAND_H - 1, w, y0 + CUE_BAND_H - 1)

        p.fillRect(0, y0, LABEL_W, CUE_BAND_H, C_TRACK_HDR)
        p.setPen(QColor(100, 80, 30))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(QRect(4, y0, LABEL_W - 4, CUE_BAND_H),
                   Qt.AlignmentFlag.AlignVCenter, "🔖 Cues")

        for cue in self._arr.cue_track.cues:
            cx = self._t_to_x(cue.time)
            if not (LABEL_W <= cx <= w):
                continue
            is_sel = (cue is self._selected_cue)
            col    = QColor(cue.color)

            # Ligne verticale
            p.setPen(QPen(col, 2 if is_sel else 1))
            p.drawLine(cx, y0, cx, y0 + CUE_BAND_H)

            # Drapeau rempli
            flag_w, flag_h = 8, CUE_BAND_H - 2
            pts = [
                QPoint(cx,          y0 + 1),
                QPoint(cx + flag_w, y0 + 1),
                QPoint(cx + flag_w, y0 + flag_h - 4),
                QPoint(cx,          y0 + flag_h),
            ]
            fill = QColor(col); fill.setAlpha(210 if is_sel else 170)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(fill))
            p.drawPolygon(*pts)

            # Label + hotkey
            label = cue.label
            if cue.hotkey:
                label = f"[{cue.hotkey}] {label}"
            p.setPen(col.lighter(170))
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            p.setClipRect(QRect(cx + flag_w + 2, y0, w - cx - flag_w - 2, CUE_BAND_H))
            p.drawText(cx + flag_w + 3, y0 + CUE_BAND_H - 5, label)
            p.setClipping(False)

    def _paint_tracks(self, p: QPainter, w: int):
        for idx, tr in enumerate(self._arr.tracks):
            y  = self._track_y(idx)
            th = tr.height

            # Fond de piste
            p.fillRect(LABEL_W, y, w - LABEL_W, th, QColor(tr.color))

            # Grille musicale
            self._paint_musical_grid(p, y, th, w)

            # Label
            p.fillRect(0, y, LABEL_W, th, C_TRACK_HDR)
            p.setPen(QColor(200, 205, 215))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.drawText(QRect(6, y, LABEL_W - 30, th),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       tr.name)

            # Indicateurs mute/solo
            if tr.muted:
                p.setPen(QColor(220, 80, 80))
                p.setFont(QFont("Segoe UI", 8))
                p.drawText(QRect(LABEL_W - 28, y, 24, th),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter,
                           "M")
            if tr.solo:
                p.setPen(QColor(220, 180, 60))
                p.setFont(QFont("Segoe UI", 8))
                p.drawText(QRect(LABEL_W - 14, y, 12, th),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter,
                           "S")

            # Séparateur bas
            p.setPen(QPen(QColor(20, 22, 32), 1))
            p.drawLine(0, y + th - 1, w, y + th - 1)

            # Blocs
            for block in tr.blocks:
                self._paint_block(p, block, tr, y, th)

    def _paint_musical_grid(self, p: QPainter, y: int, h: int, w: int):
        """Grille musicale : lignes de mesures et de beats."""
        tm      = self._arr.tempo_map
        t_start = max(0.0, self._scroll_off)
        t_end   = self._x_to_t(w)
        bpm     = tm.bpm_at(t_start)
        beat_s  = 60.0 / max(1.0, bpm)
        meas_s  = beat_s * 4

        # Seuil : afficher les beats seulement si suffisamment zoomé
        show_beats = self._pps * beat_s > 12

        t = math.floor(t_start / beat_s) * beat_s
        while t <= t_end + beat_s:
            x         = self._t_to_x(t)
            is_measure = abs(t % meas_s) < beat_s * 0.05

            if LABEL_W <= x <= w:
                if is_measure:
                    p.setPen(QPen(C_GRID_MEASURE, 1))
                    p.drawLine(x, y, x, y + h)
                elif show_beats:
                    p.setPen(QPen(C_GRID_BEAT, 1))
                    p.drawLine(x, y, x, y + h)

            t += beat_s

    def _paint_block(self, p: QPainter, block: ArrangementBlock,
                     track: ArrangementTrack, track_y: int, th: int):
        x0 = self._t_to_x(block.start)
        x1 = self._t_to_x(block.end)
        bw = max(MIN_BLOCK_W, x1 - x0)

        clip_rect = QRect(x0, track_y + 2, bw, th - 4)
        is_sel    = block in self._selected_blocks

        # Couleur de base
        base = QColor(block.color) if block.color else _color_for_name(block.scene_name)
        if block.muted:
            base = base.darker(180)
        fill = QColor(base); fill.setAlpha(200 if is_sel else 160)

        # Corps
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(clip_rect, BLOCK_RADIUS, BLOCK_RADIUS)

        # Barre de titre
        title_h  = 10
        title_r  = QRect(x0, track_y + 2, bw, title_h)
        title_c  = QColor(base.lighter(160)); title_c.setAlpha(220)
        p.setBrush(QBrush(title_c))
        p.drawRoundedRect(title_r, BLOCK_RADIUS, BLOCK_RADIUS)
        p.drawRect(QRect(x0, track_y + 2 + title_h // 2, bw, title_h // 2 + 1))

        # Liseré gauche
        lisere = QColor(base.lighter(200)); lisere.setAlpha(240)
        p.setBrush(QBrush(lisere))
        p.drawRect(QRect(x0, track_y + 2, 3, th - 4))

        # Contour (sélection ou normal)
        if is_sel:
            p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        elif block.linked_to:
            p.setPen(QPen(QColor(200, 220, 255, 160), 1, Qt.PenStyle.DashLine))
        else:
            p.setPen(QPen(base.darker(140), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(clip_rect, BLOCK_RADIUS, BLOCK_RADIUS)

        # Nom de la scène
        if bw > 20:
            p.setClipRect(clip_rect.adjusted(5, 0, -5, 0))
            name  = block.scene_name or "—"
            label = f"🔇 {name}" if block.muted else name
            p.setPen(QColor(255, 255, 255, 230))
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            p.drawText(clip_rect.adjusted(5, 2, -5, 0),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       label)
            # Durée si assez large
            if bw > 80 and th >= 36:
                p.setPen(QColor(255, 255, 255, 130))
                p.setFont(QFont("Segoe UI", 6))
                p.drawText(clip_rect.adjusted(5, -2, -5, -2),
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                           f"{block.duration:.2f}s")
            p.setClipping(False)

        # Poignées de resize
        if bw > 16:
            hc = QColor(255, 255, 255, 70)
            p.setBrush(QBrush(hc)); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRect(x1 - 6, track_y + 4, 5, th - 8), 2, 2)
            p.drawRoundedRect(QRect(x0 + 1, track_y + 4, 5, th - 8), 2, 2)

    def _paint_linked_connectors(self, p: QPainter):
        """Dessine des lignes courbes reliant les blocs liés."""
        all_blocks: dict[str, tuple[ArrangementBlock, int]] = {}
        for idx, tr in enumerate(self._arr.tracks):
            for b in tr.blocks:
                all_blocks[b.block_id] = (b, idx)

        drawn = set()
        for bid, (b, tidx) in all_blocks.items():
            if not b.linked_to or b.linked_to not in all_blocks:
                continue
            pair_key = tuple(sorted([bid, b.linked_to]))
            if pair_key in drawn:
                continue
            drawn.add(pair_key)

            b2, tidx2 = all_blocks[b.linked_to]
            x1 = self._t_to_x(b.start + b.duration * 0.5)
            x2 = self._t_to_x(b2.start + b2.duration * 0.5)
            y1 = self._track_y(tidx) + self._arr.tracks[tidx].height // 2
            y2 = self._track_y(tidx2) + self._arr.tracks[tidx2].height // 2

            p.setPen(QPen(QColor(180, 200, 255, 100), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            # Courbe de Bézier simple via drawLine par points
            steps = 20
            prev  = QPoint(x1, y1)
            for i in range(1, steps + 1):
                alpha = i / steps
                bx = int(x1 + (x2 - x1) * alpha)
                by = int(y1 + (y2 - y1) * alpha
                         + math.sin(alpha * math.pi) * -20)
                cur = QPoint(bx, by)
                p.drawLine(prev, cur)
                prev = cur

    def _paint_rubber_band(self, p: QPainter):
        if self._rubber_rect and self._rubber_rect.width() > 4:
            p.setPen(QPen(QColor(80, 140, 255, 180), 1, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(80, 140, 255, 30)))
            p.drawRect(self._rubber_rect)

    def _paint_playhead(self, p: QPainter, h: int):
        x = self._t_to_x(self._current_t)
        if LABEL_W <= x <= self.width():
            p.setPen(QPen(C_PLAYHEAD, 2))
            p.drawLine(x, 0, x, h)
            pts = [QPoint(x - 5, 0), QPoint(x + 5, 0), QPoint(x, 10)]
            p.setBrush(QBrush(C_PLAYHEAD))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(*pts)

    # ── Interactions souris ───────────────────────────────────────────────────

    def mousePressEvent(self, e: QMouseEvent):
        self.setFocus()
        x = int(e.position().x())
        y = int(e.position().y())
        t = max(0.0, self._x_to_t(x))

        # ── Règle → seek ──────────────────────────────────────────────────
        if self._in_ruler(y) and x >= LABEL_W:
            self._current_t = t
            self.time_changed.emit(t)
            self.update(); return

        # ── Bande BPM → drag point d'automation ──────────────────────────
        if self._in_tempo_band(y) and x >= LABEL_W:
            tp = self._find_tempo_point(x, y)
            if tp:
                self._drag_tempo        = tp
                self._drag_tempo_orig_b = tp.bpm
                self._drag_tempo_x0     = x
                self._drag_tempo_y0     = y
            elif e.button() == Qt.MouseButton.LeftButton:
                # Double-clic ou Shift = ajouter un point
                bpm = self._arr.tempo_map.bpm_at(t)
                self._arr.tempo_map.set_point(t, bpm)
                self.update(); self.data_changed.emit()
            return

        # ── Bande Cue → drag cue ──────────────────────────────────────────
        if self._in_cue_band(y) and x >= LABEL_W:
            cue = self._find_cue(x)
            if cue:
                self._selected_cue   = cue
                self._drag_cue       = cue
                self._drag_cue_orig_t = cue.time
                self._drag_cue_x0    = x
            else:
                self._selected_cue = None
            self.update(); return

        # ── Pistes de blocs ───────────────────────────────────────────────
        if x < LABEL_W or y < self._header_h():
            return
        tr_idx = self._track_idx_at_y(y)
        if tr_idx < 0:
            return
        tr = self._arr.tracks[tr_idx]

        # Mode liaison : attend le 2e clic
        if self._link_mode:
            block = self._find_block(tr, x)
            if block and self._link_first and block is not self._link_first:
                self._undo.push(LinkBlocksCmd(self._link_first, block, self))
            self._link_mode  = False
            self._link_first = None
            self._link_first_tr = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        block = self._find_block(tr, x)
        if not block:
            # Clic dans le vide → begin rubber-band
            if not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._selected_blocks.clear()
            self._rubber_start = QPoint(x, y)
            self._rubber_rect  = QRect(x, y, 0, 0)
            self.update(); return

        # Detect resize zones
        x0 = self._t_to_x(block.start)
        x1 = self._t_to_x(block.end)
        if abs(x - x1) <= RESIZE_ZONE:
            self._resize_block    = block
            self._resize_track    = tr
            self._resize_side     = "right"
            self._resize_orig_s   = block.start
            self._resize_orig_dur = block.duration
            self._resize_start_x  = x
            return
        if abs(x - x0) <= RESIZE_ZONE:
            self._resize_block    = block
            self._resize_track    = tr
            self._resize_side     = "left"
            self._resize_orig_s   = block.start
            self._resize_orig_dur = block.duration
            self._resize_start_x  = x
            return

        # Drag normal
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if block in self._selected_blocks:
                self._selected_blocks.discard(block)
            else:
                self._selected_blocks.add(block)
        else:
            if block not in self._selected_blocks:
                self._selected_blocks = {block}

        self._drag_block        = block
        self._drag_track        = tr
        self._drag_block_orig_s = block.start
        self._drag_start_x      = x
        self._drag_start_y      = y
        self.block_activated.emit(block.scene_name)
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent):
        x = int(e.position().x())
        y = int(e.position().y())

        # Drag tempo point
        if self._drag_tempo:
            bpm_all = [p.bpm for p in self._arr.tempo_map.points]
            bpm_min = max(20.0, min(bpm_all) - 30)
            bpm_max = min(400.0, max(bpm_all) + 30)
            bpm_rng = max(1.0, bpm_max - bpm_min)
            dy_frac = (self._drag_tempo_y0 - y) / max(1, TEMPO_H - 4)
            self._drag_tempo.bpm = max(20.0, min(400.0,
                self._drag_tempo_orig_b + dy_frac * bpm_rng))
            # Drag horizontal → temps
            dt = (x - self._drag_tempo_x0) / self._pps
            if self._arr.tempo_map.points.index(self._drag_tempo) > 0:
                self._drag_tempo.time = max(0.0, self._drag_tempo.time + dt)
                self._drag_tempo_x0 = x
            self.update(); return

        # Drag cue
        if self._drag_cue:
            dt = (x - self._drag_cue_x0) / self._pps
            self._drag_cue.time = max(0.0, self._drag_cue_orig_t + dt)
            self.update(); return

        # Resize bloc
        if self._resize_block:
            block = self._resize_block
            dt    = (x - self._resize_start_x) / self._pps
            if self._resize_side == "right":
                new_dur = max(0.1, self._resize_orig_dur + dt)
                block.duration = new_dur
            else:  # left
                new_start = max(0.0,
                    self._resize_orig_s + dt)
                new_dur   = max(0.1,
                    self._resize_orig_dur - (new_start - self._resize_orig_s))
                block.start    = new_start
                block.duration = new_dur
            self.update(); return

        # Drag bloc
        if self._drag_block:
            dt    = (x - self._drag_start_x) / self._pps
            new_s = max(0.0, self._drag_block_orig_s + dt)
            new_s = self._snap(new_s)
            delta = new_s - self._drag_block.start
            # Déplace tous les blocs sélectionnés
            for b in self._selected_blocks:
                b.start = max(0.0, b.start + delta)
            self._drag_block.start = new_s
            self._drag_start_x     = x
            self._drag_block_orig_s = new_s

            # Déplacement inter-piste
            dst_idx = self._track_idx_at_y(y)
            if dst_idx >= 0 and self._drag_track is not None:
                dst_tr = self._arr.tracks[dst_idx]
                if dst_tr is not self._drag_track:
                    self._drag_track.remove_block(self._drag_block)
                    dst_tr.blocks.append(self._drag_block)
                    dst_tr.blocks.sort(key=lambda b: b.start)
                    self._drag_track = dst_tr

            self.update(); return

        # Rubber-band
        if self._rubber_start:
            rx = min(self._rubber_start.x(), x)
            ry = min(self._rubber_start.y(), y)
            rw = abs(x - self._rubber_start.x())
            rh = abs(y - self._rubber_start.y())
            self._rubber_rect = QRect(rx, ry, rw, rh)
            self.update(); return

        # Curseur de resize
        if y >= self._header_h() and x >= LABEL_W:
            ti = self._track_idx_at_y(y)
            if ti >= 0:
                tr = self._arr.tracks[ti]
                on_edge = any(
                    abs(x - self._t_to_x(b.start)) <= RESIZE_ZONE or
                    abs(x - self._t_to_x(b.end))   <= RESIZE_ZONE
                    for b in tr.blocks)
                self.setCursor(Qt.CursorShape.SizeHorCursor if on_edge
                               else Qt.CursorShape.ArrowCursor)
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e: QMouseEvent):
        # Fin drag tempo
        if self._drag_tempo:
            self._drag_tempo = None
            self.data_changed.emit()
            self.update(); return

        # Fin drag cue
        if self._drag_cue:
            self._drag_cue = None
            self.data_changed.emit()
            self.update(); return

        # Fin resize
        if self._resize_block:
            b = self._resize_block
            if (abs(b.start - self._resize_orig_s) > 0.001 or
                    abs(b.duration - self._resize_orig_dur) > 0.001):
                self._undo.push(ResizeBlockCmd(
                    b, self._resize_orig_s, self._resize_orig_dur,
                    b.start, b.duration, self))
            self._resize_block = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update(); return

        # Fin drag bloc
        if self._drag_block:
            b = self._drag_block
            if abs(b.start - self._drag_block_orig_s) > 0.001:
                self._undo.push(MoveBlockCmd(
                    b, self._drag_block_orig_s, b.start, self))
            self._drag_block  = None
            self._drag_track  = None
            self.update(); return

        # Fin rubber-band
        if self._rubber_rect and self._rubber_rect.width() > 4:
            self._select_blocks_in_rect(self._rubber_rect)
        self._rubber_start = None
        self._rubber_rect  = None
        self.update()

    def mouseDoubleClickEvent(self, e: QMouseEvent):
        x = int(e.position().x())
        y = int(e.position().y())
        t = max(0.0, self._x_to_t(x))

        if self._in_cue_band(y) and x >= LABEL_W:
            # Double-clic dans la bande cue → ajouter
            label, ok = QInputDialog.getText(
                self, "Nouveau cue", "Nom du cue :", text=f"Cue {len(self._arr.cue_track.cues)+1}")
            if ok:
                self._arr.cue_track.add(t, label.strip() or "Cue")
                self.data_changed.emit()
                self.update()

    def wheelEvent(self, e: QWheelEvent):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.12 if e.angleDelta().y() > 0 else 0.88
            mx     = int(e.position().x())
            t_m    = self._x_to_t(mx)
            self._pps = max(3.0, min(2000.0, self._pps * factor))
            self._scroll_off = t_m - (mx - LABEL_W) / self._pps
            self.zoom_changed.emit(self._pps)
            self.update(); e.accept()
        else:
            super().wheelEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        key_str = e.text()

        # Hotkeys numériques → sauter au cue
        if key_str in "123456789":
            cue = self._arr.cue_track.at_hotkey(key_str)
            if cue:
                self._current_t = cue.time
                self.time_changed.emit(cue.time)
                self.cue_jumped.emit(cue)
                self.update()
                e.accept(); return

        if e.key() == Qt.Key.Key_Delete:
            for b in list(self._selected_blocks):
                for tr in self._arr.tracks:
                    if b in tr.blocks:
                        self._undo.push(DeleteBlockCmd(tr, b, self))
                        break
            self._selected_blocks.clear()
            e.accept(); return

        if e.key() == Qt.Key.Key_D and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+D → dupliquer
            for b in list(self._selected_blocks):
                for tr in self._arr.tracks:
                    if b in tr.blocks:
                        self._undo.push(DuplicateBlockCmd(tr, b, self))
                        break
            e.accept(); return

        if e.key() == Qt.Key.Key_S:
            # S → split au playhead
            t = self._current_t
            for tr in self._arr.tracks:
                b = tr.block_at(t)
                if b and b.start < t < b.end:
                    self._undo.push(SplitBlockCmd(tr, b, t, self))
            e.accept(); return

        if (e.key() == Qt.Key.Key_Left
                and e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            cue = self._arr.cue_track.prev(self._current_t)
            if cue:
                self._current_t = cue.time
                self.time_changed.emit(cue.time)
                self.cue_jumped.emit(cue)
                self.update()
            e.accept(); return

        if (e.key() == Qt.Key.Key_Right
                and e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            cue = self._arr.cue_track.next(self._current_t)
            if cue:
                self._current_t = cue.time
                self.time_changed.emit(cue.time)
                self.cue_jumped.emit(cue)
                self.update()
            e.accept(); return

        if e.key() == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            for tr in self._arr.tracks:
                for b in tr.blocks:
                    self._selected_blocks.add(b)
            self.update(); e.accept(); return

        if e.key() == Qt.Key.Key_Escape:
            self._selected_blocks.clear()
            self._link_mode = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update(); e.accept(); return

        super().keyPressEvent(e)

    def contextMenuEvent(self, e: QContextMenuEvent):
        x = int(e.pos().x())
        y = int(e.pos().y())
        t = max(0.0, self._x_to_t(x))

        # ── Bande BPM ─────────────────────────────────────────────────────
        if self._in_tempo_band(y) and x >= LABEL_W:
            tp  = self._find_tempo_point(x, y)
            menu = QMenu(self); menu.setStyleSheet(_MENU_STYLE)
            if tp:
                menu.addAction(f"⏱ BPM ici: {tp.bpm:.1f}").setEnabled(False)
                menu.addSeparator()
                for ease_k, ease_lbl in EASING_LABELS.items():
                    act = menu.addAction(f"  {ease_lbl}")
                    act.setCheckable(True)
                    act.setChecked(tp.easing == ease_k)
                    act.triggered.connect(
                        lambda _, k=ease_k, pt=tp: (setattr(pt, 'easing', k),
                                                    self.data_changed.emit(),
                                                    self.update()))
                menu.addSeparator()
                if len(self._arr.tempo_map.points) > 1:
                    menu.addAction("🗑 Supprimer ce point").triggered.connect(
                        lambda: (self._arr.tempo_map.remove_point(tp),
                                 self.data_changed.emit(), self.update()))
            else:
                menu.addAction(f"➕ Ajouter point BPM à {t:.2f}s").triggered.connect(
                    lambda: self._add_tempo_point(t))
            menu.exec(e.globalPos()); return

        # ── Bande Cue ─────────────────────────────────────────────────────
        if self._in_cue_band(y) and x >= LABEL_W:
            cue  = self._find_cue(x)
            menu = QMenu(self); menu.setStyleSheet(_MENU_STYLE)
            if cue:
                menu.addAction(f"✏ Renommer « {cue.label} »").triggered.connect(
                    lambda: self._rename_cue(cue))
                menu.addAction("🎹 Assigner raccourci clavier (1-9)").triggered.connect(
                    lambda: self._assign_cue_hotkey(cue))
                menu.addAction("🎨 Changer la couleur").triggered.connect(
                    lambda: self._change_cue_color(cue))
                menu.addSeparator()
                menu.addAction("🗑 Supprimer ce cue").triggered.connect(
                    lambda: (self._arr.cue_track.remove(cue),
                             self.data_changed.emit(), self.update()))
            else:
                menu.addAction(f"🔖 Ajouter un cue à {t:.2f}s").triggered.connect(
                    lambda: self._add_cue_at(t))
            menu.exec(e.globalPos()); return

        # ── Pistes de blocs ───────────────────────────────────────────────
        if x < LABEL_W:
            tr_idx = self._track_idx_at_y(y)
            if tr_idx < 0:
                return
            tr   = self._arr.tracks[tr_idx]
            menu = QMenu(self); menu.setStyleSheet(_MENU_STYLE)
            menu.addAction("✏ Renommer la piste").triggered.connect(
                lambda: self._rename_track(tr))
            menu.addAction("🎨 Couleur de la piste").triggered.connect(
                lambda: self._change_track_color(tr))
            menu.addSeparator()
            mute_lbl = "🔈 Activer" if tr.muted else "🔇 Rendre muet"
            menu.addAction(mute_lbl).triggered.connect(
                lambda: (setattr(tr, 'muted', not tr.muted),
                         self.update(), self.data_changed.emit()))
            solo_lbl = "★ Désactiver Solo" if tr.solo else "★ Solo"
            menu.addAction(solo_lbl).triggered.connect(
                lambda: self._toggle_solo(tr))
            menu.addSeparator()
            menu.addAction("🗑 Supprimer la piste").triggered.connect(
                lambda: (self._arr.remove_track(tr),
                         self.update(), self.data_changed.emit()))
            menu.exec(e.globalPos()); return

        tr_idx = self._track_idx_at_y(y)
        if tr_idx < 0:
            return
        tr    = self._arr.tracks[tr_idx]
        block = self._find_block(tr, x)
        menu  = QMenu(self); menu.setStyleSheet(_MENU_STYLE)

        if block:
            menu.addAction(f"🎬 Scène : {block.scene_name or '—'}").setEnabled(False)
            menu.addSeparator()
            menu.addAction("✏ Renommer / choisir scène").triggered.connect(
                lambda: self._rename_block(tr, block))
            menu.addAction("✂ Couper ici").triggered.connect(
                lambda: self._undo.push(SplitBlockCmd(tr, block, t, self)))
            menu.addAction("📋 Dupliquer").triggered.connect(
                lambda: self._undo.push(DuplicateBlockCmd(tr, block, self)))
            menu.addSeparator()
            if block.linked_to:
                menu.addAction("🔗 Détacher (enlever liaison)").triggered.connect(
                    lambda: (setattr(block, 'linked_to', None),
                             self.update(), self.data_changed.emit()))
            else:
                menu.addAction("🔗 Lier à un autre bloc…").triggered.connect(
                    lambda: self._start_link(block, tr))
            menu.addSeparator()
            mute_lbl = "🔈 Activer le bloc" if block.muted else "🔇 Rendre muet"
            menu.addAction(mute_lbl).triggered.connect(
                lambda: (setattr(block, 'muted', not block.muted),
                         self.update(), self.data_changed.emit()))
            menu.addSeparator()
            menu.addAction("🗑 Supprimer").triggered.connect(
                lambda: self._undo.push(DeleteBlockCmd(tr, block, self)))
        else:
            # Scènes disponibles depuis le SceneGraph (passé par ref depuis la vue parente)
            scene_names = getattr(self, '_available_scenes', [])
            if scene_names:
                add_menu = menu.addMenu("➕ Ajouter bloc — scène…")
                for sn in scene_names:
                    add_menu.addAction(sn).triggered.connect(
                        lambda _, n=sn: self._undo.push(
                            AddBlockCmd(tr, ArrangementBlock(
                                scene_name=n, start=self._snap(t),
                                duration=4.0), self)))
            else:
                menu.addAction(f"➕ Nouveau bloc à {t:.2f}s").triggered.connect(
                    lambda: self._add_block_dialog(tr, t))

        menu.exec(e.globalPos())

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_block(self, tr: ArrangementTrack, x: int) -> Optional[ArrangementBlock]:
        for b in tr.blocks:
            x0 = self._t_to_x(b.start)
            x1 = self._t_to_x(b.end)
            if x0 <= x <= x1:
                return b
        return None

    def _find_cue(self, x: int, tol: int = 8) -> Optional[CuePoint]:
        best, best_d = None, tol + 1
        for c in self._arr.cue_track.cues:
            d = abs(self._t_to_x(c.time) - x)
            if d < best_d:
                best, best_d = c, d
        return best if best_d <= tol else None

    def _find_tempo_point(self, x: int, y: int, tol: int = 8) -> Optional[TempoPoint]:
        pts      = self._arr.tempo_map.points
        all_bpms = [p.bpm for p in pts]
        bpm_min  = max(20.0, min(all_bpms) - 10)
        bpm_max  = min(400.0, max(all_bpms) + 10)
        bpm_rng  = max(1.0, bpm_max - bpm_min)
        cy0      = RULER_H + 2
        ch       = TEMPO_H - 4

        def _bpm_y(bpm: float) -> int:
            norm = (bpm - bpm_min) / bpm_rng
            return cy0 + int((1.0 - norm) * ch)

        best, best_d = None, tol + 1
        for tp in pts:
            tx = self._t_to_x(tp.time)
            ty = _bpm_y(tp.bpm)
            d  = math.hypot(x - tx, y - ty)
            if d < best_d:
                best, best_d = tp, d
        return best if best_d <= tol else None

    def _select_blocks_in_rect(self, rect: QRect):
        for tr in self._arr.tracks:
            for b in tr.blocks:
                cx = self._t_to_x(b.start + b.duration * 0.5)
                ti = self._arr.tracks.index(tr)
                cy = self._track_y(ti) + tr.height // 2
                if rect.contains(cx, cy):
                    self._selected_blocks.add(b)

    def _add_tempo_point(self, t: float):
        dlg = QDialog(self); dlg.setWindowTitle("Ajouter point BPM")
        dlg.setStyleSheet("background:#1c1e24;color:#d0d3de;"); dlg.setFixedWidth(280)
        form = QFormLayout(dlg); form.setContentsMargins(16, 14, 16, 12)
        spin = QDoubleSpinBox(); spin.setRange(20.0, 400.0); spin.setValue(120.0)
        spin.setDecimals(1); spin.setSuffix(" BPM")
        spin.setStyleSheet(_SPIN_STYLE)
        ease_cb = QComboBox()
        for k, v in EASING_LABELS.items():
            ease_cb.addItem(v, k)
        ease_cb.setStyleSheet(_COMBO_STYLE)
        form.addRow("BPM :", spin); form.addRow("Transition :", ease_cb)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet(_BTN_STYLE); btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject); form.addRow(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._arr.tempo_map.set_point(t, spin.value(), ease_cb.currentData())
            self.data_changed.emit(); self.update()

    def _add_cue_at(self, t: float):
        label, ok = QInputDialog.getText(
            self, "Nouveau Cue", "Nom du cue :",
            text=f"Cue {len(self._arr.cue_track.cues) + 1}")
        if ok:
            self._arr.cue_track.add(t, label.strip() or "Cue")
            self.data_changed.emit(); self.update()

    def _rename_cue(self, cue: CuePoint):
        new, ok = QInputDialog.getText(
            self, "Renommer le cue", "Nom :", text=cue.label)
        if ok:
            cue.label = new.strip() or cue.label
            self.data_changed.emit(); self.update()

    def _assign_cue_hotkey(self, cue: CuePoint):
        current = cue.hotkey or ""
        key, ok = QInputDialog.getText(
            self, "Raccourci clavier",
            "Touche (1-9, vide pour aucun) :", text=current)
        if ok:
            key = key.strip()
            if key and key in "123456789":
                # Désassigner si une autre cue a ce raccourci
                for c in self._arr.cue_track.cues:
                    if c is not cue and c.hotkey == key:
                        c.hotkey = None
                cue.hotkey = key
            elif not key:
                cue.hotkey = None
            self.data_changed.emit(); self.update()

    def _change_cue_color(self, cue: CuePoint):
        col = QColorDialog.getColor(QColor(cue.color), self, "Couleur du cue")
        if col.isValid():
            cue.color = col.name()
            self.data_changed.emit(); self.update()

    def _rename_track(self, tr: ArrangementTrack):
        name, ok = QInputDialog.getText(
            self, "Renommer la piste", "Nom :", text=tr.name)
        if ok and name.strip():
            tr.name = name.strip()
            self.data_changed.emit(); self.update()

    def _change_track_color(self, tr: ArrangementTrack):
        col = QColorDialog.getColor(QColor(tr.color), self, "Couleur de la piste")
        if col.isValid():
            tr.color = col.name()
            self.update()

    def _rename_block(self, tr: ArrangementTrack, block: ArrangementBlock):
        name, ok = QInputDialog.getText(
            self, "Scène du bloc", "Nom de la scène :", text=block.scene_name)
        if ok:
            block.scene_name = name.strip()
            self.data_changed.emit(); self.update()

    def _add_block_dialog(self, tr: ArrangementTrack, t: float):
        name, ok = QInputDialog.getText(
            self, "Nouveau bloc", "Nom de la scène :")
        if ok:
            self._undo.push(AddBlockCmd(
                tr, ArrangementBlock(scene_name=name.strip(),
                                     start=self._snap(t), duration=4.0), self))

    def _start_link(self, block: ArrangementBlock, tr: ArrangementTrack):
        self._link_mode     = True
        self._link_first    = block
        self._link_first_tr = tr
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _toggle_solo(self, tr: ArrangementTrack):
        tr.solo = not tr.solo
        if tr.solo:
            for t2 in self._arr.tracks:
                if t2 is not tr:
                    t2.solo = False
        self.update(); self.data_changed.emit()

    # ── API ─────────────────────────────────────────────────────────────────

    def set_current_time(self, t: float):
        self._current_t = t; self.update()

    def set_zoom(self, pps: float):
        self._pps = max(3.0, min(2000.0, pps))
        self.update()

    def sizeHint(self) -> QSize:
        h = self._header_h() + sum(tr.height for tr in self._arr.tracks) + 20
        return QSize(800, max(200, h))


# ──────────────────────────────────────────────────────────────────────────────
# CuePanel — panneau latéral de navigation live
# ──────────────────────────────────────────────────────────────────────────────

class CuePanel(QWidget):
    """Panneau latéral listant les cues pour la navigation live au clavier."""

    cue_jumped = pyqtSignal(object)   # CuePoint

    def __init__(self, cue_track: CueTrack, parent=None):
        super().__init__(parent)
        self._ct      = cue_track
        self._buttons: list[tuple[CuePoint, QPushButton]] = []
        self._build()

    def _build(self):
        self.setStyleSheet("background:#0d0f18;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QLabel(" 🔖  Cue Points")
        header.setStyleSheet(
            "background:#0a0c14; color:#6a7090; font:bold 9px 'Segoe UI';"
            "padding:5px 8px; border-bottom:1px solid #181a28;")
        lay.addWidget(header)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet("QScrollArea{border:none;}")
        self._inner = QWidget()
        self._inner.setStyleSheet("background:#0d0f18;")
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(4, 4, 4, 4)
        self._inner_lay.setSpacing(2)
        self._scroll_area.setWidget(self._inner)
        lay.addWidget(self._scroll_area, 1)

        self._hint = QLabel("Double-cliquez sur la\nbande Cue pour en ajouter.")
        self._hint.setStyleSheet("color:#3a4060; font:8px 'Segoe UI';"
                                  "padding:10px;")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._inner_lay.addWidget(self._hint)
        self._inner_lay.addStretch()

    def refresh(self):
        """Reconstruit les boutons depuis la CueTrack."""
        # Nettoyer
        for _, btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        # Retirer stretch
        while self._inner_lay.count():
            item = self._inner_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._ct.cues:
            self._inner_lay.addWidget(self._hint)
            self._inner_lay.addStretch()
            return

        for cue in self._ct.cues:
            btn = QPushButton()
            btn.setCheckable(False)
            # Couleur de la pastille
            col = QColor(cue.color)
            hotkey_badge = f" [{cue.hotkey}]" if cue.hotkey else ""
            btn.setText(f"{cue.label}{hotkey_badge}  {cue.time:.2f}s")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #13151e;
                    color: #c8ccd8;
                    border: 1px solid #202438;
                    border-left: 3px solid {cue.color};
                    border-radius: 3px;
                    padding: 5px 8px;
                    font: 9px 'Segoe UI';
                    text-align: left;
                }}
                QPushButton:hover {{ background: #1c1e2e; color: #fff; }}
                QPushButton:pressed {{ background: #22253a; }}
            """)
            btn.clicked.connect(lambda _, c=cue: self.cue_jumped.emit(c))
            self._inner_lay.addWidget(btn)
            self._buttons.append((cue, btn))

        self._inner_lay.addStretch()

    def highlight_cue(self, cue: CuePoint):
        """Met en surbrillance le bouton du cue actif (feedback live)."""
        for c, btn in self._buttons:
            active = (c is cue)
            col    = QColor(cue.color)
            if active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {col.darker(140).name()};
                        color: #fff;
                        border: 1px solid {cue.color};
                        border-left: 3px solid {cue.color};
                        border-radius: 3px;
                        padding: 5px 8px;
                        font: bold 9px 'Segoe UI';
                        text-align: left;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: #13151e; color: #c8ccd8;
                        border: 1px solid #202438;
                        border-left: 3px solid {c.color};
                        border-radius: 3px;
                        padding: 5px 8px;
                        font: 9px 'Segoe UI';
                        text-align: left;
                    }}
                    QPushButton:hover {{ background: #1c1e2e; color: #fff; }}
                """)


# ──────────────────────────────────────────────────────────────────────────────
# ArrangementView — widget principal
# ──────────────────────────────────────────────────────────────────────────────

class ArrangementView(QWidget):
    """
    Vue d'arrangement complète avec toolbar, cue panel, canvas scrollable.
    Intégration : placé dans un QDockWidget dans main_window.
    """

    time_changed            = pyqtSignal(float)
    arrangement_data_changed= pyqtSignal()
    cue_activated           = pyqtSignal(object)    # CuePoint
    scene_block_activated   = pyqtSignal(str)       # scene_name

    def __init__(self, arrangement: Arrangement, parent=None):
        super().__init__(parent)
        self._arr        = arrangement
        self._undo       = QUndoStack(self)
        self._canvas     = ArrangementCanvas(arrangement, self._undo, self)
        self._cue_panel  = CuePanel(arrangement.cue_track, self)
        self._build_ui()
        self._connect()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────
        tb = self._build_toolbar()
        root.addWidget(tb)

        # ── Splitter : canvas + cue panel ─────────────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setStyleSheet("QSplitter::handle{background:#1a1c28; width:3px;}")

        # Canvas dans un QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        self._scroll = scroll

        split.addWidget(scroll)
        self._cue_panel.setMinimumWidth(140)
        self._cue_panel.setMaximumWidth(240)
        split.addWidget(self._cue_panel)
        split.setSizes([700, 180])

        root.addWidget(split, 1)

    def _build_toolbar(self) -> QWidget:
        tb = QWidget()
        tb.setStyleSheet("background:#0f1118; border-bottom:1px solid #1a1c28;")
        tb.setFixedHeight(32)
        lay = QHBoxLayout(tb)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(5)

        # Titre
        lbl = QLabel("ARRANGEMENT")
        lbl.setStyleSheet("color:#3a4060; font:bold 9px 'Segoe UI';")
        lay.addWidget(lbl)

        def _sep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.VLine)
            s.setStyleSheet("color:#2a2d3a; max-width:1px;")
            return s

        lay.addWidget(_sep())

        # Zoom in/out
        for lbl_txt, delta in [("−", -15), ("+", 15)]:
            b = QPushButton(lbl_txt)
            b.setFixedSize(22, 22)
            b.setStyleSheet(_BTN_SMALL)
            b.clicked.connect(
                lambda _, d=delta: self._canvas.set_zoom(
                    self._canvas._pps * (1.15 if d > 0 else 0.87)))
            lay.addWidget(b)

        btn_fit = QPushButton("⊡")
        btn_fit.setFixedSize(22, 22)
        btn_fit.setStyleSheet(_BTN_SMALL)
        btn_fit.setToolTip("Ajuster à la fenêtre")
        btn_fit.clicked.connect(self._fit_zoom)
        lay.addWidget(btn_fit)

        lay.addWidget(_sep())

        # Bouton ajouter piste
        btn_add_tr = QPushButton("＋ Piste")
        btn_add_tr.setFixedHeight(22)
        btn_add_tr.setStyleSheet(_BTN)
        btn_add_tr.clicked.connect(self._add_track)
        lay.addWidget(btn_add_tr)

        # Bouton ajouter cue
        btn_add_cue = QPushButton("🔖 Cue")
        btn_add_cue.setFixedHeight(22)
        btn_add_cue.setStyleSheet(_BTN)
        btn_add_cue.clicked.connect(self._add_cue_at_playhead)
        lay.addWidget(btn_add_cue)

        # Bouton ajouter point BPM
        btn_add_bpm = QPushButton("⏱ BPM")
        btn_add_bpm.setFixedHeight(22)
        btn_add_bpm.setStyleSheet(_BTN)
        btn_add_bpm.clicked.connect(self._add_bpm_at_playhead)
        lay.addWidget(btn_add_bpm)

        lay.addWidget(_sep())

        # Undo / Redo
        btn_undo = QPushButton("↩")
        btn_undo.setFixedSize(22, 22); btn_undo.setStyleSheet(_BTN_SMALL)
        btn_undo.setToolTip("Annuler (Ctrl+Z)")
        btn_undo.clicked.connect(self._undo.undo)
        lay.addWidget(btn_undo)

        btn_redo = QPushButton("↪")
        btn_redo.setFixedSize(22, 22); btn_redo.setStyleSheet(_BTN_SMALL)
        btn_redo.setToolTip("Rétablir (Ctrl+Y)")
        btn_redo.clicked.connect(self._undo.redo)
        lay.addWidget(btn_redo)

        lay.addStretch()

        # Durée
        lbl_dur = QLabel("Durée :")
        lbl_dur.setStyleSheet("color:#4a5070; font:9px 'Segoe UI';")
        lay.addWidget(lbl_dur)

        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(1.0, 7200.0)
        self._dur_spin.setDecimals(1)
        self._dur_spin.setSuffix(" s")
        self._dur_spin.setValue(self._arr.duration)
        self._dur_spin.setFixedWidth(72)
        self._dur_spin.setFixedHeight(22)
        self._dur_spin.setStyleSheet(_SPIN_STYLE)
        self._dur_spin.valueChanged.connect(
            lambda v: (setattr(self._arr, 'duration', v),
                       self.arrangement_data_changed.emit()))
        lay.addWidget(self._dur_spin)

        return tb

    def _connect(self):
        self._canvas.time_changed.connect(self.time_changed)
        self._canvas.time_changed.connect(self._canvas.set_current_time)
        self._canvas.data_changed.connect(self.arrangement_data_changed)
        self._canvas.data_changed.connect(self._cue_panel.refresh)
        self._canvas.cue_jumped.connect(self._on_cue_jumped)
        self._canvas.block_activated.connect(self.scene_block_activated)
        self._cue_panel.cue_jumped.connect(self._on_cue_jumped)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_cue_jumped(self, cue: CuePoint):
        self._canvas.set_current_time(cue.time)
        self.time_changed.emit(cue.time)
        self._cue_panel.highlight_cue(cue)
        self.cue_activated.emit(cue)

    def _add_track(self):
        name, ok = QInputDialog.getText(self, "Nouvelle piste", "Nom :",
                                        text=f"Piste {len(self._arr.tracks)+1}")
        if ok and name.strip():
            self._arr.add_track(name.strip())
            self._canvas.update()
            self.arrangement_data_changed.emit()

    def _add_cue_at_playhead(self):
        t    = self._canvas._current_t
        n    = len(self._arr.cue_track.cues) + 1
        label, ok = QInputDialog.getText(self, "Nouveau Cue",
                                          "Nom :", text=f"Cue {n}")
        if ok:
            self._arr.cue_track.add(t, label.strip() or f"Cue {n}")
            self._cue_panel.refresh()
            self._canvas.update()
            self.arrangement_data_changed.emit()

    def _add_bpm_at_playhead(self):
        t    = self._canvas._current_t
        bpm  = self._arr.tempo_map.bpm_at(t)
        bpm_new, ok = QInputDialog.getDouble(
            self, "Nouveau point BPM", "BPM :", bpm, 20.0, 400.0, 1)
        if ok:
            self._arr.tempo_map.set_point(t, bpm_new)
            self._canvas.update()
            self.arrangement_data_changed.emit()

    def _fit_zoom(self):
        available = self._scroll.viewport().width() - LABEL_W
        dur       = self._arr.duration
        if available > 0 and dur > 0:
            self._canvas.set_zoom(available / dur)

    # ── API publique ─────────────────────────────────────────────────────────

    def set_current_time(self, t: float):
        self._canvas.set_current_time(t)

    def set_available_scenes(self, names: list[str]):
        """Injecte les noms des scènes disponibles pour les menus contextuels."""
        self._canvas._available_scenes = names

    def get_arrangement(self) -> Arrangement:
        return self._arr

    def refresh_cue_panel(self):
        self._cue_panel.refresh()


# ──────────────────────────────────────────────────────────────────────────────
# Usine : crée le dock
# ──────────────────────────────────────────────────────────────────────────────

def create_arrangement_dock(arrangement: Arrangement,
                             parent) -> tuple:
    """
    Crée et retourne (QDockWidget, ArrangementView).
    """
    from PyQt6.QtWidgets import QDockWidget
    from PyQt6.QtCore    import Qt as _Qt

    view = ArrangementView(arrangement)
    dock = QDockWidget("🎛  Arrangement", parent)
    dock.setObjectName("DockArrangement")
    dock.setWidget(view)
    dock.setAllowedAreas(
        _Qt.DockWidgetArea.BottomDockWidgetArea |
        _Qt.DockWidgetArea.TopDockWidgetArea)
    return dock, view


# ──────────────────────────────────────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────────────────────────────────────

_BTN = """
QPushButton {
    background:#1e2030; color:#8898c8;
    border:1px solid #2a2e44; border-radius:4px;
    padding:2px 8px; font:9px 'Segoe UI';
}
QPushButton:hover { background:#242840; color:#c0ccff; }
QPushButton:pressed { background:#2a3050; }
"""

_BTN_SMALL = """
QPushButton {
    background:#1a1c28; color:#6070a0;
    border:1px solid #2a2d40; border-radius:3px;
    font:bold 12px 'Segoe UI'; padding:0;
}
QPushButton:hover { background:#242840; color:#a0b0d8; }
"""

_SPIN_STYLE = """
QDoubleSpinBox {
    background:#12141a; color:#c8ccd8;
    border:1px solid #2a2d3a; border-radius:3px;
    padding:1px 3px; font:9px 'Segoe UI';
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background:#1a1c24; border:none; width:12px;
}
"""

_COMBO_STYLE = """
QComboBox {
    background:#12141a; color:#c8ccd8;
    border:1px solid #2a2d3a; border-radius:3px;
    padding:1px 4px; font:9px 'Segoe UI';
}
QComboBox::drop-down { border:none; width:14px; }
QComboBox QAbstractItemView {
    background:#1c1e24; color:#c8ccd8;
    selection-background-color:#2f3244;
}
"""

_MENU_STYLE = """
QMenu {
    background:#1c1e24; color:#c8ccd8;
    border:1px solid #3a3d4d; border-radius:4px; padding:4px;
}
QMenu::item { padding:5px 20px; border-radius:3px; }
QMenu::item:selected { background:#2f3244; }
QMenu::separator { height:1px; background:#2a2d3a; margin:3px 8px; }
"""
