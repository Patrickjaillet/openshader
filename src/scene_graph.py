"""
scene_graph.py — Éditeur de scènes multi-shaders (Scene Graph)
================================================================
Chaque scène encapsule :
  - ses propres shaders (passe Image + Post optionnelle)
  - sa propre Timeline (keyframes, durée, BPM)
  - ses propres uniforms overrides
  - son état FX (post-processing)
  - ses métadonnées (nom, couleur, thumbnail path)

Format fichier : .osdemo  (ZIP contenant scene.json + shaders/ + audio/)

API publique
------------
  SceneGraph          — modèle de données (liste de SceneItem)
  SceneGraphDock      — QDockWidget affichant l'arbre + miniatures + contrôles
  SceneItem           — une scène (données)
  SceneTransition     — paramètres de blending inter-scènes

Signaux SceneGraphDock
-----------------------
  scene_activated(SceneItem)     — l'utilisateur double-clique / active une scène
  scene_preview_requested(int)   — miniature demandée pour la scène idx
  transition_requested(int, int) — blending de la scène A → scène B demandé
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import zipfile
import uuid
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import (Qt, QMimeData, QByteArray, QDataStream, QIODevice,
                           pyqtSignal, QSize, QTimer, QPoint)
from PyQt6.QtGui  import (QFont, QColor, QPainter, QPixmap, QIcon,
                           QDrag, QBrush, QPen, QLinearGradient)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QScrollArea, QSplitter, QDockWidget, QMenu,
    QInputDialog, QMessageBox, QFileDialog, QSlider, QDoubleSpinBox,
    QColorDialog, QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QComboBox, QFrame, QSizePolicy, QAbstractItemView, QToolButton,
    QCheckBox, QProgressBar, QApplication
)

from .logger import get_logger
from .timeline import Timeline

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

THUMB_W, THUMB_H = 128, 72          # taille des miniatures dans l'arbre
OSDEMO_VERSION   = "1.0"
BLEND_MODES      = ["fade", "dissolve", "wipe_left", "wipe_right", "flash", "none"]
BLEND_LABELS     = {
    "fade":        "Fondu (alpha)",
    "dissolve":    "Dissolution",
    "wipe_left":   "Balayage ←",
    "wipe_right":  "Balayage →",
    "flash":       "Flash blanc",
    "none":        "Coupure nette",
}

# ──────────────────────────────────────────────────────────────────────────────
# Modèle de données
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneTransition:
    """Paramètres de transition vers la scène suivante."""
    mode:     str   = "fade"     # voir BLEND_MODES
    duration: float = 1.0        # secondes
    easing:   str   = "smooth"   # "linear" | "smooth" | "ease_in" | "ease_out"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "duration": self.duration, "easing": self.easing}

    @staticmethod
    def from_dict(d: dict) -> "SceneTransition":
        t = SceneTransition()
        t.mode     = d.get("mode",     "fade")
        t.duration = float(d.get("duration", 1.0))
        t.easing   = d.get("easing",   "smooth")
        return t


@dataclass
class SceneItem:
    """Une scène dans le graphe de scènes."""
    # Identité
    scene_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:       str   = "Nouvelle scène"
    color:      str   = "#3a6ea5"       # couleur de l'étiquette dans l'arbre

    # Shaders : dict pass_name → code GLSL
    shaders:    dict  = field(default_factory=dict)

    # Timeline sérialisée (dict — pas d'objet Timeline vivant ici)
    timeline:   dict  = field(default_factory=dict)

    # Uniforms overrides : dict name → float
    uniforms:   dict  = field(default_factory=dict)

    # État FX post-processing (même format que main_window._shader_fx_states)
    fx_state:   dict  = field(default_factory=dict)

    # Transition vers la scène suivante
    transition: SceneTransition = field(default_factory=SceneTransition)

    # Chemin du fichier audio associé (optionnel)
    audio_path: Optional[str] = None

    # Miniature (QPixmap en mémoire — non sérialisée)
    thumbnail:  Optional[object] = field(default=None, compare=False, repr=False)

    # ── sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "scene_id":  self.scene_id,
            "name":      self.name,
            "color":     self.color,
            "shaders":   self.shaders,
            "timeline":  self.timeline,
            "uniforms":  self.uniforms,
            "fx_state":  self.fx_state,
            "transition": self.transition.to_dict(),
            "audio_path": self.audio_path,
        }

    @staticmethod
    def from_dict(d: dict) -> "SceneItem":
        s = SceneItem()
        s.scene_id   = d.get("scene_id",  str(uuid.uuid4())[:8])
        s.name       = d.get("name",      "Scène")
        s.color      = d.get("color",     "#3a6ea5")
        s.shaders    = d.get("shaders",   {})
        s.timeline   = d.get("timeline",  {})
        s.uniforms   = d.get("uniforms",  {})
        s.fx_state   = d.get("fx_state",  {})
        s.transition = SceneTransition.from_dict(d.get("transition", {}))
        s.audio_path = d.get("audio_path")
        return s

    def clone(self) -> "SceneItem":
        c = SceneItem.from_dict(copy.deepcopy(self.to_dict()))
        c.scene_id = str(uuid.uuid4())[:8]
        c.name     = self.name + " (copie)"
        return c


class SceneGraph:
    """Modèle liste de SceneItem + curseur actif."""

    def __init__(self):
        self.scenes:       list[SceneItem] = []
        self.active_index: int             = -1

    # ── accès ────────────────────────────────────────────────────────────────

    @property
    def active_scene(self) -> Optional[SceneItem]:
        if 0 <= self.active_index < len(self.scenes):
            return self.scenes[self.active_index]
        return None

    def append(self, scene: SceneItem):
        self.scenes.append(scene)
        if self.active_index < 0:
            self.active_index = 0

    def insert_after(self, idx: int, scene: SceneItem):
        self.scenes.insert(idx + 1, scene)

    def remove(self, idx: int):
        if 0 <= idx < len(self.scenes):
            self.scenes.pop(idx)
            self.active_index = max(0, min(self.active_index, len(self.scenes) - 1))
            if not self.scenes:
                self.active_index = -1

    def move_up(self, idx: int):
        if idx > 0:
            self.scenes[idx], self.scenes[idx-1] = self.scenes[idx-1], self.scenes[idx]
            if self.active_index == idx:
                self.active_index = idx - 1
            elif self.active_index == idx - 1:
                self.active_index = idx

    def move_down(self, idx: int):
        if idx < len(self.scenes) - 1:
            self.scenes[idx], self.scenes[idx+1] = self.scenes[idx+1], self.scenes[idx]
            if self.active_index == idx:
                self.active_index = idx + 1
            elif self.active_index == idx + 1:
                self.active_index = idx

    # ── I/O ──────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "scenes":       [s.to_dict() for s in self.scenes],
            "active_index": self.active_index,
        }

    def from_dict(self, d: dict):
        self.scenes       = [SceneItem.from_dict(s) for s in d.get("scenes", [])]
        self.active_index = int(d.get("active_index", 0))

    # ── .osdemo I/O ──────────────────────────────────────────────────────────

    def save_osdemo(self, path: str) -> bool:
        """Exporte la scène active en fichier .osdemo (ZIP)."""
        scene = self.active_scene
        if scene is None:
            return False
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                meta = {
                    "version": OSDEMO_VERSION,
                    "scene":   scene.to_dict(),
                }
                zf.writestr("scene.json", json.dumps(meta, indent=2, ensure_ascii=False))
            return True
        except (OSError, ValueError) as e:
            log.error("save_osdemo: %s", e)
            return False

    def save_scene_osdemo(self, scene: SceneItem, path: str) -> bool:
        """Exporte une scène spécifique en .osdemo."""
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                meta = {"version": OSDEMO_VERSION, "scene": scene.to_dict()}
                zf.writestr("scene.json", json.dumps(meta, indent=2, ensure_ascii=False))
            return True
        except (OSError, ValueError) as e:
            log.error("save_scene_osdemo: %s", e)
            return False

    def load_osdemo(self, path: str) -> Optional[SceneItem]:
        """Importe un .osdemo et renvoie le SceneItem (sans l'ajouter)."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("scene.json") as f:
                    meta = json.load(f)
            return SceneItem.from_dict(meta.get("scene", meta))
        except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as e:
            log.error("load_osdemo: %s", e)
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers graphiques
# ──────────────────────────────────────────────────────────────────────────────

def _placeholder_thumb(color: str = "#3a6ea5", w: int = THUMB_W, h: int = THUMB_H) -> QPixmap:
    """Miniature placeholder colorée quand aucun rendu n'est disponible."""
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, w, h)
    base = QColor(color)
    grad.setColorAt(0, base.darker(80))
    grad.setColorAt(1, base.darker(160))
    p.fillRect(0, 0, w, h, QBrush(grad))
    p.setPen(QPen(QColor(255, 255, 255, 60), 1))
    p.drawRect(0, 0, w-1, h-1)
    p.end()
    return pix


def _blend_badge(mode: str, w: int = 20, h: int = 20) -> QPixmap:
    """Petit badge visuel pour le mode de transition."""
    icons = {
        "fade":      "⇝",
        "dissolve":  "⁕",
        "wipe_left": "◀",
        "wipe_right":"▶",
        "flash":     "✦",
        "none":      "✂",
    }
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setPen(QColor("#a0a8c0"))
    f = QFont("Segoe UI", 9)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, icons.get(mode, "?"))
    p.end()
    return pix


# ──────────────────────────────────────────────────────────────────────────────
# Widget : une ligne de scène dans l'arbre
# ──────────────────────────────────────────────────────────────────────────────

class SceneRowWidget(QWidget):
    """Widget inline affiché dans chaque QTreeWidgetItem de la scène."""

    activate_requested   = pyqtSignal(int)
    rename_requested     = pyqtSignal(int)
    delete_requested     = pyqtSignal(int)
    export_requested     = pyqtSignal(int)
    thumb_refresh_req    = pyqtSignal(int)
    edit_transition_req  = pyqtSignal(int)

    def __init__(self, idx: int, scene: SceneItem, parent=None):
        super().__init__(parent)
        self._idx   = idx
        self._scene = scene
        self._build()

    # ── construction ────────────────────────────────────────────────────────

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 3, 4, 3)
        root.setSpacing(6)

        # Miniature
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setFixedSize(THUMB_W, THUMB_H)
        self._thumb_lbl.setScaledContents(True)
        self._thumb_lbl.setStyleSheet("border:1px solid #2a2d3a; border-radius:3px;")
        self.refresh_thumbnail()
        root.addWidget(self._thumb_lbl)

        # Colonne centrale : nom + méta
        center = QVBoxLayout()
        center.setSpacing(3)

        # Pastille couleur + nom
        name_row = QHBoxLayout()
        name_row.setSpacing(5)

        self._color_dot = QLabel()
        self._color_dot.setFixedSize(10, 10)
        self._color_dot.setStyleSheet(
            f"background:{self._scene.color}; border-radius:5px;")
        name_row.addWidget(self._color_dot)

        self._name_lbl = QLabel(self._scene.name)
        self._name_lbl.setStyleSheet("color:#e0e4f0; font-weight:600; font-size:12px;")
        name_row.addWidget(self._name_lbl, 1)
        center.addLayout(name_row)

        # Infos : nbre de passes + durée timeline
        passes = list(self._scene.shaders.keys())
        dur    = self._scene.timeline.get("duration", 0.0)
        info   = f"{len(passes)} pass{'es' if len(passes)!=1 else ''}"
        if dur:
            info += f"  ·  {dur:.1f}s"
        self._info_lbl = QLabel(info)
        self._info_lbl.setStyleSheet("color:#6070a0; font-size:10px;")
        center.addWidget(self._info_lbl)

        # Transition badge
        trans_row = QHBoxLayout()
        trans_row.setSpacing(4)
        badge = QLabel()
        badge.setPixmap(_blend_badge(self._scene.transition.mode))
        badge.setFixedSize(20, 16)
        trans_row.addWidget(badge)
        trans_lbl = QLabel(
            f"{BLEND_LABELS.get(self._scene.transition.mode,'?')} "
            f"({self._scene.transition.duration:.1f}s)")
        trans_lbl.setStyleSheet("color:#5060a0; font-size:10px;")
        trans_row.addWidget(trans_lbl, 1)
        center.addLayout(trans_row)

        root.addLayout(center, 1)

        # Boutons d'action
        btn_col = QVBoxLayout()
        btn_col.setSpacing(3)

        def _mk(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setFixedWidth(26)
            b.setStyleSheet(
                "QToolButton{background:#1e2030;color:#9098b8;border:1px solid #2a2e44;"
                "border-radius:3px;font-size:11px;}"
                "QToolButton:hover{background:#2a3050;color:#d0d8ff;}")
            b.clicked.connect(slot)
            return b

        btn_col.addWidget(_mk("▶", "Activer cette scène",           lambda: self.activate_requested.emit(self._idx)))
        btn_col.addWidget(_mk("⤓", "Exporter .osdemo",              lambda: self.export_requested.emit(self._idx)))
        btn_col.addWidget(_mk("⇄", "Modifier la transition",        lambda: self.edit_transition_req.emit(self._idx)))
        btn_col.addWidget(_mk("⟳", "Rafraîchir la miniature",       lambda: self.thumb_refresh_req.emit(self._idx)))
        btn_col.addWidget(_mk("✎", "Renommer",                       lambda: self.rename_requested.emit(self._idx)))
        btn_col.addWidget(_mk("✕", "Supprimer la scène",             lambda: self.delete_requested.emit(self._idx)))
        btn_col.addStretch()

        root.addLayout(btn_col)

    # ── API ─────────────────────────────────────────────────────────────────

    def refresh_thumbnail(self):
        pix = self._scene.thumbnail
        if pix is None or pix.isNull():
            pix = _placeholder_thumb(self._scene.color)
        self._thumb_lbl.setPixmap(pix.scaled(
            THUMB_W, THUMB_H, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation))

    def update_index(self, new_idx: int):
        self._idx = new_idx

    def update_scene(self, scene: SceneItem):
        self._scene = scene
        self._name_lbl.setText(scene.name)
        self._color_dot.setStyleSheet(
            f"background:{scene.color}; border-radius:5px;")
        passes = list(scene.shaders.keys())
        dur    = scene.timeline.get("duration", 0.0)
        info   = f"{len(passes)} pass{'es' if len(passes)!=1 else ''}"
        if dur:
            info += f"  ·  {dur:.1f}s"
        self._info_lbl.setText(info)
        self.refresh_thumbnail()


# ──────────────────────────────────────────────────────────────────────────────
# Dialogue : éditer la transition d'une scène
# ──────────────────────────────────────────────────────────────────────────────

class TransitionDialog(QDialog):
    def __init__(self, transition: SceneTransition, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transition vers la scène suivante")
        self.setMinimumWidth(340)
        self._trans = copy.deepcopy(transition)
        self._build()

    def _build(self):
        lay = QFormLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 16)

        self._mode_cb = QComboBox()
        for m in BLEND_MODES:
            self._mode_cb.addItem(BLEND_LABELS[m], m)
        idx = BLEND_MODES.index(self._trans.mode) if self._trans.mode in BLEND_MODES else 0
        self._mode_cb.setCurrentIndex(idx)
        lay.addRow("Mode :", self._mode_cb)

        self._dur_sb = QDoubleSpinBox()
        self._dur_sb.setRange(0.0, 30.0)
        self._dur_sb.setSingleStep(0.1)
        self._dur_sb.setValue(self._trans.duration)
        self._dur_sb.setSuffix(" s")
        lay.addRow("Durée :", self._dur_sb)

        self._ease_cb = QComboBox()
        for e in ["linear", "smooth", "ease_in", "ease_out"]:
            self._ease_cb.addItem(e, e)
        ease_idx = ["linear", "smooth", "ease_in", "ease_out"].index(
            self._trans.easing) if self._trans.easing in ["linear","smooth","ease_in","ease_out"] else 1
        self._ease_cb.setCurrentIndex(ease_idx)
        lay.addRow("Easing :", self._ease_cb)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

    def result_transition(self) -> SceneTransition:
        t          = SceneTransition()
        t.mode     = self._mode_cb.currentData()
        t.duration = self._dur_sb.value()
        t.easing   = self._ease_cb.currentData()
        return t


# ──────────────────────────────────────────────────────────────────────────────
# Widget principal : SceneGraphDock
# ──────────────────────────────────────────────────────────────────────────────

class SceneGraphWidget(QWidget):
    """Widget de gestion du graphe de scènes (placé dans un QDockWidget)."""

    # ── signaux ──────────────────────────────────────────────────────────────
    scene_activated         = pyqtSignal(int)       # idx → charger la scène
    scene_preview_requested = pyqtSignal(int)       # idx → générer miniature
    scene_order_changed     = pyqtSignal()
    scene_list_changed      = pyqtSignal()          # ajout/suppr/renommage

    def __init__(self, graph: SceneGraph, parent=None):
        super().__init__(parent)
        self._graph       = graph
        self._row_widgets: list[SceneRowWidget] = []
        self._build_ui()
        self.refresh()

    # ── construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre d'outils ────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("SceneToolbar")
        toolbar.setStyleSheet(
            "#SceneToolbar{background:#0f1118;border-bottom:1px solid #1e2030;}")
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(8, 5, 8, 5)
        tb_lay.setSpacing(4)

        title = QLabel("🎬  Scènes")
        title.setStyleSheet("color:#8090c0;font-weight:700;font-size:12px;")
        tb_lay.addWidget(title)
        tb_lay.addStretch()

        for text, tip, slot in [
            ("＋ Nouvelle",    "Ajouter une scène vide",       self._add_new_scene),
            ("⤓ Importer…",   "Importer un fichier .osdemo",   self._import_osdemo),
            ("⤒ Exporter tts","Exporter toutes les scènes",    self._export_all),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet(
                "QPushButton{background:#1a1c2c;color:#8898c8;border:1px solid #2a2e44;"
                "border-radius:4px;padding:3px 8px;font-size:11px;}"
                "QPushButton:hover{background:#242840;color:#c0ccff;}")
            b.clicked.connect(slot)
            tb_lay.addWidget(b)

        root.addWidget(toolbar)

        # ── Liste de scènes ───────────────────────────────────────────────
        self._list = QTreeWidget()
        self._list.setObjectName("SceneList")
        self._list.setHeaderHidden(True)
        self._list.setIndentation(0)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setAnimated(True)
        self._list.setStyleSheet("""
            QTreeWidget {
                background: #0d0f16;
                border: none;
                outline: none;
            }
            QTreeWidget::item {
                border-bottom: 1px solid #16192a;
                padding: 0;
            }
            QTreeWidget::item:selected {
                background: #141830;
            }
            QTreeWidget::item:hover {
                background: #10121e;
            }
        """)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        root.addWidget(self._list, 1)

        # ── Pied de page : indicateur actif ──────────────────────────────
        footer = QWidget()
        footer.setStyleSheet("background:#0a0c14;border-top:1px solid #1a1d2e;")
        ft_lay = QHBoxLayout(footer)
        ft_lay.setContentsMargins(8, 4, 8, 4)
        self._status_lbl = QLabel("Aucune scène active")
        self._status_lbl.setStyleSheet("color:#4a5878;font-size:10px;")
        ft_lay.addWidget(self._status_lbl)
        root.addWidget(footer)

    # ── rafraîchissement ──────────────────────────────────────────────────

    def refresh(self):
        """Reconstruit entièrement la liste depuis self._graph."""
        self._list.clear()
        self._row_widgets.clear()

        for idx, scene in enumerate(self._graph.scenes):
            item = QTreeWidgetItem(self._list)
            row  = SceneRowWidget(idx, scene)
            self._connect_row(row)
            self._row_widgets.append(row)
            item.setSizeHint(0, QSize(0, THUMB_H + 14))
            self._list.setItemWidget(item, 0, row)
            if idx == self._graph.active_index:
                item.setBackground(0, QBrush(QColor("#141a2e")))

        self._update_status()

    def _update_status(self):
        sc = self._graph.active_scene
        if sc:
            self._status_lbl.setText(f"Active : 「{sc.name}」  (index {self._graph.active_index})")
        else:
            self._status_lbl.setText("Aucune scène active")

    def _connect_row(self, row: SceneRowWidget):
        row.activate_requested.connect(self._on_activate)
        row.rename_requested.connect(self._on_rename)
        row.delete_requested.connect(self._on_delete)
        row.export_requested.connect(self._on_export_single)
        row.thumb_refresh_req.connect(self.scene_preview_requested)
        row.edit_transition_req.connect(self._on_edit_transition)

    # ── slots utilisateur ─────────────────────────────────────────────────

    def _on_activate(self, idx: int):
        self._graph.active_index = idx
        self.scene_activated.emit(idx)
        self.refresh()

    def _on_rename(self, idx: int):
        scene = self._graph.scenes[idx]
        name, ok = QInputDialog.getText(
            self, "Renommer la scène", "Nouveau nom :", text=scene.name)
        if ok and name.strip():
            scene.name = name.strip()
            self.refresh()
            self.scene_list_changed.emit()

    def _on_delete(self, idx: int):
        scene = self._graph.scenes[idx]
        reply = QMessageBox.question(
            self, "Supprimer la scène",
            f"Supprimer « {scene.name} » ? Cette action est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._graph.remove(idx)
            self.refresh()
            self.scene_list_changed.emit()

    def _on_export_single(self, idx: int):
        scene = self._graph.scenes[idx]
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la scène", f"{scene.name}.osdemo",
            "OpenShader Demo (*.osdemo)")
        if path:
            ok = self._graph.save_scene_osdemo(scene, path)
            if ok:
                QMessageBox.information(self, "Export réussi",
                                        f"Scène exportée :\n{path}")
            else:
                QMessageBox.critical(self, "Erreur export",
                                     "Impossible d'écrire le fichier .osdemo.")

    def _on_edit_transition(self, idx: int):
        scene = self._graph.scenes[idx]
        dlg   = TransitionDialog(scene.transition, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            scene.transition = dlg.result_transition()
            self.refresh()
            self.scene_list_changed.emit()

    def _on_rows_moved(self, *_):
        """Synchronise l'ordre du modèle après drag-drop dans l'arbre."""
        # L'arbre Qt a déjà déplacé les items ; reconstruire l'ordre dans le graph
        new_order = []
        for i in range(self._list.topLevelItemCount()):
            item = self._list.topLevelItem(i)
            w    = self._list.itemWidget(item, 0)
            if w is not None:
                new_order.append(self._graph.scenes[w._idx])
        self._graph.scenes = new_order
        # Recalibrer les indices dans les row widgets
        for i, scene in enumerate(self._graph.scenes):
            item = self._list.topLevelItem(i)
            w    = self._list.itemWidget(item, 0)
            if w is not None:
                w.update_index(i)
        self.scene_order_changed.emit()

    # ── Ajout / import ────────────────────────────────────────────────────

    def _add_new_scene(self):
        scene = SceneItem(name=f"Scène {len(self._graph.scenes)+1}")
        self._graph.append(scene)
        self.refresh()
        self.scene_list_changed.emit()

    def _import_osdemo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Importer une scène .osdemo", "",
            "OpenShader Demo (*.osdemo);;Tous les fichiers (*)")
        if path:
            scene = self._graph.load_osdemo(path)
            if scene:
                self._graph.append(scene)
                self.refresh()
                self.scene_list_changed.emit()
                QMessageBox.information(self, "Import réussi",
                                        f"Scène « {scene.name} » importée.")
            else:
                QMessageBox.critical(self, "Erreur import",
                                     "Impossible de lire le fichier .osdemo.")

    def _export_all(self):
        if not self._graph.scenes:
            QMessageBox.information(self, "Aucune scène", "Il n'y a aucune scène à exporter.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Dossier de destination pour les .osdemo")
        if not folder:
            return
        ok_count = 0
        for scene in self._graph.scenes:
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in scene.name)
            path = os.path.join(folder, f"{safe}_{scene.scene_id}.osdemo")
            if self._graph.save_scene_osdemo(scene, path):
                ok_count += 1
        QMessageBox.information(
            self, "Export terminé",
            f"{ok_count}/{len(self._graph.scenes)} scène(s) exportée(s) dans :\n{folder}")

    # ── API externe ───────────────────────────────────────────────────────

    def set_thumbnail(self, idx: int, pixmap: QPixmap):
        """Appelé par main_window quand la miniature d'une scène est prête."""
        if 0 <= idx < len(self._graph.scenes):
            self._graph.scenes[idx].thumbnail = pixmap
            if idx < len(self._row_widgets):
                self._row_widgets[idx].refresh_thumbnail()

    def add_scene_from_current(self, scene: SceneItem):
        """Appelé par main_window pour sauvegarder la scène courante dans le graphe."""
        self._graph.append(scene)
        self.refresh()
        self.scene_list_changed.emit()

    def replace_scene(self, idx: int, scene: SceneItem):
        """Met à jour une scène existante (ex: 'Sauvegarder dans la scène courante')."""
        if 0 <= idx < len(self._graph.scenes):
            scene.thumbnail = self._graph.scenes[idx].thumbnail
            self._graph.scenes[idx] = scene
            if idx < len(self._row_widgets):
                self._row_widgets[idx].update_scene(scene)
            self.scene_list_changed.emit()


# ──────────────────────────────────────────────────────────────────────────────
# Usine : crée le dock prêt à être branché dans main_window
# ──────────────────────────────────────────────────────────────────────────────

def create_scene_graph_dock(graph: SceneGraph,
                             parent: "QMainWindow") -> tuple["QDockWidget", SceneGraphWidget]:
    """
    Crée et retourne (dock, widget).
    Le dock a déjà son objectName positionné.
    """
    from PyQt6.QtWidgets import QDockWidget
    from PyQt6.QtCore    import Qt

    widget = SceneGraphWidget(graph)
    dock   = QDockWidget("🎬  Scene Graph", parent)
    dock.setObjectName("DockSceneGraph")
    dock.setWidget(widget)
    dock.setAllowedAreas(
        Qt.DockWidgetArea.LeftDockWidgetArea |
        Qt.DockWidgetArea.RightDockWidgetArea)
    return dock, widget
