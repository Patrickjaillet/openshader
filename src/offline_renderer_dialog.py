"""
offline_renderer_dialog.py
--------------------------
UI PyQt6 du moteur de rendu différé (Offline Renderer) — OpenShader v6.1

Interface :
  • 3 onglets : Sortie | Anti-Aliasing & Flou | DCP / Avancé
  • Preview de la commande FFmpeg générée
  • Barre de progression temps-réel (QTimer polling)
  • Annulation propre
  • QTimer polling au lieu de signaux inter-thread (plus robuste)
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QLineEdit,
    QGroupBox, QFrame, QTextEdit, QSizePolicy,
)

from .offline_renderer import (
    OfflineRenderConfig, OfflineRenderEngine, OfflineRenderProgress,
)
from .logger import get_logger

log = get_logger(__name__)

# ── Design tokens ──────────────────────────────────────────────────────────────

_BG   = "#0c0e18"
_SURF = "#12141f"
_BORD = "#1c2030"
_TEXT = "#c0c8e0"
_DIM  = "#505878"
_ACC  = "#4e7fff"
_GRN  = "#40d090"
_WARN = "#e07840"
_SANS = "Segoe UI, Arial, sans-serif"
_MONO = "Cascadia Code, Consolas, monospace"


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {_BORD};")
    return f


def _label(text: str, muted: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {'#6070a0' if muted else _TEXT}; font: 9px '{_SANS}';")
    lbl.setWordWrap(True)
    return lbl


def _badge(text: str, color: str = _ACC) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {color}; background: {color}18; border: 1px solid {color}44;"
        f"border-radius: 3px; padding: 1px 6px; font: 8px '{_SANS}';"
    )
    return lbl


# ── Codec presets (offline quality) ───────────────────────────────────────────

OFFLINE_CODECS: dict[str, dict] = {
    "H.264 — MP4 (universel)": {
        "ext": "mp4", "vcodec": "libx264", "pix_fmt": "yuv420p",
        "extra": ["-preset", "veryslow", "-movflags", "+faststart"],
        "crf_range": (0, 51), "crf_default": 14,
        "desc": "Compatibilité maximale. CRF 14 = qualité mastering.",
    },
    "H.265 — MP4 (haute compression)": {
        "ext": "mp4", "vcodec": "libx265", "pix_fmt": "yuv420p",
        "extra": ["-preset", "veryslow", "-tag:v", "hvc1"],
        "crf_range": (0, 51), "crf_default": 16,
        "desc": "−50% taille vs H.264. Recommandé pour archivage.",
    },
    "ProRes 4444 — MOV (mastering alpha)": {
        "ext": "mov", "vcodec": "prores_ks", "pix_fmt": "yuva444p10le",
        "extra": ["-profile:v", "4", "-vendor", "apl0", "-bits_per_mb", "8000"],
        "crf_range": None, "crf_default": None,
        "desc": "10-bit avec canal alpha. Standard post-production.",
    },
    "VP9 — WebM (web sans perte approx.)": {
        "ext": "webm", "vcodec": "libvpx-vp9", "pix_fmt": "yuv420p",
        "extra": ["-row-mt", "1", "-tile-columns", "2", "-frame-parallel", "1"],
        "crf_range": (0, 63), "crf_default": 20,
        "desc": "Compression web efficace. CRF 20 ≈ sans perte visible.",
    },
    "AV1 — MKV (nouvelle génération)": {
        "ext": "mkv", "vcodec": "libaom-av1", "pix_fmt": "yuv420p",
        "extra": ["-cpu-used", "2", "-row-mt", "1"],
        "crf_range": (0, 63), "crf_default": 22,
        "desc": "Meilleure compression. Encodage très lent.",
    },
    "GIF animé — palette optimisée": {
        "ext": "gif", "vcodec": "gif", "pix_fmt": None,
        "extra": [], "crf_range": None, "crf_default": None,
        "desc": "256 couleurs + dithering Floyd-Steinberg.",
    },
    "Séquence PNG": {
        "ext": "png_seq", "vcodec": None, "pix_fmt": None,
        "extra": [], "crf_range": None, "crf_default": None,
        "desc": "Chaque frame en PNG 8-bit RGBA. Dossier de sortie requis.",
    },
    "Séquence EXR (16-bit)": {
        "ext": "exr_seq", "vcodec": None, "pix_fmt": None,
        "extra": [], "crf_range": None, "crf_default": None,
        "desc": "Chaque frame en OpenEXR 16-bit half. Requiert le module OpenEXR.",
    },
    "DCP — Digital Cinema Package": {
        "ext": "dcp", "vcodec": "libopenjpeg", "pix_fmt": "rgb48le",
        "extra": [], "crf_range": None, "crf_default": None,
        "desc": "Standard projection festival. MXF JPEG2000 XYZ + métadonnées DCI.",
    },
}

RESOLUTION_PRESETS: dict[str, Optional[tuple[int, int]]] = {
    "Viewport actuel":       None,
    "720p  (1280×720)":      (1280, 720),
    "1080p (1920×1080)":     (1920, 1080),
    "2K DCI (2048×1080)":    (2048, 1080),
    "1440p (2560×1440)":     (2560, 1440),
    "4K DCI (4096×2160)":    (4096, 2160),
    "4K UHD (3840×2160)":    (3840, 2160),
    "Carré 1080 (1080×1080)":(1080, 1080),
    "Personnalisée…":        "custom",
}

TAA_SAMPLES = [1, 2, 4, 8, 16, 32, 64]
MB_SAMPLES  = [2, 4, 8, 16, 32]


# ═══════════════════════════════════════════════════════════════════════════════
#  OfflineRendererDialog
# ═══════════════════════════════════════════════════════════════════════════════

class OfflineRendererDialog(QDialog):
    """
    Dialog complet de rendu différé ultra-qualité.

    L'appelant fournit une callable render_fn(t, jitter_x, jitter_y) → np.ndarray.
    """

    def __init__(self, parent=None, *,
                 viewport_w: int = 1920,
                 viewport_h: int = 1080,
                 timeline_duration: float = 10.0,
                 audio_path: Optional[str] = None,
                 render_fn: Optional[Callable] = None):
        super().__init__(parent)
        self.setWindowTitle("🎞 Rendu Offline Haute Qualité — v6.1")
        self.setMinimumWidth(620)
        self.resize(660, 660)
        self.setStyleSheet(f"""
            QDialog, QWidget  {{ background: {_BG}; color: {_TEXT}; }}
            QTabWidget::pane  {{ border: 1px solid {_BORD}; background: {_SURF}; }}
            QTabBar::tab      {{ background: {_SURF}; color: {_DIM};
                                 padding: 5px 16px; border: 1px solid {_BORD};
                                 margin-right: 2px; font: 9px '{_SANS}'; }}
            QTabBar::tab:selected {{ color: {_TEXT}; border-bottom: 2px solid {_ACC}; }}
            QGroupBox         {{ border: 1px solid {_BORD}; border-radius: 4px;
                                 margin-top: 8px; padding-top: 6px;
                                 font: bold 9px '{_SANS}'; color: {_DIM}; }}
            QGroupBox::title  {{ subcontrol-origin: margin; left: 8px; color: {_DIM}; }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
                background: {_SURF}; color: {_TEXT}; border: 1px solid {_BORD};
                border-radius: 3px; padding: 3px 6px; font: 9px '{_SANS}'; }}
            QComboBox:focus, QSpinBox:focus {{ border-color: {_ACC}88; }}
            QCheckBox         {{ color: {_TEXT}; font: 9px '{_SANS}'; spacing: 6px; }}
            QCheckBox::indicator {{ width: 14px; height: 14px;
                                    border: 1px solid {_BORD}; border-radius: 2px;
                                    background: {_SURF}; }}
            QCheckBox::indicator:checked {{ background: {_ACC}; border-color: {_ACC}; }}
            QSlider::groove:horizontal   {{ height: 4px; background: {_BORD}; border-radius: 2px; }}
            QSlider::handle:horizontal   {{ width: 14px; height: 14px; margin: -5px 0;
                                            background: {_ACC}; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: {_ACC}44; border-radius: 2px; }}
            QPushButton {{ background: {_SURF}; color: {_TEXT}; border: 1px solid {_BORD};
                           border-radius: 3px; padding: 5px 14px; font: 9px '{_SANS}'; }}
            QPushButton:hover   {{ background: {_ACC}22; border-color: {_ACC}66; }}
            QPushButton:pressed {{ background: {_ACC}44; }}
            QPushButton:disabled {{ color: {_DIM}; border-color: {_BORD}; }}
            QProgressBar {{ background: {_BORD}; border-radius: 3px; border: none; height: 8px; }}
            QProgressBar::chunk {{ background: {_ACC}; border-radius: 3px; }}
        """)

        self._viewport_w = viewport_w
        self._viewport_h = viewport_h
        self._duration   = timeline_duration
        self._audio_path = audio_path
        self._render_fn  = render_fn

        self._progress:  Optional[OfflineRenderProgress] = None
        self._engine_thread: Optional[threading.Thread]  = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(80)   # 80 ms polling
        self._poll_timer.timeout.connect(self._poll_progress)

        self._build_ui()
        self._on_codec_changed()
        self._on_res_changed()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Titre ────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        t = QLabel("Rendu Offline Haute Qualité")
        t.setStyleSheet(f"color: {_TEXT}; font: bold 13px '{_SANS}';")
        title_row.addWidget(t)
        title_row.addStretch()
        for txt, col in [("TAA", _GRN), ("Motion Blur", _ACC), ("DCP", _WARN)]:
            title_row.addWidget(_badge(txt, col))
        root.addLayout(title_row)
        root.addWidget(_sep())

        # ── Onglets ───────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.addTab(self._tab_output(),   "🎬 Sortie")
        tabs.addTab(self._tab_quality(),  "✨ Anti-Aliasing & Flou")
        tabs.addTab(self._tab_advanced(), "⚙ DCP & Avancé")
        root.addWidget(tabs, 1)

        # ── Progression ───────────────────────────────────────────────────
        prog_box = QGroupBox("Progression")
        pb_lay = QVBoxLayout(prog_box)
        pb_lay.setSpacing(4)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(10)
        self._progress_label = QLabel("En attente…")
        self._progress_label.setStyleSheet(f"color: {_DIM}; font: 9px '{_MONO}';")
        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color: {_DIM}; font: 8px '{_SANS}';")
        pb_lay.addWidget(self._progress_bar)

        stats_row = QHBoxLayout()
        stats_row.addWidget(self._progress_label, 1)
        stats_row.addWidget(self._eta_label)
        pb_lay.addLayout(stats_row)
        root.addWidget(prog_box)

        # ── Boutons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("🔍 Aperçu FFmpeg…")
        self._btn_preview.clicked.connect(self._show_preview)
        btn_row.addWidget(self._btn_preview)
        btn_row.addStretch()

        self._btn_cancel = QPushButton("⏹ Annuler")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._do_cancel)
        btn_row.addWidget(self._btn_cancel)

        self._btn_render = QPushButton("🎞 Lancer le rendu…")
        self._btn_render.setStyleSheet(
            f"QPushButton {{ background: {_ACC}28; color: {_ACC}; "
            f"border: 1px solid {_ACC}66; font: bold 9px '{_SANS}'; }}"
            f"QPushButton:hover {{ background: {_ACC}44; }}"
        )
        self._btn_render.clicked.connect(self._do_render)
        btn_row.addWidget(self._btn_render)

        self._btn_close = QPushButton("Fermer")
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_close)
        root.addLayout(btn_row)

    # ── Onglet Sortie ──────────────────────────────────────────────────────────

    def _tab_output(self) -> QWidget:
        w   = QWidget()
        lay = QFormLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(12, 12, 12, 12)

        # Format / codec
        self._codec_cb = QComboBox()
        self._codec_cb.addItems(list(OFFLINE_CODECS.keys()))
        self._codec_cb.currentTextChanged.connect(self._on_codec_changed)
        lay.addRow("Format / codec :", self._codec_cb)

        self._codec_desc_lbl = _label("", muted=True)
        lay.addRow("", self._codec_desc_lbl)
        lay.addRow(_sep())

        # Résolution
        self._res_cb = QComboBox()
        self._res_cb.addItems(list(RESOLUTION_PRESETS.keys()))
        self._res_cb.setCurrentText("1080p (1920×1080)")
        self._res_cb.currentTextChanged.connect(self._on_res_changed)
        lay.addRow("Résolution :", self._res_cb)

        res_row = QHBoxLayout()
        self._width_sb  = QSpinBox(); self._width_sb.setRange(16, 8192)
        self._width_sb.setValue(1920)
        self._height_sb = QSpinBox(); self._height_sb.setRange(16, 4320)
        self._height_sb.setValue(1080)
        res_row.addWidget(self._width_sb)
        res_row.addWidget(_label(" × "))
        res_row.addWidget(self._height_sb)
        res_row.addStretch()
        lay.addRow("Taille :", res_row)

        # FPS
        self._fps_cb = QComboBox()
        for fps_val in ["23.976", "24", "25", "29.97", "30", "48", "50", "60"]:
            self._fps_cb.addItem(f"{fps_val} fps", float(fps_val))
        self._fps_cb.setCurrentText("24 fps")
        lay.addRow("Fréquence :", self._fps_cb)

        # Durée + début
        dur_row = QHBoxLayout()
        self._dur_sb = QDoubleSpinBox()
        self._dur_sb.setRange(0.1, 7200.0)
        self._dur_sb.setValue(self._duration)
        self._dur_sb.setSuffix(" s")
        self._dur_sb.valueChanged.connect(self._update_frame_count)
        dur_row.addWidget(self._dur_sb)
        dur_row.addWidget(_label("  début :"))
        self._start_sb = QDoubleSpinBox()
        self._start_sb.setRange(0.0, 7200.0)
        self._start_sb.setValue(0.0)
        self._start_sb.setSuffix(" s")
        dur_row.addWidget(self._start_sb)
        dur_row.addStretch()
        lay.addRow("Durée :", dur_row)

        self._frames_lbl = _label("", muted=True)
        lay.addRow("", self._frames_lbl)
        self._update_frame_count()
        lay.addRow(_sep())

        # CRF
        crf_row = QHBoxLayout()
        self._crf_slider = QSlider(Qt.Orientation.Horizontal)
        self._crf_slider.setRange(0, 51)
        self._crf_slider.setValue(14)
        self._crf_label  = QLabel("14")
        self._crf_label.setStyleSheet(f"color: {_GRN}; font: 9px '{_MONO}'; min-width: 28px;")
        self._crf_slider.valueChanged.connect(
            lambda v: self._crf_label.setText(str(v)))
        crf_row.addWidget(self._crf_slider)
        crf_row.addWidget(self._crf_label)
        lay.addRow("Qualité (CRF) :", crf_row)

        # Audio
        self._audio_cb = QCheckBox("Inclure l'audio")
        self._audio_cb.setChecked(bool(self._audio_path))
        self._audio_cb.setEnabled(bool(self._audio_path))
        lay.addRow("Audio :", self._audio_cb)
        if not self._audio_path:
            lay.addRow("", _label("Aucun fichier audio chargé.", muted=True))

        return w

    # ── Onglet Anti-Aliasing & Motion Blur ────────────────────────────────────

    def _tab_quality(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        # ── TAA ─────────────────────────────────────────────────────────
        taa_box = QGroupBox("TAA — Temporal Anti-Aliasing (jitter Halton 2D)")
        taa_lay = QFormLayout(taa_box)
        taa_lay.setSpacing(8)

        self._taa_cb = QCheckBox("Activer le TAA")
        self._taa_cb.setChecked(True)
        self._taa_cb.toggled.connect(self._on_quality_toggled)
        taa_lay.addRow("", self._taa_cb)

        self._taa_samples_cb = QComboBox()
        for s in TAA_SAMPLES:
            self._taa_samples_cb.addItem(f"{s} sample{'s' if s > 1 else ''}", s)
        self._taa_samples_cb.setCurrentText("8 samples")
        self._taa_samples_cb.currentIndexChanged.connect(self._update_quality_label)
        taa_lay.addRow("Samples :", self._taa_samples_cb)

        self._jitter_sb = QDoubleSpinBox()
        self._jitter_sb.setRange(0.1, 2.0)
        self._jitter_sb.setSingleStep(0.1)
        self._jitter_sb.setValue(0.5)
        self._jitter_sb.setSuffix(" px")
        taa_lay.addRow("Rayon jitter :", self._jitter_sb)

        taa_lay.addRow("", _label(
            "Le TAA accumule N rendus avec des décalages sub-pixel (séquence Halton).\n"
            "8 samples = excellent. 32+ = résultats quasi parfaits mais très lent.",
            muted=True))
        lay.addWidget(taa_box)

        # ── Motion Blur ─────────────────────────────────────────────────
        mb_box = QGroupBox("Motion Blur — Accumulation temporelle")
        mb_lay = QFormLayout(mb_box)
        mb_lay.setSpacing(8)

        self._mb_cb = QCheckBox("Activer le motion blur")
        self._mb_cb.setChecked(False)
        self._mb_cb.toggled.connect(self._on_quality_toggled)
        mb_lay.addRow("", self._mb_cb)

        self._mb_samples_cb = QComboBox()
        for s in MB_SAMPLES:
            self._mb_samples_cb.addItem(f"{s} sub-frames", s)
        self._mb_samples_cb.setCurrentText("8 sub-frames")
        self._mb_samples_cb.currentIndexChanged.connect(self._update_quality_label)
        mb_lay.addRow("Sub-frames :", self._mb_samples_cb)

        shutter_row = QHBoxLayout()
        self._shutter_slider = QSlider(Qt.Orientation.Horizontal)
        self._shutter_slider.setRange(5, 100)
        self._shutter_slider.setValue(50)
        self._shutter_label  = QLabel("50%")
        self._shutter_label.setStyleSheet(f"color: {_ACC}; font: 9px '{_MONO}'; min-width: 36px;")
        self._shutter_slider.valueChanged.connect(
            lambda v: self._shutter_label.setText(f"{v}%"))
        shutter_row.addWidget(self._shutter_slider)
        shutter_row.addWidget(self._shutter_label)
        mb_lay.addRow("Obturation :", shutter_row)

        mb_lay.addRow("", _label(
            "N sous-frames sont rendues sur l'intervalle [t, t + shutter/fps]\n"
            "et moyennées. 8 sub-frames + 50% = flou cinématographique naturel.",
            muted=True))
        lay.addWidget(mb_box)

        # ── Estimation de durée ──────────────────────────────────────────
        self._quality_estimate_lbl = QLabel("")
        self._quality_estimate_lbl.setStyleSheet(
            f"color: {_WARN}; font: 9px '{_SANS}'; padding: 6px 10px;"
            f"background: {_WARN}12; border-radius: 3px;"
        )
        lay.addWidget(self._quality_estimate_lbl)

        lay.addStretch()
        self._update_quality_label()
        return w

    # ── Onglet DCP & Avancé ───────────────────────────────────────────────────

    def _tab_advanced(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        # ── DCP ─────────────────────────────────────────────────────────
        dcp_box = QGroupBox("DCP — Digital Cinema Package")
        dcp_lay = QFormLayout(dcp_box)
        dcp_lay.setSpacing(8)

        self._dcp_title_le = QLineEdit("OpenShader")
        dcp_lay.addRow("Titre :", self._dcp_title_le)

        self._dcp_issuer_le = QLineEdit("OpenShader v6")
        dcp_lay.addRow("Émetteur :", self._dcp_issuer_le)

        dcp_lay.addRow("", _label(
            "Le DCP produit un dossier MXF/JPEG2000 lisible par les serveurs\n"
            "de cinéma DCI (Barco, NEC, Christie…). Requiert FFmpeg + libopenjpeg.",
            muted=True))

        # Vérification FFmpeg + libopenjpeg
        ffmpeg_ok = bool(shutil.which("ffmpeg"))
        status_color = _GRN if ffmpeg_ok else _WARN
        status_text  = "✓ FFmpeg trouvé" if ffmpeg_ok else "⚠ FFmpeg introuvable"
        dcp_lay.addRow("FFmpeg :", _badge(status_text, status_color))
        lay.addWidget(dcp_box)

        # ── Avancé ──────────────────────────────────────────────────────
        adv_box = QGroupBox("Options avancées")
        adv_lay = QFormLayout(adv_box)
        adv_lay.setSpacing(8)

        self._threads_sb = QSpinBox()
        self._threads_sb.setRange(0, 128)
        self._threads_sb.setValue(0)
        self._threads_sb.setSpecialValueText("auto")
        adv_lay.addRow("Threads FFmpeg :", self._threads_sb)

        self._exr_cb = QCheckBox("Séquences EXR 16-bit half (si OpenEXR Python installé)")
        adv_lay.addRow("", self._exr_cb)

        self._open_dir_cb = QCheckBox("Ouvrir le dossier après le rendu")
        self._open_dir_cb.setChecked(True)
        adv_lay.addRow("", self._open_dir_cb)

        lay.addWidget(adv_box)
        lay.addStretch()
        return w

    # ── Slots UI ──────────────────────────────────────────────────────────────

    def _on_codec_changed(self):
        name = self._codec_cb.currentText()
        info = OFFLINE_CODECS.get(name, {})
        self._codec_desc_lbl.setText(info.get("desc", ""))

        crf_range   = info.get("crf_range")
        crf_default = info.get("crf_default")
        has_crf     = crf_range is not None

        self._crf_slider.setEnabled(has_crf)
        if has_crf:
            lo, hi = crf_range
            self._crf_slider.setRange(lo, hi)
            if crf_default is not None:
                self._crf_slider.setValue(crf_default)
        else:
            self._crf_label.setText("n/a")

        # Séquence ou DCP → choix de dossier
        ext = info.get("ext", "")
        is_dir = ext in ("png_seq", "exr_seq", "dcp")
        # Mise à jour de l'estimation
        self._update_quality_label()

    def _on_res_changed(self):
        text = self._res_cb.currentText()
        val  = RESOLUTION_PRESETS.get(text)
        if val is None:
            self._width_sb.setValue(self._viewport_w)
            self._height_sb.setValue(self._viewport_h)
            self._width_sb.setEnabled(False)
            self._height_sb.setEnabled(False)
        elif val == "custom":
            self._width_sb.setEnabled(True)
            self._height_sb.setEnabled(True)
        else:
            self._width_sb.setValue(val[0])
            self._height_sb.setValue(val[1])
            self._width_sb.setEnabled(False)
            self._height_sb.setEnabled(False)

    def _on_quality_toggled(self):
        self._update_quality_label()

    def _update_frame_count(self):
        fps = self._fps_cb.currentData() or 24.0
        n   = int(self._dur_sb.value() * fps)
        self._frames_lbl.setText(f"{n} frames @ {fps:.3g} fps")

    def _update_quality_label(self):
        taa_n = self._taa_samples_cb.currentData() or 1 if self._taa_cb.isChecked() else 1
        mb_n  = self._mb_samples_cb.currentData() or 1 if self._mb_cb.isChecked() else 1
        mult  = taa_n * mb_n
        fps   = self._fps_cb.currentData() or 24.0
        n     = int(self._dur_sb.value() * fps)

        # Estimation très grossière : ~30ms par render call @ 1080p
        ms_per_call = 30
        total_calls = n * mult
        est_s = total_calls * ms_per_call / 1000

        def _fmt(s: float) -> str:
            if s < 60: return f"{s:.0f}s"
            if s < 3600: return f"{s/60:.0f}min"
            return f"{s/3600:.1f}h"

        if mult == 1:
            self._quality_estimate_lbl.setText(
                f"Rendu standard : {n} frames  ≈ {_fmt(est_s)} estimé")
        else:
            self._quality_estimate_lbl.setText(
                f"×{mult} render calls/frame ({taa_n} TAA × {mb_n} MB) × {n} frames"
                f" ≈ {_fmt(est_s)} estimé (GPU dépendant)")

    # ── Aperçu FFmpeg ─────────────────────────────────────────────────────────

    def _show_preview(self):
        cfg  = self._build_config("/tmp/output.mp4")
        info = OFFLINE_CODECS.get(self._codec_cb.currentText(), {})
        lines = [
            f"# Résolution   : {cfg.width}×{cfg.height}",
            f"# FPS          : {cfg.fps}",
            f"# TAA          : {'×' + str(cfg.taa_samples) if cfg.taa_enabled else 'désactivé'}",
            f"# Motion blur  : {'×' + str(cfg.mb_samples) + ' shutter=' + str(cfg.mb_shutter) if cfg.mb_enabled else 'désactivé'}",
            "",
            "ffmpeg -y \\",
            f"  -framerate {cfg.fps:.4g} \\",
            "  -i frame_%06d.png \\",
        ]
        if cfg.audio_path:
            lines += [f"  -i {cfg.audio_path} -t {cfg.duration:.2f} \\"]
        if info.get("vcodec"):
            lines += [f"  -c:v {info['vcodec']} \\"]
        if info.get("pix_fmt"):
            lines += [f"  -pix_fmt {info['pix_fmt']} \\"]
        if info.get("crf_range") and cfg.crf >= 0:
            lines += [f"  -crf {cfg.crf} \\"]
        for e in info.get("extra", []):
            lines += [f"  {e} \\"]
        if cfg.threads > 0:
            lines += [f"  -threads {cfg.threads} \\"]
        lines += ["  output." + info.get("ext", "mp4")]

        dlg = QDialog(self)
        dlg.setWindowTitle("Aperçu commande FFmpeg")
        dlg.resize(580, 340)
        dlg.setStyleSheet(self.styleSheet())
        vl  = QVBoxLayout(dlg)
        te  = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText("\n".join(lines))
        te.setStyleSheet(f"font: 11px '{_MONO}'; background: {_SURF}; color: {_TEXT};")
        vl.addWidget(te)
        ok = QPushButton("Fermer"); ok.clicked.connect(dlg.accept)
        vl.addWidget(ok)
        dlg.exec()

    # ── Lancement du rendu ────────────────────────────────────────────────────

    def _do_render(self):
        # Vérification render_fn
        render_fn = self._render_fn
        if render_fn is None:
            # Essaie via parent
            parent = self.parent()
            if hasattr(parent, "_offline_render_fn"):
                render_fn = parent._offline_render_fn
            elif hasattr(parent, "_export_render_frame"):
                # Adapte l'API existante (t, w, h) → array numpy
                def _adapt(t: float, jx: float, jy: float) -> np.ndarray:
                    from PyQt6.QtGui import QImage
                    img = parent._export_render_frame(
                        t + jx * 0.001,   # jitter infime sur le temps si pas de vrai jitter GL
                        self._width_sb.value(),
                        self._height_sb.value(),
                    )
                    # QImage → numpy RGBA
                    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
                    ptr = img.constBits()
                    ptr.setsize(img.width() * img.height() * 4)
                    return np.frombuffer(ptr, dtype=np.uint8).reshape(
                        img.height(), img.width(), 4).copy()
                render_fn = _adapt
        if render_fn is None:
            QMessageBox.critical(self, "Erreur",
                "Aucune fonction de rendu disponible.\n"
                "La fenêtre principale doit implémenter _offline_render_fn ou _export_render_frame.")
            return

        if not shutil.which("ffmpeg"):
            codec_name = self._codec_cb.currentText()
            ext = OFFLINE_CODECS.get(codec_name, {}).get("ext", "")
            if ext not in ("png_seq", "exr_seq"):
                QMessageBox.critical(self, "FFmpeg manquant",
                    "FFmpeg est introuvable dans le PATH.\n"
                    "Téléchargez-le sur https://ffmpeg.org/download.html")
                return

        # Choix du fichier / dossier de sortie
        out_path = self._choose_output()
        if not out_path:
            return

        cfg = self._build_config(out_path)
        self._start_render(cfg, render_fn)

    def _choose_output(self) -> str:
        codec_name = self._codec_cb.currentText()
        info       = OFFLINE_CODECS.get(codec_name, {})
        ext        = info.get("ext", "mp4")
        is_dir     = ext in ("png_seq", "exr_seq", "dcp")

        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self, "Choisir le dossier de sortie",
                os.path.expanduser("~"))
            if path:
                # Ajoute un sous-dossier nommé selon le titre DCP ou "render"
                dcp_title = self._dcp_title_le.text().strip() or "render"
                safe = "".join(c if c.isalnum() or c in "._- " else "_"
                               for c in dcp_title).strip()
                return os.path.join(path, safe + ("_dcp" if ext == "dcp" else "_frames"))
            return ""
        else:
            filt = f"*.{ext}"
            default = os.path.expanduser(f"~/render.{ext}")
            path, _ = QFileDialog.getSaveFileName(
                self, f"Enregistrer la vidéo ({ext.upper()})", default, filt)
            if path and not path.lower().endswith(f".{ext}"):
                path += f".{ext}"
            return path

    def _build_config(self, out_path: str) -> OfflineRenderConfig:
        codec_name = self._codec_cb.currentText()
        info       = OFFLINE_CODECS.get(codec_name, {})
        fps        = self._fps_cb.currentData() or 24.0
        shutter    = self._shutter_slider.value() / 100.0

        return OfflineRenderConfig(
            output_path    = out_path,
            format         = info.get("ext", "mp4"),
            width          = self._width_sb.value(),
            height         = self._height_sb.value(),
            fps            = fps,
            duration       = self._dur_sb.value(),
            start_time     = self._start_sb.value(),
            taa_enabled    = self._taa_cb.isChecked(),
            taa_samples    = self._taa_samples_cb.currentData() or 8,
            taa_jitter_radius = self._jitter_sb.value(),
            mb_enabled     = self._mb_cb.isChecked(),
            mb_samples     = self._mb_samples_cb.currentData() or 8,
            mb_shutter     = shutter,
            video_codec    = info.get("vcodec") or "libx264",
            crf            = self._crf_slider.value(),
            pixel_format   = info.get("pix_fmt") or "yuv420p",
            ffmpeg_extra   = list(info.get("extra", [])),
            dcp_title      = self._dcp_title_le.text().strip() or "OpenShader",
            dcp_issuer     = self._dcp_issuer_le.text().strip() or "OpenShader v6",
            audio_path     = self._audio_path if self._audio_cb.isChecked() and self._audio_cb.isEnabled() else None,
            threads        = self._threads_sb.value(),
            use_exr        = self._exr_cb.isChecked(),
        )

    def _start_render(self, cfg: OfflineRenderConfig, render_fn: Callable):
        self._progress = OfflineRenderProgress()
        engine = OfflineRenderEngine(cfg, render_fn, self._progress)

        self._btn_render.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_close.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Initialisation…")

        self._engine_thread = engine.start()
        self._poll_timer.start()

    # ── Annulation ────────────────────────────────────────────────────────────

    def _do_cancel(self):
        if self._progress:
            self._progress.cancel()

    # ── Polling progression ───────────────────────────────────────────────────

    def _poll_progress(self):
        p = self._progress
        if p is None:
            return

        frac = p.fraction
        self._progress_bar.setValue(int(frac * 1000))
        self._progress_label.setText(p.phase or "…")

        eta = p.eta_s
        if eta > 0:
            self._eta_label.setText(
                f"ETA {int(eta//60)}min {int(eta%60)}s" if eta >= 60 else f"ETA {eta:.0f}s")
        else:
            self._eta_label.setText("")

        if p.is_done:
            self._poll_timer.stop()
            self._btn_render.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._btn_close.setEnabled(True)

            if p.error:
                if "annulé" not in p.error.lower():
                    QMessageBox.critical(self, "Erreur de rendu", p.error)
                self._progress_label.setText(f"⚠ {p.error}")
            else:
                out = p.output
                size_info = ""
                if os.path.isfile(out):
                    size_info = f"  ({os.path.getsize(out)/1024/1024:.1f} Mo)"
                elif os.path.isdir(out):
                    n = len([f for f in os.listdir(out) if not f.startswith(".")])
                    size_info = f"  ({n} fichiers)"

                self._progress_label.setText(f"✓ Terminé{size_info}")
                self._progress_bar.setValue(1000)
                self._progress_bar.setStyleSheet(
                    "QProgressBar::chunk { background: #40d090; border-radius: 3px; }")

                QMessageBox.information(self, "Rendu terminé",
                    f"✓ Rendu offline terminé !\n\n{out}{size_info}")

                if self._open_dir_cb.isChecked():
                    target = out if os.path.isdir(out) else os.path.dirname(out)
                    try:
                        if sys.platform == "win32":
                            subprocess.Popen(["explorer", "/select,", out])
                        elif sys.platform == "darwin":
                            subprocess.Popen(["open", "-R", out])
                        else:
                            subprocess.Popen(["xdg-open", target])
                    except OSError:
                        pass

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._progress and not self._progress.is_done:
            self._progress.cancel()
        self._poll_timer.stop()
        super().closeEvent(event)
