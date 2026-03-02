"""
export_dialog.py
----------------
v2.3 — Dialog d'export vidéo haute qualité.

Codecs supportés :
  H.264  (libx264)   — MP4/MKV, universel
  H.265  (libx265)   — MP4/MKV, meilleur taux compression
  ProRes (prores_ks) — MOV, mastering post-prod
  VP9    (libvpx-vp9)— WebM, web sans perte approx.
  AV1    (libaom-av1)— WebM/MKV, nouvelle génération
  GIF    — GIF animé palettisé Floyd-Steinberg
  WebP   — WebP animé (lossy/lossless)

Fonctionnalités :
  - Supersampling 2× puis downscale Lanczos (SSAA)
  - Audio muxé automatiquement si disponible
  - Barre de progression en temps réel (QThread worker)
  - Estimation ETA
  - Aperçu des options FFmpeg générées
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QCheckBox, QPushButton, QProgressBar,
    QDialogButtonBox, QFileDialog, QMessageBox, QTextEdit,
    QTabWidget, QWidget, QSizePolicy,
)

from .logger import get_logger

log = get_logger(__name__)


# ── Codec presets ─────────────────────────────────────────────────────────────

CODEC_PRESETS = {
    "H.264 — MP4 (universel)": {
        "ext": "mp4", "vcodec": "libx264", "container": "mp4",
        "pix_fmt": "yuv420p", "extra": ["-preset", "slow", "-movflags", "+faststart"],
        "acodec": "aac", "ab": "192k", "crf_range": (0, 51), "crf_default": 18,
        "desc": "Compatible partout. Qualité professionnelle à CRF 18.",
    },
    "H.265 — MP4 (haute compression)": {
        "ext": "mp4", "vcodec": "libx265", "container": "mp4",
        "pix_fmt": "yuv420p", "extra": ["-preset", "slow", "-tag:v", "hvc1"],
        "acodec": "aac", "ab": "192k", "crf_range": (0, 51), "crf_default": 20,
        "desc": "~50% plus léger que H.264 à qualité égale. Requis macOS/iOS.",
    },
    "ProRes 4444 — MOV (master)": {
        "ext": "mov", "vcodec": "prores_ks", "container": "mov",
        "pix_fmt": "yuva444p10le",
        "extra": ["-profile:v", "4", "-vendor", "apl0", "-bits_per_mb", "8000"],
        "acodec": "pcm_s24le", "ab": None, "crf_range": None, "crf_default": None,
        "desc": "Qualité mastering avec canal alpha. Fichiers volumineux.",
    },
    "VP9 — WebM (web)": {
        "ext": "webm", "vcodec": "libvpx-vp9", "container": "webm",
        "pix_fmt": "yuv420p", "extra": ["-row-mt", "1", "-tile-columns", "2"],
        "acodec": "libopus", "ab": "128k", "crf_range": (0, 63), "crf_default": 28,
        "desc": "Optimal pour le web. Bonne transparence via yuva420p.",
    },
    "AV1 — WebM (nouvelle génération)": {
        "ext": "webm", "vcodec": "libaom-av1", "container": "webm",
        "pix_fmt": "yuv420p",
        "extra": ["-cpu-used", "4", "-row-mt", "1"],
        "acodec": "libopus", "ab": "128k", "crf_range": (0, 63), "crf_default": 30,
        "desc": "Meilleure compression. Encodage lent — CPU-intensif.",
    },
    "GIF animé": {
        "ext": "gif", "vcodec": "gif", "container": "gif",
        "pix_fmt": None, "extra": [], "acodec": None, "ab": None,
        "crf_range": None, "crf_default": None,
        "desc": "GIF 256 couleurs avec dithering Floyd-Steinberg.",
    },
    "WebP animé": {
        "ext": "webp", "vcodec": "libwebp_anim", "container": "webp",
        "pix_fmt": "yuva420p", "extra": ["-loop", "0", "-lossless", "0"],
        "acodec": None, "ab": None, "crf_range": (0, 100), "crf_default": 80,
        "desc": "WebP animé. Qualité = 80 par défaut (0=meilleure compression).",
    },
}

RESOLUTION_PRESETS = {
    "Actuelle (viewport)": None,
    "720p  (1280×720)":   (1280, 720),
    "1080p (1920×1080)":  (1920, 1080),
    "1440p (2560×1440)":  (2560, 1440),
    "4K    (3840×2160)":  (3840, 2160),
    "Carré 1080 (1080×1080)": (1080, 1080),
    "Personnalisée…": "custom",
}


# ── Worker Thread ─────────────────────────────────────────────────────────────

class ExportWorker(QObject):
    """Worker lancé dans un QThread pour le rendu + encodage FFmpeg."""

    progress    = pyqtSignal(float, str)   # (0‒1, message)
    finished    = pyqtSignal(str)          # chemin du fichier final
    error       = pyqtSignal(str)          # message d'erreur

    def __init__(self, params: dict, render_fn, audio_path: str | None):
        super().__init__()
        self._params     = params
        self._render_fn  = render_fn   # callable(t: float) -> QImage
        self._audio_path = audio_path
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        p          = self._params
        out_path   = p["out_path"]
        codec_info = p["codec_info"]
        width      = p["width"]
        height     = p["height"]
        fps        = p["fps"]
        duration   = p["duration"]
        crf        = p.get("crf")
        ssaa       = p.get("ssaa", False)
        with_audio = p.get("with_audio", True) and self._audio_path

        render_w = width * 2 if ssaa else width
        render_h = height * 2 if ssaa else height

        total_frames = max(1, int(duration * fps))
        ffmpeg       = shutil.which("ffmpeg")

        tmp_dir = tempfile.mkdtemp(prefix="openshader_v23_")
        t0 = time.perf_counter()

        try:
            # ── Phase 1 : rendu frames ───────────────────────────────────────
            for i in range(total_frames):
                if self._cancelled:
                    self.error.emit("Export annulé.")
                    return

                t = i / fps
                img = self._render_fn(t, render_w, render_h)

                frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
                img.save(frame_path)

                elapsed = time.perf_counter() - t0
                pct_raw = (i + 1) / total_frames
                pct     = pct_raw * 0.70   # 70% du total pour le rendu
                eta     = elapsed / max(pct_raw, 1e-6) * (1.0 - pct_raw)
                self.progress.emit(pct, f"Rendu {i+1}/{total_frames}  ETA {eta:.0f}s")

            if self._cancelled:
                self.error.emit("Export annulé.")
                return

            # ── Phase 2 : encodage FFmpeg ────────────────────────────────────
            self.progress.emit(0.72, "Encodage FFmpeg…")
            ext        = codec_info["ext"]
            vcodec     = codec_info["vcodec"]
            pix_fmt    = codec_info["pix_fmt"]
            extra      = codec_info["extra"]
            acodec     = codec_info.get("acodec")
            ab         = codec_info.get("ab")
            input_pat  = os.path.join(tmp_dir, "frame_%06d.png")

            if ext == "gif":
                out_path = self._encode_gif(ffmpeg, input_pat, out_path,
                                            width, height, fps)
            elif ext == "webp":
                out_path = self._encode_webp(ffmpeg, input_pat, out_path,
                                             width, height, fps, crf or 80)
            else:
                cmd = [ffmpeg, "-y",
                       "-framerate", str(fps),
                       "-i", input_pat]

                # Supersampling downscale
                vf_parts = []
                if ssaa:
                    vf_parts.append(f"scale={width}:{height}:flags=lanczos")

                # Audio
                if with_audio and self._audio_path and acodec:
                    cmd += ["-i", self._audio_path, "-t", str(duration)]

                # Codec vidéo
                cmd += ["-c:v", vcodec]
                if pix_fmt:
                    cmd += ["-pix_fmt", pix_fmt]
                if crf is not None and codec_info["crf_range"]:
                    if vcodec in ("libvpx-vp9", "libaom-av1"):
                        cmd += ["-crf", str(crf), "-b:v", "0"]
                    else:
                        cmd += ["-crf", str(crf)]
                if vf_parts:
                    cmd += ["-vf", ",".join(vf_parts)]
                cmd += extra

                # Audio
                if with_audio and self._audio_path and acodec:
                    cmd += ["-c:a", acodec]
                    if ab:
                        cmd += ["-b:a", ab]
                elif acodec is None or not with_audio:
                    cmd += ["-an"]

                cmd.append(out_path)

                self.progress.emit(0.75, "Encodage en cours…")
                result = subprocess.run(cmd, capture_output=True, timeout=1200)
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace")
                    raise RuntimeError(f"FFmpeg erreur :\n{err[-1000:]}")

            elapsed = time.perf_counter() - t0
            self.progress.emit(1.0, f"✓ Terminé en {elapsed:.1f}s")
            self.finished.emit(out_path)

        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.error.emit(str(exc))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _encode_gif(self, ffmpeg, input_pat, out_path, w, h, fps):
        tmp_palette = out_path + ".palette.png"
        try:
            # Passe 1 : palette
            r1 = subprocess.run([
                ffmpeg, "-y", "-framerate", str(fps), "-i", input_pat,
                "-vf", f"scale={w}:{h}:flags=lanczos,palettegen=max_colors=256:reserve_transparent=0",
                tmp_palette,
            ], capture_output=True, timeout=300)
            if r1.returncode != 0:
                raise RuntimeError(r1.stderr.decode(errors="replace"))

            self.progress.emit(0.85, "Assemblage GIF (dithering Floyd-Steinberg)…")

            # Passe 2 : GIF avec dithering
            r2 = subprocess.run([
                ffmpeg, "-y", "-framerate", str(fps), "-i", input_pat,
                "-i", tmp_palette,
                "-lavfi",
                f"scale={w}:{h}:flags=lanczos[x];[x][1:v]paletteuse=dither=floyd_steinberg",
                "-loop", "0",
                out_path,
            ], capture_output=True, timeout=600)
            if r2.returncode != 0:
                raise RuntimeError(r2.stderr.decode(errors="replace"))
        finally:
            if os.path.isfile(tmp_palette):
                os.remove(tmp_palette)
        return out_path

    def _encode_webp(self, ffmpeg, input_pat, out_path, w, h, fps, quality):
        r = subprocess.run([
            ffmpeg, "-y",
            "-framerate", str(fps), "-i", input_pat,
            "-c:v", "libwebp_anim",
            "-vf", f"scale={w}:{h}:flags=lanczos",
            "-quality", str(quality),
            "-loop", "0",
            out_path,
        ], capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace"))
        return out_path


# ── Main Dialog ───────────────────────────────────────────────────────────────

class ExportDialog(QDialog):
    """
    Dialog d'export vidéo haute qualité v2.3.

    Paramètres émis via export_requested(dict) une fois validés,
    ou l'export peut être lancé directement via start_export().
    """

    def __init__(self, parent=None, *, viewport_w=800, viewport_h=450,
                 timeline_duration=10.0, audio_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export vidéo — v2.3")
        self.setMinimumWidth(560)
        self.resize(600, 580)

        self._viewport_w  = viewport_w
        self._viewport_h  = viewport_h
        self._duration    = timeline_duration
        self._audio_path  = audio_path
        self._worker      = None
        self._thread      = None

        self._build_ui()
        self._on_codec_changed()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._tab_video(),   "🎬 Vidéo")
        tabs.addTab(self._tab_audio(),   "🔊 Audio")
        tabs.addTab(self._tab_advanced(),"⚙ Avancé")
        root.addWidget(tabs)

        # Barre de progression
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        root.addWidget(self._progress_bar)
        root.addWidget(self._progress_label)

        # Boutons
        btn_box = QHBoxLayout()
        self._btn_preview = QPushButton("🔍 Aperçu commande…")
        self._btn_preview.clicked.connect(self._show_ffmpeg_preview)
        btn_box.addWidget(self._btn_preview)
        btn_box.addStretch()
        self._btn_cancel_export = QPushButton("⏹ Annuler l'export")
        self._btn_cancel_export.setVisible(False)
        self._btn_cancel_export.clicked.connect(self._cancel_export)
        btn_box.addWidget(self._btn_cancel_export)
        self._btn_export = QPushButton("⬇ Exporter…")
        self._btn_export.setDefault(True)
        self._btn_export.clicked.connect(self._start_export_dialog)
        btn_box.addWidget(self._btn_export)
        self._btn_close = QPushButton("Fermer")
        self._btn_close.clicked.connect(self.accept)
        btn_box.addWidget(self._btn_close)
        root.addLayout(btn_box)

    def _tab_video(self) -> QWidget:
        w   = QWidget()
        lay = QFormLayout(w)
        lay.setSpacing(8)

        # Codec
        self._codec_cb = QComboBox()
        self._codec_cb.addItems(list(CODEC_PRESETS.keys()))
        self._codec_cb.currentTextChanged.connect(self._on_codec_changed)
        lay.addRow("Codec :", self._codec_cb)

        self._codec_desc = QLabel()
        self._codec_desc.setWordWrap(True)
        self._codec_desc.setStyleSheet("color: #888; font-size: 11px;")
        lay.addRow("", self._codec_desc)

        lay.addRow(self._sep())

        # Résolution
        self._res_cb = QComboBox()
        self._res_cb.addItems(list(RESOLUTION_PRESETS.keys()))
        self._res_cb.currentTextChanged.connect(self._on_res_changed)
        lay.addRow("Résolution :", self._res_cb)

        res_row = QHBoxLayout()
        self._width_sb  = QSpinBox(); self._width_sb.setRange(16, 7680); self._width_sb.setValue(self._viewport_w)
        self._height_sb = QSpinBox(); self._height_sb.setRange(16, 4320); self._height_sb.setValue(self._viewport_h)
        res_row.addWidget(self._width_sb)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self._height_sb)
        res_row.addStretch()
        lay.addRow("Taille :", res_row)

        # FPS
        self._fps_cb = QComboBox()
        for fps in ["24", "25", "30", "50", "60", "120"]:
            self._fps_cb.addItem(f"{fps} fps", float(fps))
        self._fps_cb.setCurrentText("60 fps")
        lay.addRow("Fréquence :", self._fps_cb)

        # Durée
        self._dur_sb = QDoubleSpinBox()
        self._dur_sb.setRange(0.1, 3600.0)
        self._dur_sb.setValue(self._duration)
        self._dur_sb.setSuffix(" s")
        lay.addRow("Durée :", self._dur_sb)

        lay.addRow(self._sep())

        # CRF
        crf_row = QHBoxLayout()
        self._crf_slider = QSlider(Qt.Orientation.Horizontal)
        self._crf_slider.setRange(0, 51)
        self._crf_slider.setValue(18)
        self._crf_slider.valueChanged.connect(
            lambda v: self._crf_label.setText(f"{v}  (0=sans perte, 51=min)"))
        self._crf_label  = QLabel("18  (0=sans perte, 51=min)")
        crf_row.addWidget(self._crf_slider)
        crf_row.addWidget(self._crf_label)
        lay.addRow("Qualité (CRF) :", crf_row)

        # Supersampling
        self._ssaa_cb = QCheckBox("Supersampling 2× (rendu 2× puis downscale Lanczos)")
        lay.addRow("SSAA :", self._ssaa_cb)

        return w

    def _tab_audio(self) -> QWidget:
        w   = QWidget()
        lay = QFormLayout(w)
        lay.setSpacing(8)

        self._audio_cb = QCheckBox("Inclure l'audio dans la vidéo")
        self._audio_cb.setChecked(True)
        self._audio_cb.setEnabled(bool(self._audio_path))
        lay.addRow("Audio :", self._audio_cb)

        audio_status = QLabel(
            f"Fichier audio : {os.path.basename(self._audio_path)}"
            if self._audio_path else "Aucun fichier audio chargé."
        )
        audio_status.setStyleSheet("color: #888; font-size: 11px;")
        lay.addRow("", audio_status)

        lay.addRow(self._sep())
        note = QLabel(
            "Le codec audio est choisi automatiquement selon le format :\n"
            "  MP4/MOV → AAC 192 kbps\n"
            "  WebM    → Opus 128 kbps\n"
            "  ProRes  → PCM 24 bits (sans perte)\n"
            "  GIF/WebP → pas d'audio"
        )
        note.setStyleSheet("color: #666; font-size: 11px;")
        lay.addRow(note)

        return w

    def _tab_advanced(self) -> QWidget:
        w   = QWidget()
        lay = QFormLayout(w)
        lay.setSpacing(8)

        self._threads_sb = QSpinBox()
        self._threads_sb.setRange(0, 64)
        self._threads_sb.setValue(0)
        self._threads_sb.setSpecialValueText("auto")
        lay.addRow("Threads FFmpeg :", self._threads_sb)

        self._hwaccel_cb = QCheckBox("Activer hwaccel si disponible (NVENC / VideoToolbox)")
        lay.addRow("Accélération HW :", self._hwaccel_cb)

        lay.addRow(self._sep())

        self._open_dir_cb = QCheckBox("Ouvrir le dossier après export")
        self._open_dir_cb.setChecked(True)
        lay.addRow("Post-export :", self._open_dir_cb)

        return w

    def _sep(self) -> QWidget:
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #444;")
        return line

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_codec_changed(self):
        name = self._codec_cb.currentText()
        info = CODEC_PRESETS.get(name, {})
        self._codec_desc.setText(info.get("desc", ""))

        crf_range   = info.get("crf_range")
        crf_default = info.get("crf_default")
        has_crf     = crf_range is not None

        self._crf_slider.setEnabled(has_crf)
        if has_crf:
            lo, hi = crf_range
            self._crf_slider.setRange(lo, hi)
            if crf_default is not None:
                self._crf_slider.setValue(crf_default)
                if name.startswith("WebP"):
                    self._crf_label.setText(f"{crf_default}  (100=meilleure qualité, 0=min)")
                else:
                    self._crf_label.setText(f"{crf_default}  (0=sans perte, {hi}=min)")
        else:
            self._crf_label.setText("n/a")

    def _on_res_changed(self, text: str):
        val = RESOLUTION_PRESETS.get(text)
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

    # ── Export params ─────────────────────────────────────────────────────────

    def _build_params(self, out_path: str) -> dict:
        codec_name = self._codec_cb.currentText()
        fps_idx    = self._fps_cb.currentIndex()
        fps        = self._fps_cb.itemData(fps_idx) or 60.0
        return {
            "out_path":   out_path,
            "codec_name": codec_name,
            "codec_info": CODEC_PRESETS[codec_name],
            "width":      self._width_sb.value(),
            "height":     self._height_sb.value(),
            "fps":        fps,
            "duration":   self._dur_sb.value(),
            "crf":        self._crf_slider.value(),
            "ssaa":       self._ssaa_cb.isChecked(),
            "with_audio": self._audio_cb.isChecked() if self._audio_cb.isEnabled() else False,
            "threads":    self._threads_sb.value(),
            "hwaccel":    self._hwaccel_cb.isChecked(),
            "open_dir":   self._open_dir_cb.isChecked(),
        }

    def _show_ffmpeg_preview(self):
        """Affiche la commande FFmpeg qui sera exécutée (à titre informatif)."""
        codec_name = self._codec_cb.currentText()
        info       = CODEC_PRESETS[codec_name]
        fps        = self._fps_cb.itemData(self._fps_cb.currentIndex()) or 60.0
        w, h       = self._width_sb.value(), self._height_sb.value()
        crf        = self._crf_slider.value()
        ssaa       = self._ssaa_cb.isChecked()
        rw = w * 2 if ssaa else w
        rh = h * 2 if ssaa else h

        lines = [
            f"# Résolution rendu : {rw}×{rh}  (SSAA={'oui' if ssaa else 'non'})",
            f"# Résolution finale : {w}×{h}",
            "",
            "ffmpeg -y \\",
            f"  -framerate {fps:.0f} \\",
            "  -i frame_%06d.png \\",
        ]
        if info.get("acodec") and self._audio_cb.isChecked():
            lines.append("  -i <audio_file> \\")
        lines.append(f"  -c:v {info['vcodec']} \\")
        if info.get("pix_fmt"):
            lines.append(f"  -pix_fmt {info['pix_fmt']} \\")
        if info.get("crf_range"):
            lines.append(f"  -crf {crf} \\")
        for e in info.get("extra", []):
            lines.append(f"  {e} \\")
        if ssaa:
            lines.append(f"  -vf scale={w}:{h}:flags=lanczos \\")
        if info.get("acodec"):
            lines.append(f"  -c:a {info['acodec']} \\")
            if info.get("ab"):
                lines.append(f"  -b:a {info['ab']} \\")
        lines.append(f"  output.{info['ext']}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Aperçu commande FFmpeg")
        dlg.resize(560, 320)
        vl  = QVBoxLayout(dlg)
        te  = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText("\n".join(lines))
        te.setStyleSheet("font-family: monospace; font-size: 12px;")
        vl.addWidget(te)
        ok = QPushButton("Fermer"); ok.clicked.connect(dlg.accept)
        vl.addWidget(ok)
        dlg.exec()

    # ── Export launch ─────────────────────────────────────────────────────────

    def _start_export_dialog(self):
        """Ouvre le dialog de sauvegarde puis lance le worker."""
        codec_name = self._codec_cb.currentText()
        ext        = CODEC_PRESETS[codec_name]["ext"]
        filt       = f"*.{ext}"
        default    = os.path.expanduser(f"~/export.{ext}")
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Exporter en {ext.upper()}", default, filt)
        if not out_path:
            return
        if not out_path.lower().endswith(f".{ext}"):
            out_path += f".{ext}"

        # Vérifie FFmpeg
        if not shutil.which("ffmpeg"):
            QMessageBox.critical(self, "FFmpeg manquant",
                "FFmpeg est introuvable dans le PATH.\n"
                "Téléchargez-le sur https://ffmpeg.org/download.html")
            return

        params = self._build_params(out_path)

        # Demande au parent de fournir la fonction de rendu
        parent = self.parent()
        if not hasattr(parent, "_export_render_frame"):
            QMessageBox.critical(self, "Erreur", "La fenêtre principale ne supporte pas le rendu export.")
            return

        self._launch_worker(params, parent._export_render_frame)

    def _launch_worker(self, params: dict, render_fn):
        """Lance le thread worker d'export."""
        self._btn_export.setEnabled(False)
        self._btn_close.setEnabled(False)
        self._btn_cancel_export.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)
        self._progress_bar.setValue(0)

        self._worker = ExportWorker(params, render_fn, self._audio_path)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._params_cache = params
        self._thread.start()

    def _cancel_export(self):
        if self._worker:
            self._worker.cancel()

    def _on_worker_progress(self, pct: float, msg: str):
        self._progress_bar.setValue(int(pct * 1000))
        self._progress_label.setText(msg)

    def _on_worker_finished(self, out_path: str):
        self._reset_ui()
        size_mb = os.path.getsize(out_path) / 1024 / 1024 if os.path.isfile(out_path) else 0
        QMessageBox.information(self, "Export terminé",
            f"✓ Fichier généré :\n{out_path}\n({size_mb:.1f} Mo)")

        if self._params_cache.get("open_dir"):
            target = os.path.dirname(out_path)
            import sys as _sys
            if _sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", out_path])
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", "-R", out_path])
            else:
                subprocess.Popen(["xdg-open", target])

    def _on_worker_error(self, msg: str):
        self._reset_ui()
        if "annulé" not in msg.lower():
            QMessageBox.critical(self, "Erreur d'export", msg)

    def _reset_ui(self):
        self._btn_export.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._btn_cancel_export.setVisible(False)
        self._progress_bar.setValue(0)
        self._worker = None
        self._thread = None

    def closeEvent(self, event):
        if self._worker:
            self._worker.cancel()
        super().closeEvent(event)
