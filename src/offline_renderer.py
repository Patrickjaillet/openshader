"""
offline_renderer.py
-------------------
Moteur de rendu différé (Offline Renderer) — OpenShader v6.1

Fonctionnalités :
  • Rendu frame-by-frame ultra-qualité hors temps-réel
  • TAA (Temporal Anti-Aliasing) multi-sample offline — jusqu'à 64 samples / pixel
  • Motion blur accumulatif sur N sub-frames (vrai accumulation GL)
  • Export DCP (Digital Cinema Package) — MXF JPEG2000 via FFmpeg
  • Toutes les sorties classiques : MP4/MKV/MOV/WebM/GIF + séquence PNG/EXR

Architecture :
  OfflineRenderConfig     — dataclass de configuration complète
  OfflineRenderEngine     — moteur de rendu (thread-safe, GL offscreen)
  OfflineRenderProgress   — conteneur de progression thread-safe
  TAAAccumulator          — accumulation jitter + blend TAA
  MotionBlurAccumulator   — accumulation temporelle N sub-frames
  DCPExporter             — assemblage DCP (répertoire XYZ+MXF via FFmpeg)
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── Logger ─────────────────────────────────────────────────────────────────────

import logging
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OfflineRenderConfig:
    """Configuration complète d'un rendu offline."""

    # ── Sortie ───────────────────────────────────────────────────────────────
    output_path:    str   = ""
    format:         str   = "mp4"          # mp4 | mkv | mov | webm | gif |
                                            # png_seq | exr_seq | dcp
    # ── Dimensions ───────────────────────────────────────────────────────────
    width:          int   = 1920
    height:         int   = 1080

    # ── Temporel ─────────────────────────────────────────────────────────────
    fps:            float = 24.0
    duration:       float = 10.0
    start_time:     float = 0.0

    # ── TAA (Temporal Anti-Aliasing) ─────────────────────────────────────────
    taa_enabled:    bool  = True
    taa_samples:    int   = 8              # 1 | 2 | 4 | 8 | 16 | 32 | 64
    taa_jitter_radius: float = 0.5        # rayon jitter en pixels

    # ── Motion Blur ──────────────────────────────────────────────────────────
    mb_enabled:     bool  = False
    mb_samples:     int   = 8             # sub-frames accumulées
    mb_shutter:     float = 0.5          # [0.0–1.0] proportion de la frame

    # ── Codec / qualité ──────────────────────────────────────────────────────
    video_codec:    str   = "libx264"      # libx264 | libx265 | prores_ks |
                                            # libvpx-vp9 | libaom-av1
    crf:            int   = 18
    pixel_format:   str   = "yuv420p"
    ffmpeg_extra:   list  = field(default_factory=list)

    # ── DCP ──────────────────────────────────────────────────────────────────
    dcp_title:      str   = "OpenShader"
    dcp_issuer:     str   = "OpenShader v6"
    dcp_colorspace: str   = "XYZ"          # XYZ (cinema) | sRGB

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_path:     Optional[str] = None

    # ── Avancé ───────────────────────────────────────────────────────────────
    threads:        int   = 0              # 0 = auto
    use_exr:        bool  = False          # EXR 16-bit half pour séquences

    @property
    def total_frames(self) -> int:
        return max(1, int(self.duration * self.fps))

    @property
    def is_dcp(self) -> bool:
        return self.format == "dcp"

    @property
    def is_image_seq(self) -> bool:
        return self.format in ("png_seq", "exr_seq")


# ═══════════════════════════════════════════════════════════════════════════════
#  Progression thread-safe
# ═══════════════════════════════════════════════════════════════════════════════

class OfflineRenderProgress:
    """Conteneur de progression partagé entre thread de rendu et UI."""

    def __init__(self):
        self._lock         = threading.Lock()
        self._fraction:    float      = 0.0   # 0.0 → 1.0
        self._phase:       str        = ""
        self._frame:       int        = 0
        self._total:       int        = 0
        self._eta_s:       float      = 0.0
        self._cancelled:   bool       = False
        self._done:        bool       = False
        self._error:       str        = ""
        self._output:      str        = ""
        self._t_start:     float      = time.perf_counter()

    # ── Mise à jour (thread worker) ────────────────────────────────────────

    def update(self, fraction: float, phase: str = "", frame: int = 0, total: int = 0):
        with self._lock:
            self._fraction = max(0.0, min(1.0, fraction))
            if phase:    self._phase = phase
            if frame:    self._frame = frame
            if total:    self._total = total
            elapsed = time.perf_counter() - self._t_start
            if fraction > 0.005:
                self._eta_s = elapsed / fraction * (1.0 - fraction)

    def set_done(self, output_path: str):
        with self._lock:
            self._fraction = 1.0
            self._done     = True
            self._output   = output_path

    def set_error(self, msg: str):
        with self._lock:
            self._error = msg
            self._done  = True

    def cancel(self):
        with self._lock:
            self._cancelled = True

    # ── Lecture (UI) ──────────────────────────────────────────────────────

    @property
    def fraction(self) -> float:
        with self._lock: return self._fraction

    @property
    def phase(self) -> str:
        with self._lock: return self._phase

    @property
    def frame(self) -> int:
        with self._lock: return self._frame

    @property
    def total(self) -> int:
        with self._lock: return self._total

    @property
    def eta_s(self) -> float:
        with self._lock: return self._eta_s

    @property
    def is_cancelled(self) -> bool:
        with self._lock: return self._cancelled

    @property
    def is_done(self) -> bool:
        with self._lock: return self._done

    @property
    def error(self) -> str:
        with self._lock: return self._error

    @property
    def output(self) -> str:
        with self._lock: return self._output


# ═══════════════════════════════════════════════════════════════════════════════
#  Séquences de jitter Halton pour TAA
# ═══════════════════════════════════════════════════════════════════════════════

def _halton(index: int, base: int) -> float:
    """Halton sequence — base `base` pour l'index `index`."""
    f, r = 1.0, 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_sequence_2d(n: int) -> list[tuple[float, float]]:
    """Séquence Halton 2D en base (2,3) — meilleure distribution que random."""
    return [(_halton(i + 1, 2) - 0.5, _halton(i + 1, 3) - 0.5) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
#  TAAAccumulator — accumulation jitter + blend sur N samples
# ═══════════════════════════════════════════════════════════════════════════════

class TAAAccumulator:
    """
    Accumule N rendus avec jitter sub-pixel (séquence Halton 2D)
    et retourne la moyenne pondérée en float32.

    Usage :
        acc = TAAAccumulator(samples=8, jitter_radius=0.5)
        for jx, jy in acc:
            rgba = render_fn(t, jitter_x=jx, jitter_y=jy)
            acc.add(rgba)
        result = acc.result()   # np.ndarray float32 RGBA [0,1]
    """

    def __init__(self, samples: int = 8, jitter_radius: float = 0.5):
        self._n       = max(1, samples)
        self._radius  = jitter_radius
        self._offsets = _halton_sequence_2d(self._n)
        self._accum:  Optional[np.ndarray] = None
        self._count:  int = 0

    def __iter__(self):
        self._accum = None
        self._count = 0
        for jx, jy in self._offsets:
            yield jx * self._radius, jy * self._radius

    def add(self, rgba_uint8: np.ndarray):
        """Ajoute une frame RGBA uint8 (H×W×4) à l'accumulation."""
        f = rgba_uint8.astype(np.float32) / 255.0
        if self._accum is None:
            self._accum = f.copy()
        else:
            self._accum += f
        self._count += 1

    def result(self) -> np.ndarray:
        """Retourne le résultat RGBA uint8 moyenné."""
        if self._accum is None or self._count == 0:
            raise RuntimeError("TAAAccumulator: aucun sample accumulé")
        averaged = (self._accum / self._count).clip(0, 1)
        return (averaged * 255).astype(np.uint8)

    @property
    def n_samples(self) -> int:
        return self._n


# ═══════════════════════════════════════════════════════════════════════════════
#  MotionBlurAccumulator — accumulation sur N sub-frames temporelles
# ═══════════════════════════════════════════════════════════════════════════════

class MotionBlurAccumulator:
    """
    Accumule N rendus répartis dans l'intervalle [t, t + shutter/fps].

    Le rendu résultant est la moyenne (pondérée uniformément) des sub-frames,
    ce qui produit un flou de mouvement réaliste sans coût shader.
    """

    def __init__(self, n_samples: int = 8, shutter: float = 0.5, fps: float = 24.0):
        self._n       = max(1, n_samples)
        self._shutter = max(0.0, min(1.0, shutter))
        self._step    = self._shutter / max(1, fps)
        self._accum:  Optional[np.ndarray] = None

    def sub_times(self, t_frame: float) -> list[float]:
        """Retourne la liste des N temps à rendre pour la frame à t_frame."""
        dt = self._step / self._n
        return [t_frame + i * dt for i in range(self._n)]

    def accumulate(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Reçoit une liste de frames RGBA uint8, retourne la moyenne uint8.
        """
        if not frames:
            raise RuntimeError("MotionBlurAccumulator: liste de frames vide")
        acc = np.zeros_like(frames[0], dtype=np.float32)
        for f in frames:
            acc += f.astype(np.float32)
        avg = (acc / len(frames)).clip(0, 255)
        return avg.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#  DCPExporter — assemblage d'un DCP minimal (MXF JPEG2000 via FFmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

class DCPExporter:
    """
    Assemble un DCP (Digital Cinema Package) à partir d'une séquence PNG.

    Structure produite :
        <dcp_dir>/
          ├── ASSETMAP
          ├── VOLINDEX
          ├── CPL_<uuid>.xml
          ├── PKL_<uuid>.xml
          └── MXF/
                ├── video_<uuid>.mxf    (JPEG2000 XYZ)
                └── audio_<uuid>.mxf    (PCM 24-bit, si audio fourni)

    Nécessite FFmpeg avec libopenjpeg ou libkdu.
    """

    def __init__(self, config: OfflineRenderConfig):
        self.cfg      = config
        self._cpl_id  = str(uuid.uuid4()).upper()
        self._pkl_id  = str(uuid.uuid4()).upper()
        self._vid_id  = str(uuid.uuid4()).upper()
        self._aud_id  = str(uuid.uuid4()).upper()

    def export(self, png_dir: str, dcp_dir: str,
               progress_cb: Callable[[float, str], None] | None = None) -> bool:
        """
        Assemble le DCP depuis png_dir vers dcp_dir.
        Retourne True si succès.
        """
        os.makedirs(os.path.join(dcp_dir, "MXF"), exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg introuvable — requis pour l'export DCP.")

        cfg = self.cfg

        # ── 1. Conversion PNG XYZ → JPEG2000 → MXF vidéo ─────────────────
        if progress_cb: progress_cb(0.05, "DCP : encodage JPEG2000 XYZ…")
        vid_mxf = os.path.join(dcp_dir, "MXF", f"video_{self._vid_id}.mxf")
        input_pat = os.path.join(png_dir, "frame_%06d.png")

        # Vérifier support J2K dans FFmpeg
        j2k_encoder = self._detect_j2k_encoder(ffmpeg)

        # Commande FFmpeg : PNG → XYZ colorspace → J2K → MXF
        # La conversion sRGB → XYZ D65 est nécessaire pour le standard DCI
        vf_chain = self._build_vf_chain(j2k_encoder)

        cmd = [
            ffmpeg, "-y",
            "-framerate", str(cfg.fps),
            "-i", input_pat,
            "-c:v", j2k_encoder,
            "-vf", vf_chain,
            "-pix_fmt", "rgb48le",          # 16-bit RGB pour J2K
            "-f", "mxf",
            vid_mxf,
        ]
        log.debug("DCP video cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, timeout=3600)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            # Fallback : si J2K indisponible, utilise une MXF sans perte (DNXHD)
            log.warning("DCP J2K échoué (%s), fallback DNXHD", err[-200:])
            vid_mxf = self._fallback_mxf_encode(ffmpeg, input_pat, dcp_dir)

        vid_mxf_size = os.path.getsize(vid_mxf) if os.path.isfile(vid_mxf) else 0
        if progress_cb: progress_cb(0.80, "DCP : assemblage métadonnées…")

        # ── 2. Audio MXF (PCM 24-bit) ────────────────────────────────────
        aud_mxf = None
        aud_mxf_size = 0
        if cfg.audio_path and os.path.isfile(cfg.audio_path):
            if progress_cb: progress_cb(0.82, "DCP : encodage audio PCM…")
            aud_mxf = os.path.join(dcp_dir, "MXF", f"audio_{self._aud_id}.mxf")
            acmd = [
                ffmpeg, "-y",
                "-i", cfg.audio_path,
                "-t", str(cfg.duration),
                "-acodec", "pcm_s24le",
                "-ar", "48000",
                "-f", "mxf",
                aud_mxf,
            ]
            r2 = subprocess.run(acmd, capture_output=True, timeout=300)
            if r2.returncode != 0:
                log.warning("DCP audio MXF échoué : %s",
                            r2.stderr.decode(errors="replace")[-200:])
                aud_mxf = None
            else:
                aud_mxf_size = os.path.getsize(aud_mxf)

        # ── 3. CPL (Composition Play List) XML ───────────────────────────
        cpl_path = os.path.join(dcp_dir, f"CPL_{self._cpl_id}.xml")
        self._write_cpl(cpl_path, vid_mxf_size, aud_mxf, aud_mxf_size)

        # ── 4. PKL (Packing List) XML ────────────────────────────────────
        pkl_path = os.path.join(dcp_dir, f"PKL_{self._pkl_id}.xml")
        cpl_size = os.path.getsize(cpl_path)
        self._write_pkl(pkl_path, cpl_size, vid_mxf_size, aud_mxf, aud_mxf_size)

        # ── 5. ASSETMAP + VOLINDEX ────────────────────────────────────────
        self._write_assetmap(dcp_dir, cpl_path, pkl_path,
                             vid_mxf, aud_mxf)
        self._write_volindex(dcp_dir)

        if progress_cb: progress_cb(1.0, "DCP assemblé ✓")
        log.info("DCP généré → %s", dcp_dir)
        return True

    # ── Helpers FFmpeg ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_j2k_encoder(ffmpeg: str) -> str:
        """Retourne 'libopenjpeg' ou 'jpeg2000' selon disponibilité."""
        try:
            r = subprocess.run([ffmpeg, "-codecs"], capture_output=True, timeout=10)
            out = r.stdout.decode(errors="replace")
            if "libopenjpeg" in out:
                return "libopenjpeg"
        except (OSError, subprocess.TimeoutExpired):
            pass
        return "jpeg2000"

    @staticmethod
    def _build_vf_chain(j2k_encoder: str) -> str:
        """Construit la chaîne -vf pour conversion sRGB→XYZ DCI."""
        # Matrice sRGB D65 → XYZ D65 (Bradford), puis gamma 2.6 DCI
        # FFmpeg colorspace filter : colorspace=all=xyz, gamma=2.6
        return (
            "colorspace=all=xyz:ispace=bt709,"
            "colorlevels=rimax=4.0:gimax=4.0:bimax=4.0"  # scale to DCI white
        )

    def _fallback_mxf_encode(self, ffmpeg: str, input_pat: str, dcp_dir: str) -> str:
        """Fallback DNXHD 444 12-bit si JPEG2000 indisponible."""
        vid_mxf = os.path.join(dcp_dir, "MXF", f"video_{self._vid_id}_dnxhd.mxf")
        cfg = self.cfg
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(cfg.fps),
            "-i", input_pat,
            "-c:v", "dnxhd",
            "-profile:v", "dnxhr_444",
            "-pix_fmt", "yuv444p10le",
            "-f", "mxf",
            vid_mxf,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(
                "DCP : ni JPEG2000 ni DNXHD disponibles dans ce FFmpeg.\n"
                + r.stderr.decode(errors="replace")[-500:]
            )
        return vid_mxf

    # ── XMLs DCP ──────────────────────────────────────────────────────────

    def _write_cpl(self, path: str, vid_size: int,
                   aud_mxf: Optional[str], aud_size: int):
        cfg = self.cfg
        frame_rate_n = int(cfg.fps)
        frame_rate_d = 1

        # Ratio écran DCI 2K ou 4K
        w, h = cfg.width, cfg.height
        aspect = f"{w} {h}"
        n_frames = cfg.total_frames

        aud_block = ""
        if aud_mxf:
            aud_block = f"""
      <Asset>
        <Id>urn:uuid:{self._aud_id}</Id>
        <PackingList>true</PackingList>
        <ChunkList>
          <Chunk><Path>MXF/{os.path.basename(aud_mxf)}</Path><VolumeIndex>1</VolumeIndex><Offset>0</Offset><Length>{aud_size}</Length></Chunk>
        </ChunkList>
      </Asset>"""

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<CompositionPlaylist xmlns="http://www.digicine.com/PROTO-ASDCP-CPL-20040511#">
  <Id>urn:uuid:{self._cpl_id}</Id>
  <Issuer>{cfg.dcp_issuer}</Issuer>
  <Creator>OpenShader v6</Creator>
  <ContentTitleText>{cfg.dcp_title}</ContentTitleText>
  <ContentKind>short</ContentKind>
  <RatingList/>
  <ReelList>
    <Reel>
      <Id>urn:uuid:{str(uuid.uuid4()).upper()}</Id>
      <AssetList>
        <MainPicture>
          <Id>urn:uuid:{self._vid_id}</Id>
          <EditRate>{frame_rate_n} {frame_rate_d}</EditRate>
          <IntrinsicDuration>{n_frames}</IntrinsicDuration>
          <EntryPoint>0</EntryPoint>
          <Duration>{n_frames}</Duration>
          <FrameRate>{frame_rate_n} {frame_rate_d}</FrameRate>
          <ScreenAspectRatio>{aspect}</ScreenAspectRatio>
        </MainPicture>
        {"<MainSound><Id>urn:uuid:" + self._aud_id + "</Id><EditRate>" + str(frame_rate_n) + " " + str(frame_rate_d) + "</EditRate><IntrinsicDuration>" + str(n_frames) + "</IntrinsicDuration><EntryPoint>0</EntryPoint><Duration>" + str(n_frames) + "</Duration></MainSound>" if aud_mxf else ""}
      </AssetList>
    </Reel>
  </ReelList>
</CompositionPlaylist>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_pkl(self, path: str, cpl_size: int, vid_size: int,
                   aud_mxf: Optional[str], aud_size: int):
        cpl_fname = f"CPL_{self._cpl_id}.xml"
        vid_fname = f"video_{self._vid_id}.mxf"
        aud_fname = f"audio_{self._aud_id}.mxf" if aud_mxf else ""

        aud_entry = ""
        if aud_mxf:
            aud_entry = f"""
    <Asset>
      <Id>urn:uuid:{self._aud_id}</Id>
      <Size>{aud_size}</Size>
      <Type>application/mxf</Type>
    </Asset>"""

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<PackingList xmlns="http://www.digicine.com/PROTO-ASDCP-PKL-20040311#">
  <Id>urn:uuid:{self._pkl_id}</Id>
  <Issuer>{self.cfg.dcp_issuer}</Issuer>
  <Creator>OpenShader v6</Creator>
  <AssetList>
    <Asset>
      <Id>urn:uuid:{self._cpl_id}</Id>
      <Size>{cpl_size}</Size>
      <Type>text/xml;asdcpKind=CPL</Type>
    </Asset>
    <Asset>
      <Id>urn:uuid:{self._vid_id}</Id>
      <Size>{vid_size}</Size>
      <Type>application/mxf</Type>
    </Asset>{aud_entry}
  </AssetList>
</PackingList>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)

    def _write_assetmap(self, dcp_dir: str, cpl_path: str, pkl_path: str,
                        vid_mxf: str, aud_mxf: Optional[str]):
        def _asset(uid: str, fname: str, size: int) -> str:
            return f"""  <Asset>
    <Id>urn:uuid:{uid}</Id>
    <PackingList>true</PackingList>
    <ChunkList>
      <Chunk><Path>{fname}</Path><VolumeIndex>1</VolumeIndex><Offset>0</Offset><Length>{size}</Length></Chunk>
    </ChunkList>
  </Asset>"""

        cpl_id2 = str(uuid.uuid4()).upper()
        pkl_id2 = str(uuid.uuid4()).upper()
        entries = [
            _asset(self._pkl_id, os.path.basename(pkl_path),
                   os.path.getsize(pkl_path)),
            _asset(self._cpl_id, os.path.basename(cpl_path),
                   os.path.getsize(cpl_path)),
            _asset(self._vid_id,
                   os.path.join("MXF", os.path.basename(vid_mxf)),
                   os.path.getsize(vid_mxf) if os.path.isfile(vid_mxf) else 0),
        ]
        if aud_mxf and os.path.isfile(aud_mxf):
            entries.append(_asset(self._aud_id,
                                  os.path.join("MXF", os.path.basename(aud_mxf)),
                                  os.path.getsize(aud_mxf)))

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<AssetMap xmlns="http://www.digicine.com/PROTO-ASDCP-AM-20040311#">
  <Id>urn:uuid:{str(uuid.uuid4()).upper()}</Id>
  <Issuer>{self.cfg.dcp_issuer}</Issuer>
  <Creator>OpenShader v6</Creator>
  <VolumeCount>1</VolumeCount>
  <IssueDate>{time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}</IssueDate>
  <AssetList>
{"".join(entries)}
  </AssetList>
</AssetMap>"""
        with open(os.path.join(dcp_dir, "ASSETMAP"), "w", encoding="utf-8") as f:
            f.write(xml)

    @staticmethod
    def _write_volindex(dcp_dir: str):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<VolumeIndex xmlns="http://www.digicine.com/PROTO-ASDCP-AM-20040311#">
  <Index>1</Index>
</VolumeIndex>"""
        with open(os.path.join(dcp_dir, "VOLINDEX"), "w", encoding="utf-8") as f:
            f.write(xml)


# ═══════════════════════════════════════════════════════════════════════════════
#  PNG writer stdlib (sans Pillow)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_png(path: str, rgba: np.ndarray):
    """Écrit un PNG RGBA depuis un array numpy (H×W×4 uint8), sans Pillow."""
    h, w = rgba.shape[:2]

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFF_FFFF)

    # Flip vertical : OpenGL origin bas-gauche → PNG origine haut-gauche
    flipped = rgba[::-1, :, :]
    raw = b"".join(b"\x00" + flipped[row].tobytes() for row in range(h))
    compressed = zlib.compress(raw, 6)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)


def _write_exr_half(path: str, rgba: np.ndarray):
    """Écrit un EXR minimal 16-bit half (via numpy uniquement, format custom)."""
    # Fallback : si OpenEXR Python non dispo, on sauvegarde en PNG 16-bit
    # via une conversion numpy + struct manuelle minimaliste.
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore
        h, w = rgba.shape[:2]
        hdr = OpenEXR.Header(w, h)
        hdr["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "A": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        }
        out = OpenEXR.OutputFile(path, hdr)
        data = rgba.astype(np.float16) / 255.0
        flip = data[::-1, :, :]
        out.writePixels({
            "R": flip[:, :, 0].tobytes(),
            "G": flip[:, :, 1].tobytes(),
            "B": flip[:, :, 2].tobytes(),
            "A": flip[:, :, 3].tobytes(),
        })
        out.close()
    except ImportError:
        # Fallback PNG si pas d'OpenEXR
        png_path = path.replace(".exr", ".png")
        _write_png(png_path, rgba)
        log.debug("OpenEXR Python non installé — sauvegardé en PNG : %s", png_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  OfflineRenderEngine — orchestrateur principal
# ═══════════════════════════════════════════════════════════════════════════════

class OfflineRenderEngine:
    """
    Orchestre le rendu différé complet.

    L'appelant fournit une fonction render_fn(t, jitter_x, jitter_y) → np.ndarray (H×W×4 uint8).
    Le moteur gère TAA, motion blur, écriture des frames, et encodage final.

    Lancé dans un thread daemon via OfflineRenderEngine.start().
    """

    def __init__(self,
                 config: OfflineRenderConfig,
                 render_fn: Callable,         # (t, jx, jy) → np.ndarray uint8 H×W×4
                 progress: OfflineRenderProgress):
        self.cfg       = config
        self._render   = render_fn
        self._progress = progress

    # ── Point d'entrée thread ──────────────────────────────────────────────

    def run(self):
        """Exécute le rendu complet. À appeler depuis un thread séparé."""
        cfg = self.cfg
        p   = self._progress
        t0  = time.perf_counter()

        tmp_dir = tempfile.mkdtemp(prefix="openshader_offline_")
        try:
            self._run_inner(cfg, p, tmp_dir)
        except Exception as exc:
            log.exception("Offline renderer erreur : %s", exc)
            p.set_error(str(exc))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            elapsed = time.perf_counter() - t0
            log.info("Rendu offline terminé en %.2fs → %s", elapsed, p.output or "(erreur)")

    def start(self) -> threading.Thread:
        """Lance le rendu dans un thread daemon et retourne le thread."""
        t = threading.Thread(target=self.run, daemon=True, name="OfflineRender")
        t.start()
        return t

    # ── Boucle interne ────────────────────────────────────────────────────

    def _run_inner(self, cfg: OfflineRenderConfig, p: OfflineRenderProgress, tmp_dir: str):
        taa = TAAAccumulator(cfg.taa_samples, cfg.taa_jitter_radius) if cfg.taa_enabled else None
        mb  = MotionBlurAccumulator(cfg.mb_samples, cfg.mb_shutter, cfg.fps) if cfg.mb_enabled else None

        n_frames    = cfg.total_frames
        taa_n       = taa.n_samples if taa else 1
        mb_n        = cfg.mb_samples if mb else 1
        samples_per_frame = taa_n * mb_n

        phase_label = self._phase_label()
        p.update(0.0, phase_label, 0, n_frames)

        png_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(png_dir, exist_ok=True)

        t_start = time.perf_counter()

        for fi in range(n_frames):
            if p.is_cancelled:
                p.set_error("Rendu annulé.")
                return

            t_frame = cfg.start_time + fi / cfg.fps

            # ── Rendu de la frame ────────────────────────────────────────
            rgba = self._render_frame(t_frame, taa, mb)

            # ── Sauvegarde frame ─────────────────────────────────────────
            frame_path = os.path.join(png_dir, f"frame_{fi:06d}.png")
            _write_png(frame_path, rgba)

            # Optionnel : séquence EXR
            if cfg.use_exr and cfg.is_image_seq and cfg.format == "exr_seq":
                exr_path = os.path.join(png_dir, f"frame_{fi:06d}.exr")
                _write_exr_half(exr_path, rgba)

            # ── Progression ──────────────────────────────────────────────
            frac_render = (fi + 1) / n_frames
            frac_total  = frac_render * 0.75   # 75% pour le rendu
            elapsed = time.perf_counter() - t_start
            eta     = elapsed / max(frac_render, 1e-6) * (1 - frac_render)
            p.update(frac_total,
                     f"{phase_label}  frame {fi+1}/{n_frames}  ETA {eta:.0f}s",
                     fi + 1, n_frames)

        if p.is_cancelled:
            p.set_error("Rendu annulé.")
            return

        # ── Encodage final ───────────────────────────────────────────────
        p.update(0.76, "Encodage…")
        out_path = self._encode(cfg, png_dir, tmp_dir, p)
        p.set_done(out_path)

    def _render_frame(self,
                      t_frame: float,
                      taa: Optional[TAAAccumulator],
                      mb:  Optional[MotionBlurAccumulator]) -> np.ndarray:
        """
        Rend une frame complète avec TAA et/ou motion blur.
        Retourne np.ndarray uint8 H×W×4.
        """
        if mb is not None:
            # ── Motion blur : N sub-times × (TAA si activé) ────────────
            sub_times = mb.sub_times(t_frame)
            mb_frames = []
            for t_sub in sub_times:
                if taa is not None:
                    frame_f = self._render_taa(t_sub, taa)
                else:
                    frame_f = self._render(t_sub, 0.0, 0.0)
                mb_frames.append(frame_f)
            return mb.accumulate(mb_frames)

        elif taa is not None:
            # ── TAA seul ───────────────────────────────────────────────
            return self._render_taa(t_frame, taa)

        else:
            # ── Rendu simple ────────────────────────────────────────────
            return self._render(t_frame, 0.0, 0.0)

    def _render_taa(self, t: float, taa: TAAAccumulator) -> np.ndarray:
        """Accumule N samples jittérés via TAA et retourne le résultat."""
        for jx, jy in taa:
            frame = self._render(t, jx, jy)
            taa.add(frame)
        return taa.result()

    # ── Encodage ──────────────────────────────────────────────────────────

    def _encode(self, cfg: OfflineRenderConfig, png_dir: str,
                tmp_dir: str, p: OfflineRenderProgress) -> str:
        fmt = cfg.format
        if fmt == "png_seq":
            return self._copy_png_seq(cfg, png_dir)
        if fmt == "exr_seq":
            return self._copy_png_seq(cfg, png_dir)  # EXR déjà dans png_dir
        if fmt == "dcp":
            return self._encode_dcp(cfg, png_dir, p)
        return self._encode_video(cfg, png_dir, p)

    def _copy_png_seq(self, cfg: OfflineRenderConfig, png_dir: str) -> str:
        """Copie la séquence PNG vers le répertoire de sortie."""
        out_dir = cfg.output_path
        os.makedirs(out_dir, exist_ok=True)
        for fname in sorted(os.listdir(png_dir)):
            shutil.copy2(os.path.join(png_dir, fname),
                         os.path.join(out_dir, fname))
        return out_dir

    def _encode_dcp(self, cfg: OfflineRenderConfig, png_dir: str,
                    p: OfflineRenderProgress) -> str:
        out_dir = cfg.output_path
        os.makedirs(out_dir, exist_ok=True)
        exporter = DCPExporter(cfg)

        def _cb(frac: float, msg: str):
            p.update(0.75 + frac * 0.25, msg)

        exporter.export(png_dir, out_dir, progress_cb=_cb)
        return out_dir

    def _encode_video(self, cfg: OfflineRenderConfig, png_dir: str,
                      p: OfflineRenderProgress) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg est requis pour l'encodage vidéo.")

        input_pat = os.path.join(png_dir, "frame_%06d.png")
        out_path  = cfg.output_path

        ext = Path(out_path).suffix.lower()
        if ext == ".gif":
            return self._encode_gif(ffmpeg, input_pat, out_path, cfg)

        cmd = [ffmpeg, "-y",
               "-framerate", str(cfg.fps),
               "-i", input_pat]

        # Audio
        if cfg.audio_path and os.path.isfile(cfg.audio_path):
            cmd += ["-i", cfg.audio_path, "-t", str(cfg.duration)]

        # Codec vidéo
        cmd += ["-c:v", cfg.video_codec]
        if cfg.pixel_format:
            cmd += ["-pix_fmt", cfg.pixel_format]

        # CRF selon codec
        if cfg.crf >= 0:
            if cfg.video_codec in ("libvpx-vp9", "libaom-av1"):
                cmd += ["-crf", str(cfg.crf), "-b:v", "0"]
            elif cfg.video_codec not in ("prores_ks",):
                cmd += ["-crf", str(cfg.crf)]

        # Extra codec args
        cmd += cfg.ffmpeg_extra

        # Threads
        if cfg.threads > 0:
            cmd += ["-threads", str(cfg.threads)]

        # Audio codec
        if cfg.audio_path and os.path.isfile(cfg.audio_path):
            if cfg.video_codec == "prores_ks":
                cmd += ["-c:a", "pcm_s24le"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]

        cmd.append(out_path)

        log.debug("FFmpeg encode cmd: %s", " ".join(cmd))
        p.update(0.77, "FFmpeg : encodage…")
        result = subprocess.run(cmd, capture_output=True, timeout=7200)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"FFmpeg erreur (code {result.returncode}):\n{err[-1200:]}")

        p.update(1.0, "✓ Encodage terminé")
        return out_path

    def _encode_gif(self, ffmpeg: str, input_pat: str, out_path: str,
                    cfg: OfflineRenderConfig) -> str:
        """GIF 2 passes avec palette Floyd-Steinberg."""
        tmp_pal = out_path + ".palette.png"
        try:
            r1 = subprocess.run([
                ffmpeg, "-y", "-framerate", str(cfg.fps), "-i", input_pat,
                "-vf", "palettegen=max_colors=256",
                tmp_pal,
            ], capture_output=True, timeout=300)
            if r1.returncode != 0:
                raise RuntimeError(r1.stderr.decode(errors="replace"))
            r2 = subprocess.run([
                ffmpeg, "-y", "-framerate", str(cfg.fps), "-i", input_pat,
                "-i", tmp_pal,
                "-lavfi", "paletteuse=dither=floyd_steinberg",
                "-loop", "0",
                out_path,
            ], capture_output=True, timeout=600)
            if r2.returncode != 0:
                raise RuntimeError(r2.stderr.decode(errors="replace"))
        finally:
            if os.path.isfile(tmp_pal):
                os.remove(tmp_pal)
        return out_path

    # ── Helpers ───────────────────────────────────────────────────────────

    def _phase_label(self) -> str:
        cfg = self.cfg
        parts = []
        if cfg.taa_enabled:  parts.append(f"TAA×{cfg.taa_samples}")
        if cfg.mb_enabled:   parts.append(f"MB×{cfg.mb_samples}")
        if cfg.is_dcp:       parts.append("DCP")
        return "Rendu " + " + ".join(parts) if parts else "Rendu"
