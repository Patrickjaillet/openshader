"""
renderer.py
-----------
High-level headless renderer built on top of openshader-core.

Usage::

    from openshader_headless import HeadlessRenderer

    with HeadlessRenderer("demo.glsl", width=1920, height=1080) as r:
        r.render_video("output.mp4", duration=10, fps=60)
        r.render_png("preview.png", t=0.0)
        r.render_sequence("frames/frame_%05d.png", duration=5, fps=30)
"""

from __future__ import annotations

import logging
import os
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Iterator

from .context import HeadlessContext, create_headless_context

log = logging.getLogger("openshader.headless.renderer")


# ── PNG writer (stdlib only, no Pillow) ───────────────────────────────────────

def _write_png(path: str | Path, rgba: bytes, width: int, height: int) -> None:
    """Write a raw RGBA byte buffer as a PNG file (pure stdlib, no Pillow needed)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row_size = width * 4
    # Flip vertical: OpenGL origin = bottom-left, PNG origin = top-left
    rows = [rgba[i * row_size: (i + 1) * row_size] for i in range(height - 1, -1, -1)]
    raw = b"".join(b"\x00" + row for row in rows)  # filter type 0 (None) per row
    compressed = zlib.compress(raw, 6)

    png  = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")

    Path(path).write_bytes(png)


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _open_ffmpeg_pipe(
    output: str, width: int, height: int, fps: float
) -> subprocess.Popen:
    ext = Path(output).suffix.lower()

    if ext == ".gif":
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgba",
            "-r", str(fps), "-i", "pipe:0",
            "-vf", (
                f"vflip,fps={fps},"
                f"split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer"
            ),
            output,
        ]
    elif ext == ".webm":
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgba",
            "-r", str(fps), "-i", "pipe:0",
            "-vf", "vflip",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
            "-b:v", "0", "-crf", "30", output,
        ]
    else:  # mp4 / mkv / mov
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgba",
            "-r", str(fps), "-i", "pipe:0",
            "-vf", "vflip",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "18", output,
        ]

    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


# ── HeadlessRenderer ──────────────────────────────────────────────────────────

class HeadlessRenderer:
    """
    Headless GLSL shader renderer.

    Parameters
    ----------
    shader_source : str | Path
        Path to a .glsl / .st shader file, or a raw GLSL source string.
    width, height : int
        Output resolution.
    pass_name : str
        Shader pipeline pass to render (default: 'Image').
    defines : dict | None
        Extra ``#define`` directives injected before compilation.
    lib_dir : str | None
        Directory for GLSL ``#include`` resolution.
    context : HeadlessContext | None
        Provide an existing context (e.g. to share across multiple renderers).
        If None, a new context is created automatically.

    Examples
    --------
    Context manager (recommended)::

        with HeadlessRenderer("fire.glsl", 1920, 1080) as r:
            r.render_video("fire.mp4", duration=10, fps=60)

    Manual lifecycle::

        r = HeadlessRenderer("fire.glsl")
        r.open()
        rgba = r.render_frame(t=2.5)
        r.close()
    """

    def __init__(
        self,
        shader_source: str | Path,
        width: int = 1920,
        height: int = 1080,
        *,
        pass_name: str = "Image",
        defines: dict | None = None,
        lib_dir: str | None = None,
        context: HeadlessContext | None = None,
    ):
        self.width  = width
        self.height = height
        self.pass_name = pass_name
        self.defines   = defines or {}
        self.lib_dir   = lib_dir

        # Resolve shader source
        src_path = Path(str(shader_source))
        if src_path.exists():
            self._source_path = str(src_path.resolve())
            self._source_text = src_path.read_text(encoding="utf-8")
        else:
            # Treat as raw GLSL source string
            self._source_path = None
            self._source_text = str(shader_source)

        self._ext_context = context
        self._hctx: HeadlessContext | None = None
        self._engine: ShaderEngine | None = None
        self._owns_context = context is None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> "HeadlessRenderer":
        """Initialize context and compile shader. Called automatically by __enter__."""
        if self._engine is not None:
            return self

        if self._ext_context:
            self._hctx = self._ext_context
        else:
            self._hctx = create_headless_context()

        log.info(
            "HeadlessRenderer: %s  [%dx%d  pass=%s]",
            self._hctx, self.width, self.height, self.pass_name,
        )

        from openshader_core.engine import ShaderEngine as _SE
        self._engine = _SE(
            width=self.width, height=self.height, lib_dir=self.lib_dir
        )
        self._engine.initialize(self._hctx.ctx)

        ok, err = self._engine.load_shader_source(
            self._source_text, self.pass_name, source_path=self._source_path
        )
        if not ok:
            self.close()
            from openshader_core.exceptions import ShaderCompileError
            raise ShaderCompileError(
                f"Shader compilation failed: {err}", pass_name=self.pass_name, log=err
            )

        log.info("Shader compiled OK  (pass=%s)", self.pass_name)
        return self

    def close(self) -> None:
        """Release engine and context. Called automatically by __exit__."""
        if self._engine:
            self._engine.cleanup()
            self._engine = None
        if self._hctx and self._owns_context:
            self._hctx.release()
            self._hctx = None

    def __enter__(self) -> "HeadlessRenderer":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ── Rendering API ──────────────────────────────────────────────────────────

    def render_frame(self, t: float = 0.0) -> bytes:
        """
        Render a single frame at time *t* (seconds).

        Returns
        -------
        bytes
            Raw RGBA8 pixel data, row-major, bottom-left origin (OpenGL convention).
            Size = width × height × 4 bytes.
        """
        if self._engine is None:
            raise RuntimeError("HeadlessRenderer is not open — call open() first")
        return self._engine.render_frame(t)

    def render_png(self, output: str | Path, t: float = 0.0) -> None:
        """Render a single frame and write it as a PNG file."""
        rgba = self.render_frame(t)
        _write_png(output, rgba, self.width, self.height)
        log.info("PNG written → %s", output)

    def render_frames(
        self, fps: float = 60.0, duration: float = 10.0
    ) -> Iterator[tuple[int, float, bytes]]:
        """
        Iterate over all frames as (frame_index, timestamp, rgba_bytes).

        Useful for custom output pipelines::

            with HeadlessRenderer("demo.glsl") as r:
                for idx, t, rgba in r.render_frames(fps=30, duration=5):
                    process(rgba)
        """
        total = max(1, int(duration * fps))
        for i in range(total):
            t = i / fps
            yield i, t, self.render_frame(t)

    def render_sequence(
        self,
        pattern: str | Path,
        fps: float = 60.0,
        duration: float = 10.0,
        *,
        show_progress: bool = True,
    ) -> int:
        """
        Render a PNG sequence.

        Parameters
        ----------
        pattern : str
            Output path pattern with a ``%05d`` (or similar) placeholder,
            e.g. ``"frames/frame_%05d.png"``.
        fps, duration : float
            Frame rate and total duration.
        show_progress : bool
            Print a CLI progress bar to stdout.

        Returns
        -------
        int
            Number of frames written.
        """
        pattern = str(pattern)
        Path(pattern).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        total = max(1, int(duration * fps))
        for i, t, rgba in self.render_frames(fps=fps, duration=duration):
            if show_progress:
                _print_progress(i + 1, total, t0)
            _write_png(pattern % i, rgba, self.width, self.height)
        if show_progress:
            print()

        elapsed = time.perf_counter() - t0
        log.info("Sequence done: %d frames in %.2fs → %s", total, elapsed, pattern)
        return total

    def render_video(
        self,
        output: str | Path,
        fps: float = 60.0,
        duration: float = 10.0,
        *,
        show_progress: bool = True,
    ) -> None:
        """
        Render directly to a video file via FFmpeg.

        Parameters
        ----------
        output : str | Path
            Destination file. Extension determines codec:
            ``.mp4`` → H.264/yuv420p,  ``.webm`` → VP9,  ``.gif`` → palette GIF.
        fps, duration : float
            Frame rate and total duration.
        show_progress : bool
            Print a CLI progress bar to stdout.
        """
        output = str(output)
        if not _ffmpeg_available():
            raise RuntimeError(
                "FFmpeg is required for video export but was not found in PATH.\n"
                "Install with: sudo apt-get install ffmpeg  (or brew install ffmpeg)"
            )

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        proc = _open_ffmpeg_pipe(output, self.width, self.height, fps)

        t0 = time.perf_counter()
        total = max(1, int(duration * fps))
        try:
            for i, t, rgba in self.render_frames(fps=fps, duration=duration):
                if show_progress:
                    _print_progress(i + 1, total, t0)
                proc.stdin.write(rgba)
        except (BrokenPipeError, KeyboardInterrupt):
            proc.stdin.close()
            proc.wait()
            raise
        finally:
            if show_progress:
                print()

        proc.stdin.close()
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {proc.returncode}):\n"
                + stderr.decode(errors="replace")
            )

        elapsed = time.perf_counter() - t0
        log.info("Video rendered in %.2fs → %s", elapsed, output)


# ── Progress bar ──────────────────────────────────────────────────────────────

def _print_progress(done: int, total: int, t0: float) -> None:
    pct = done / total
    bar_w = 40
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    elapsed = time.perf_counter() - t0
    eta = elapsed / pct * (1 - pct) if pct > 0 else 0
    print(
        f"\r  [{bar}] {pct*100:5.1f}%  {done}/{total}  ETA {eta:.0f}s   ",
        end="",
        flush=True,
    )
