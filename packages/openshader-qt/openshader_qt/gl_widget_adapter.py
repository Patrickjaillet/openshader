"""
gl_widget_adapter.py
--------------------
PyQt6 OpenGL widget adapter that delegates rendering to ``openshader_core.ShaderEngine``.

This adapter allows the existing ``src/gl_widget.py`` (which contains Qt-specific
painting and interaction logic) to be gradually migrated toward the microkernel
architecture while still running inside the full ``openshader-qt`` GUI.

Design
------
- ``CoreGLWidget`` is a thin subclass of ``QOpenGLWidget`` that wires Qt's
  ``initializeGL / resizeGL / paintGL`` lifecycle to a ``ShaderEngine`` from
  openshader-core.
- All Qt signal/slot connections for uniform updates, audio RMS, etc. are handled
  here so that ``ShaderEngine`` itself stays Qt-free.
- The full ``GLWidget`` (in ``src/gl_widget.py``) continues to work unchanged;
  this adapter is the migration path for users who want only the core engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore    import QTimer, pyqtSignal
from PyQt6.QtWidgets import QOpenGLWidget
from PyQt6.QtGui     import QSurfaceFormat

from openshader_core import ShaderEngine

if TYPE_CHECKING:
    from PyQt6.QtCore import QSize

log = logging.getLogger("openshader.qt.gl_widget_adapter")


def _configure_surface_format(samples: int = 4, depth: int = 24) -> None:
    """Apply a QSurfaceFormat with MSAA + depth buffer globally."""
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(samples)
    fmt.setDepthBufferSize(depth)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(1)  # vsync
    QSurfaceFormat.setDefaultFormat(fmt)


class CoreGLWidget(QOpenGLWidget):
    """
    Minimal QOpenGLWidget that renders via ``openshader_core.ShaderEngine``.

    Signals
    -------
    frame_rendered(float)
        Emitted after each frame with the current render time.
    fps_updated(float)
        Emitted every second with the measured FPS.
    compile_error(str, str)
        Emitted on shader compile error: (pass_name, error_log).
    """

    frame_rendered = pyqtSignal(float)
    fps_updated    = pyqtSignal(float)
    compile_error  = pyqtSignal(str, str)

    def __init__(self, parent=None, *, target_fps: float = 60.0):
        super().__init__(parent)
        self._engine: ShaderEngine | None = None
        self._time: float = 0.0
        self._running: bool = False
        self._target_fps = target_fps

        # FPS counter
        self._frame_count = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        # Render timer
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self.update)  # schedule repaint

        # Uniforms injected from outside (audio RMS, MIDI, OSC …)
        self._extra_uniforms: dict[str, float | tuple] = {}

    # ── Qt GL lifecycle ───────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        """Called once by Qt after the OpenGL context is ready."""
        import moderngl
        try:
            # Obtain the moderngl context from Qt's current context
            ctx = moderngl.create_context()
        except moderngl.Error as exc:
            log.error("Failed to create ModernGL context: %s", exc)
            return

        self._engine = ShaderEngine(
            width=self.width(), height=self.height()
        )
        self._engine.initialize(ctx)
        log.info("CoreGLWidget: ModernGL context initialised (%dx%d)", self.width(), self.height())

    def resizeGL(self, w: int, h: int) -> None:
        """Called by Qt when the widget is resized."""
        if self._engine:
            self._engine.resize(w, h)

    def paintGL(self) -> None:
        """Called by Qt to render a frame."""
        if not self._engine:
            return
        if self._running:
            import time as _time
            self._time += 1.0 / self._target_fps  # deterministic step

        # Inject extra uniforms
        for name, val in self._extra_uniforms.items():
            self._engine.set_uniform(name, val)

        try:
            self._engine.render(self._time)
        except Exception as exc:
            log.error("Render error: %s", exc)

        self._frame_count += 1
        self.frame_rendered.emit(self._time)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_shader_source(self, source: str, pass_name: str = "Image") -> bool:
        """Compile and load GLSL source. Returns True on success."""
        if not self._engine:
            log.warning("CoreGLWidget.load_shader_source called before initializeGL")
            return False
        ok, err = self._engine.load_shader_source(source, pass_name)
        if not ok:
            self.compile_error.emit(pass_name, err)
        return ok

    def play(self) -> None:
        """Start rendering loop."""
        self._running = True
        interval_ms = max(1, int(1000 / self._target_fps))
        self._render_timer.start(interval_ms)

    def pause(self) -> None:
        """Pause rendering loop (keeps current time)."""
        self._running = False
        self._render_timer.stop()

    def stop(self) -> None:
        """Stop and reset time to 0."""
        self.pause()
        self._time = 0.0
        self.update()

    def seek(self, t: float) -> None:
        """Jump to a specific time."""
        self._time = t
        self.update()

    def set_uniform(self, name: str, value) -> None:
        """Set an extra uniform value that will be injected on every frame."""
        self._extra_uniforms[name] = value

    def set_audio_rms(self, rms: float) -> None:
        """Convenience helper: inject iAudioRMS uniform."""
        self.set_uniform("iAudioRMS", rms)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        fps = self._frame_count  # frames in the last second
        self._frame_count = 0
        self.fps_updated.emit(float(fps))
