"""
gl_widget.py — QOpenGLWidget + ModernGL.

FIX ÉCRAN NOIR : QOpenGLWidget rend dans son propre FBO interne, PAS le FBO 0.
ctx.screen = FBO 0 → noir. Solution : defaultFramebufferObject() → detect_framebuffer().

Améliorations v1.1 :
  - Logger centralisé (plus de print() de debug)
  - iMouse interactif : mouseMoveEvent / mousePressEvent → uniform iMouse
  - FFT + Waveform mic injectés comme uniforms GLSL (iAudioAmplitude, iFFTBands, iAudioRMS)
"""

import numpy as np
import time
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui  import QSurfaceFormat, QMouseEvent
import moderngl
from .shader_engine import ShaderEngine
# v2.7 — Support backend alternatif Vulkan
from .vulkan_shader_engine import VulkanShaderEngine, load_backend_pref, BACKEND_VULKAN
from .logger import get_logger
from .gpu_profiler import GPUProfiler  # v2.1
from .ai_upscaler  import UpscalerController  # v2.3 — Upscaling IA

log = get_logger(__name__)

VIEWPORT_W, VIEWPORT_H = 800, 450

# Résolutions prédéfinies disponibles dans l'UI
PRESET_RESOLUTIONS: list[tuple[int, int]] = [
    (400, 225),
    (640, 360),
    (800, 450),
    (1280, 720),
    (1920, 1080),
]


class GLWidget(QOpenGLWidget):
    fps_updated  = pyqtSignal(float)
    render_error = pyqtSignal(str)

    def __init__(self, shader_engine: ShaderEngine, parent=None):
        super().__init__(parent)
        self.shader_engine = shader_engine
        self.setFixedSize(VIEWPORT_W, VIEWPORT_H)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(0)
        self.setFormat(fmt)

        # Activer le suivi de la souris même sans clic enfoncé
        self.setMouseTracking(True)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)

        self._fps_frames    = 0
        self._fps_last_time = time.perf_counter()
        self._current_time  = 0.0
        self._ctx           = None
        self._qt_fbo        = None
        self._last_qt_fbo_id = -1

        # iMouse : (x, y, click_x, click_y) — coords GL (origine bas-gauche)
        self._mouse_x       = 0.0
        self._mouse_y       = 0.0
        self._mouse_click_x = 0.0
        self._mouse_click_y = 0.0
        self._mouse_down    = False

        # Visualisation overlay FFT / oscilloscope
        self._show_fft = False; self._fft_prog = None
        self._fft_vao  = None;  self._fft_vbo  = None
        self._show_osc = False; self._osc_prog = None
        self._osc_vao  = None;  self._osc_vbo  = None

        # Données audio courantes
        self._audio_amplitude: float = 0.0

        # v2.0 — Benchmark
        self._benchmark_enabled: bool = False
        self._bench_frame_times: list = []   # timestamps perf_counter des frames
        self._bench_start:       float = 0.0
        self._bench_max_samples: int  = 3600  # ~60s @ 60fps
        self.gpu_profiler = GPUProfiler()  # v2.1

        # v2.3 — Upscaler IA (initialisé dans initializeGL)
        self.upscaler_ctrl: UpscalerController | None = None

    # ── OpenGL lifecycle ────────────────────────────────────────────────────

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        log.debug("Contexte ModernGL créé : %s", self._ctx.info.get("GL_RENDERER", "?"))

        # v2.7 — Si le backend Vulkan est sélectionné, tenter l'init Vulkan.
        # En cas d'échec, on reste sur le ShaderEngine OpenGL courant.
        if isinstance(self.shader_engine, VulkanShaderEngine):
            try:
                self.shader_engine.initialize(ctx=None)
                log.info("Backend Vulkan initialisé avec succès")
            except VulkanShaderEngine.VulkanNotAvailableError as e:
                log.warning("Backend Vulkan indisponible — basculement OpenGL : %s", e)
                from .shader_engine import ShaderEngine
                self.shader_engine = ShaderEngine(self.shader_engine.width, self.shader_engine.height)
                self.shader_engine.initialize(self._ctx)
        else:
            self.shader_engine.initialize(self._ctx)

        self.gpu_profiler.initialize(self._ctx)  # v2.1
        self._init_fft_renderer()
        self._init_osc_renderer()

        # v2.3 — Upscaler IA
        self.upscaler_ctrl = UpscalerController(self.shader_engine, self)
        try:
            self.upscaler_ctrl.initialize(self._ctx)
        except Exception as e:
            log.warning("UpscalerController init échoué : %s", e)
            self.upscaler_ctrl = None

        self._timer.start()

    def resizeGL(self, w, h):
        self.shader_engine.resize(VIEWPORT_W, VIEWPORT_H)
        self._qt_fbo = None

    def paintGL(self):
        # Guard : initializeGL peut ne pas encore être terminé
        # (ex: update() déclenché très tôt, ou resize avant init)
        if self._ctx is None:
            return
        # ── iMouse : coords GL, click positif si bouton enfoncé ───────────
        self.gpu_profiler.begin_frame()  # v2.1
        click_x = self._mouse_click_x if self._mouse_down else -self._mouse_click_x
        click_y = self._mouse_click_y if self._mouse_down else -self._mouse_click_y
        self.shader_engine.set_uniform(
            'iMouse', (self._mouse_x, self._mouse_y, click_x, click_y)
        )

        # ── Uniforms audio ─────────────────────────────────────────────────
        self.shader_engine.set_uniform('iAudioAmplitude', self._audio_amplitude)

        qt_id = self.defaultFramebufferObject()
        if self._qt_fbo is None or self._last_qt_fbo_id != qt_id:
            self._qt_fbo = self._ctx.detect_framebuffer(qt_id)
            self._last_qt_fbo_id = qt_id

        # v2.3 — Upscaling IA : intercale entre image_fbo et screen_fbo
        _upscaler = self.upscaler_ctrl
        if _upscaler is not None and _upscaler.is_active:
            # Passe 1+2 : rend à résolution réduite, laisse image_texture prête
            self.shader_engine.render_to_texture(self._current_time)
            # Passe 3 (upscaling) : image_texture → qt_fbo à résolution native
            ow = self.width()
            oh = self.height()
            applied = _upscaler.apply_upscale(
                src_texture=self.shader_engine.image_texture,
                dst_fbo=self._qt_fbo,
                output_viewport=(0, 0, ow, oh),
            )
            if not applied:
                # Fallback : blit direct si upscale échoué
                self._ctx.copy_framebuffer(dst=self._qt_fbo,
                                           src=self.shader_engine.image_fbo)
        else:
            self.shader_engine.render(self._current_time, screen_fbo=self._qt_fbo)

        if self._show_fft and self._fft_vao:
            self._qt_fbo.use()
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._fft_vao.render(moderngl.LINE_STRIP)
            self._ctx.disable(moderngl.BLEND)

        if self._show_osc and self._osc_vao:
            self._qt_fbo.use()
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._osc_vao.render(moderngl.LINE_STRIP)
            self._ctx.disable(moderngl.BLEND)

        self.gpu_profiler.end_frame()  # v2.1
        self._fps_frames += 1
        now = time.perf_counter()
        d   = now - self._fps_last_time
        if d >= 1.0:
            self.fps_updated.emit(self._fps_frames / d)
            self._fps_frames = 0; self._fps_last_time = now

        # v2.0 — Benchmark : enregistrement du timestamp de frame
        if self._benchmark_enabled:
            self._bench_frame_times.append(now)
            if len(self._bench_frame_times) > self._bench_max_samples:
                self._bench_frame_times.pop(0)

    # ── Souris → iMouse ────────────────────────────────────────────────────

    def _qt_to_gl_y(self, y: int) -> float:
        """Convertit Y Qt (origine haut) en Y GL (origine bas)."""
        return float(VIEWPORT_H - y)

    def mouseMoveEvent(self, event: QMouseEvent):
        self._mouse_x = float(event.position().x())
        self._mouse_y = self._qt_to_gl_y(int(event.position().y()))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_down    = True
            self._mouse_x       = float(event.position().x())
            self._mouse_y       = self._qt_to_gl_y(int(event.position().y()))
            self._mouse_click_x = self._mouse_x
            self._mouse_click_y = self._mouse_y

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_down = False

    # ── Overlay FFT / oscilloscope ──────────────────────────────────────────

    def _init_fft_renderer(self):
        try:
            self._fft_prog = self._ctx.program(
                vertex_shader="""
#version 330 core
in float in_mag; uniform int u_num_bins;
void main() {
    float x = -1.0 + 2.0*(float(gl_VertexID)/float(u_num_bins-1));
    gl_Position = vec4(x, -1.0+in_mag*1.5, 0.0, 1.0);
}""",
                fragment_shader="""
#version 330 core
out vec4 fragColor;
void main() { fragColor = vec4(0.2,0.8,0.4,0.7); }""")
        except moderngl.Error as e:
            log.error("Compilation shader FFT overlay : %s", e)

    def _init_osc_renderer(self):
        try:
            self._osc_prog = self._ctx.program(
                vertex_shader="""
#version 330 core
in float in_val; uniform int u_num_samples;
void main() {
    float x = -1.0 + 2.0*(float(gl_VertexID)/float(u_num_samples-1));
    gl_Position = vec4(x, in_val*0.5, 0.0, 1.0);
}""",
                fragment_shader="""
#version 330 core
out vec4 fragColor;
void main() { fragColor = vec4(0.8,0.4,0.2,0.7); }""")
        except moderngl.Error as e:
            log.error("Compilation shader oscilloscope overlay : %s", e)

    # ── Slots audio ─────────────────────────────────────────────────────────

    @pyqtSlot(float)
    def on_amplitude_data(self, amplitude: float):
        """Amplitude mic → uniform iAudioAmplitude."""
        self._audio_amplitude = amplitude

    @pyqtSlot(object)
    def on_fft_data(self, data: np.ndarray):
        """FFT mic → uniform iFFTBands (8 bandes moyennées) + overlay visuel."""
        # Injection GLSL : 8 bandes bass→treble normalisées [0,1]
        if data is not None and len(data) >= 8:
            bands = np.array_split(data, 8)
            band_means = tuple(float(b.mean()) for b in bands)
            self.shader_engine.set_uniform('iFFTBands', band_means)

        # Overlay visuel
        if not self._show_fft or not self._fft_prog or self._ctx is None:
            return
        arr = data.astype('f4')
        n = len(arr)
        # Les opérations GL nécessitent que le contexte OpenGL soit courant
        self.makeCurrent()
        try:
            if self._fft_vbo is None or self._fft_vbo.size != arr.nbytes:
                if self._fft_vbo: self._fft_vbo.release()
                if self._fft_vao: self._fft_vao.release()
                self._fft_vbo = self._ctx.buffer(arr, dynamic=True)
                self._fft_vao = self._ctx.vertex_array(self._fft_prog,
                                                        [(self._fft_vbo, 'f', 'in_mag')])
                if 'u_num_bins' in self._fft_prog:
                    self._fft_prog['u_num_bins'].value = n
            else:
                self._fft_vbo.write(arr)
        finally:
            self.doneCurrent()

    @pyqtSlot(object)
    def on_waveform_data(self, data: np.ndarray):
        """Waveform mic → uniform iAudioRMS + overlay visuel."""
        if data is not None and len(data) > 0:
            rms = float(np.sqrt(np.mean(data ** 2)))
            self.shader_engine.set_uniform('iAudioRMS', rms)

        if not self._show_osc or not self._osc_prog or self._ctx is None:
            return
        arr = data.astype('f4')
        n = len(arr)
        self.makeCurrent()
        try:
            if self._osc_vbo is None or self._osc_vbo.size != arr.nbytes:
                if self._osc_vbo: self._osc_vbo.release()
                if self._osc_vao: self._osc_vao.release()
                self._osc_vbo = self._ctx.buffer(arr, dynamic=True)
                self._osc_vao = self._ctx.vertex_array(self._osc_prog,
                                                        [(self._osc_vbo, 'f', 'in_val')])
                if 'u_num_samples' in self._osc_prog:
                    self._osc_prog['u_num_samples'].value = n
            else:
                self._osc_vbo.write(arr)
        finally:
            self.doneCurrent()

    # ── API publique ────────────────────────────────────────────────────────

    def toggle_fft(self, show: bool):        self._show_fft = show
    def toggle_oscilloscope(self, show: bool): self._show_osc = show
    def set_time(self, t: float):            self._current_time = t
    def stop_render(self):                   self._timer.stop()
    def start_render(self):
        if not self._timer.isActive():
            self._timer.start()

    def set_resolution(self, w: int, h: int):
        """Change la résolution de rendu et recrée les FBOs.
        Le widget Qt reste à taille fixe ; le viewport interne est redimensionné.
        """
        if w == self.shader_engine.width and h == self.shader_engine.height:
            return
        self.makeCurrent()
        self.shader_engine.resize(w, h)
        self._qt_fbo = None   # force re-détection du FBO Qt
        self.doneCurrent()
        log.info("Résolution changée → %dx%d", w, h)

    # ── v2.0 — Benchmark mode ────────────────────────────────────────────────

    def enable_benchmark(self, enabled: bool):
        """Active/désactive la collecte de métriques de performance."""
        self._benchmark_enabled = enabled
        if enabled:
            self._bench_frame_times.clear()
            self._bench_start = time.perf_counter()
            log.info("Benchmark activé.")
        else:
            log.info("Benchmark désactivé.")

    def get_benchmark_stats(self) -> dict:
        """
        Retourne un dict de statistiques de performance :
          - fps_mean, fps_min, fps_max
          - frame_time_mean_ms, frame_time_p95_ms
          - total_frames, elapsed_s
        """
        times = list(self._bench_frame_times)
        if len(times) < 2:
            return {}

        import statistics
        dts = [times[i] - times[i-1] for i in range(1, len(times))]
        dts_ms = [dt * 1000 for dt in dts]
        fps_list = [1.0 / dt for dt in dts if dt > 0]
        elapsed = (times[-1] - times[0]) if times else 0.0

        dts_sorted = sorted(dts_ms)
        p95_idx = max(0, int(len(dts_sorted) * 0.95) - 1)

        return {
            'fps_mean':          round(statistics.mean(fps_list), 1) if fps_list else 0.0,
            'fps_min':           round(min(fps_list), 1) if fps_list else 0.0,
            'fps_max':           round(max(fps_list), 1) if fps_list else 0.0,
            'frame_time_mean_ms': round(statistics.mean(dts_ms), 2) if dts_ms else 0.0,
            'frame_time_p95_ms':  round(dts_sorted[p95_idx], 2) if dts_sorted else 0.0,
            'total_frames':      len(times),
            'elapsed_s':         round(elapsed, 2),
        }

    def reset_benchmark(self):
        """Remet les métriques à zéro."""
        self._bench_frame_times.clear()
        self._bench_start = time.perf_counter()
