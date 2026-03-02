"""
openshader_core
===============
Pure-Python core of OpenShader.  No Qt dependency.

Installable standalone::

    pip install openshader-core
    pip install "openshader-core[images]"   # + Pillow for texture loading

Quick start (headless render)::

    from openshader_core import ShaderEngine, create_offscreen_context

    ctx    = create_offscreen_context()
    engine = ShaderEngine(width=1920, height=1080)
    engine.initialize(ctx)
    engine.load_shader_source(open("demo.glsl").read(), "Image")
    rgba   = engine.render_frame(t=0.0)   # bytes, RGBA8
    engine.cleanup()
    ctx.release()
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openshader-core")
except PackageNotFoundError:
    __version__ = "5.0.0-dev"

# ── Pure-Python API (always available, no moderngl required) ──────────────────

from .logger     import get_logger, set_debug
from .bezier     import (
    cubic_bezier, cubic_bezier_tuple,
    solve_t_for_x, bezier_interpolate,
    catmull_rom_tangent, auto_tangent_endpoint,
)
from .marker     import Marker, MarkerTrack
from .timeline   import Timeline, Track, Keyframe
from .exceptions import (
    OpenShaderError,
    ShaderCompileError,
    ProjectLoadError,
    ContextError,
)

# ── GPU-dependent API (lazy-imported — moderngl required at call time) ─────────
# import openshader_core succeeds even without moderngl installed.
# Pure-Python use cases (timeline scripting, project serialisation) work anywhere.

def __getattr__(name: str):
    if name == "ShaderEngine":
        from .engine import ShaderEngine
        return ShaderEngine
    if name == "preprocess_glsl":
        from .engine import preprocess_glsl
        return preprocess_glsl
    if name == "ShaderCache":
        from .shader_cache import ShaderCache
        return ShaderCache
    if name == "create_offscreen_context":
        return create_offscreen_context
    raise AttributeError(f"module 'openshader_core' has no attribute {name!r}")


def create_offscreen_context(width: int = 800, height: int = 450):
    """
    Create a ModernGL offscreen context for headless rendering.

    Tries EGL -> OSMesa -> native. Raises ContextError when unavailable.
    For LLVMpipe/Mesa CI helpers install openshader-headless.
    """
    import moderngl
    import logging
    _log = logging.getLogger("openshader.core.context")

    for backend in ("egl", "osmesa", None):
        try:
            kwargs = {"backend": backend} if backend else {}
            ctx = moderngl.create_standalone_context(**kwargs)
            _log.info("OpenGL context created (backend=%s)", backend or "native")
            return ctx
        except (moderngl.Error, RuntimeError) as exc:
            _log.debug("Backend %s unavailable: %s", backend, exc)

    raise ContextError(
        "No offscreen OpenGL backend available. "
        "Install libEGL (Mesa) or libOSMesa, or set LIBGL_ALWAYS_SOFTWARE=1."
    )


__all__ = [
    "__version__",
    "get_logger", "set_debug",
    "cubic_bezier", "cubic_bezier_tuple", "solve_t_for_x",
    "bezier_interpolate", "catmull_rom_tangent", "auto_tangent_endpoint",
    "Marker", "MarkerTrack",
    "Timeline", "Track", "Keyframe",
    "OpenShaderError", "ShaderCompileError", "ProjectLoadError", "ContextError",
    "ShaderCache",
    "ShaderEngine", "preprocess_glsl",
    "create_offscreen_context",
]
