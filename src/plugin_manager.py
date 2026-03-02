"""
plugin_manager.py
-----------------
v2.0 — Système de plugins : effets post-process, sources audio, extensions UI.

Architecture :
  - BasePlugin          : classe de base abstraite
  - PostProcessPlugin   : shader GLSL post-processing injectable dans la chaîne FX
  - AudioSourcePlugin   : source audio alternative (générateur, stream réseau…)
  - PluginManager       : découverte, chargement, activation des plugins

Les plugins sont des fichiers Python placés dans le répertoire plugins/ à la
racine du projet, ou dans ~/.demomaker/plugins/ pour les plugins utilisateur.

Format d'un plugin (plugins/my_plugin.py) :
    from src.plugin_manager import PostProcessPlugin, register

    class ChromaShift(PostProcessPlugin):
        name        = "Chroma Shift"
        description = "Aberration chromatique paramétrable"
        version     = "1.0"
        author      = "OpenShader"

        GLSL = '''
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;
            float amount = uPluginParam0 * 0.02;
            fragColor = vec4(
                texture(iChannel0, vec2(uv.x + amount, uv.y)).r,
                texture(iChannel0, uv).g,
                texture(iChannel0, vec2(uv.x - amount, uv.y)).b,
                1.0
            );
        }
        '''
        PARAMS = [
            {'name': 'Amount', 'uniform': 'uPluginParam0', 'default': 0.5, 'min': 0.0, 'max': 1.0},
        ]

    register(ChromaShift)
"""

from __future__ import annotations

import os
import sys
import importlib.util
import traceback
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger
from .native_plugin import NativePluginManager, NativePlugin  # v3.0 — plugins C++ natifs

log = get_logger(__name__)


# ── Classes de base des plugins ──────────────────────────────────────────────


# ── Classes de base des plugins ──────────────────────────────────────────────

class BasePlugin:
    """Classe de base abstraite pour tous les plugins OpenShader."""
    name:        str = "Unnamed Plugin"
    description: str = ""
    version:     str = "1.0"
    author:      str = "Unknown"
    plugin_type: str = "base"

    def __init__(self):
        self._enabled: bool = True
        self._params:  dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        self.on_enable()

    def disable(self):
        self._enabled = False
        self.on_disable()

    def set_param(self, name: str, value: Any):
        self._params[name] = value
        self.on_param_changed(name, value)

    def get_param(self, name: str, default: Any = None) -> Any:
        return self._params.get(name, default)

    # ── À surcharger ─────────────────────────────────────────────────────────

    def on_enable(self):
        pass

    def on_disable(self):
        pass

    def on_param_changed(self, name: str, value: Any):
        pass

    def to_dict(self) -> dict:
        return {'name': self.name, 'enabled': self._enabled, 'params': dict(self._params)}

    def from_dict(self, data: dict):
        self._enabled = data.get('enabled', True)
        self._params  = dict(data.get('params', {}))


@dataclass
class PluginParam:
    """Descripteur d'un paramètre de plugin (pour générer l'UI)."""
    name:     str
    uniform:  str
    default:  float = 0.5
    min:      float = 0.0
    max:      float = 1.0
    label:    str   = ""
    type:     str   = "float"  # 'float' | 'int' | 'bool' | 'color'

    def __post_init__(self):
        if not self.label:
            self.label = self.name


class PostProcessPlugin(BasePlugin):
    """Plugin d'effet post-process : fournit un fragment shader GLSL."""
    plugin_type = "post_process"

    # ── À définir dans la sous-classe ────────────────────────────────────────
    GLSL:   str             = ""   # fragment shader GLSL (format Shadertoy)
    PARAMS: list[dict]      = []   # liste de PluginParam descriptors

    def get_glsl(self) -> str:
        return self.GLSL

    def get_uniforms(self) -> dict[str, float]:
        """Retourne le dictionnaire uniform → valeur courante."""
        result = {}
        for p in self.PARAMS:
            uniform = p.get('uniform', '')
            default = p.get('default', 0.0)
            result[uniform] = float(self._params.get(p.get('name', ''), default))
        return result

    def get_param_descriptors(self) -> list[PluginParam]:
        return [
            PluginParam(
                name=p.get('name', ''),
                uniform=p.get('uniform', ''),
                default=float(p.get('default', 0.5)),
                min=float(p.get('min', 0.0)),
                max=float(p.get('max', 1.0)),
                label=p.get('label', p.get('name', '')),
                type=p.get('type', 'float'),
            )
            for p in self.PARAMS
        ]


class AudioSourcePlugin(BasePlugin):
    """Plugin de source audio alternative."""
    plugin_type = "audio_source"

    def get_amplitude(self) -> float:
        return 0.0

    def get_fft(self):
        return None

    def start(self): pass
    def stop(self):  pass


# ── Registre global (utilisé par register()) ────────────────────────────────

_PLUGIN_REGISTRY: list[type[BasePlugin]] = []

def register(cls: type[BasePlugin]):
    """Enregistre une classe de plugin (appelé depuis le module du plugin)."""
    if cls not in _PLUGIN_REGISTRY:
        _PLUGIN_REGISTRY.append(cls)
        log.debug("Plugin enregistré : %s (%s)", cls.name, cls.plugin_type)


# ── Gestionnaire de plugins ──────────────────────────────────────────────────

class PluginManager(QObject):
    """Découverte, chargement et gestion du cycle de vie des plugins."""

    plugin_loaded    = pyqtSignal(str)   # nom du plugin chargé
    plugin_unloaded  = pyqtSignal(str)
    plugin_toggled   = pyqtSignal(str, bool)  # (nom, enabled)
    plugins_changed  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instances: list[BasePlugin] = []
        self._search_dirs: list[str] = []
        # v3.0 — Gestionnaire de plugins natifs C++
        self.native = NativePluginManager(self)
        self.native.plugin_loaded.connect(self.plugin_loaded)
        self.native.plugin_unloaded.connect(self.plugin_unloaded)
        # v3.6 — Marketplace (instancié lazily)
        self._marketplace = None

    # ── Répertoires de recherche ──────────────────────────────────────────────

    def add_search_dir(self, path: str):
        if path not in self._search_dirs and os.path.isdir(path):
            self._search_dirs.append(path)

    def get_default_dirs(self) -> list[str]:
        """Retourne les répertoires de plugins par défaut."""
        dirs = []
        # Répertoire du projet
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_plugins = os.path.join(base, 'plugins')
        if os.path.isdir(project_plugins):
            dirs.append(project_plugins)
        # Répertoire utilisateur
        user_plugins = os.path.join(os.path.expanduser('~'), '.demomaker', 'plugins')
        os.makedirs(user_plugins, exist_ok=True)
        dirs.append(user_plugins)
        return dirs

    # ── Découverte et chargement ──────────────────────────────────────────────

    def scan_and_load(self) -> int:
        """Scanne tous les répertoires et charge les plugins disponibles."""
        dirs = self._search_dirs or self.get_default_dirs()
        loaded = 0

        for d in dirs:
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.py') and not fname.startswith('_'):
                    path = os.path.join(d, fname)
                    try:
                        self._load_file(path)
                        loaded += 1
                    except (OSError, ImportError, AttributeError):
                        log.warning("Échec chargement plugin '%s' :\n%s",
                                    fname, traceback.format_exc())

        # Instancier les classes enregistrées
        existing_names = {p.name for p in self._instances}
        for cls in _PLUGIN_REGISTRY:
            if cls.name not in existing_names:
                try:
                    inst = cls()
                    # Charger les paramètres par défaut
                    for p in getattr(cls, 'PARAMS', []):
                        inst._params[p.get('name', '')] = p.get('default', 0.0)
                    self._instances.append(inst)
                    self.plugin_loaded.emit(inst.name)
                    log.info("Plugin instancié : %s v%s", inst.name, inst.version)
                except (TypeError, AttributeError):
                    log.warning("Échec instanciation plugin '%s':\n%s",
                                cls.name, traceback.format_exc())

        if loaded > 0 or _PLUGIN_REGISTRY:
            self.plugins_changed.emit()

        # v3.0 — Scan des plugins natifs C++ dans plugins/native/
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        native_dir = os.path.join(base, 'plugins', 'native')
        if os.path.isdir(native_dir):
            ext = {'Windows': '.dll', 'Darwin': '.dylib'}.get(
                __import__('platform').system(), '.so'
            )
            for fname in sorted(os.listdir(native_dir)):
                if fname.endswith(ext):
                    npath = os.path.join(native_dir, fname)
                    try:
                        self.native.load(npath)
                    except Exception as e:
                        log.warning("Plugin natif '%s' ignoré : %s", fname, e)

        return len(self._instances)

    def _load_file(self, path: str):
        spec = importlib.util.spec_from_file_location(
            f"demomaker_plugin_{os.path.basename(path)[:-3]}", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    # ── Accès aux plugins ─────────────────────────────────────────────────────

    def get_all(self) -> list[BasePlugin]:
        return list(self._instances)

    def get_by_type(self, plugin_type: str) -> list[BasePlugin]:
        return [p for p in self._instances if p.plugin_type == plugin_type]

    def get_post_process(self) -> list[PostProcessPlugin]:
        return [p for p in self._instances
                if isinstance(p, PostProcessPlugin) and p.enabled]

    def get_by_name(self, name: str) -> BasePlugin | None:
        for p in self._instances:
            if p.name == name:
                return p
        return None

    def toggle(self, name: str, enabled: bool):
        p = self.get_by_name(name)
        if p:
            if enabled:
                p.enable()
            else:
                p.disable()
            self.plugin_toggled.emit(name, enabled)
            self.plugins_changed.emit()

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> list[dict]:
        return [p.to_dict() for p in self._instances]

    def from_dict(self, data: list[dict]):
        name_map = {p.name: p for p in self._instances}
        for d in data:
            name = d.get('name', '')
            if name in name_map:
                name_map[name].from_dict(d)

    # ── Marketplace ───────────────────────────────────────────────────────────

    @property
    def marketplace(self):
        """Retourne (en créant si nécessaire) le MarketplaceManager."""
        if self._marketplace is None:
            from .marketplace import MarketplaceManager
            self._marketplace = MarketplaceManager(self, self.parent())
        return self._marketplace


# ── Plugins intégrés (livrés avec OpenShader) ─────────────────────────────────

class _VignettePlugin(PostProcessPlugin):
    name        = "Vignette"
    description = "Assombrit les bords de l'image"
    version     = "1.0"
    author      = "OpenShader"
    GLSL = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 src = texture(iChannel0, uv);
    vec2 q = uv - 0.5;
    float vig = 1.0 - dot(q, q) * uPluginParam0 * 4.0;
    fragColor = vec4(src.rgb * clamp(vig, 0.0, 1.0), src.a);
}
"""
    PARAMS = [
        {'name': 'Strength', 'uniform': 'uPluginParam0', 'default': 0.4, 'min': 0.0, 'max': 1.0},
    ]

register(_VignettePlugin)


class _FilmGrainPlugin(PostProcessPlugin):
    name        = "Film Grain"
    description = "Grain analogique sur le rendu"
    version     = "1.0"
    author      = "OpenShader"
    GLSL = """
float rand(vec2 co) { return fract(sin(dot(co, vec2(12.9898,78.233))) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 src = texture(iChannel0, uv);
    float grain = (rand(uv + fract(iTime)) - 0.5) * uPluginParam0 * 0.15;
    fragColor = vec4(clamp(src.rgb + grain, 0.0, 1.0), src.a);
}
"""
    PARAMS = [
        {'name': 'Amount', 'uniform': 'uPluginParam0', 'default': 0.5, 'min': 0.0, 'max': 1.0},
    ]

register(_FilmGrainPlugin)


class _ColorGradePlugin(PostProcessPlugin):
    name        = "Color Grade"
    description = "Teinte colorimétrique globale"
    version     = "1.0"
    author      = "OpenShader"
    GLSL = """
vec3 hueShift(vec3 c, float h) {
    float angle = h * 6.28318;
    vec3 k = vec3(0.57735);
    return c * cos(angle) + cross(k, c) * sin(angle) + k * dot(k,c)*(1.0-cos(angle));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 src = texture(iChannel0, uv);
    vec3 col = hueShift(src.rgb, uPluginParam0);
    float luma = dot(col, vec3(0.299,0.587,0.114));
    col = mix(vec3(luma), col, 1.0 + uPluginParam1 * 0.8);
    col = pow(clamp(col, 0.001, 1.0), vec3(1.0 / (0.5 + uPluginParam2)));
    fragColor = vec4(col, src.a);
}
"""
    PARAMS = [
        {'name': 'Hue',        'uniform': 'uPluginParam0', 'default': 0.0,  'min': 0.0, 'max': 1.0},
        {'name': 'Saturation', 'uniform': 'uPluginParam1', 'default': 0.0,  'min': -1.0, 'max': 1.0},
        {'name': 'Gamma',      'uniform': 'uPluginParam2', 'default': 0.0,  'min': -0.5, 'max': 0.5},
    ]

register(_ColorGradePlugin)


class _BloomPlugin(PostProcessPlugin):
    name        = "Bloom"
    description = "Halo lumineux sur les zones claires"
    version     = "1.0"
    author      = "OpenShader"
    GLSL = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 src = texture(iChannel0, uv);
    vec4 bloom = vec4(0.0);
    float thresh = uPluginParam0 * 0.8 + 0.1;
    float spread = 0.003 + uPluginParam1 * 0.012;
    for (int x=-3; x<=3; x++) for (int y=-3; y<=3; y++) {
        vec2 off = vec2(x, y) * spread;
        vec4 s = texture(iChannel0, uv + off);
        float luma = dot(s.rgb, vec3(0.299,0.587,0.114));
        bloom += s * max(0.0, luma - thresh);
    }
    bloom /= 49.0;
    fragColor = vec4(src.rgb + bloom.rgb * uPluginParam2 * 3.0, src.a);
}
"""
    PARAMS = [
        {'name': 'Threshold', 'uniform': 'uPluginParam0', 'default': 0.6, 'min': 0.0, 'max': 1.0},
        {'name': 'Spread',    'uniform': 'uPluginParam1', 'default': 0.5, 'min': 0.0, 'max': 1.0},
        {'name': 'Intensity', 'uniform': 'uPluginParam2', 'default': 0.5, 'min': 0.0, 'max': 2.0},
    ]

register(_BloomPlugin)
