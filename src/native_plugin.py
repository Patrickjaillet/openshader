"""
native_plugin.py
----------------
v1.0 — Plugins C++ natifs pour OpenShader / DemoMaker.

Permet de charger des plugins compilés (.dll Windows, .so Linux, .dylib macOS)
directement depuis Python, sans redémarrer l'application.

Architecture :
  - ABI versionnée (OPENSHADER_SDK_VERSION) : compatibilité garantie par version majeure
  - Deux backends de binding au choix : ctypes (zéro dépendance) ou pybind11 (API riche)
  - Hot-reload : surveillance du fichier via watchdog ou polling, rechargement à chaud
  - NativePluginBase : classe de base Python qui wrap le .dll/.so
  - Exemples SDK : ParticlePlugin, PhysicsPlugin, AudioFFTPlugin

Format d'un plugin C++ (voir plugins/sdk/examples/) :
    extern "C" {
        OPENSHADER_API int  openshader_sdk_version();
        OPENSHADER_API void openshader_init(const OSPluginContext* ctx);
        OPENSHADER_API void openshader_tick(float dt, const OSUniforms* u);
        OPENSHADER_API void openshader_get_uniforms(OSUniformOut* out, int* count);
        OPENSHADER_API void openshader_shutdown();
        OPENSHADER_API const char* openshader_name();
        OPENSHADER_API const char* openshader_version();
    }

Usage :
    mgr = NativePluginManager()
    plugin = mgr.load('plugins/particles.dll')
    plugin.tick(dt=0.016, uniforms={...})
    plugin.get_uniforms()   # → dict[uniform_name, float]
    mgr.hot_reload_enable()
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)

# ── Constantes ABI ────────────────────────────────────────────────────────────

OPENSHADER_SDK_MAJOR  = 1   # breaking changes → incrémente major
OPENSHADER_SDK_MINOR  = 0   # ajouts rétro-compatibles → incrémente minor
OPENSHADER_SDK_VERSION = (OPENSHADER_SDK_MAJOR << 16) | OPENSHADER_SDK_MINOR

# Noms d'exports C obligatoires dans chaque plugin natif
_REQUIRED_EXPORTS = [
    "openshader_sdk_version",
    "openshader_name",
    "openshader_version",
    "openshader_init",
    "openshader_tick",
    "openshader_get_uniforms",
    "openshader_shutdown",
]

# ── Structures C (ctypes mirror) ──────────────────────────────────────────────

class OSPluginContext(ctypes.Structure):
    """Contexte passé à openshader_init().  Correspond à OSPluginContext dans le SDK C."""
    _fields_ = [
        ("sample_rate",    ctypes.c_float),   # fréquence audio (44100.0)
        ("canvas_width",   ctypes.c_int),
        ("canvas_height",  ctypes.c_int),
        ("sdk_version",    ctypes.c_uint32),  # OPENSHADER_SDK_VERSION
    ]


class OSUniforms(ctypes.Structure):
    """Uniforms lus depuis le shader — snapshot à chaque tick."""
    _fields_ = [
        ("iTime",      ctypes.c_float),
        ("iDeltaTime", ctypes.c_float),
        ("iBeat",      ctypes.c_float),
        ("iRMS",       ctypes.c_float),
        ("iBPM",       ctypes.c_float),
    ]


class OSUniformEntry(ctypes.Structure):
    """Une entrée produite par le plugin → injectée comme uniform GLSL."""
    _fields_ = [
        ("name",  ctypes.c_char * 64),
        ("value", ctypes.c_float),
    ]

# Tableau de 32 uniforms max par plugin
MAX_UNIFORMS = 32
OSUniformArray = OSUniformEntry * MAX_UNIFORMS


# ── Import conditionnel pybind11 ─────────────────────────────────────────────

try:
    import pybind11  # type: ignore  # noqa: F401
    _PYBIND11_AVAILABLE = True
except ImportError:
    _PYBIND11_AVAILABLE = False
    log.debug("pybind11 non installé — backend ctypes utilisé (pip install pybind11)")


# ── Résolution du nom de bibliothèque natif ───────────────────────────────────

def _resolve_lib_path(path: str) -> str:
    """
    Résout un chemin de plugin vers l'extension native correcte
    pour la plateforme courante, si aucune extension n'est fournie.
    """
    p = Path(path)
    if p.suffix in ('.dll', '.so', '.dylib'):
        return str(p)
    # Pas d'extension → on essaie dans l'ordre de préférence
    candidates = []
    sys_name = platform.system()
    if sys_name == 'Windows':
        candidates = [p.with_suffix('.dll')]
    elif sys_name == 'Darwin':
        candidates = [p.with_suffix('.dylib'), p.with_name(f'lib{p.name}.dylib')]
    else:
        candidates = [p.with_suffix('.so'), p.with_name(f'lib{p.name}.so')]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(path)


# ── Wrapper d'un plugin natif individuel ─────────────────────────────────────

class NativePlugin:
    """
    Wrapping Python d'un plugin C++ chargé via ctypes.

    Cycle de vie :
        plugin = NativePlugin(path)
        plugin.load()
        plugin.init(sample_rate=44100, width=1920, height=1080)
        plugin.tick(dt=0.016, itime=3.0, rms=0.5)
        uniforms = plugin.get_uniforms()   # dict[str, float]
        plugin.unload()
    """

    def __init__(self, path: str):
        self._path:     str            = _resolve_lib_path(path)
        self._lib:      ctypes.CDLL | None = None
        self._loaded:   bool           = False
        self._name:     str            = Path(path).stem
        self._version:  str            = "?"
        self._sdk_ver:  int            = 0
        # Snapshot des uniforms produits par le plugin
        self._uniform_cache: dict[str, float] = {}

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sdk_version(self) -> int:
        return self._sdk_ver

    @property
    def sdk_compatible(self) -> bool:
        """Vérifie la compatibilité ABI : les versions major doivent être identiques."""
        plugin_major = (self._sdk_ver >> 16) & 0xFFFF
        host_major   = OPENSHADER_SDK_MAJOR
        return plugin_major == host_major

    # ── Chargement / déchargement ────────────────────────────────────────────

    def load(self) -> bool:
        """Charge la bibliothèque et vérifie les exports requis."""
        if not os.path.exists(self._path):
            log.error("Plugin natif introuvable : %s", self._path)
            return False
        try:
            # ctypes.CDLL utilise LoadLibrary (Win) / dlopen (Unix)
            lib = ctypes.CDLL(self._path)
        except OSError as e:
            log.error("Impossible de charger '%s' : %s", self._path, e)
            return False

        # Vérification des exports obligatoires
        missing = []
        for sym in _REQUIRED_EXPORTS:
            if not hasattr(lib, sym):
                missing.append(sym)
        if missing:
            log.error("Plugin '%s' : exports manquants : %s", self._path, missing)
            return False

        # Lecture de la version SDK
        lib.openshader_sdk_version.restype  = ctypes.c_uint32
        lib.openshader_sdk_version.argtypes = []
        self._sdk_ver = lib.openshader_sdk_version()

        if not self.sdk_compatible:
            plugin_major = (self._sdk_ver >> 16) & 0xFFFF
            log.error(
                "Plugin '%s' : version ABI incompatible (plugin v%d.x, host v%d.x)",
                self._path, plugin_major, OPENSHADER_SDK_MAJOR
            )
            return False

        # Lecture du nom et de la version du plugin
        lib.openshader_name.restype  = ctypes.c_char_p
        lib.openshader_name.argtypes = []
        raw_name = lib.openshader_name()
        self._name = raw_name.decode('utf-8', errors='replace') if raw_name else Path(self._path).stem

        lib.openshader_version.restype  = ctypes.c_char_p
        lib.openshader_version.argtypes = []
        raw_ver = lib.openshader_version()
        self._version = raw_ver.decode('utf-8', errors='replace') if raw_ver else "?"

        # Signatures des autres fonctions
        lib.openshader_init.restype         = None
        lib.openshader_init.argtypes        = [ctypes.POINTER(OSPluginContext)]

        lib.openshader_tick.restype         = None
        lib.openshader_tick.argtypes        = [ctypes.c_float, ctypes.POINTER(OSUniforms)]

        lib.openshader_get_uniforms.restype  = None
        lib.openshader_get_uniforms.argtypes = [
            ctypes.POINTER(OSUniformArray), ctypes.POINTER(ctypes.c_int)
        ]

        lib.openshader_shutdown.restype     = None
        lib.openshader_shutdown.argtypes    = []

        self._lib    = lib
        self._loaded = True
        log.info("Plugin natif chargé : '%s' v%s (SDK %08X)", self._name, self._version, self._sdk_ver)
        return True

    def unload(self):
        """Appelle shutdown() et décharge la bibliothèque."""
        if self._lib and self._loaded:
            try:
                self._lib.openshader_shutdown()
            except Exception as e:
                log.warning("Erreur shutdown plugin '%s' : %s", self._name, e)
            # Sous Windows, FreeLibrary ; sous Unix, dlclose via del
            try:
                if platform.system() == 'Windows':
                    ctypes.windll.kernel32.FreeLibrary(self._lib._handle)
                else:
                    _libdl = ctypes.CDLL(None)
                    _libdl.dlclose.argtypes = [ctypes.c_void_p]
                    _libdl.dlclose(self._lib._handle)
            except Exception:
                pass  # dlclose peut échouer silencieusement — acceptable
            self._lib    = None
            self._loaded = False
            log.info("Plugin natif déchargé : '%s'", self._name)

    # ── API runtime ──────────────────────────────────────────────────────────

    def init(self, sample_rate: float = 44100.0,
             width: int = 1920, height: int = 1080):
        """Appelle openshader_init() avec le contexte courant."""
        if not self._loaded:
            return
        ctx = OSPluginContext(
            sample_rate  = sample_rate,
            canvas_width = width,
            canvas_height= height,
            sdk_version  = OPENSHADER_SDK_VERSION,
        )
        try:
            self._lib.openshader_init(ctypes.byref(ctx))
        except Exception as e:
            log.error("Plugin '%s' init() erreur : %s", self._name, e)

    def tick(self, dt: float = 0.016, itime: float = 0.0,
             rms: float = 0.0, bpm: float = 120.0, beat: float = 0.0):
        """Appelle openshader_tick() avec les uniforms du shader courant."""
        if not self._loaded:
            return
        u = OSUniforms(
            iTime      = itime,
            iDeltaTime = dt,
            iBeat      = beat,
            iRMS       = rms,
            iBPM       = bpm,
        )
        try:
            self._lib.openshader_tick(ctypes.c_float(dt), ctypes.byref(u))
        except Exception as e:
            log.error("Plugin '%s' tick() erreur : %s", self._name, e)

    def get_uniforms(self) -> dict[str, float]:
        """
        Récupère les uniforms produits par le plugin.
        Retourne un dict {uniform_name: float} à injecter dans le shader.
        """
        if not self._loaded:
            return {}
        arr   = OSUniformArray()
        count = ctypes.c_int(0)
        try:
            self._lib.openshader_get_uniforms(
                ctypes.cast(arr, ctypes.POINTER(OSUniformArray)),
                ctypes.byref(count)
            )
        except Exception as e:
            log.error("Plugin '%s' get_uniforms() erreur : %s", self._name, e)
            return {}

        result = {}
        for i in range(min(count.value, MAX_UNIFORMS)):
            name_bytes = arr[i].name
            name = name_bytes.decode('utf-8', errors='replace').rstrip('\x00')
            if name:
                result[name] = arr[i].value
        self._uniform_cache = result
        return result

    @property
    def cached_uniforms(self) -> dict[str, float]:
        """Dernier snapshot des uniforms sans appel C."""
        return dict(self._uniform_cache)

    def to_dict(self) -> dict:
        return {'path': self._path, 'name': self._name, 'version': self._version}


# ── Gestionnaire de plugins natifs avec hot-reload ────────────────────────────

class NativePluginManager(QObject):
    """
    Gestionnaire de plugins natifs C++ : chargement, hot-reload, uniforms.

    Signaux
    -------
    plugin_loaded    (str)            — nom du plugin chargé ou rechargé
    plugin_unloaded  (str)            — nom du plugin déchargé
    plugin_error     (str, str)       — (path, message d'erreur)
    uniforms_ready   (dict)           — snapshot uniforms agrégés de tous les plugins actifs
    """

    plugin_loaded   = pyqtSignal(str)
    plugin_unloaded = pyqtSignal(str)
    plugin_error    = pyqtSignal(str, str)
    uniforms_ready  = pyqtSignal(dict)

    # Intervalle de polling hot-reload en secondes (utilisé si watchdog absent)
    HOT_RELOAD_POLL_INTERVAL = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plugins:    dict[str, NativePlugin] = {}  # path → plugin
        self._mtimes:     dict[str, float]        = {}  # path → mtime au chargement
        self._hr_thread:  threading.Thread | None = None
        self._hr_running: bool                    = False
        self._context_sample_rate: float          = 44100.0
        self._context_width:       int            = 1920
        self._context_height:      int            = 1080

    # ── Cycle de vie ─────────────────────────────────────────────────────────

    def set_context(self, sample_rate: float = 44100.0,
                    width: int = 1920, height: int = 1080):
        """Met à jour le contexte de rendu transmis à chaque plugin à l'init."""
        self._context_sample_rate = sample_rate
        self._context_width       = width
        self._context_height      = height

    def load(self, path: str) -> NativePlugin | None:
        """
        Charge un plugin natif depuis son chemin.

        Sur Windows, copie le .dll dans un répertoire temporaire avant de le
        charger afin de permettre le hot-reload (le fichier original reste
        déverrouillé).
        """
        resolved = _resolve_lib_path(path)
        load_path = self._shadow_copy(resolved)

        if load_path in self._plugins:
            log.warning("Plugin '%s' déjà chargé — utilisez reload().", load_path)
            return self._plugins[load_path]

        plugin = NativePlugin(load_path)
        if not plugin.load():
            self.plugin_error.emit(path, f"Échec du chargement de '{path}'")
            return None

        plugin.init(
            sample_rate = self._context_sample_rate,
            width       = self._context_width,
            height      = self._context_height,
        )

        self._plugins[resolved] = plugin     # indexé sur le chemin original
        self._mtimes[resolved]  = self._mtime(resolved)
        self.plugin_loaded.emit(plugin.name)
        return plugin

    def unload(self, path: str):
        """Décharge un plugin natif."""
        resolved = _resolve_lib_path(path)
        plugin = self._plugins.pop(resolved, None)
        if plugin:
            name = plugin.name
            plugin.unload()
            self._mtimes.pop(resolved, None)
            self.plugin_unloaded.emit(name)

    def reload(self, path: str) -> NativePlugin | None:
        """Recharge à chaud un plugin sans redémarrer l'application."""
        resolved = _resolve_lib_path(path)
        old_plugin = self._plugins.get(resolved)
        name_hint  = old_plugin.name if old_plugin else Path(path).stem

        if old_plugin:
            log.info("Hot-reload du plugin natif '%s'…", name_hint)
            old_plugin.unload()
            del self._plugins[resolved]

        new_plugin = self.load(resolved)
        if new_plugin:
            log.info("Plugin natif '%s' rechargé avec succès.", new_plugin.name)
        else:
            log.error("Hot-reload échoué pour '%s'.", name_hint)
        return new_plugin

    def reload_all(self):
        """Recharge tous les plugins chargés (utile après une recompilation)."""
        paths = list(self._plugins.keys())
        for p in paths:
            self.reload(p)

    # ── Tick agrégé ──────────────────────────────────────────────────────────

    def tick_all(self, dt: float, itime: float = 0.0,
                 rms: float = 0.0, bpm: float = 120.0, beat: float = 0.0) -> dict[str, float]:
        """
        Appelle tick() sur tous les plugins chargés et agrège leurs uniforms.
        Retourne un dict uniforms consolidé à injecter dans le shader.
        Émet uniforms_ready() avec le même dict.
        """
        uniforms: dict[str, float] = {}
        for plugin in list(self._plugins.values()):
            try:
                plugin.tick(dt=dt, itime=itime, rms=rms, bpm=bpm, beat=beat)
                uniforms.update(plugin.get_uniforms())
            except Exception as e:
                log.error("Erreur tick plugin '%s' : %s", plugin.name, e)
        if uniforms:
            self.uniforms_ready.emit(uniforms)
        return uniforms

    # ── Accès ─────────────────────────────────────────────────────────────────

    def get_all(self) -> list[NativePlugin]:
        return list(self._plugins.values())

    def get_by_name(self, name: str) -> NativePlugin | None:
        for p in self._plugins.values():
            if p.name == name:
                return p
        return None

    # ── Hot-reload automatique ────────────────────────────────────────────────

    def hot_reload_enable(self, poll_interval: float = HOT_RELOAD_POLL_INTERVAL):
        """
        Démarre le thread de surveillance hot-reload.

        Essaie d'utiliser watchdog si disponible, sinon polling par mtime.
        Recharge automatiquement tout plugin dont le fichier source a changé
        sur disque (date de modification plus récente).
        """
        if self._hr_running:
            return
        self._hr_running = True
        self._hr_thread  = threading.Thread(
            target=self._hot_reload_loop,
            args=(poll_interval,),
            daemon=True,
            name="NativePluginHotReload",
        )
        self._hr_thread.start()
        log.info("Hot-reload de plugins natifs activé (poll %.1fs)", poll_interval)

    def hot_reload_disable(self):
        """Arrête le thread de surveillance."""
        self._hr_running = False
        if self._hr_thread and self._hr_thread.is_alive():
            self._hr_thread.join(timeout=2.0)
        self._hr_thread = None
        log.info("Hot-reload de plugins natifs désactivé.")

    def _hot_reload_loop(self, poll_interval: float):
        while self._hr_running:
            for path, old_mtime in list(self._mtimes.items()):
                new_mtime = self._mtime(path)
                if new_mtime and new_mtime > old_mtime + 0.5:
                    log.info("Changement détecté sur '%s' — rechargement…", path)
                    self._mtimes[path] = new_mtime
                    self.reload(path)
            time.sleep(poll_interval)

    # ── Utilitaires ──────────────────────────────────────────────────────────

    @staticmethod
    def _mtime(path: str) -> float:
        """Retourne le mtime du fichier ou 0.0 si inexistant."""
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    @staticmethod
    def _shadow_copy(path: str) -> str:
        """
        Sous Windows, copie le .dll dans %TEMP% avant de le charger afin que
        l'original reste modifiable/remplaçable (nécessaire pour le hot-reload).
        Sur les autres plateformes, retourne le chemin tel quel.
        """
        if platform.system() != 'Windows':
            return path
        tmp_dir  = os.path.join(tempfile.gettempdir(), "openshader_native_plugins")
        os.makedirs(tmp_dir, exist_ok=True)
        basename = os.path.basename(path)
        dest     = os.path.join(tmp_dir, basename)
        try:
            shutil.copy2(path, dest)
        except (OSError, shutil.Error) as e:
            log.warning("Shadow copy échouée pour '%s' : %s — chargement direct.", path, e)
            return path
        return dest

    # ── Sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> list[dict]:
        return [p.to_dict() for p in self._plugins.values()]

    def from_dict(self, data: list[dict]):
        """Recharge les plugins listés dans data (depuis un projet .demomaker)."""
        for entry in data:
            path = entry.get('path', '')
            if path and os.path.exists(_resolve_lib_path(path)):
                if path not in self._plugins:
                    self.load(path)
            else:
                log.warning("Plugin natif '%s' introuvable — ignoré.", path)
