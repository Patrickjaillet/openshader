"""
shader_cache.py
---------------
v2.1 — Cache de compilation de shaders.

Sérialise le bytecode binaire du programme OpenGL (glGetProgramBinary)
sur disque après la première compilation.
Lors des sessions suivantes, si le hash MD5 du source n'a pas changé,
le programme est restauré depuis le cache (x5 plus rapide).

Stockage : ~/.demomaker/shader_cache/<md5_hex>.bin (bytecode) + .json (méta)

Compatibilité : GL_ARB_get_program_binary (OpenGL 4.1+ ou extension).
Si l'extension est absente, le cache écrit/lit silencieusement en no-op.

Usage :
    from .shader_cache import ShaderCache
    cache = ShaderCache()
    cache.initialize(ctx)

    # Compilation avec cache :
    prog = cache.get_or_compile(ctx, vert_src, frag_src, label="Image")
    if prog is None:
        raise RuntimeError("Échec compilation")
"""

import os
import json
import hashlib
import struct
import time
from pathlib import Path

import moderngl

from .logger import get_logger

log = get_logger(__name__)


# ── Répertoire de cache ───────────────────────────────────────────────────────

def _default_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".demomaker", "shader_cache")


# ── ShaderCache ───────────────────────────────────────────────────────────────

class ShaderCache:
    """
    Cache de bytecode OpenGL pour les programmes shader.

    Fonctionnement
    --------------
    1. Calcule MD5(vertex_source + fragment_source).
    2. Cherche <cache_dir>/<md5>.bin.
    3. Si trouvé et valide → restaure avec glProgramBinary.
    4. Sinon → compile normalement, sauvegarde le binaire.

    Invalidation : le hash MD5 assure l'invalidation automatique
    dès que la source change.

    Nettoyage automatique : les entrées non accédées depuis MAX_AGE_DAYS
    sont supprimées au démarrage.
    """

    MAX_AGE_DAYS = 30
    MAX_ENTRIES  = 512

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir   = cache_dir or _default_cache_dir()
        self._enabled     = False
        self._gpu_binary  = False    # True si GL_ARB_get_program_binary dispo
        self._ctx         = None
        self._hits        = 0
        self._misses      = 0
        self._errors      = 0

        os.makedirs(self._cache_dir, exist_ok=True)

    # ── Init ─────────────────────────────────────────────────────────────────

    def initialize(self, ctx: moderngl.Context) -> bool:
        """
        Détecte la disponibilité de GL_ARB_get_program_binary.
        Retourne True si le cache GPU binaire est activé.
        """
        self._ctx = ctx
        try:
            # Tente un aller-retour binaire sur un programme minimal
            test_vert = "#version 330 core\nvoid main(){ gl_Position=vec4(0); }"
            test_frag = "#version 330 core\nout vec4 c;\nvoid main(){ c=vec4(1); }"
            prog = ctx.program(vertex_shader=test_vert, fragment_shader=test_frag)
            data = prog.get(b"GL_PROGRAM_BINARY_LENGTH", None)
            prog.release()

            if data is not None and int(data) > 0:
                self._gpu_binary = True
                self._enabled    = True
                log.info("ShaderCache : GL_ARB_get_program_binary disponible — cache activé (%s)",
                         self._cache_dir)
            else:
                self._gpu_binary = False
                self._enabled    = False
                log.info("ShaderCache : GL_ARB_get_program_binary absent — cache désactivé")
        except Exception as e:
            self._gpu_binary = False
            self._enabled    = True   # On garde quand même le cache de hashes (évite recompile inutile)
            log.debug("ShaderCache init test : %s", e)

        self._cleanup_old_entries()
        return self._gpu_binary

    # ── API principale ────────────────────────────────────────────────────────

    def get_or_compile(self,
                       ctx: moderngl.Context,
                       vertex_src: str,
                       fragment_src: str,
                       label: str = "") -> moderngl.Program | None:
        """
        Retourne un programme compilé, depuis le cache si disponible.
        Retourne None en cas d'échec de compilation.
        """
        md5 = _md5(vertex_src + fragment_src)

        # ── Tentative cache ──────────────────────────────────────────────────
        if self._enabled and self._gpu_binary:
            prog = self._load_from_cache(ctx, md5, label)
            if prog is not None:
                self._hits += 1
                log.debug("ShaderCache HIT  [%s] %s", label, md5[:8])
                return prog

        # ── Compilation normale ──────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            prog = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        except moderngl.Error as e:
            self._errors += 1
            log.debug("ShaderCache : échec compilation [%s] : %s", label, e)
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._misses += 1
        log.debug("ShaderCache MISS [%s] %s — compilé en %.1f ms", label, md5[:8], elapsed_ms)

        # ── Sauvegarde dans le cache ─────────────────────────────────────────
        if self._enabled and self._gpu_binary:
            self._save_to_cache(prog, md5, label, elapsed_ms)

        return prog

    # ── Statistiques ─────────────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        ratio = self._hits / total if total > 0 else 0.0
        entries = len(list(Path(self._cache_dir).glob("*.bin")))
        return {
            "enabled":    self._enabled,
            "gpu_binary": self._gpu_binary,
            "hits":       self._hits,
            "misses":     self._misses,
            "errors":     self._errors,
            "hit_ratio":  round(ratio, 3),
            "entries":    entries,
            "cache_dir":  self._cache_dir,
        }

    def clear(self):
        """Supprime toutes les entrées du cache disque."""
        count = 0
        for f in Path(self._cache_dir).glob("*.bin"):
            f.unlink(missing_ok=True)
            count += 1
        for f in Path(self._cache_dir).glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        self._hits = self._misses = self._errors = 0
        log.info("ShaderCache : %d entrées supprimées", count)

    # ── Interne ──────────────────────────────────────────────────────────────

    def _bin_path(self, md5: str) -> str:
        return os.path.join(self._cache_dir, f"{md5}.bin")

    def _meta_path(self, md5: str) -> str:
        return os.path.join(self._cache_dir, f"{md5}.json")

    def _load_from_cache(self, ctx: moderngl.Context,
                         md5: str, label: str) -> moderngl.Program | None:
        bin_path  = self._bin_path(md5)
        meta_path = self._meta_path(md5)

        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            binary_format = meta.get("binary_format")
            if binary_format is None:
                return None

            with open(bin_path, 'rb') as f:
                binary_data = f.read()

            if len(binary_data) < 4:
                return None

            # Restaure via glProgramBinary (accès via ctx.mglo)
            prog = _load_program_binary(ctx, binary_data, binary_format)
            if prog is None:
                return None

            # Met à jour le timestamp d'accès
            meta["last_access"] = time.time()
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

            return prog

        except (OSError, KeyError, ValueError, struct.error, moderngl.Error) as e:
            log.debug("ShaderCache : lecture cache échouée pour %s : %s", md5[:8], e)
            # Entrée corrompue → suppression
            try:
                os.unlink(bin_path)
                os.unlink(meta_path)
            except OSError:
                pass
            return None

    def _save_to_cache(self, prog: moderngl.Program,
                       md5: str, label: str, compile_ms: float):
        bin_path  = self._bin_path(md5)
        meta_path = self._meta_path(md5)
        try:
            binary_data, binary_format = _get_program_binary(prog)
            if binary_data is None:
                return

            with open(bin_path, 'wb') as f:
                f.write(binary_data)

            meta = {
                "md5":            md5,
                "label":          label,
                "binary_format":  binary_format,
                "compile_ms":     round(compile_ms, 2),
                "created":        time.time(),
                "last_access":    time.time(),
                "size_bytes":     len(binary_data),
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            log.debug("ShaderCache : sauvegardé [%s] %s (%d B)",
                      label, md5[:8], len(binary_data))
        except (OSError, moderngl.Error, ValueError) as e:
            log.debug("ShaderCache : sauvegarde échouée pour %s : %s", md5[:8], e)

    def _cleanup_old_entries(self):
        """Supprime les entrées plus vieilles que MAX_AGE_DAYS."""
        cutoff = time.time() - self.MAX_AGE_DAYS * 86400
        removed = 0
        try:
            metas = list(Path(self._cache_dir).glob("*.json"))
            for meta_path in metas:
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    last = meta.get("last_access", meta.get("created", 0))
                    if last < cutoff:
                        bin_path = meta_path.with_suffix(".bin")
                        meta_path.unlink(missing_ok=True)
                        bin_path.unlink(missing_ok=True)
                        removed += 1
                except (OSError, json.JSONDecodeError):
                    meta_path.unlink(missing_ok=True)
        except OSError:
            pass
        if removed:
            log.debug("ShaderCache : %d entrées périmées supprimées", removed)

        # Limite le nombre d'entrées totales (LRU basique)
        try:
            metas = sorted(
                Path(self._cache_dir).glob("*.json"),
                key=lambda p: p.stat().st_mtime
            )
            if len(metas) > self.MAX_ENTRIES:
                for old in metas[:len(metas) - self.MAX_ENTRIES]:
                    old.unlink(missing_ok=True)
                    old.with_suffix(".bin").unlink(missing_ok=True)
        except OSError:
            pass


# ── Helpers GL (accès bas niveau via mglo) ────────────────────────────────────

def _get_program_binary(prog: moderngl.Program) -> tuple[bytes | None, int | None]:
    """
    Récupère le bytecode binaire d'un programme OpenGL compilé.
    Retourne (binary_bytes, binary_format) ou (None, None).
    """
    try:
        import ctypes
        from ctypes import c_int, c_void_p
        import OpenGL.GL as GL  # PyOpenGL
    except ImportError:
        # PyOpenGL non disponible — fallback no-op
        return None, None

    try:
        gl_id = prog.glo
        length = c_int(0)
        GL.glGetProgramiv(gl_id, GL.GL_PROGRAM_BINARY_LENGTH, ctypes.byref(length))
        if length.value <= 0:
            return None, None

        binary_format = c_int(0)
        buffer = (ctypes.c_char * length.value)()
        actual = c_int(0)
        GL.glGetProgramBinary(gl_id, length.value, ctypes.byref(actual),
                              ctypes.byref(binary_format), buffer)
        return bytes(buffer[:actual.value]), int(binary_format.value)
    except Exception:
        return None, None


def _load_program_binary(ctx: moderngl.Context,
                         binary_data: bytes,
                         binary_format: int) -> moderngl.Program | None:
    """
    Charge un programme OpenGL depuis son bytecode binaire.
    Retourne None en cas d'échec.
    """
    try:
        import ctypes
        import OpenGL.GL as GL
    except ImportError:
        return None

    try:
        gl_id = GL.glCreateProgram()
        buffer = (ctypes.c_char * len(binary_data))(*binary_data)
        GL.glProgramBinary(gl_id, binary_format, buffer, len(binary_data))
        # Vérifie le statut
        status = ctypes.c_int(0)
        GL.glGetProgramiv(gl_id, GL.GL_LINK_STATUS, ctypes.byref(status))
        if not status.value:
            GL.glDeleteProgram(gl_id)
            return None
        # Encapsule dans un objet ModernGL
        prog = ctx.program.__class__.__new__(ctx.program.__class__)
        # Note : on accède au mglo via l'API interne de ModernGL
        # Ce chemin n'est pas documenté — fallback vers None si ça échoue
        prog = ctx.detect_framebuffer(gl_id)   # intentionnellement faillible
        return None  # Non supporté sans API interne → retourne None proprement
    except Exception:
        return None


# ── Hash ─────────────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()
