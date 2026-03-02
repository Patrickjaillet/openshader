"""
shader_engine.py
----------------
Moteur OpenGL via ModernGL. Gère shaders Shadertoy ET GLSL pur.
"""

import moderngl
import numpy as np
import time
import re
from PyQt6.QtGui import QImage

from .logger import get_logger

log = get_logger(__name__)

# ── Préprocesseur GLSL ────────────────────────────────────────────────────────
# Supporte : #define, #undef, #if/#ifdef/#ifndef/#elif/#else/#endif, #include

def preprocess_glsl(source: str,
                    extra_defines: dict | None = None,
                    include_dirs: list[str] | None = None,
                    _visited: set | None = None,
                    _source_file: str | None = None) -> str:
    """
    Préprocesseur GLSL minimaliste.

    - ``#define NAME`` / ``#define NAME value``
    - ``#undef NAME``
    - ``#ifdef NAME`` / ``#ifndef NAME`` / ``#if expr``
    - ``#elif expr`` / ``#else`` / ``#endif``
    - ``#include "file.glsl"``  (résolu depuis include_dirs, chemins relatifs)
    - ``#pragma hw_performance`` → injecte ``#define HW_PERFORMANCE 1``

    Les blocs GLSL ignorés sont remplacés par des lignes vides pour conserver
    la numérotation et donc la correspondance erreur ↔ ligne source.
    """
    import os

    if _visited is None:
        _visited = set()
    if include_dirs is None:
        include_dirs = []

    defines: dict[str, str] = {}
    if extra_defines:
        defines.update({k: str(v) for k, v in extra_defines.items()})

    def _eval_condition(expr: str) -> bool:
        """Évalue une expression #if/#elif simple avec les defines courants."""
        expr = expr.strip()
        # Substitue defined(X) et defined X
        def _defined(m):
            return '1' if m.group(1) in defines else '0'
        expr = re.sub(r'defined\s*\(\s*(\w+)\s*\)', _defined, expr)
        expr = re.sub(r'defined\s+(\w+)', _defined, expr)
        # Substitue les macros connues
        for k, v in defines.items():
            expr = re.sub(r'\b' + re.escape(k) + r'\b', v or '1', expr)
        # Remplace les identifiants inconnus par 0
        expr = re.sub(r'\b[A-Za-z_]\w*\b', '0', expr)
        # Opérateurs C → Python
        expr = expr.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
        try:
            return bool(eval(expr))  # noqa: S307 — expression GLSL, pas user input
        except (SyntaxError, NameError, TypeError, ValueError, ZeroDivisionError):
            return False

    def _resolve_include(fname: str, current_file: str | None) -> str | None:
        """Cherche fname dans include_dirs + répertoire du fichier courant."""
        search = list(include_dirs)
        if current_file:
            search.insert(0, os.path.dirname(current_file))
        for d in search:
            candidate = os.path.join(d, fname)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _process(src: str, current_file: str | None = None) -> str:
        lines = src.splitlines()
        out   = []
        # Pile : chaque niveau = [enabled, seen_true, in_else]
        stack: list[list[bool]] = []
        active = lambda: all(s[0] for s in stack)

        for line in lines:
            stripped = line.strip()

            # ── #pragma hw_performance ──────────────────────────────────────
            if stripped == '#pragma hw_performance':
                if active():
                    defines['HW_PERFORMANCE'] = '1'
                out.append('' if not active() else line)
                continue

            # ── #define ─────────────────────────────────────────────────────
            m = re.match(r'#define\s+(\w+)(?:\s+(.*))?$', stripped)
            if m:
                if active():
                    defines[m.group(1)] = (m.group(2) or '').strip()
                out.append('')
                continue

            # ── #undef ──────────────────────────────────────────────────────
            m = re.match(r'#undef\s+(\w+)', stripped)
            if m:
                if active():
                    defines.pop(m.group(1), None)
                out.append('')
                continue

            # ── #ifdef / #ifndef / #if ──────────────────────────────────────
            m = re.match(r'#ifdef\s+(\w+)', stripped)
            if m:
                cond = m.group(1) in defines
                stack.append([cond, cond, False])
                out.append('')
                continue

            m = re.match(r'#ifndef\s+(\w+)', stripped)
            if m:
                cond = m.group(1) not in defines
                stack.append([cond, cond, False])
                out.append('')
                continue

            m = re.match(r'#if\s+(.+)', stripped)
            if m:
                cond = _eval_condition(m.group(1))
                stack.append([cond, cond, False])
                out.append('')
                continue

            # ── #elif ───────────────────────────────────────────────────────
            m = re.match(r'#elif\s+(.+)', stripped)
            if m and stack:
                top = stack[-1]
                if top[2]:  # déjà dans #else → erreur, on ignore
                    out.append('')
                    continue
                if top[1]:  # un bloc précédent était vrai → désactive
                    top[0] = False
                else:
                    cond = _eval_condition(m.group(1))
                    top[0] = cond
                    if cond: top[1] = True
                out.append('')
                continue

            # ── #else ───────────────────────────────────────────────────────
            if stripped == '#else' and stack:
                top = stack[-1]
                top[2] = True
                top[0] = not top[1]
                out.append('')
                continue

            # ── #endif ──────────────────────────────────────────────────────
            if stripped == '#endif' and stack:
                stack.pop()
                out.append('')
                continue

            # ── #include ────────────────────────────────────────────────────
            m = re.match(r'#include\s+"([^"]+)"', stripped)
            if m:
                if active():
                    fname   = m.group(1)
                    fpath   = _resolve_include(fname, current_file)
                    if fpath and fpath not in _visited:
                        _visited.add(fpath)
                        try:
                            inc_src = open(fpath, encoding='utf-8').read()
                            inc_out = _process(inc_src, current_file=fpath)
                            out.append(f'// --- include: {fname} ---')
                            out.append(inc_out)
                            out.append(f'// --- end include: {fname} ---')
                        except OSError as e:
                            log.warning("#include '%s' introuvable : %s", fpath, e)
                            out.append(f'// #include "{fname}" — ERREUR: {e}')
                    elif fpath in _visited:
                        out.append(f'// #include "{fname}" — déjà inclus (guard)')
                    else:
                        log.warning("#include '%s' : fichier introuvable dans %s", fname, include_dirs)
                        out.append(f'// #include "{fname}" — INTROUVABLE')
                else:
                    out.append('')
                continue

            # ── Ligne ordinaire ─────────────────────────────────────────────
            if not active():
                out.append('')  # ligne masquée → vide (préserve numérotation)
                continue

            # Substitution des macros dans le code actif
            result = line
            for k, v in defines.items():
                if v:  # ne substitue que les macros avec valeur
                    result = re.sub(r'\b' + re.escape(k) + r'\b', v, result)
            out.append(result)

        return '\n'.join(out)

    return _process(source, current_file=_source_file)


VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
void main() { gl_Position = vec4(in_position, 0.0, 1.0); }
"""

SHADERTOY_HEADER = """
#version 330 core
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform vec3  iChannelResolution[4];
out vec4 _fragColor;
"""

SHADERTOY_FOOTER = """
void main() { mainImage(_fragColor, gl_FragCoord.xy); }
"""

GLSL_HEADER = """
#version 330 core
uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeDelta;
uniform int   uFrame;
out vec4 fragColor;
"""

# Header spécifique aux shaders de transition
# iChannel0 = scène A (entrant/sortant), iChannel1 = scène B (entrant)
# iProgress  = avancement de la transition [0.0 → 1.0]
TRANS_HEADER = """
#version 330 core
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform float iProgress;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
out vec4 _fragColor;
"""

TRANS_FOOTER = """
void main() { mainImage(_fragColor, gl_FragCoord.xy); }
"""

ERROR_SHADER = """
#version 330 core
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / 800.0;
    float c = mod(floor(uv.x*20.0)+floor(uv.y*20.0), 2.0);
    fragColor = vec4(c*0.6, 0.0, 0.0, 1.0);
}
"""


def detect_type(source: str) -> str:
    """Détecte 'shadertoy' (mainImage) ou 'glsl' (main)."""
    if re.search(r'\bvoid\s+mainImage\s*\(', source):
        return 'shadertoy'
    if re.search(r'\bvoid\s+main\s*\(', source):
        return 'glsl'
    return 'shadertoy'  # défaut


def detect_trans_type(source: str) -> bool:
    """Retourne True si le source est un shader de transition (mainImage + iProgress)."""
    has_main  = bool(re.search(r'\bvoid\s+mainImage\s*\(', source))
    has_prog  = bool(re.search(r'\biProgress\b', source))
    return has_main and has_prog


def _count_header_lines(header: str) -> int:
    """Retourne le nombre de lignes du header injecté (sans compter la ligne vide finale)."""
    return len(header.splitlines())


def build_source(source: str,
                 defines: dict | None = None,
                 include_dirs: list[str] | None = None,
                 source_path: str | None = None) -> tuple[str, str]:
    """Retourne (source_compilable, 'shadertoy'|'glsl')."""
    processed = preprocess_glsl(source, extra_defines=defines, include_dirs=include_dirs, _source_file=source_path)
    t = detect_type(processed)
    if t == 'shadertoy':
        clean = re.sub(r'#version\s+\d+\s*(core|compatibility)?\s*\n', '', processed)
        return SHADERTOY_HEADER + clean + SHADERTOY_FOOTER, 'shadertoy'
    else:
        if processed.strip().startswith('#version'):
            return processed, 'glsl'
        return GLSL_HEADER + processed, 'glsl'


def build_trans_source(source: str,
                       defines: dict | None = None,
                       include_dirs: list[str] | None = None,
                       source_path: str | None = None) -> str:
    """Retourne la source compilable d'un shader de transition."""
    processed = preprocess_glsl(source, extra_defines=defines, include_dirs=include_dirs, _source_file=source_path)
    clean = re.sub(r'#version\s+\d+\s*(core|compatibility)?\s*\n', '', processed)
    return TRANS_HEADER + clean + TRANS_FOOTER


def get_header_line_count(source: str) -> int:
    """Retourne le nombre de lignes du header injecté pour un source donné.
    Utilisé par CodeEditor pour calculer l'offset d'erreur GLSL dynamiquement."""
    t = detect_type(source)
    if t == 'shadertoy':
        return _count_header_lines(SHADERTOY_HEADER)
    # GLSL pur avec #version déjà présent → pas de header injecté
    if source.strip().startswith('#version'):
        return 0
    return _count_header_lines(GLSL_HEADER)



import re as _re

def _strip_unresolved_includes(source: str) -> str:
    """
    Remplace les directives #include résiduelles par un commentaire GLSL.
    Évite que moderngl lève une KeyError en tentant de les résoudre lui-même
    quand le preprocesseur Python n'a pas pu les résoudre (lib_dir absent,
    fichier introuvable…).
    """
    def _replace(m):
        return f'// [include non résolu: {m.group(1)}]'
    return _re.sub(r'#include\s+"([^"]+)"', _replace, source)


class ShaderEngine:
    def __init__(self, width=800, height=450, lib_dir=None):
        self.width  = width
        self.height = height
        self.ctx    = None

        import os as _os
        import sys as _sys
        # Résolution robuste du dossier shaders/ (lib_dir).
        # Plusieurs stratégies en cascade pour fonctionner quel que soit
        # le répertoire de lancement (IDE, double-clic, terminal, etc.).
        if lib_dir is None:
            _here = _os.path.dirname(_os.path.abspath(__file__))

            # Stratégie 1 : src/../shaders  (structure normale du projet)
            _candidate = _os.path.normpath(_os.path.join(_here, '..', 'shaders'))

            if not _os.path.isdir(_candidate):
                # Stratégie 2 : CWD/shaders  (lancé depuis la racine du projet)
                _candidate = _os.path.normpath(_os.path.join(_os.getcwd(), 'shaders'))

            if not _os.path.isdir(_candidate):
                # Stratégie 3 : dossier de main.py/shaders  (sys.argv[0])
                _main_dir = _os.path.dirname(_os.path.abspath(_sys.argv[0]))
                _candidate = _os.path.normpath(_os.path.join(_main_dir, 'shaders'))

            if not _os.path.isdir(_candidate):
                # Stratégie 4 : src/shaders  (layout alternatif)
                _candidate = _os.path.normpath(_os.path.join(_here, 'shaders'))

            lib_dir = _candidate

        self.lib_dir: str = lib_dir if _os.path.isdir(str(lib_dir)) else ''
        if self.lib_dir:
            log.debug("ShaderEngine lib_dir: %s", self.lib_dir)
        else:
            log.warning(
                "ShaderEngine: dossier shaders/ introuvable — "
                "les #include GLSL ne seront pas résolus. Cherché: %s", lib_dir)

        self.pass_names = ['Image', 'Buffer A', 'Buffer B', 'Buffer C', 'Buffer D', 'Post']
        self.programs  = {p: None for p in self.pass_names}
        self.vaos      = {p: None for p in self.pass_names}
        self.sources   = {p: '' for p in self.pass_names}
        self.errors    = {p: None for p in self.pass_names}
        self.types     = {p: 'shadertoy' for p in self.pass_names}

        self.buffer_fbos     = {p: [None, None] for p in self.pass_names if p != 'Image'}
        self.buffer_textures = {p: [None, None] for p in self.pass_names if p != 'Image'}

        self.image_fbo     = None
        self.image_texture = None
        self.textures      = [None] * 4

        # ── Passe Transition ─────────────────────────────────────────────────
        # Shaders de transition : iChannel0=scèneA, iChannel1=scèneB, iProgress
        self.trans_program  = None
        self.trans_vao      = None
        self.trans_source   = ''
        self.trans_error    = None
        # FBOs intermédiaires pour rendre scèneA et scèneB indépendamment
        self.scene_a_fbo      = None
        self.scene_a_texture  = None
        self.scene_b_fbo      = None
        self.scene_b_texture  = None
        # Programme/VAO de la scène B (source GLSL chargé séparément)
        self.scene_b_program  = None
        self.scene_b_vao      = None
        self.scene_b_source   = ''
        self.scene_b_error    = None
        self.scene_b_type     = 'shadertoy'
        # Paramètre de progression de la transition (0→1)
        self._trans_progress  = 0.0
        # Indique si la transition est active ce frame
        self._trans_active    = False

        self._error_prog = None
        self._error_vao  = None
        self._frame      = 0
        self._last_time  = 0.0
        self.extra_uniforms: dict = {}

        # ── Layers multi-pistes ──────────────────────────────────────────────
        # Chaque piste shader active contribue à un layer rendu indépendamment
        # puis composé par alpha-blend dans l'ordre (layer 0 = fond, layer N = dessus).
        # MAX_LAYERS layers simultanés.
        MAX_LAYERS = 8
        self._layer_programs:  list = [None] * MAX_LAYERS
        self._layer_vaos:      list = [None] * MAX_LAYERS
        self._layer_sources:   list = [''] * MAX_LAYERS
        self._layer_types:     list = ['shadertoy'] * MAX_LAYERS
        self._layer_errors:    list = [None] * MAX_LAYERS
        self._layer_fbos:      list = [None] * MAX_LAYERS
        self._layer_textures:  list = [None] * MAX_LAYERS
        self._active_layer_paths: list[str] = []   # chemins dans l'ordre des pistes
        self.MAX_LAYERS = MAX_LAYERS

    # ── Init ─────────────────────────────────────────────────────────────────

    def initialize(self, ctx: moderngl.Context):
        self.ctx = ctx
        verts = np.array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1], dtype='f4')
        self._vbo = ctx.buffer(verts)

        self.image_texture = ctx.texture((self.width, self.height), 4, dtype='f1')
        self.image_fbo     = ctx.framebuffer(color_attachments=[self.image_texture])

        for name in self.buffer_fbos:
            t1 = ctx.texture((self.width, self.height), 4, dtype='f1')
            t2 = ctx.texture((self.width, self.height), 4, dtype='f1')
            self.buffer_textures[name] = [t1, t2]
            self.buffer_fbos[name]     = [ctx.framebuffer(color_attachments=[t1]),
                                          ctx.framebuffer(color_attachments=[t2])]

        # FBOs de scène A/B pour les transitions
        self.scene_a_texture = ctx.texture((self.width, self.height), 4, dtype='f1')
        self.scene_a_fbo     = ctx.framebuffer(color_attachments=[self.scene_a_texture])
        self.scene_b_texture = ctx.texture((self.width, self.height), 4, dtype='f1')
        self.scene_b_fbo     = ctx.framebuffer(color_attachments=[self.scene_b_texture])

        # FBOs pour le rendu multi-layer (pistes shader superposées)
        for i in range(self.MAX_LAYERS):
            tex = ctx.texture((self.width, self.height), 4, dtype='f1')
            self._layer_textures[i] = tex
            self._layer_fbos[i]     = ctx.framebuffer(color_attachments=[tex])

        # Shader de blend alpha : compose deux textures (src over dst)
        BLEND_FRAG = """
#version 330 core
uniform sampler2D uDst;
uniform sampler2D uSrc;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uDst, 0));
    vec4 dst = texture(uDst, uv);
    vec4 src_col = texture(uSrc, uv);
    // Porter-Duff src-over
    float a_out = src_col.a + dst.a * (1.0 - src_col.a);
    vec3 rgb = a_out > 0.0
        ? (src_col.rgb * src_col.a + dst.rgb * dst.a * (1.0 - src_col.a)) / a_out
        : vec3(0.0);
    fragColor = vec4(rgb, a_out);
}
"""
        try:
            self._blend_prog = ctx.program(vertex_shader=VERTEX_SHADER,
                                           fragment_shader=BLEND_FRAG)
            self._blend_vao  = ctx.vertex_array(self._blend_prog,
                                                [(self._vbo, '2f', 'in_position')])
        except moderngl.Error as e:
            log.error("Impossible de compiler le shader de blend : %s", e)
            self._blend_prog = None
            self._blend_vao  = None

        try:
            self._error_prog = ctx.program(vertex_shader=VERTEX_SHADER,
                                           fragment_shader=ERROR_SHADER)
            self._error_vao  = ctx.vertex_array(self._error_prog,
                                                [(self._vbo, '2f', 'in_position')])
        except moderngl.Error as e:
            log.error("Impossible de compiler le shader d'erreur : %s", e)

    # ── Compile ───────────────────────────────────────────────────────────────

    def load_shader_source(self, source: str, pass_name: str,
                           source_path: str | None = None) -> tuple[bool, str]:
        """source_path : chemin absolu du fichier .st/.glsl, utilisé pour
        résoudre les #include relatifs (ex: #include "lib/noise.glsl")."""
        if pass_name not in self.pass_names:
            return False, f"Pass inconnue: {pass_name}"
        if not source.strip():
            self._release_pass(pass_name)
            return True, ''
        _idirs = [self.lib_dir] if self.lib_dir else []
        compiled, stype = build_source(source, include_dirs=_idirs, source_path=source_path)
        self.types[pass_name] = stype
        # Filet de sécurité : si un #include n'a pas pu être résolu par le
        # preprocesseur (lib_dir absent, fichier introuvable…), moderngl lève
        # une KeyError en tentant de le résoudre lui-même. On remplace les
        # #include résiduels par un commentaire pour éviter ce crash.
        compiled = _strip_unresolved_includes(compiled)
        try:
            prog = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                    fragment_shader=compiled)
            vao  = self.ctx.vertex_array(prog, [(self._vbo, '2f', 'in_position')])
            self._release_pass(pass_name)
            self.programs[pass_name] = prog
            self.vaos[pass_name]     = vao
            self.sources[pass_name]  = source
            self.errors[pass_name]   = None
            return True, ''
        except moderngl.Error as e:
            self.errors[pass_name] = str(e)
            return False, str(e)

    def load_trans_source(self, source: str,
                          source_path: str | None = None) -> tuple[bool, str]:
        """Charge et compile un shader de transition.
        iChannel0 = scène A, iChannel1 = scène B, iProgress = avancement."""
        if not source.strip():
            self._release_trans()
            return True, ''
        _idirs = [self.lib_dir] if self.lib_dir else []
        compiled = build_trans_source(source, include_dirs=_idirs, source_path=source_path)
        compiled = _strip_unresolved_includes(compiled)
        try:
            prog = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                    fragment_shader=compiled)
            vao  = self.ctx.vertex_array(prog, [(self._vbo, '2f', 'in_position')])
            self._release_trans()
            self.trans_program = prog
            self.trans_vao     = vao
            self.trans_source  = source
            self.trans_error   = None
            return True, ''
        except moderngl.Error as e:
            self.trans_error = str(e)
            return False, str(e)

    def load_scene_b_source(self, source: str,
                            source_path: str | None = None) -> tuple[bool, str]:
        """Charge la scène B pour le rendu de transition."""
        if not source.strip():
            self._release_scene_b()
            return True, ''
        _idirs = [self.lib_dir] if self.lib_dir else []
        compiled, stype = build_source(source, include_dirs=_idirs, source_path=source_path)
        self.scene_b_type = stype
        try:
            prog = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                    fragment_shader=compiled)
            vao  = self.ctx.vertex_array(prog, [(self._vbo, '2f', 'in_position')])
            self._release_scene_b()
            self.scene_b_program = prog
            self.scene_b_vao     = vao
            self.scene_b_source  = source
            self.scene_b_error   = None
            return True, ''
        except moderngl.Error as e:
            self.scene_b_error = str(e)
            return False, str(e)

    def set_transition(self, progress: float, active: bool):
        """Met à jour l'état de la transition (progress 0→1, active=True si en cours)."""
        self._trans_progress = max(0.0, min(1.0, progress))
        self._trans_active   = active

    def _release_trans(self):
        if self.trans_program: self.trans_program.release()
        if self.trans_vao:     self.trans_vao.release()
        self.trans_program = None
        self.trans_vao     = None
        self.trans_source  = ''
        self.trans_error   = None

    def load_layer_source(self, idx: int, source: str,
                          source_path: str | None = None) -> tuple[bool, str]:
        """Compile et stocke le programme GLSL pour le layer idx."""
        if idx < 0 or idx >= self.MAX_LAYERS:
            return False, f"Index layer {idx} hors limites (max {self.MAX_LAYERS - 1})"
        if not source.strip():
            self._release_layer(idx)
            return True, ''
        _idirs = [self.lib_dir] if self.lib_dir else []
        compiled, stype = build_source(source, include_dirs=_idirs, source_path=source_path)
        self._layer_types[idx] = stype
        try:
            prog = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                    fragment_shader=compiled)
            vao  = self.ctx.vertex_array(prog, [(self._vbo, '2f', 'in_position')])
            self._release_layer(idx)
            self._layer_programs[idx] = prog
            self._layer_vaos[idx]     = vao
            self._layer_sources[idx]  = source
            self._layer_errors[idx]   = None
            log.debug("Layer %d compilé (%s) — %s", idx, stype,
                      source_path or '<direct>')
            return True, ''
        except moderngl.Error as e:
            self._layer_errors[idx] = str(e)
            return False, str(e)

    def _release_layer(self, idx: int):
        try:
            if self._layer_vaos[idx]:     self._layer_vaos[idx].release()
            if self._layer_programs[idx]: self._layer_programs[idx].release()
        except moderngl.Error:
            pass
        self._layer_vaos[idx]     = None
        self._layer_programs[idx] = None
        self._layer_sources[idx]  = ''
        self._layer_errors[idx]   = None

    def set_active_layers(self, paths: list[str]):
        """
        Met à jour les layers actifs depuis une liste de chemins de fichiers shader.
        Appelé par _tick() avec la liste ordonnée des pistes shader actives au temps t.
        paths[0] = piste la plus basse (fond), paths[-1] = piste la plus haute (dessus).
        """
        import os as _os
        paths = [p for p in paths if p]  # filtre les vides
        self._active_layer_paths = paths

        for i in range(self.MAX_LAYERS):
            if i < len(paths):
                path = paths[i]
                # Recompile uniquement si le source a changé
                if self._layer_sources[i] != path:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        ok, err = self.load_layer_source(i, source, source_path=path)
                        if not ok:
                            log.error("Erreur layer %d '%s': %s", i,
                                      _os.path.basename(path), err)
                        else:
                            # Stocker le chemin comme "source" pour la comparaison
                            self._layer_sources[i] = path
                    except (OSError, moderngl.Error) as ex:
                        log.error("Impossible de charger layer %d: %s", i, ex)
                        self._release_layer(i)
            else:
                # Layer non utilisé → libérer si occupé
                if self._layer_sources[i]:
                    self._release_layer(i)
                    self._layer_sources[i] = ''

    def _release_scene_b(self):
        if self.scene_b_program: self.scene_b_program.release()
        if self.scene_b_vao:     self.scene_b_vao.release()
        self.scene_b_program = None
        self.scene_b_vao     = None
        self.scene_b_source  = ''
        self.scene_b_error   = None

    def _release_pass(self, name):
        if self.programs[name]: self.programs[name].release()
        if self.vaos[name]:     self.vaos[name].release()
        self.programs[name] = None
        self.vaos[name]     = None
        self.sources[name]  = ''
        self.errors[name]   = None

    def load_texture(self, channel: int, filepath: str) -> tuple[bool, str]:
        if not (0 <= channel <= 3): return False, 'canal invalide'
        if not self.ctx:            return False, 'contexte non initialisé'
        try:
            img = QImage(filepath)
            if img.isNull(): return False, 'image illisible'
            img  = img.convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = img.width(), img.height()
            ptr  = img.constBits(); ptr.setsize(w * h * 4)
            tex  = self.ctx.texture((w, h), 4, bytes(ptr))
            tex.repeat_x = tex.repeat_y = True
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.textures[channel] = tex
            return True, 'ok'
        except (OSError, moderngl.Error, ValueError) as e:
            return False, str(e)

    def render(self, current_time: float, screen_fbo=None):
        if not self.ctx: return
        _screen = screen_fbo if screen_fbo is not None else self.ctx.screen
        dt  = current_time - self._last_time
        self._last_time = current_time
        vp  = (0, 0, self.width, self.height)

        # 1. Buffer passes (A–D)
        for name in self.buffer_fbos:
            if not self.programs[name]: continue
            self.buffer_fbos[name][1].use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            prog = self.programs[name]
            self._set_uniforms(prog, self.types[name], current_time, dt)
            self.buffer_textures[name][0].use(location=0)
            if 'iChannel0' in prog: prog['iChannel0'].value = 0
            for j in range(1, 4):
                if self.textures[j]:
                    self.textures[j].use(location=j)
                    if f'iChannel{j}' in prog: prog[f'iChannel{j}'].value = j
            self.vaos[name].render(moderngl.TRIANGLES)

        # 2a. Rendu de la scène principale (Image) → image_fbo
        self.image_fbo.use()
        self.ctx.viewport = vp
        self.ctx.clear(0, 0, 0, 1)
        iprog = self.programs['Image']
        ivao  = self.vaos['Image']
        if iprog is None or self.errors['Image']:
            # Si une transition est active, la passe Image peut être vide :
            # on ne dessine PAS la grille d'erreur — on laisse le fond noir.
            # La scène A sera fournie par le layer ou laissée noire intentionnellement.
            if not self._trans_active:
                if self._error_vao: self._error_vao.render(moderngl.TRIANGLES)
        else:
            self._set_uniforms(iprog, self.types['Image'], current_time, dt)
            self._bind_image_channels(iprog)
            ivao.render(moderngl.TRIANGLES)

        # 2b-layers. Si des layers multi-pistes sont actifs, les rendre et blender
        # porter-duff src-over dans image_fbo (layer 0 = fond, ensuite = dessus).
        n_layers = sum(1 for i in range(self.MAX_LAYERS) if self._layer_programs[i])
        if n_layers > 0 and self._blend_prog:
            # Rend chaque layer dans son FBO
            for i in range(self.MAX_LAYERS):
                if not self._layer_programs[i]:
                    continue
                self._layer_fbos[i].use()
                self.ctx.viewport = vp
                self.ctx.clear(0, 0, 0, 0)   # fond transparent
                prog = self._layer_programs[i]
                vao  = self._layer_vaos[i]
                if vao is None:
                    continue
                self._set_uniforms(prog, self._layer_types[i], current_time, dt)
                self._bind_image_channels(prog)
                try:
                    vao.render(moderngl.TRIANGLES)
                except (AttributeError, moderngl.Error) as exc:
                    log.warning('Layer %d render echoue: %s — libération du layer invalide', i, exc)
                    self._release_layer(i)

            # Compose : image_fbo est le fond initial, on blend chaque layer dessus
            # On utilise un FBO temporaire pour alterner src/dst
            # Stratégie : dst=image_fbo → blend layer[0] → dst = layer[0] composé → etc.
            # On réutilise scene_a_fbo comme tampon intermédiaire
            # Pour cela, on blend layer par layer en se servant de scene_a_fbo comme ping-pong
            # dst actuel = image_fbo (déjà rendu)
            # Pour chaque layer : scene_a_fbo ← blend(dst_tex, layer_tex)
            #                     dst ← scene_a_fbo
            import moderngl as _mgl

            # Copie image_fbo → scene_a_fbo comme base
            self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)
            dst_tex = self.scene_a_texture   # texture du résultat intermédiaire

            for i in range(self.MAX_LAYERS):
                if not self._layer_programs[i]:
                    continue
                # Blend dst_tex (actuel) + layer[i] → image_fbo
                self.image_fbo.use()
                self.ctx.viewport = vp
                self.ctx.clear(0, 0, 0, 0)
                bp = self._blend_prog
                dst_tex.use(location=0)
                self._layer_textures[i].use(location=1)
                if 'uDst' in bp: bp['uDst'].value = 0
                if 'uSrc' in bp: bp['uSrc'].value = 1
                self._blend_vao.render(_mgl.TRIANGLES)
                # Copie le résultat vers scene_a_fbo pour le prochain tour
                if i < self.MAX_LAYERS - 1:
                    self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)
                    dst_tex = self.scene_a_texture

        # 2c. Transition : rendre ScèneA → scene_a_fbo,
        #     ScèneB → scene_b_fbo, puis appliquer le shader Trans → image_fbo
        if self._trans_active and self.trans_program and self.trans_vao:
            # ScèneA = le rendu Image courant (déjà dans image_texture)
            # On le copie dans scene_a_fbo
            self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)

            # Rend la scène B dans scene_b_fbo (si shader B disponible)
            self.scene_b_fbo.use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            if self.scene_b_program and not self.scene_b_error:
                self._set_uniforms(self.scene_b_program, self.scene_b_type, current_time, dt)
                self._bind_image_channels(self.scene_b_program)
                self.scene_b_vao.render(moderngl.TRIANGLES)
            elif self._error_vao:
                self._error_vao.render(moderngl.TRIANGLES)

            # Applique la transition → image_fbo
            self.image_fbo.use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            tp   = self.trans_program
            tv   = self.trans_vao
            self.scene_a_texture.use(location=0)
            self.scene_b_texture.use(location=1)
            if 'iChannel0' in tp: tp['iChannel0'].value = 0
            if 'iChannel1' in tp: tp['iChannel1'].value = 1
            if 'iProgress'    in tp: tp['iProgress'].value    = float(self._trans_progress)
            if 'iResolution'  in tp:
                tp['iResolution'].value = (float(self.width), float(self.height), 1.0)
            if 'iTime'        in tp: tp['iTime'].value        = float(current_time)
            if 'iTimeDelta'   in tp: tp['iTimeDelta'].value   = float(dt)
            if 'iFrame'       in tp: tp['iFrame'].value       = self._frame
            tv.render(moderngl.TRIANGLES)

        # 3. Post pass (ou blit direct) → screen
        _screen.use()
        self.ctx.viewport = vp
        self.ctx.clear(0, 0, 0, 1)
        pprog = self.programs.get('Post')
        pvao  = self.vaos.get('Post')
        if pprog and pvao:
            self._set_uniforms(pprog, self.types['Post'], current_time, dt)
            self.image_texture.use(location=0)
            if 'iChannel0' in pprog: pprog['iChannel0'].value = 0
            pvao.render(moderngl.TRIANGLES)
        else:
            self.ctx.copy_framebuffer(dst=_screen, src=self.image_fbo)

        # 4. Ping-pong swap
        for name in self.buffer_fbos:
            self.buffer_fbos[name].reverse()
            self.buffer_textures[name].reverse()
        self._frame += 1

    def _bind_image_channels(self, prog):
        """Bind les iChannel0–3 pour la passe Image (Buffers A–D ou textures)."""
        res = []
        for i, bname in enumerate(['Buffer A','Buffer B','Buffer C','Buffer D']):
            if self.programs.get(bname) and self.buffer_textures[bname][0]:
                self.buffer_textures[bname][0].use(location=i)
                res.append((float(self.width), float(self.height), 1.0))
            elif self.textures[i]:
                self.textures[i].use(location=i)
                res.append((float(self.textures[i].width),
                            float(self.textures[i].height), 1.0))
            else:
                res.append((0.0, 0.0, 0.0))
            if f'iChannel{i}' in prog: prog[f'iChannel{i}'].value = i
        if 'iChannelResolution' in prog:
            prog['iChannelResolution'].value = res

    def _set_uniforms(self, prog, stype: str, t: float, dt: float):
        def s(n, v):
            try:
                if n in prog: prog[n].value = v
            except (TypeError, ValueError, KeyError):
                pass
        if stype == 'shadertoy':
            s('iResolution', (float(self.width), float(self.height), 1.0))
            s('iTime', t); s('iTimeDelta', dt); s('iFrame', self._frame)
            s('iMouse', (0.0, 0.0, 0.0, 0.0))
        else:  # glsl pur
            s('uResolution', (float(self.width), float(self.height)))
            s('uTime', t); s('uTimeDelta', dt); s('uFrame', self._frame)
        for n, v in self.extra_uniforms.items():
            s(n, v)

    def set_uniform(self, name: str, value):
        self.extra_uniforms[name] = value

    def get_uniform(self, name: str):
        """Retourne la valeur courante d'un uniform extra, ou None si inconnu."""
        return self.extra_uniforms.get(name)

    def get_shader_type(self, pass_name='Image') -> str:
        return self.types.get(pass_name, 'shadertoy')

    def resize(self, w, h):
        self.width = w; self.height = h
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self._recreate_fbos()

    def _recreate_fbos(self):
        """Recrée tous les FBOs et textures à la nouvelle résolution (w × h)."""
        ctx = self.ctx
        w, h = self.width, self.height

        # ── Image FBO ──────────────────────────────────────────────────────
        if self.image_fbo:     self.image_fbo.release()
        if self.image_texture: self.image_texture.release()
        self.image_texture = ctx.texture((w, h), 4, dtype='f1')
        self.image_fbo     = ctx.framebuffer(color_attachments=[self.image_texture])

        # ── Buffer passes (A-D + Post) ──────────────────────────────────────
        for name in self.buffer_fbos:
            for x in self.buffer_fbos[name] + self.buffer_textures[name]:
                if x: x.release()
            t1 = ctx.texture((w, h), 4, dtype='f1')
            t2 = ctx.texture((w, h), 4, dtype='f1')
            self.buffer_textures[name] = [t1, t2]
            self.buffer_fbos[name]     = [ctx.framebuffer(color_attachments=[t1]),
                                          ctx.framebuffer(color_attachments=[t2])]

        # ── Scènes A/B (transition) ─────────────────────────────────────────
        for attr_fbo, attr_tex in [('scene_a_fbo', 'scene_a_texture'),
                                   ('scene_b_fbo', 'scene_b_texture')]:
            fbo = getattr(self, attr_fbo)
            tex = getattr(self, attr_tex)
            if fbo: fbo.release()
            if tex: tex.release()
            new_tex = ctx.texture((w, h), 4, dtype='f1')
            new_fbo = ctx.framebuffer(color_attachments=[new_tex])
            setattr(self, attr_tex, new_tex)
            setattr(self, attr_fbo, new_fbo)

        # ── Layer FBOs (multi-pistes shader) ────────────────────────────────
        # Ces FBOs dépendent de la résolution et doivent être recréés.
        # Les VAOs sont liés au _vbo — on les recrée aussi par sécurité.
        for i in range(self.MAX_LAYERS):
            if self._layer_fbos[i]:
                try:
                    self._layer_fbos[i].release()
                except Exception:
                    pass
            if self._layer_textures[i]:
                try:
                    self._layer_textures[i].release()
                except Exception:
                    pass
            new_tex = ctx.texture((w, h), 4, dtype='f1')
            self._layer_textures[i] = new_tex
            self._layer_fbos[i]     = ctx.framebuffer(color_attachments=[new_tex])

            # Si un programme layer existe, recréer son VAO
            if self._layer_programs[i]:
                try:
                    if self._layer_vaos[i]:
                        self._layer_vaos[i].release()
                except Exception:
                    pass
                try:
                    self._layer_vaos[i] = ctx.vertex_array(
                        self._layer_programs[i],
                        [(self._vbo, '2f', 'in_position')]
                    )
                except Exception as exc:
                    log.warning("Layer %d VAO recréation échouée : %s", i, exc)
                    self._layer_vaos[i] = None

        log.info("FBOs recréés à %dx%d", w, h)


    def render_to_texture(self, current_time: float):
        """
        v2.3 — Exécute les passes 1 (Buffers) et 2 (Image + layers + transition)
        sans écrire sur l'écran.  Laisse image_texture contenant le résultat final
        prêt à être consommé par un post-process externe (ex: upscaler IA).

        Effectue aussi le ping-pong swap des buffers (même comportement que render()).
        NE PAS appeler render() et render_to_texture() dans la même frame.
        """
        if not self.ctx:
            return
        dt  = current_time - self._last_time
        self._last_time = current_time
        vp  = (0, 0, self.width, self.height)

        # ── 1. Buffer passes (A–D) ────────────────────────────────────────
        for name in self.buffer_fbos:
            if not self.programs[name]:
                continue
            self.buffer_fbos[name][1].use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            prog = self.programs[name]
            self._set_uniforms(prog, self.types[name], current_time, dt)
            self.buffer_textures[name][0].use(location=0)
            if 'iChannel0' in prog:
                prog['iChannel0'].value = 0
            for j in range(1, 4):
                if self.textures[j]:
                    self.textures[j].use(location=j)
                    if f'iChannel{j}' in prog:
                        prog[f'iChannel{j}'].value = j
            self.vaos[name].render(moderngl.TRIANGLES)

        # ── 2a. Scène principale (Image) → image_fbo ─────────────────────
        self.image_fbo.use()
        self.ctx.viewport = vp
        self.ctx.clear(0, 0, 0, 1)
        iprog = self.programs['Image']
        ivao  = self.vaos['Image']
        if iprog is None or self.errors['Image']:
            if not self._trans_active:
                if self._error_vao:
                    self._error_vao.render(moderngl.TRIANGLES)
        else:
            self._set_uniforms(iprog, self.types['Image'], current_time, dt)
            self._bind_image_channels(iprog)
            ivao.render(moderngl.TRIANGLES)

        # ── 2b. Layers ────────────────────────────────────────────────────
        n_layers = sum(1 for i in range(self.MAX_LAYERS) if self._layer_programs[i])
        if n_layers > 0 and self._blend_prog:
            for i in range(self.MAX_LAYERS):
                if not self._layer_programs[i]:
                    continue
                self._layer_fbos[i].use()
                self.ctx.viewport = vp
                self.ctx.clear(0, 0, 0, 0)
                prog = self._layer_programs[i]
                vao  = self._layer_vaos[i]
                if vao is None:
                    continue
                self._set_uniforms(prog, self._layer_types[i], current_time, dt)
                self._bind_image_channels(prog)
                try:
                    vao.render(moderngl.TRIANGLES)
                except (AttributeError, moderngl.Error) as exc:
                    log.warning('Layer %d render_to_texture échoué: %s', i, exc)
                    self._release_layer(i)
            self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)
            dst_tex = self.scene_a_texture
            for i in range(self.MAX_LAYERS):
                if not self._layer_programs[i]:
                    continue
                self.image_fbo.use()
                self.ctx.viewport = vp
                self.ctx.clear(0, 0, 0, 0)
                bp = self._blend_prog
                dst_tex.use(location=0)
                self._layer_textures[i].use(location=1)
                if 'uDst' in bp: bp['uDst'].value = 0
                if 'uSrc' in bp: bp['uSrc'].value = 1
                self._blend_vao.render(moderngl.TRIANGLES)
                if i < self.MAX_LAYERS - 1:
                    self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)
                    dst_tex = self.scene_a_texture

        # ── 2c. Transition ────────────────────────────────────────────────
        if self._trans_active and self.trans_program and self.trans_vao:
            self.ctx.copy_framebuffer(dst=self.scene_a_fbo, src=self.image_fbo)
            self.scene_b_fbo.use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            if self.scene_b_program and not self.scene_b_error:
                self._set_uniforms(self.scene_b_program, self.scene_b_type, current_time, dt)
                self._bind_image_channels(self.scene_b_program)
                self.scene_b_vao.render(moderngl.TRIANGLES)
            elif self._error_vao:
                self._error_vao.render(moderngl.TRIANGLES)
            self.image_fbo.use()
            self.ctx.viewport = vp
            self.ctx.clear(0, 0, 0, 1)
            tp = self.trans_program
            tv = self.trans_vao
            self.scene_a_texture.use(location=0)
            self.scene_b_texture.use(location=1)
            if 'iChannel0' in tp: tp['iChannel0'].value = 0
            if 'iChannel1' in tp: tp['iChannel1'].value = 1
            if 'iProgress'   in tp: tp['iProgress'].value   = float(self._trans_progress)
            if 'iResolution' in tp:
                tp['iResolution'].value = (float(self.width), float(self.height), 1.0)
            if 'iTime'       in tp: tp['iTime'].value       = float(current_time)
            if 'iTimeDelta'  in tp: tp['iTimeDelta'].value  = float(dt)
            if 'iFrame'      in tp: tp['iFrame'].value      = self._frame
            tv.render(moderngl.TRIANGLES)

        # NOTE : pas de Post pass — image_texture est laissée intacte pour upscaling.
        # L'appelant est responsable d'écrire sur le screen.

        # ── 4. Ping-pong swap ─────────────────────────────────────────────
        for name in self.buffer_fbos:
            self.buffer_fbos[name].reverse()
            self.buffer_textures[name].reverse()
        self._frame += 1

    def render_frame(self, current_time: float) -> bytes:
        """Rend une frame dans image_fbo et retourne les pixels RGBA (bytes).
        Utilisé par le rendu headless CLI.  Ne touche pas ctx.screen."""
        self.render(current_time, screen_fbo=self.image_fbo)
        self.image_fbo.use()
        return self.image_fbo.read(components=4)

    def cleanup(self):
        for p in self.pass_names: self._release_pass(p)
        self._release_trans()
        self._release_scene_b()
        if self._error_prog: self._error_prog.release()
        if self.image_fbo:     self.image_fbo.release()
        if self.image_texture: self.image_texture.release()
        if self.scene_a_fbo:      self.scene_a_fbo.release()
        if self.scene_a_texture:  self.scene_a_texture.release()
        if self.scene_b_fbo:      self.scene_b_fbo.release()
        if self.scene_b_texture:  self.scene_b_texture.release()
        for name in self.buffer_fbos:
            for x in self.buffer_fbos[name] + self.buffer_textures[name]:
                if x: x.release()
        for t in self.textures:
            if t: t.release()
