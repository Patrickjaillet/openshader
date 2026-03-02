"""
intro_toolkit.py
----------------
v2.8 — 4K / 64K intro toolkit pour DemoMaker.

Composants
----------
ShaderMinifier
    Minification GLSL : suppression commentaires, whitespace, renommage
    identifiants longs, inline #define constants, normalisation tokens.
    Objectif : réduire la taille brute avant compression.

ShaderPacker
    Packing multi-passes en blob LZMA auto-décompressé.
    Génère un script Python stub (~512 B compressé) + payload LZMA qui
    décompresse et exécute les shaders à la volée via moderngl/pygame.

ProceduralSynth
    Synthétiseur audio procédural inspiré de 4klang.
    Oscillateurs : sin, saw, square, triangle, noise.
    Enveloppes ADSR, LFO, effets : reverb simple, délai, distorsion.
    Export WAV 44100 Hz stéréo. Piloté par un patch dict JSON-sérialisable.

IntroSizeEstimator
    Estime la taille compressée (LZMA niveau 9) des shaders + audio patch
    en temps réel. Calcule les budgets 4K / 64K et le ratio restant.

IntroBuilderDialog
    Dialog principal : onglets Minifier / Packer / Synth / Estimation.
"""

from __future__ import annotations

import io
import json
import lzma
import math
import os
import re
import struct
import wave
import zlib
from typing import NamedTuple

from PyQt6.QtCore    import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QTextEdit, QPlainTextEdit,
    QGroupBox, QFrame, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QSizePolicy, QSplitter, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView,
)
from PyQt6.QtGui import QFont, QColor, QPalette

from .logger import get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  GLSL token helpers
# ══════════════════════════════════════════════════════════════════════════════

# Mots-clés GLSL à ne jamais renommer
_GLSL_KEYWORDS = frozenset("""
    void bool int uint float double vec2 vec3 vec4 bvec2 bvec3 bvec4
    ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 dvec2 dvec3 dvec4
    mat2 mat3 mat4 mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4
    mat4x2 mat4x3 mat4x4 sampler2D sampler3D samplerCube sampler2DShadow
    samplerCubeArray sampler2DArray
    if else for while do break continue return discard
    in out inout uniform attribute varying const struct
    layout location binding precision highp mediump lowp
    gl_Position gl_FragCoord gl_FragColor gl_PointSize
    gl_VertexID gl_InstanceID
    true false
    abs acos acosh all any asin asinh atan atanh ceil clamp cos cosh
    cross dFdx dFdy degrees determinant distance dot equal exp exp2
    faceforward floor fract greaterThan greaterThanEqual inversesqrt
    isinf isnan length lessThan lessThanEqual log log2 matrixCompMult
    max min mix mod modf normalize not notEqual outerProduct pow
    radians reflect refract roundEven sign sin sinh smoothstep sqrt
    step tan tanh texelFetch texture textureGrad textureLod textureSize
    transpose trunc
    mainImage main fragColor fragCoord iTime iResolution iMouse
    iFrame iTimeDelta iChannel0 iChannel1 iChannel2 iChannel3
    iChannelResolution iSampleRate
""".split())

# Noms courts disponibles pour renommage (ordre croissant de longueur)
_SHORT_NAMES = (
    [c for c in 'abcdefghjkmnopqrstuwxyz'] +
    [f'{a}{b}' for a in 'abcdefghjkmnopqrstuwxyz'
               for b in '0123456789abcdefghjkmnopqrstuwxyz']
)


# ══════════════════════════════════════════════════════════════════════════════
#  ShaderMinifier
# ══════════════════════════════════════════════════════════════════════════════

class MinifyResult(NamedTuple):
    original_bytes: int
    minified_bytes: int
    compressed_bytes: int        # LZMA level 9
    identifiers_renamed: int
    defines_inlined: int
    source: str

    @property
    def ratio(self) -> float:
        return self.minified_bytes / max(1, self.original_bytes)

    @property
    def savings_pct(self) -> float:
        return (1 - self.ratio) * 100


class ShaderMinifier:
    """
    Minificateur GLSL multi-passes.

    Passes appliquées dans l'ordre :
      1. Strip commentaires (/* */ et //)
      2. Inline des #define constants numériques
      3. Suppression whitespace superflu (préserve les tokens)
      4. Renommage identifiants locaux courts (optionnel)
    """

    def __init__(self,
                 rename_identifiers: bool = True,
                 inline_defines: bool = True,
                 preserve_precision: bool = True):
        self.rename_identifiers = rename_identifiers
        self.inline_defines     = inline_defines
        self.preserve_precision = preserve_precision

    # ── API publique ──────────────────────────────────────────────────────────

    def minify(self, source: str) -> MinifyResult:
        orig_bytes = len(source.encode('utf-8'))
        src = source

        defines_inlined = 0
        if self.inline_defines:
            src, defines_inlined = self._inline_defines(src)

        src = self._strip_comments(src)
        src = self._collapse_whitespace(src)

        renamed = 0
        if self.rename_identifiers:
            src, renamed = self._rename_identifiers(src)

        src = src.strip()
        mini_bytes = len(src.encode('utf-8'))
        comp_bytes = len(lzma.compress(src.encode('utf-8'),
                                       preset=lzma.PRESET_EXTREME))
        return MinifyResult(
            original_bytes   = orig_bytes,
            minified_bytes   = mini_bytes,
            compressed_bytes = comp_bytes,
            identifiers_renamed = renamed,
            defines_inlined  = defines_inlined,
            source           = src,
        )

    # ── Passes de minification ────────────────────────────────────────────────

    @staticmethod
    def _strip_comments(src: str) -> str:
        """Supprime // et /* */ sans toucher aux strings (GLSL n'en a pas)."""
        # Block comments
        src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
        # Line comments
        src = re.sub(r'//[^\n]*', '', src)
        return src

    @staticmethod
    def _inline_defines(src: str) -> tuple[str, int]:
        """
        Inline les #define numériques simples.
        Ex : #define PI 3.14159  →  remplace PI partout + supprime la ligne.
        Ignore les macros avec paramètres.
        """
        defines: dict[str, str] = {}
        # Trouver les defines numériques simples (pas de parenthèses → pas de macro-fonction)
        def _collect(m: re.Match) -> str:
            name, val = m.group(1), m.group(2).strip()
            if '(' not in name and re.match(r'^[\d\.\-\+eEfF_u]+$', val):
                defines[name] = val
            return ''  # supprime la ligne

        src = re.sub(
            r'^\s*#define\s+([A-Z_][A-Z0-9_]*)\s+([\d\.\-\+eEfF_u]+)\s*$',
            _collect, src, flags=re.MULTILINE
        )
        count = 0
        for name, val in defines.items():
            # Remplace seulement les occurrences qui sont des tokens isolés
            pattern = r'\b' + re.escape(name) + r'\b'
            new, n = re.subn(pattern, val, src)
            src = new
            count += n
        return src, len(defines)

    @staticmethod
    def _collapse_whitespace(src: str) -> str:
        """
        Réduit le whitespace au strict minimum tout en préservant la validité
        des tokens GLSL (les séparateurs entre deux identifiants/mots-clés
        doivent rester).
        """
        lines = []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith('#'):
                # Les directives de préprocesseur doivent rester sur leur ligne
                lines.append(s)
            else:
                lines.append(s)

        src = '\n'.join(lines)
        # Collapse séquences de whitespace (hors newlines) en espace unique
        src = re.sub(r'[ \t]+', ' ', src)
        # Supprime espace autour des opérateurs et ponctuations (sauf identifiant↔identifiant)
        for op in (r'\(', r'\)', r'\{', r'\}', r'\[', r'\]',
                   r';', r',', r'\.'):
            src = re.sub(r'\s*' + op + r'\s*', op.replace('\\', ''), src)
        # Collapse lignes vides multiples
        src = re.sub(r'\n{3,}', '\n\n', src)
        # Supprime lignes entièrement vides
        src = '\n'.join(l for l in src.splitlines() if l.strip())
        return src

    def _rename_identifiers(self, src: str) -> tuple[str, int]:
        """
        Renomme les identifiants non-keyword de longueur >= 3 en noms courts.
        Préserve tous les mots-clés GLSL et les uniforms standard.
        """
        # Trouve tous les identifiants
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', src)
        candidates: dict[str, int] = {}
        for tok in tokens:
            if tok not in _GLSL_KEYWORDS and len(tok) >= 3:
                candidates[tok] = candidates.get(tok, 0) + 1

        # Tri par fréquence décroissante (les plus fréquents = gains maximaux)
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])

        name_gen = iter(_SHORT_NAMES)
        mapping: dict[str, str] = {}
        for name, freq in sorted_cands:
            try:
                short = next(name_gen)
                # S'assurer que le nom court n'est pas déjà un keyword ou existant
                while short in _GLSL_KEYWORDS or short in src:
                    short = next(name_gen)
                mapping[name] = short
            except StopIteration:
                break  # Plus de noms courts disponibles

        for old, new in mapping.items():
            src = re.sub(r'\b' + re.escape(old) + r'\b', new, src)

        return src, len(mapping)


# ══════════════════════════════════════════════════════════════════════════════
#  ShaderPacker
# ══════════════════════════════════════════════════════════════════════════════

# Stub Python minimal qui se décompresse et joue les shaders
# ~450 octets source / ~300 compressé (LZMA)
_STUB_TEMPLATE = '''\
import lzma,json,sys
_P={payload_repr}
_D=json.loads(lzma.decompress(_P).decode())
try:
 import moderngl,pygame,numpy as np,time
 pygame.init()
 _W,_H=_D.get('w',800),_D.get('h',450)
 pygame.display.set_mode((_W,_H),pygame.OPENGL|pygame.DOUBLEBUF)
 _C=moderngl.create_context()
 _VS="#version 330 core\\nin vec2 p;void main(){{gl_Position=vec4(p,0,1);}}"
 _FS=_D['shaders'].get('Image',list(_D['shaders'].values())[0])
 _IS=bool(__import__('re').search(r'void\\s+mainImage\\s*\\(',_FS))
 _H2="#version 330 core\\nuniform vec3 iResolution;uniform float iTime;\\nuniform vec4 iMouse;uniform int iFrame;\\nout vec4 _o;\\n"
 _FS2=_H2+_FS+("\\nvoid main(){{mainImage(_o,gl_FragCoord.xy);}}" if _IS else "")
 _PG=_C.program(vertex_shader=_VS,fragment_shader=_FS2)
 _VA=_C.vertex_array(_PG,_C.buffer(np.array([-1,-1,1,-1,-1,1,1,-1,1,1,-1,1],np.float32).tobytes()),[('p',moderngl.FLOAT,'p',2)])
 _T0=time.time();_F=0;_M=[0,0,0,0]
 while True:
  for e in pygame.event.get():
   if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):sys.exit()
   if e.type==pygame.MOUSEMOTION:_M[:2]=list(e.pos)
   if e.type==pygame.MOUSEBUTTONDOWN:_M[2:]=list(e.pos)
   if e.type==pygame.MOUSEBUTTONUP:_M[2:]=[0,0]
  _t=time.time()-_T0
  if 'iResolution' in _PG:_PG['iResolution']=(_W,_H,1.0)
  if 'iTime' in _PG:_PG['iTime']=_t
  if 'iMouse' in _PG:_PG['iMouse']=tuple(_M)
  if 'iFrame' in _PG:_PG['iFrame']=_F
  _C.clear();_VA.render();pygame.display.flip();_F+=1
except Exception as _e:
 print("Error:",_e);input()
'''


class PackResult(NamedTuple):
    stub_bytes: int          # taille du stub Python source
    payload_bytes: int       # taille du payload LZMA (shaders + meta)
    total_bytes: int         # stub + payload (ce qui ira dans l'exe final)
    stub_source: str         # source Python complète du stub
    passes: list[str]        # noms des passes incluses


class ShaderPacker:
    """
    Emballe plusieurs passes GLSL dans un stub Python auto-extractible.

    Le résultat est un fichier .py qui :
      - Décompresse les shaders depuis un payload LZMA inline
      - Crée un contexte OpenGL via moderngl + pygame
      - Joue le shader Image (ou le premier disponible)
    """

    def __init__(self,
                 shaders: dict[str, str],
                 width:   int = 800,
                 height:  int = 450,
                 minify:  bool = True):
        self.shaders = shaders
        self.width   = width
        self.height  = height
        self.minify  = minify
        self._minifier = ShaderMinifier() if minify else None

    def pack(self) -> PackResult:
        # Minification des shaders si demandé
        packed_shaders: dict[str, str] = {}
        for name, src in self.shaders.items():
            if self._minifier and src.strip():
                result = self._minifier.minify(src)
                packed_shaders[name] = result.source
                log.debug("Pack minify '%s': %d→%d B", name,
                          result.original_bytes, result.minified_bytes)
            else:
                packed_shaders[name] = src

        # Payload JSON → LZMA
        payload_dict = {
            'w':       self.width,
            'h':       self.height,
            'shaders': packed_shaders,
        }
        payload_json  = json.dumps(payload_dict, separators=(',', ':'),
                                   ensure_ascii=True)
        payload_lzma  = lzma.compress(payload_json.encode('utf-8'),
                                      preset=lzma.PRESET_EXTREME)
        payload_bytes = len(payload_lzma)

        # Représentation du payload dans le stub (bytes literal)
        payload_repr  = repr(payload_lzma)

        stub = _STUB_TEMPLATE.format(payload_repr=payload_repr)
        stub_bytes = len(stub.encode('utf-8'))

        return PackResult(
            stub_bytes    = stub_bytes,
            payload_bytes = payload_bytes,
            total_bytes   = stub_bytes + payload_bytes,
            stub_source   = stub,
            passes        = list(packed_shaders.keys()),
        )

    def pack_to_file(self, path: str) -> PackResult:
        result = self.pack()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result.stub_source)
        log.info("Stub écrit : %s (%d B)", path, result.stub_bytes)
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ProceduralSynth
# ══════════════════════════════════════════════════════════════════════════════

class SynthPatch:
    """
    Patch de synthétiseur procédural.

    Structure :
      tracks : list[TrackDef]
        Chaque track est un oscillateur + enveloppe + effets.
      bpm : float
      duration : float  (secondes)
    """

    def __init__(self):
        self.bpm: float = 120.0
        self.duration: float = 8.0
        self.tracks: list[dict] = []

    def add_track(self,
                  wave:     str   = 'sin',   # sin | saw | sqr | tri | noise
                  freq:     float = 440.0,
                  amp:      float = 0.5,
                  attack:   float = 0.01,
                  decay:    float = 0.1,
                  sustain:  float = 0.7,
                  release:  float = 0.2,
                  pan:      float = 0.0,     # -1 left … +1 right
                  detune:   float = 0.0,     # cents
                  lfo_rate: float = 0.0,     # Hz (0 = désactivé)
                  lfo_depth:float = 0.0,     # fraction de freq
                  delay_ms: float = 0.0,
                  delay_fb: float = 0.0,
                  dist:     float = 0.0):    # drive 0..1
        self.tracks.append(dict(
            wave=wave, freq=freq, amp=amp,
            attack=attack, decay=decay, sustain=sustain, release=release,
            pan=pan, detune=detune,
            lfo_rate=lfo_rate, lfo_depth=lfo_depth,
            delay_ms=delay_ms, delay_fb=delay_fb, dist=dist,
        ))

    def to_dict(self) -> dict:
        return {'bpm': self.bpm, 'duration': self.duration,
                'tracks': self.tracks}

    @classmethod
    def from_dict(cls, d: dict) -> 'SynthPatch':
        p = cls()
        p.bpm = float(d.get('bpm', 120))
        p.duration = float(d.get('duration', 8))
        p.tracks = d.get('tracks', [])
        return p

    @classmethod
    def default_4k(cls) -> 'SynthPatch':
        """Patch typique 4K intro : bass + lead + pad."""
        p = cls()
        p.bpm = 138.0
        p.duration = 16.0
        p.add_track(wave='saw',  freq=55.0,  amp=0.6,
                    attack=0.005, decay=0.2, sustain=0.5, release=0.3,
                    dist=0.3)
        p.add_track(wave='sqr',  freq=440.0, amp=0.3,
                    attack=0.01, decay=0.1, sustain=0.6, release=0.2,
                    detune=7.0, lfo_rate=3.0, lfo_depth=0.02, pan=0.2)
        p.add_track(wave='sin',  freq=220.0, amp=0.2,
                    attack=0.2, decay=0.4, sustain=0.8, release=0.8,
                    pan=-0.2, delay_ms=125.0, delay_fb=0.4)
        return p


class ProceduralSynth:
    """
    Synthétiseur audio procédural inspiré de 4klang.

    Génère un buffer numpy stéréo 44100 Hz float32, exportable en WAV.
    """

    SAMPLE_RATE = 44100

    def __init__(self, patch: SynthPatch):
        self.patch = patch

    def render(self) -> 'np.ndarray':
        """Retourne un array numpy (N, 2) float32 [-1, 1]."""
        import numpy as np

        sr  = self.SAMPLE_RATE
        dur = self.patch.duration
        n   = int(sr * dur)
        out = np.zeros((n, 2), dtype=np.float32)

        for track in self.patch.tracks:
            track_buf = self._render_track(track, n, sr)
            out += track_buf

        # Normalisation douce (peak limit)
        peak = np.max(np.abs(out))
        if peak > 0.95:
            out *= 0.95 / peak

        return out

    def _render_track(self, t: dict, n: int, sr: int) -> 'np.ndarray':
        import numpy as np

        ts   = np.arange(n, dtype=np.float64) / sr
        freq = float(t.get('freq', 440))
        amp  = float(t.get('amp',  0.5))

        # Detune en cents
        detune_cents = float(t.get('detune', 0))
        freq *= 2 ** (detune_cents / 1200)

        # LFO sur la fréquence
        lfo_rate  = float(t.get('lfo_rate',  0))
        lfo_depth = float(t.get('lfo_depth', 0))
        if lfo_rate > 0 and lfo_depth > 0:
            lfo = 1.0 + lfo_depth * np.sin(2 * math.pi * lfo_rate * ts)
        else:
            lfo = 1.0

        # Phase cumulée
        phase = np.cumsum(2 * math.pi * freq * lfo / sr)

        # Oscillateur
        wave = t.get('wave', 'sin')
        if wave == 'sin':
            sig = np.sin(phase)
        elif wave == 'saw':
            sig = 2 * (phase / (2 * math.pi) % 1) - 1
        elif wave == 'sqr':
            sig = np.sign(np.sin(phase))
        elif wave == 'tri':
            p = phase / (2 * math.pi) % 1
            sig = 2 * np.where(p < 0.5, p, 1 - p) * 2 - 1
        elif wave == 'noise':
            rng = np.random.default_rng(42)
            sig = rng.uniform(-1, 1, n)
        else:
            sig = np.sin(phase)

        # Distorsion (soft clipping tanh)
        dist = float(t.get('dist', 0))
        if dist > 0:
            drive = 1 + dist * 9
            sig = np.tanh(sig * drive) / math.tanh(drive)

        # Enveloppe ADSR (appliquée sur toute la durée)
        envelope = self._adsr_envelope(
            n, sr,
            attack  = float(t.get('attack',  0.01)),
            decay   = float(t.get('decay',   0.1)),
            sustain = float(t.get('sustain', 0.7)),
            release = float(t.get('release', 0.2)),
        )
        sig *= envelope * amp

        # Délai
        delay_ms = float(t.get('delay_ms', 0))
        delay_fb = float(t.get('delay_fb', 0))
        if delay_ms > 0 and delay_fb > 0:
            sig = self._apply_delay(sig, delay_ms, delay_fb, sr)

        # Pan stéréo
        pan = float(t.get('pan', 0.0))
        pan = max(-1.0, min(1.0, pan))
        vol_l = math.sqrt((1 - pan) / 2)
        vol_r = math.sqrt((1 + pan) / 2)

        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = sig * vol_l
        out[:, 1] = sig * vol_r
        return out

    @staticmethod
    def _adsr_envelope(n: int, sr: int,
                       attack: float, decay: float,
                       sustain: float, release: float) -> 'np.ndarray':
        import numpy as np
        env = np.zeros(n, dtype=np.float64)
        a = min(int(attack  * sr), n)
        d = min(int(decay   * sr), n - a)
        r = min(int(release * sr), n)
        s_end = max(0, n - r)

        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if d > 0:
            env[a:a+d] = np.linspace(1, sustain, d)
        if a + d < s_end:
            env[a+d:s_end] = sustain
        if r > 0 and s_end < n:
            env[s_end:] = np.linspace(sustain, 0, n - s_end)
        return env

    @staticmethod
    def _apply_delay(sig: 'np.ndarray', delay_ms: float,
                     feedback: float, sr: int) -> 'np.ndarray':
        import numpy as np
        delay_samples = int(delay_ms * sr / 1000)
        out = sig.copy()
        for i in range(delay_samples, len(out)):
            out[i] += feedback * out[i - delay_samples]
        return out

    def export_wav(self, path: str) -> int:
        """Exporte en WAV 44100 Hz stéréo 16 bits. Retourne nb d'échantillons."""
        buf = self.render()
        pcm = (buf * 32767).clip(-32767, 32767).astype('<i2')
        with wave.open(path, 'w') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        log.info("WAV exporté : %s (%d samples)", path, len(buf))
        return len(buf)


# ══════════════════════════════════════════════════════════════════════════════
#  IntroSizeEstimator
# ══════════════════════════════════════════════════════════════════════════════

_BUDGET_4K  = 4096
_BUDGET_64K = 65536
_BUDGET_96K = 98304

# Overhead fixe estimé du runtime (moderngl + pygame + stub Python interpréteur)
# En pratique: exe PyInstaller ~3.5 Mo, mais on cible script pur ou Nuitka
_RUNTIME_OVERHEAD_BYTES = 256  # stub Python minimal compressé


class SizeReport(NamedTuple):
    shaders_raw_bytes:   int
    shaders_mini_bytes:  int
    shaders_lzma_bytes:  int
    audio_lzma_bytes:    int
    stub_bytes:          int
    total_estimated:     int
    budget_4k_free:      int   # octets restants avant limite 4K
    budget_64k_free:     int   # octets restants avant limite 64K
    budget_96k_free:     int
    minify_savings_pct:  float


class IntroSizeEstimator:
    """
    Calcule en temps réel la taille estimée d'une intro compressée.

    Utilise LZMA preset=9 (EXTREME) comme approximation de Crinkler/kkrunchy.
    """

    def __init__(self, shaders: dict[str, str],
                 synth_patch: SynthPatch | None = None):
        self.shaders     = shaders
        self.synth_patch = synth_patch
        self._minifier   = ShaderMinifier()

    def estimate(self) -> SizeReport:
        # Shaders bruts
        raw_src = '\n'.join(self.shaders.values())
        raw_bytes = len(raw_src.encode('utf-8'))

        # Shaders minifiés
        mini_parts = []
        for src in self.shaders.values():
            if src.strip():
                mini_parts.append(self._minifier.minify(src).source)
        mini_src   = '\n'.join(mini_parts)
        mini_bytes = len(mini_src.encode('utf-8'))

        # Compression LZMA des shaders minifiés
        lzma_bytes = len(lzma.compress(mini_src.encode('utf-8'),
                                       preset=lzma.PRESET_EXTREME))

        # Audio patch (si présent)
        audio_lzma = 0
        if self.synth_patch:
            patch_json = json.dumps(self.synth_patch.to_dict(),
                                    separators=(',', ':'))
            audio_lzma = len(lzma.compress(patch_json.encode('utf-8'),
                                           preset=lzma.PRESET_EXTREME))

        stub = len(_STUB_TEMPLATE.encode('utf-8'))
        total = lzma_bytes + audio_lzma + stub + _RUNTIME_OVERHEAD_BYTES

        savings = (1 - mini_bytes / max(1, raw_bytes)) * 100

        return SizeReport(
            shaders_raw_bytes   = raw_bytes,
            shaders_mini_bytes  = mini_bytes,
            shaders_lzma_bytes  = lzma_bytes,
            audio_lzma_bytes    = audio_lzma,
            stub_bytes          = stub,
            total_estimated     = total,
            budget_4k_free      = _BUDGET_4K  - total,
            budget_64k_free     = _BUDGET_64K - total,
            budget_96k_free     = _BUDGET_96K - total,
            minify_savings_pct  = savings,
        )

    @staticmethod
    def format_bytes(n: int) -> str:
        if abs(n) < 1024:
            return f"{n} B"
        return f"{n/1024:.1f} KB"

    @staticmethod
    def budget_bar(used: int, budget: int) -> str:
        """Barre ASCII de remplissage du budget."""
        pct = min(100, max(0, used / max(1, budget) * 100))
        filled = int(pct / 5)
        return f"[{'█' * filled}{'░' * (20 - filled)}] {pct:.0f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  IntroBuilderDialog — UI principale
# ══════════════════════════════════════════════════════════════════════════════

class IntroBuilderDialog(QDialog):
    """
    Dialog 4K/64K intro toolkit.

    Onglets :
      Minifier    — minification GLSL avec stats + diff avant/après
      Packer      — génération du stub Python auto-extractible
      Synthétiseur — patch audio procédural + export WAV
      Estimation  — budget 4K/64K en temps réel
    """

    def __init__(self,
                 shaders: dict[str, str],
                 parent=None):
        super().__init__(parent)
        self.shaders      = shaders
        self._synth_patch = SynthPatch.default_4k()

        self.setWindowTitle("🎛 4K / 64K Intro Toolkit")
        self.setMinimumSize(780, 620)
        self.resize(860, 680)

        # Timer de mise à jour estimation (toutes les 2s)
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setInterval(2000)
        self._estimate_timer.timeout.connect(self._refresh_estimate)

        self._build_ui()
        self._refresh_estimate()
        self._estimate_timer.start()

    def closeEvent(self, event):
        self._estimate_timer.stop()
        super().closeEvent(event)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        tabs.addTab(self._build_tab_minifier(),    "🗜 Minifier")
        tabs.addTab(self._build_tab_packer(),      "📦 Packer")
        tabs.addTab(self._build_tab_synth(),       "🎵 Synthétiseur")
        tabs.addTab(self._build_tab_estimate(),    "📐 Estimation")

        # Bouton fermer
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 12)
        self._lbl_global_status = QLabel("")
        self._lbl_global_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn_row.addWidget(self._lbl_global_status, 1)
        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    # ── Onglet Minifier ───────────────────────────────────────────────────────

    def _build_tab_minifier(self) -> QWidget:
        w   = QWidget()
        vl  = QVBoxLayout(w)
        vl.setContentsMargins(16, 12, 16, 12)
        vl.setSpacing(10)

        # Options
        grp_opts = QGroupBox("Options de minification")
        fo = QHBoxLayout(grp_opts)
        fo.setContentsMargins(12, 8, 12, 8)
        self._chk_inline_defines    = QCheckBox("Inline #define numériques")
        self._chk_inline_defines.setChecked(True)
        self._chk_rename_ids        = QCheckBox("Renommer identifiants")
        self._chk_rename_ids.setChecked(True)
        fo.addWidget(self._chk_inline_defines)
        fo.addWidget(self._chk_rename_ids)
        fo.addStretch()

        btn_run_mini = QPushButton("▶ Minifier")
        btn_run_mini.clicked.connect(self._run_minifier)
        fo.addWidget(btn_run_mini)
        vl.addWidget(grp_opts)

        # Stats
        self._lbl_mini_stats = QLabel("Cliquez sur ▶ Minifier pour analyser les shaders chargés.")
        self._lbl_mini_stats.setWordWrap(True)
        self._lbl_mini_stats.setStyleSheet("color: palette(mid); font-size: 11px;")
        vl.addWidget(self._lbl_mini_stats)

        # Sélecteur de passe
        hl_pass = QHBoxLayout()
        hl_pass.addWidget(QLabel("Passe :"))
        self._cmb_mini_pass = QComboBox()
        self._cmb_mini_pass.addItems(list(self.shaders.keys()))
        self._cmb_mini_pass.currentIndexChanged.connect(self._on_mini_pass_changed)
        hl_pass.addWidget(self._cmb_mini_pass)
        hl_pass.addStretch()
        btn_copy_mini = QPushButton("📋 Copier minifié")
        btn_copy_mini.clicked.connect(self._copy_minified)
        hl_pass.addWidget(btn_copy_mini)
        btn_save_mini = QPushButton("💾 Sauvegarder…")
        btn_save_mini.clicked.connect(self._save_minified)
        hl_pass.addWidget(btn_save_mini)
        vl.addLayout(hl_pass)

        # Split avant / après
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._txt_mini_before = QPlainTextEdit()
        self._txt_mini_before.setReadOnly(True)
        self._txt_mini_before.setFont(QFont('Consolas', 9))
        self._txt_mini_before.setPlaceholderText("Source originale…")
        splitter.addWidget(self._wrap_with_label(self._txt_mini_before, "Avant"))

        self._txt_mini_after = QPlainTextEdit()
        self._txt_mini_after.setReadOnly(True)
        self._txt_mini_after.setFont(QFont('Consolas', 9))
        self._txt_mini_after.setPlaceholderText("Source minifiée…")
        splitter.addWidget(self._wrap_with_label(self._txt_mini_after, "Après"))

        vl.addWidget(splitter, 1)
        self._mini_results: dict[str, MinifyResult] = {}
        return w

    def _run_minifier(self):
        m = ShaderMinifier(
            rename_identifiers = self._chk_rename_ids.isChecked(),
            inline_defines     = self._chk_inline_defines.isChecked(),
        )
        self._mini_results = {}
        total_orig = total_mini = total_lzma = 0
        for name, src in self.shaders.items():
            if not src.strip():
                continue
            r = m.minify(src)
            self._mini_results[name] = r
            total_orig += r.original_bytes
            total_mini += r.minified_bytes
            total_lzma += r.compressed_bytes

        if not self._mini_results:
            self._lbl_mini_stats.setText("⚠ Aucun shader chargé.")
            return

        savings = (1 - total_mini / max(1, total_orig)) * 100
        self._lbl_mini_stats.setText(
            f"Total :  {total_orig:,} B brut  →  {total_mini:,} B minifié  "
            f"({savings:.1f}% économisé)  →  {total_lzma:,} B LZMA"
        )
        self._cmb_mini_pass.clear()
        self._cmb_mini_pass.addItems(list(self._mini_results.keys()))
        self._on_mini_pass_changed(0)

    def _on_mini_pass_changed(self, _idx: int):
        name = self._cmb_mini_pass.currentText()
        if name in self._mini_results:
            r = self._mini_results[name]
            self._txt_mini_before.setPlainText(
                self.shaders.get(name, ''))
            self._txt_mini_after.setPlainText(r.source)
        elif name in self.shaders:
            self._txt_mini_before.setPlainText(self.shaders[name])
            self._txt_mini_after.setPlainText("")

    def _copy_minified(self):
        from PyQt6.QtWidgets import QApplication
        name = self._cmb_mini_pass.currentText()
        if name in self._mini_results:
            QApplication.clipboard().setText(self._mini_results[name].source)
            self._lbl_global_status.setText("✓ Copié dans le presse-papier.")

    def _save_minified(self):
        name = self._cmb_mini_pass.currentText()
        if name not in self._mini_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder le shader minifié",
            f"{name.lower().replace(' ', '_')}_min.glsl",
            "GLSL (*.glsl *.st *.frag)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._mini_results[name].source)
            self._lbl_global_status.setText(f"✓ Sauvegardé : {os.path.basename(path)}")

    # ── Onglet Packer ─────────────────────────────────────────────────────────

    def _build_tab_packer(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(16, 12, 16, 12)
        vl.setSpacing(10)

        grp_opts = QGroupBox("Options de packing")
        fo = QFormLayout(grp_opts)
        fo.setContentsMargins(12, 10, 12, 10)
        fo.setSpacing(8)

        self._chk_pack_minify = QCheckBox("Minifier les shaders avant packing")
        self._chk_pack_minify.setChecked(True)
        fo.addRow("", self._chk_pack_minify)

        self._spn_pack_w = QSpinBox()
        self._spn_pack_w.setRange(320, 3840)
        self._spn_pack_w.setValue(800)
        self._spn_pack_w.setSuffix(" px")
        fo.addRow("Largeur", self._spn_pack_w)

        self._spn_pack_h = QSpinBox()
        self._spn_pack_h.setRange(240, 2160)
        self._spn_pack_h.setValue(450)
        self._spn_pack_h.setSuffix(" px")
        fo.addRow("Hauteur", self._spn_pack_h)

        vl.addWidget(grp_opts)

        # Résultat
        self._lbl_pack_stats = QLabel(
            "Le packer génère un fichier .py auto-extractible qui décompresse\n"
            "et joue vos shaders via moderngl + pygame, sans dépendances Qt.")
        self._lbl_pack_stats.setWordWrap(True)
        self._lbl_pack_stats.setStyleSheet("color: palette(mid); font-size: 11px;")
        vl.addWidget(self._lbl_pack_stats)

        self._txt_pack_preview = QPlainTextEdit()
        self._txt_pack_preview.setReadOnly(True)
        self._txt_pack_preview.setFont(QFont('Consolas', 8))
        self._txt_pack_preview.setMaximumHeight(160)
        self._txt_pack_preview.setPlaceholderText("Aperçu du stub généré…")
        vl.addWidget(self._txt_pack_preview)

        hl_btns = QHBoxLayout()
        btn_gen = QPushButton("▶ Générer le stub")
        btn_gen.clicked.connect(self._run_packer)
        hl_btns.addWidget(btn_gen)

        self._btn_save_stub = QPushButton("💾 Sauvegarder .py…")
        self._btn_save_stub.setEnabled(False)
        self._btn_save_stub.clicked.connect(self._save_stub)
        hl_btns.addWidget(self._btn_save_stub)
        hl_btns.addStretch()
        vl.addLayout(hl_btns)

        vl.addStretch()
        self._pack_result: PackResult | None = None
        return w

    def _run_packer(self):
        packer = ShaderPacker(
            shaders = self.shaders,
            width   = self._spn_pack_w.value(),
            height  = self._spn_pack_h.value(),
            minify  = self._chk_pack_minify.isChecked(),
        )
        result = packer.pack()
        self._pack_result = result

        lines = [
            f"Passes incluses : {', '.join(result.passes)}",
            f"Stub source     : {result.stub_bytes:,} B",
            f"Payload LZMA    : {result.payload_bytes:,} B",
            f"Total estimé    : {result.total_bytes:,} B"
            f"  ({result.total_bytes/1024:.1f} KB)",
            "",
            "─" * 40,
            "Extrait du stub :",
        ]
        lines += result.stub_source.splitlines()[:18]
        lines.append("…")

        self._txt_pack_preview.setPlainText('\n'.join(lines))
        self._lbl_pack_stats.setText(
            f"✓ Stub généré — {result.total_bytes:,} B total  "
            f"({'✅ sous 4K' if result.total_bytes < _BUDGET_4K else '⚠ dépasse 4K'}  "
            f"{'✅ sous 64K' if result.total_bytes < _BUDGET_64K else '⚠ dépasse 64K'})"
        )
        self._btn_save_stub.setEnabled(True)

    def _save_stub(self):
        if not self._pack_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder le stub Python",
            "intro.py", "Python (*.py)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._pack_result.stub_source)
            self._lbl_global_status.setText(
                f"✓ Stub sauvegardé : {os.path.basename(path)} "
                f"({self._pack_result.total_bytes:,} B)")

    # ── Onglet Synthétiseur ───────────────────────────────────────────────────

    def _build_tab_synth(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(16, 12, 16, 12)
        vl.setSpacing(10)

        # Info
        info = QLabel(
            "Synthétiseur procédural inspiré de 4klang. Génère l'audio du patch "
            "défini ci-dessous en Python pur (numpy). Export WAV 44100 Hz stéréo.")
        info.setWordWrap(True)
        info.setStyleSheet("color: palette(mid); font-size: 11px;")
        vl.addWidget(info)

        # Patch global
        grp_global = QGroupBox("Paramètres globaux")
        fg = QFormLayout(grp_global)
        fg.setContentsMargins(12, 8, 12, 8)
        fg.setSpacing(6)

        self._spn_bpm = QDoubleSpinBox()
        self._spn_bpm.setRange(40, 300)
        self._spn_bpm.setValue(self._synth_patch.bpm)
        self._spn_bpm.setSuffix(" BPM")
        fg.addRow("BPM", self._spn_bpm)

        self._spn_dur = QDoubleSpinBox()
        self._spn_dur.setRange(0.5, 300)
        self._spn_dur.setValue(self._synth_patch.duration)
        self._spn_dur.setSuffix(" s")
        fg.addRow("Durée", self._spn_dur)
        vl.addWidget(grp_global)

        # Tableau des tracks
        grp_tracks = QGroupBox("Pistes")
        tv = QVBoxLayout(grp_tracks)
        tv.setContentsMargins(8, 8, 8, 8)

        self._tbl_tracks = QTableWidget(0, 9)
        self._tbl_tracks.setHorizontalHeaderLabels(
            ["Wave", "Fréq (Hz)", "Amp", "A", "D", "S", "R", "Pan", "Dist"])
        self._tbl_tracks.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._tbl_tracks.setMinimumHeight(160)
        self._populate_tracks_table()
        tv.addWidget(self._tbl_tracks)

        hl_track_btns = QHBoxLayout()
        btn_add = QPushButton("＋ Ajouter piste")
        btn_add.clicked.connect(self._add_synth_track)
        hl_track_btns.addWidget(btn_add)
        btn_del = QPushButton("✕ Supprimer")
        btn_del.clicked.connect(self._del_synth_track)
        hl_track_btns.addWidget(btn_del)
        btn_default = QPushButton("↺ Patch 4K par défaut")
        btn_default.clicked.connect(self._load_default_patch)
        hl_track_btns.addWidget(btn_default)
        hl_track_btns.addStretch()
        tv.addLayout(hl_track_btns)
        vl.addWidget(grp_tracks)

        # Boutons export
        hl_export = QHBoxLayout()
        self._lbl_synth_status = QLabel("")
        self._lbl_synth_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hl_export.addWidget(self._lbl_synth_status, 1)

        btn_preview = QPushButton("▶ Prévisualiser (console)")
        btn_preview.clicked.connect(self._preview_synth)
        hl_export.addWidget(btn_preview)

        btn_export_wav = QPushButton("💾 Exporter WAV…")
        btn_export_wav.clicked.connect(self._export_synth_wav)
        hl_export.addWidget(btn_export_wav)

        btn_export_json = QPushButton("💾 Exporter patch JSON…")
        btn_export_json.clicked.connect(self._export_synth_json)
        hl_export.addWidget(btn_export_json)
        vl.addLayout(hl_export)
        return w

    def _populate_tracks_table(self):
        self._tbl_tracks.setRowCount(0)
        for t in self._synth_patch.tracks:
            row = self._tbl_tracks.rowCount()
            self._tbl_tracks.insertRow(row)
            vals = [
                t.get('wave', 'sin'),
                str(t.get('freq', 440)),
                f"{t.get('amp', 0.5):.2f}",
                f"{t.get('attack', 0.01):.3f}",
                f"{t.get('decay', 0.1):.2f}",
                f"{t.get('sustain', 0.7):.2f}",
                f"{t.get('release', 0.2):.2f}",
                f"{t.get('pan', 0.0):.2f}",
                f"{t.get('dist', 0.0):.2f}",
            ]
            for col, val in enumerate(vals):
                self._tbl_tracks.setItem(row, col, QTableWidgetItem(val))

    def _add_synth_track(self):
        self._synth_patch.add_track()
        self._populate_tracks_table()

    def _del_synth_track(self):
        row = self._tbl_tracks.currentRow()
        if 0 <= row < len(self._synth_patch.tracks):
            self._synth_patch.tracks.pop(row)
            self._populate_tracks_table()

    def _load_default_patch(self):
        self._synth_patch = SynthPatch.default_4k()
        self._spn_bpm.setValue(self._synth_patch.bpm)
        self._spn_dur.setValue(self._synth_patch.duration)
        self._populate_tracks_table()

    def _collect_patch_from_ui(self) -> SynthPatch:
        """Lit le tableau UI et retourne un SynthPatch."""
        patch = SynthPatch()
        patch.bpm      = self._spn_bpm.value()
        patch.duration = self._spn_dur.value()
        waves = ('sin', 'saw', 'sqr', 'tri', 'noise')
        for row in range(self._tbl_tracks.rowCount()):
            def cell(c):
                item = self._tbl_tracks.item(row, c)
                return item.text() if item else ''
            wave = cell(0) if cell(0) in waves else 'sin'
            try:
                patch.add_track(
                    wave    = wave,
                    freq    = float(cell(1) or 440),
                    amp     = float(cell(2) or 0.5),
                    attack  = float(cell(3) or 0.01),
                    decay   = float(cell(4) or 0.1),
                    sustain = float(cell(5) or 0.7),
                    release = float(cell(6) or 0.2),
                    pan     = float(cell(7) or 0.0),
                    dist    = float(cell(8) or 0.0),
                )
            except ValueError:
                pass
        return patch

    def _preview_synth(self):
        patch = self._collect_patch_from_ui()
        json_str = json.dumps(patch.to_dict(), indent=2)
        n_tracks = len(patch.tracks)
        dur = patch.duration
        est_size = len(lzma.compress(json_str.encode(), preset=lzma.PRESET_EXTREME))
        self._lbl_synth_status.setText(
            f"{n_tracks} piste(s) · {dur:.1f}s · patch LZMA ≈ {est_size} B")

    def _export_synth_wav(self):
        patch = self._collect_patch_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter audio WAV",
            "synth_audio.wav", "WAV (*.wav)")
        if not path:
            return
        try:
            synth = ProceduralSynth(patch)
            n = synth.export_wav(path)
            self._lbl_synth_status.setText(
                f"✓ WAV exporté : {os.path.basename(path)}  ({n:,} samples · {n/synth.SAMPLE_RATE:.1f}s)")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur export WAV :\n{e}")

    def _export_synth_json(self):
        patch = self._collect_patch_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter patch JSON",
            "synth_patch.json", "JSON (*.json)")
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(patch.to_dict(), f, indent=2)
        self._lbl_synth_status.setText(f"✓ Patch sauvegardé : {os.path.basename(path)}")

    # ── Onglet Estimation ─────────────────────────────────────────────────────

    def _build_tab_estimate(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(16, 14, 16, 14)
        vl.setSpacing(12)

        info = QLabel(
            "Estimation en temps réel de la taille compressée (LZMA niveau 9).\n"
            "Mise à jour automatique toutes les 2 secondes.")
        info.setStyleSheet("color: palette(mid); font-size: 11px;")
        vl.addWidget(info)

        self._tbl_estimate = QTableWidget(0, 2)
        self._tbl_estimate.setHorizontalHeaderLabels(["Composant", "Taille"])
        self._tbl_estimate.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._tbl_estimate.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl_estimate.setFixedHeight(250)
        vl.addWidget(self._tbl_estimate)

        # Barres de budget
        grp_budgets = QGroupBox("Budgets intro")
        fb = QVBoxLayout(grp_budgets)
        fb.setContentsMargins(12, 10, 12, 10)
        fb.setSpacing(8)

        self._lbl_budget_4k  = QLabel("")
        self._lbl_budget_64k = QLabel("")
        self._lbl_budget_96k = QLabel("")
        for lbl in (self._lbl_budget_4k, self._lbl_budget_64k, self._lbl_budget_96k):
            lbl.setFont(QFont('Consolas', 9))
            fb.addWidget(lbl)

        vl.addWidget(grp_budgets)

        btn_refresh = QPushButton("↺ Recalculer maintenant")
        btn_refresh.clicked.connect(self._refresh_estimate)
        vl.addWidget(btn_refresh, alignment=Qt.AlignmentFlag.AlignLeft)
        vl.addStretch()
        return w

    def _refresh_estimate(self):
        """Recalcule l'estimation et met à jour l'onglet + le label status bar."""
        est = IntroSizeEstimator(self.shaders, self._synth_patch)
        r   = est.estimate()

        rows = [
            ("Shaders bruts",          est.format_bytes(r.shaders_raw_bytes)),
            ("Shaders minifiés",       est.format_bytes(r.shaders_mini_bytes)
                                       + f"  (−{r.minify_savings_pct:.0f}%)"),
            ("Shaders LZMA",           est.format_bytes(r.shaders_lzma_bytes)),
            ("Patch audio LZMA",       est.format_bytes(r.audio_lzma_bytes)),
            ("Stub Python",            est.format_bytes(r.stub_bytes)),
            ("─" * 20,                 "─" * 10),
            ("Total estimé",           est.format_bytes(r.total_estimated)),
        ]
        self._tbl_estimate.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._tbl_estimate.setItem(i, 0, QTableWidgetItem(k))
            self._tbl_estimate.setItem(i, 1, QTableWidgetItem(v))

        used = r.total_estimated
        self._lbl_budget_4k.setText(
            f"4K  (4096 B) : {est.budget_bar(used, _BUDGET_4K)} "
            f"libre : {est.format_bytes(r.budget_4k_free)}"
        )
        self._lbl_budget_64k.setText(
            f"64K (64 KB)  : {est.budget_bar(used, _BUDGET_64K)} "
            f"libre : {est.format_bytes(r.budget_64k_free)}"
        )
        self._lbl_budget_96k.setText(
            f"96K (96 KB)  : {est.budget_bar(used, _BUDGET_96K)} "
            f"libre : {est.format_bytes(r.budget_96k_free)}"
        )

        # Signal vers la status bar (via attribut public)
        self._last_size_report = r

    # ── Helpers UI ────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_with_label(widget: QWidget, label: str) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: palette(mid); font-size: 10px;")
        vl.addWidget(lbl)
        vl.addWidget(widget)
        return w
