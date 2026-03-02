"""
ai_param_detector.py
---------------------
v1.0 — Détection automatique de paramètres exposables (uniforms) dans un shader GLSL.

Fonctionnalités :
  - Scan des uniforms déjà déclarés (float/int/vec) dans le code source
  - Détection des "magic numbers" (constantes numériques inline exposables)
  - Nommage intelligent selon le contexte : uSpeed, uColorIntensity, uDistortion, etc.
  - Estimation des bornes min/max/step selon le contexte sémantique
  - Suggestions d'exposition avec priorité (score de pertinence)

Retourne une liste de ShaderParam prête à être injectée dans le panneau gauche.

Usage :
    detector = AIParamDetector()
    params = detector.detect(glsl_code)   # → list[ShaderParam]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  Données
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShaderParam:
    """Un paramètre exposable détecté dans le shader."""
    name:        str           # nom uniform : uSpeed
    label:       str           # libellé UI : Speed
    glsl_type:   str           # float | int | vec2 | vec3 | vec4
    default:     float         # valeur par défaut
    min_val:     float         # borne minimum
    max_val:     float         # borne maximum
    step:        float         # pas du slider
    source:      str           # 'declared' | 'magic_number'
    original:    str           # valeur originale dans le code (pour magic numbers)
    context:     str = ""      # ligne de code contextuelle
    score:       int  = 0      # score de pertinence (+ = plus important)
    category:    str  = ""     # 'animation' | 'color' | 'geometry' | 'audio' | 'misc'


# ═══════════════════════════════════════════════════════════════════════════
#  Uniforms système déjà fournis par l'hôte — on ne les réexpose pas
# ═══════════════════════════════════════════════════════════════════════════

_HOST_UNIFORMS = {
    "iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse",
    "iSampleRate", "uRMS", "uBeat", "uBPM",
    # Post-processing FX (left_panel.py)
    "uChromatic", "uBloom", "uVignette", "uBlurRadius", "uGlitch",
    "uScanlines", "uCurvature", "uGrain", "uSaturation", "uContrast",
    "uBrightness", "uPixelSize", "uColors", "uKaleido",
    "uMirrorX", "uMirrorY", "uHueShift", "uSharpen", "uEdge",
    "uPosterize", "uDuoR1", "uDuoG1", "uDuoB1", "uDuoR2", "uDuoG2", "uDuoB2",
    "uNeon", "uThermal", "uOldFilm", "uHalftone", "uOilRadius",
    "uFisheye", "uRGBSplit", "uRGBAngle", "uWarp", "uWarpFreq",
    "uZoom", "uZoomX", "uZoomY", "uTiltFocus", "uTiltBlur",
    "uDither", "uRecolorHue", "uRecolorSat",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Règles sémantiques pour nommage et bornes automatiques
# ═══════════════════════════════════════════════════════════════════════════

# (regex sur le contexte de la ligne) → (nom suggéré, label, min, max, step, catégorie, score)
_SEMANTIC_RULES: list[tuple[str, str, str, float, float, float, str, int]] = [
    # pattern_context        name_hint          label               mn    mx    step  cat        score
    (r'speed|velocity|vel',  "uSpeed",          "Speed",            0.0,  5.0,  0.01, "animation", 10),
    (r'freq|frequency',      "uFrequency",      "Frequency",        0.1, 20.0,  0.1,  "animation", 9),
    (r'amplitude|amp\b',     "uAmplitude",      "Amplitude",        0.0,  2.0,  0.01, "animation", 9),
    (r'phase|offset\b',      "uPhase",          "Phase",            0.0,  6.28, 0.01, "animation", 7),
    (r'distort|warp\b',      "uDistortion",     "Distortion",       0.0,  2.0,  0.01, "geometry",  9),
    (r'scale\b|zoom\b',      "uScale",          "Scale",            0.1,  5.0,  0.01, "geometry",  8),
    (r'rotate|rotation|angle', "uRotation",     "Rotation",         0.0,  6.28, 0.01, "geometry",  7),
    (r'twist|swirl',         "uTwist",          "Twist",            0.0,  5.0,  0.01, "geometry",  8),
    (r'color.*intens|intens.*color', "uColorIntensity", "Color Intensity", 0.0, 3.0, 0.01, "color", 10),
    (r'hue\b|color\b',       "uHue",            "Hue",              0.0,  1.0,  0.005,"color",     8),
    (r'bright|luminan',      "uBrightness",     "Brightness",       0.0,  2.0,  0.01, "color",     8),
    (r'contrast\b',          "uContrast",       "Contrast",         0.0,  3.0,  0.01, "color",     8),
    (r'satur',               "uSaturation",     "Saturation",       0.0,  2.0,  0.01, "color",     8),
    (r'glow|bloom',          "uGlow",           "Glow",             0.0,  2.0,  0.01, "color",     7),
    (r'noise\b|grain\b',     "uNoise",          "Noise",            0.0,  1.0,  0.005,"misc",      8),
    (r'smooth|softness',     "uSmoothness",     "Smoothness",       0.0,  1.0,  0.005,"misc",      7),
    (r'thick|width|stroke',  "uThickness",      "Thickness",        0.0,  1.0,  0.005,"geometry",  7),
    (r'iter|octave|step\b',  "uIterations",     "Iterations",       1.0, 12.0,  1.0,  "misc",      6),
    (r'time\b|t\b',          "uTimeScale",      "Time Scale",       0.0,  3.0,  0.01, "animation", 6),
    (r'radius\b|r\b',        "uRadius",         "Radius",           0.0,  2.0,  0.01, "geometry",  7),
    (r'mix\b|blend\b|fade',  "uBlend",          "Blend",            0.0,  1.0,  0.005,"color",     7),
    (r'density|density',     "uDensity",        "Density",          0.0,  5.0,  0.01, "misc",      6),
    (r'sharp|detail',        "uDetail",         "Detail",           0.0,  1.0,  0.005,"misc",      6),
    (r'shadow|dark',         "uShadow",         "Shadow",           0.0,  1.0,  0.005,"color",     6),
    (r'light|shine',         "uLight",          "Light",            0.0,  2.0,  0.01, "color",     6),
    (r'grid|tile',           "uGridSize",       "Grid Size",        1.0, 20.0,  0.5,  "geometry",  7),
]

# Noms génériques de fallback selon le type de valeur
def _generic_name_from_value(val: float, context: str) -> tuple[str, str, float, float, float, str, int]:
    """Fallback nommage quand aucune règle sémantique ne matche."""
    if 0.0 <= val <= 1.0:
        return "uMix", "Mix", 0.0, 1.0, 0.005, "misc", 3
    elif 1.0 < val <= 10.0:
        return "uScale", "Scale", 0.0, 10.0, 0.01, "geometry", 3
    elif val > 10.0:
        return "uIntensity", "Intensity", 0.0, val * 2.0, val * 0.1, "misc", 2
    else:  # négatif
        return "uOffset", "Offset", val * 2.0, -val * 2.0, abs(val) * 0.1, "misc", 2


# ═══════════════════════════════════════════════════════════════════════════
#  Estimateur de bornes pour les magic numbers
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_bounds(val: float, context: str) -> tuple[float, float, float]:
    """
    Estime min/max/step pour un magic number selon sa valeur et son contexte.
    """
    ctx = context.lower()

    # Constantes mathématiques communes — skip
    if abs(val - 3.14159) < 0.01 or abs(val - 6.28318) < 0.01:
        return val * 0.5, val * 2.0, val * 0.01

    # Bornes selon le contexte
    if any(k in ctx for k in ("color", "col.", "rgb", "hue", "sat")):
        if 0.0 <= val <= 1.0:
            return 0.0, 1.0, 0.005
        return 0.0, 3.0, 0.01

    if any(k in ctx for k in ("time", "itime", "speed", "freq")):
        return 0.0, max(val * 5.0, 10.0), max(val * 0.01, 0.01)

    if any(k in ctx for k in ("resolution", "pixel", "px", "size")):
        return 1.0, max(val * 4.0, 100.0), max(1.0, val * 0.1)

    if any(k in ctx for k in ("uv", "coord", "pos", "vec2")):
        abs_v = abs(val)
        return -abs_v * 3.0, abs_v * 3.0, abs_v * 0.01

    # Règles génériques par magnitude
    if val == 0.0:
        return -1.0, 1.0, 0.01
    elif 0.0 < val <= 1.0:
        return 0.0, 1.0, 0.005
    elif 1.0 < val <= 5.0:
        return 0.0, val * 3.0, 0.01
    elif 5.0 < val <= 50.0:
        return 0.0, val * 2.0, 0.1
    elif val > 50.0:
        return 0.0, val * 2.0, val * 0.05
    else:  # négatif
        return val * 3.0, -val * 3.0, abs(val) * 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Sémantique contextuelle → nommage
# ═══════════════════════════════════════════════════════════════════════════

def _semantic_name(context: str, val: float) -> tuple[str, str, float, float, float, str, int]:
    """
    Cherche la meilleure règle sémantique pour le contexte donné.
    Retourne (name, label, min, max, step, category, score).
    """
    ctx = context.lower()
    for pattern, name, label, mn, mx, step, cat, score in _SEMANTIC_RULES:
        if re.search(pattern, ctx, re.IGNORECASE):
            # Ajuster les bornes si la valeur est hors-bornes
            if val < mn:
                mn = val * 2.0
            if val > mx:
                mx = val * 2.0
            return name, label, mn, mx, step, cat, score

    return _generic_name_from_value(val, context)


def _make_label(uniform_name: str) -> str:
    """
    uColorIntensity → 'Color Intensity'
    uSpeed          → 'Speed'
    """
    # Retire le préfixe 'u'
    name = uniform_name
    if name.startswith("u") and len(name) > 1 and name[1].isupper():
        name = name[1:]
    # CamelCase → mots
    words = re.sub(r'([A-Z])', r' \1', name).strip()
    return words


def _deduplicate_name(name: str, existing: set[str]) -> str:
    """Évite les collisions de noms : uSpeed → uSpeed2 → uSpeed3…"""
    if name not in existing:
        return name
    i = 2
    while f"{name}{i}" in existing:
        i += 1
    return f"{name}{i}"


# ═══════════════════════════════════════════════════════════════════════════
#  Détecteur principal
# ═══════════════════════════════════════════════════════════════════════════

# Regex pour les uniforms déjà déclarés
_RE_UNIFORM_DECL = re.compile(
    r'uniform\s+(float|int|vec[234]|bool)\s+(\w+)\s*;',
    re.MULTILINE,
)

# Regex pour les magic numbers dans le code (float uniquement pour éviter les faux positifs)
_RE_MAGIC_NUMBER = re.compile(
    r'(?<![.\w])(\d+\.\d+|\d+\.|\.\d+)(?![.\w\d])',
)

# Valeurs banales à ne pas exposer (constantes de normalisation, epsilon, etc.)
_SKIP_VALUES = {0.0, 1.0, 0.5, 2.0, 0.25, 0.75, 4.0, 8.0, 16.0, 0.0}
_SKIP_VALUES_CLOSE = [(3.14159, 0.01), (6.28318, 0.01), (1.5708, 0.01),
                      (0.333, 0.005), (0.299, 0.005), (0.587, 0.005),
                      (0.114, 0.005), (43758.5, 1.0), (12.989, 0.01),
                      (78.233, 0.01), (127.1, 0.1), (311.7, 0.1)]


def _is_trivial_value(val: float) -> bool:
    """Retourne True pour les constantes qui ne valent pas la peine d'être exposées."""
    if val in _SKIP_VALUES:
        return True
    for ref, tol in _SKIP_VALUES_CLOSE:
        if abs(val - ref) < tol:
            return True
    return False


def _get_line(code: str, pos: int) -> str:
    """Retourne la ligne de code à la position `pos`."""
    start = code.rfind('\n', 0, pos) + 1
    end   = code.find('\n', pos)
    if end == -1:
        end = len(code)
    return code[start:end].strip()


class AIParamDetector:
    """
    Analyse un shader GLSL et retourne les paramètres exposables.

    Deux sources :
    1. Uniforms déjà déclarés dans le shader (hors uniforms système)
    2. Magic numbers détectés inline avec suggestion de les transformer en uniforms

    Usage :
        detector = AIParamDetector()
        params = detector.detect(glsl_code)
    """

    def detect(self, glsl: str) -> list[ShaderParam]:
        """
        Analyse complète du shader.
        Retourne une liste triée par score décroissant.
        """
        if not glsl or not glsl.strip():
            return []

        used_names: set[str] = set(_HOST_UNIFORMS)
        results: list[ShaderParam] = []

        # ── 1. Uniforms déjà déclarés ─────────────────────────────────────
        for match in _RE_UNIFORM_DECL.finditer(glsl):
            glsl_type = match.group(1)
            uname     = match.group(2)

            if uname in _HOST_UNIFORMS:
                continue  # Uniforme système, skip

            used_names.add(uname)
            label = _make_label(uname)

            # Bornes par défaut selon le nom
            ctx_line = _get_line(glsl, match.start())
            name_hint, _, mn, mx, step, cat, score = _semantic_name(
                uname + " " + ctx_line, 1.0
            )

            results.append(ShaderParam(
                name      = uname,
                label     = label,
                glsl_type = glsl_type,
                default   = 1.0 if glsl_type == "float" else 0.0,
                min_val   = mn,
                max_val   = mx,
                step      = step,
                source    = "declared",
                original  = uname,
                context   = ctx_line,
                score     = score + 5,  # bonus car déjà déclaré
                category  = cat,
            ))

        # ── 2. Magic numbers ──────────────────────────────────────────────
        # On restreint au corps de mainImage pour éviter les déclarations globales
        body_match = re.search(r'void\s+mainImage\s*\([^)]*\)\s*\{', glsl)
        body_start = body_match.start() if body_match else 0
        body = glsl[body_start:]

        # Exclure les lignes de déclaration uniform
        body_clean = re.sub(r'^\s*uniform\s+.*$', '', body, flags=re.MULTILINE)

        seen_values: set[str] = set()

        for match in _RE_MAGIC_NUMBER.finditer(body_clean):
            raw_val = match.group(1)
            if raw_val in seen_values:
                continue

            try:
                val = float(raw_val)
            except ValueError:
                continue

            if _is_trivial_value(val):
                continue

            seen_values.add(raw_val)

            # Contexte de la ligne
            ctx_line = _get_line(body_clean, match.start())

            # Nommage sémantique
            name_hint, label_hint, mn, mx, step, cat, score = _semantic_name(ctx_line, val)

            # Dédupliquer le nom
            uname = _deduplicate_name(name_hint, used_names)
            used_names.add(uname)

            label = _make_label(uname)

            # Bornes affinées selon la valeur
            mn_v, mx_v, step_v = _estimate_bounds(val, ctx_line)
            # On préfère les bornes sémantiques si la valeur est dans la plage
            if not (mn <= val <= mx):
                mn, mx, step = mn_v, mx_v, step_v

            results.append(ShaderParam(
                name      = uname,
                label     = label,
                glsl_type = "float",
                default   = val,
                min_val   = mn,
                max_val   = mx,
                step      = step,
                source    = "magic_number",
                original  = raw_val,
                context   = ctx_line[:80],
                score     = score,
                category  = cat,
            ))

        # ── Tri : déclarés > magic numbers, puis par score ────────────────
        results.sort(key=lambda p: (0 if p.source == "declared" else 1, -p.score))

        # Limite raisonnable
        return results[:20]

    def apply_param_to_shader(self, glsl: str, param: ShaderParam) -> str:
        """
        Transforme un magic number en uniform dans le code GLSL.
        Ajoute la déclaration et remplace la valeur originale.

        Ne modifie que la première occurrence exacte pour éviter les faux positifs.
        """
        if param.source != "magic_number":
            return glsl

        # Ajoute la déclaration uniform avant mainImage
        decl = f"uniform float {param.name}; // auto-param: {param.label}\n"

        # Insertion avant void mainImage (ou en tête si pas trouvé)
        insert_pat = re.search(r'void\s+mainImage', glsl)
        if insert_pat:
            pos = insert_pat.start()
            glsl = glsl[:pos] + decl + glsl[pos:]
        else:
            glsl = decl + glsl

        # Remplace la première occurrence de la valeur originale dans le corps
        # On cherche la valeur exacte entourée de non-chiffres pour éviter les faux positifs
        pattern = r'(?<![.\w])' + re.escape(param.original) + r'(?![.\w\d])'
        glsl = re.sub(pattern, param.name, glsl, count=1)

        return glsl
