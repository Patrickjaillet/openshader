"""
code_editor.py
--------------
Éditeur de code GLSL intégré — v2.2

Nouvelles fonctionnalités v2.2 :
  - Color Picker inline : Ctrl+clic sur vec3/vec4 → color picker flottant (HSV/Hex)
    Mise à jour en temps réel du littéral dans le source GLSL.

Nouvelles fonctionnalités v1.4 :
  - LineNumberGutter : numérotation des lignes dans un gutter latéral
  - FindReplaceBar   : barre Rechercher / Remplacer (Ctrl+H) avec regex optionnel
  - Pliage de blocs  : fold/unfold des fonctions GLSL (Ctrl+Shift+[ / ])
  - Hover-docs       : tooltip de signature GLSL au survol du curseur
  - Autocomplétion contextuelle : uniforms déclarés dans le shader + mots-clés
  - Snippets         : bibliothèque insérables via Ctrl+J
  - Split view       : deux éditeurs côte à côte pour comparer deux passes
  - Thèmes           : Monokai, Dracula, Solarized Dark, Solarized Light

Améliorations v1.1 héritées :
  - Offset d'erreur calculé dynamiquement depuis le header compilé
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt6.QtWidgets import (
    QPlainTextEdit, QCompleter, QTextEdit, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QCheckBox, QLabel,
    QToolTip, QSplitter, QFrame, QApplication, QMenu, QSizePolicy,
    QColorDialog,
)
from PyQt6.QtCore import (
    pyqtSignal, QTimer, Qt, QRect, QSize, QPoint, QRegularExpression,
)
from PyQt6.QtGui import (
    QFont, QTextCursor, QTextFormat, QColor, QPainter, QTextBlock,
    QFontMetrics, QPalette, QSyntaxHighlighter, QTextCharFormat,
    QKeySequence, QShortcut, QTextDocument,
)

from .glsl_highlighter import GLSLHighlighter
from .glsl_ai_completion import AICompletionMixin

# ── Constantes GLSL ──────────────────────────────────────────────────────────

GLSL_KEYWORDS = [
    # Types
    "float", "int", "uint", "bool",
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "bvec2", "bvec3", "bvec4",
    "mat2", "mat3", "mat4",
    "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4",
    "mat4x2", "mat4x3", "mat4x4",
    "sampler2D", "sampler3D", "samplerCube",
    "void",
    # Qualifiers
    "in", "out", "inout", "uniform", "const", "varying",
    "lowp", "mediump", "highp", "precision",
    # Flow control
    "if", "else", "for", "while", "do", "break", "continue",
    "return", "switch", "case", "default", "discard",
    # Fonctions natives GLSL
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
    "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract",
    "mod", "modf", "min", "max", "clamp", "mix", "step", "smoothstep",
    "length", "distance", "dot", "cross", "normalize",
    "faceforward", "reflect", "refract",
    "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse",
    "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
    "equal", "notEqual", "any", "all", "not",
    "texture", "texture2D", "textureCube", "textureSize", "texelFetch",
    "dFdx", "dFdy", "fwidth",
    # Shadertoy
    "mainImage", "iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse",
    "iChannel0", "iChannel1", "iChannel2", "iChannel3",
    "iAudioAmplitude", "iAudioRMS",
    # OpenShader GLSL pur
    "uResolution", "uTime", "uTimeDelta", "uFrame",
]

# ── Hover-docs : signature GLSL → description ────────────────────────────────

GLSL_DOCS: dict[str, str] = {
    "sin":         "float sin(float angle)\nSinus de l'angle (radians).",
    "cos":         "float cos(float angle)\nCosinus de l'angle (radians).",
    "tan":         "float tan(float angle)\nTangente.",
    "atan":        "float atan(float y, float x)  — ou —\nfloat atan(float y_over_x)\nArctangente.",
    "pow":         "genType pow(genType x, genType y)\nx élevé à la puissance y.",
    "exp":         "genType exp(genType x)\nExponentielle naturelle : e^x.",
    "log":         "genType log(genType x)\nLogarithme naturel.",
    "sqrt":        "genType sqrt(genType x)\nRacine carrée.",
    "inversesqrt": "genType inversesqrt(genType x)\n1.0 / sqrt(x).",
    "abs":         "genType abs(genType x)\nValeur absolue.",
    "sign":        "genType sign(genType x)\n-1, 0 ou 1 selon le signe de x.",
    "floor":       "genType floor(genType x)\nArrondi vers le bas.",
    "ceil":        "genType ceil(genType x)\nArrondi vers le haut.",
    "fract":       "genType fract(genType x)\nPartie fractionnaire : x - floor(x).",
    "mod":         "genType mod(genType x, genType y)\nModulo : x - y * floor(x/y).",
    "min":         "genType min(genType x, genType y)\nMinimum composante par composante.",
    "max":         "genType max(genType x, genType y)\nMaximum composante par composante.",
    "clamp":       "genType clamp(genType x, genType minVal, genType maxVal)\nLimite x dans [minVal, maxVal].",
    "mix":         "genType mix(genType x, genType y, genType a)\nInterpolation linéaire : x*(1-a) + y*a.",
    "step":        "genType step(genType edge, genType x)\n0.0 si x < edge, sinon 1.0.",
    "smoothstep":  "genType smoothstep(genType edge0, genType edge1, genType x)\nInterpolation Hermite lisse entre 0 et 1.",
    "length":      "float length(genType x)\nLongueur (norme L2) du vecteur.",
    "distance":    "float distance(genType p0, genType p1)\nDistance euclidienne entre p0 et p1.",
    "dot":         "float dot(genType x, genType y)\nProduit scalaire.",
    "cross":       "vec3 cross(vec3 x, vec3 y)\nProduit vectoriel.",
    "normalize":   "genType normalize(genType x)\nVecteur unitaire dans la même direction.",
    "reflect":     "genType reflect(genType I, genType N)\nDirection réfléchie par rapport à la normale N.",
    "refract":     "genType refract(genType I, genType N, float eta)\nDirection réfractée (loi de Snell).",
    "texture":     "vec4 texture(sampler2D sampler, vec2 P)\nÉchantillonnage de texture.",
    "texelFetch":  "vec4 texelFetch(sampler2D sampler, ivec2 P, int lod)\nAccès direct à un texel (sans filtrage).",
    "dFdx":        "genType dFdx(genType p)\nDérivée partielle selon x (espace écran).",
    "dFdy":        "genType dFdy(genType p)\nDérivée partielle selon y (espace écran).",
    "fwidth":      "genType fwidth(genType p)\nabs(dFdx(p)) + abs(dFdy(p)) — utile pour l'antialiasing.",
    "mainImage":   "void mainImage(out vec4 fragColor, in vec2 fragCoord)\nPoint d'entrée Shadertoy. fragCoord en pixels.",
    "iResolution": "vec3 iResolution\nRésolution du viewport (x=largeur, y=hauteur, z=ratio).",
    "iTime":       "float iTime\nTemps écoulé en secondes depuis le démarrage.",
    "iTimeDelta":  "float iTimeDelta\nDurée du frame précédent en secondes.",
    "iFrame":      "int iFrame\nNuméro du frame courant.",
    "iMouse":      "vec4 iMouse\n(x,y) = position souris. (z,w) = position au clic.\nz/w négatifs si bouton relâché (convention Shadertoy).",
    "iChannel0":   "sampler2D iChannel0\nTexture/Buffer d'entrée canal 0.",
    "iChannel1":   "sampler2D iChannel1\nTexture/Buffer d'entrée canal 1.",
    "iAudioAmplitude": "float iAudioAmplitude\nAmplitude audio instantanée [0.0, 1.0].",
    "iAudioRMS":   "float iAudioRMS\nRMS de la waveform audio [0.0, 1.0].",
    "uResolution": "vec2 uResolution\nRésolution du viewport (GLSL pur).",
    "uTime":       "float uTime\nTemps en secondes (GLSL pur).",
}

# ── Snippets ─────────────────────────────────────────────────────────────────

GLSL_SNIPPETS: dict[str, str] = {
    "noise2": (
        "// Hash + value noise 2D\nfloat hash(vec2 p) {\n"
        "    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);\n}\n"
        "float noise(vec2 p) {\n"
        "    vec2 i = floor(p), f = fract(p);\n"
        "    vec2 u = f * f * (3.0 - 2.0 * f);\n"
        "    return mix(mix(hash(i), hash(i+vec2(1,0)), u.x),\n"
        "               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);\n}\n"
    ),
    "fbm": (
        "// FBM (Fractal Brownian Motion)\nfloat fbm(vec2 p) {\n"
        "    float v = 0.0, a = 0.5;\n"
        "    for (int i = 0; i < 6; i++) {\n"
        "        v += a * noise(p);\n"
        "        p = p * 2.0 + vec2(1.7, 9.2);\n"
        "        a *= 0.5;\n"
        "    }\n"
        "    return v;\n}\n"
    ),
    "sdf_circle": (
        "// SDF — Cercle (centre c, rayon r)\nfloat sdCircle(vec2 p, vec2 c, float r) {\n"
        "    return length(p - c) - r;\n}\n"
    ),
    "sdf_box": (
        "// SDF — Boîte 2D (demi-dimensions b)\nfloat sdBox(vec2 p, vec2 b) {\n"
        "    vec2 d = abs(p) - b;\n"
        "    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);\n}\n"
    ),
    "raymarching": (
        "// Raymarching template\nfloat map(vec3 p) {\n    return length(p) - 1.0; // SDF scène\n}\n\n"
        "vec3 getNormal(vec3 p) {\n    const float e = 0.001;\n"
        "    return normalize(vec3(\n"
        "        map(p+vec3(e,0,0)) - map(p-vec3(e,0,0)),\n"
        "        map(p+vec3(0,e,0)) - map(p-vec3(0,e,0)),\n"
        "        map(p+vec3(0,0,e)) - map(p-vec3(0,0,e))\n    ));\n}\n\n"
        "float raymarch(vec3 ro, vec3 rd) {\n    float t = 0.0;\n"
        "    for (int i = 0; i < 100; i++) {\n"
        "        float d = map(ro + rd * t);\n"
        "        if (d < 0.001) return t;\n"
        "        t += d;\n        if (t > 100.0) break;\n"
        "    }\n    return -1.0;\n}\n"
    ),
    "pbr_lighting": (
        "// Éclairage PBR simplifié (Blinn-Phong)\nvec3 pbr(vec3 N, vec3 V, vec3 L, vec3 albedo,\n"
        "         float roughness, vec3 lightCol) {\n"
        "    vec3 H = normalize(V + L);\n"
        "    float NdL = max(dot(N, L), 0.0);\n"
        "    float NdH = max(dot(N, H), 0.0);\n"
        "    float spec = pow(NdH, mix(4.0, 256.0, 1.0 - roughness));\n"
        "    return lightCol * (albedo * NdL + vec3(spec) * NdL);\n}\n"
    ),
    "hue_shift": (
        "// Rotation de teinte (angle en radians)\nvec3 hueShift(vec3 col, float angle) {\n"
        "    const vec3 k = vec3(0.57735);\n"
        "    float c = cos(angle), s = sin(angle);\n"
        "    return col*c + cross(k,col)*s + k*dot(k,col)*(1.0-c);\n}\n"
    ),
    "palette": (
        "// Palette cosinus (Inigo Quilez)\nvec3 palette(float t,\n"
        "    vec3 a, vec3 b, vec3 c, vec3 d) {\n"
        "    return a + b * cos(6.28318 * (c * t + d));\n}\n"
    ),
    "polar_uv": (
        "// Coordonnées polaires depuis UV centré\nvec2 toPolar(vec2 uv) {\n"
        "    return vec2(length(uv), atan(uv.y, uv.x));\n}\n"
    ),
}

# ── Thèmes ───────────────────────────────────────────────────────────────────

THEMES: dict[str, dict] = {
    "OpenShader": {
        "bg":          "#0d0f14",
        "fg":          "#cdd6f4",
        "gutter_bg":   "#12141a",
        "gutter_fg":   "#3a4060",
        "cursor_line": "#1a1d2e",
        "selection":   "#2a3060",
        "error_line":  "#3c1414",
    },
    "Monokai": {
        "bg":          "#272822",
        "fg":          "#f8f8f2",
        "gutter_bg":   "#1e1f1c",
        "gutter_fg":   "#75715e",
        "cursor_line": "#3e3d32",
        "selection":   "#49483e",
        "error_line":  "#4d1a1a",
    },
    "Dracula": {
        "bg":          "#282a36",
        "fg":          "#f8f8f2",
        "gutter_bg":   "#21222c",
        "gutter_fg":   "#6272a4",
        "cursor_line": "#44475a",
        "selection":   "#44475a",
        "error_line":  "#5c1f2e",
    },
    "Solarized Dark": {
        "bg":          "#002b36",
        "fg":          "#839496",
        "gutter_bg":   "#073642",
        "gutter_fg":   "#586e75",
        "cursor_line": "#073642",
        "selection":   "#0d3d52",
        "error_line":  "#3a1010",
    },
    "Solarized Light": {
        "bg":          "#fdf6e3",
        "fg":          "#657b83",
        "gutter_bg":   "#eee8d5",
        "gutter_fg":   "#93a1a1",
        "cursor_line": "#eee8d5",
        "selection":   "#d2cba0",
        "error_line":  "#f5cccc",
    },
}

_DEFAULT_THEME = "OpenShader"

# ── LineNumberGutter ──────────────────────────────────────────────────────────

class LineNumberGutter(QWidget):
    """Gutter latéral affichant les numéros de ligne."""

    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self._editor = editor
        self.setFont(editor.font())

    def sizeHint(self) -> QSize:
        return QSize(self._editor._gutter_width(), 0)

    def paintEvent(self, event):
        self._editor._paint_gutter(event)


# ── FindReplaceBar ────────────────────────────────────────────────────────────

class FindReplaceBar(QFrame):
    """Barre Rechercher / Remplacer flottante sous la toolbar."""

    closed = pyqtSignal()

    def __init__(self, parent: "CodeEditor"):
        super().__init__(parent)
        self._editor = parent

        self.setStyleSheet(
            "FindReplaceBar { background:#12141a; border-top:1px solid #1e2030; }"
            "QLineEdit { background:#0d0f14; color:#cdd6f4; border:1px solid #2a3060;"
            "            border-radius:3px; padding:2px 6px; }"
            "QCheckBox { color:#6272a4; font-size:9px; }"
            "QPushButton { background:#1a1d2e; color:#89b4fa; border:1px solid #2a3060;"
            "              border-radius:3px; padding:2px 8px; font-size:9px; }"
            "QPushButton:hover { background:#2a3060; }"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        # Chercher
        lay.addWidget(QLabel("🔍", styleSheet="color:#6272a4; font:10px;"))
        self.find_edit = QLineEdit(placeholderText="Rechercher…")
        self.find_edit.setFixedHeight(22)
        self.find_edit.textChanged.connect(self._on_find_changed)
        self.find_edit.returnPressed.connect(self.find_next)
        lay.addWidget(self.find_edit)

        # Remplacer
        lay.addWidget(QLabel("→", styleSheet="color:#6272a4;"))
        self.replace_edit = QLineEdit(placeholderText="Remplacer par…")
        self.replace_edit.setFixedHeight(22)
        lay.addWidget(self.replace_edit)

        # Options
        self.chk_case = QCheckBox("Casse")
        self.chk_regex = QCheckBox("Regex")
        lay.addWidget(self.chk_case)
        lay.addWidget(self.chk_regex)

        # Boutons
        btn_prev = QPushButton("◀")
        btn_prev.setFixedWidth(28)
        btn_prev.setToolTip("Occurrence précédente")
        btn_prev.clicked.connect(self.find_prev)
        lay.addWidget(btn_prev)

        btn_next = QPushButton("▶")
        btn_next.setFixedWidth(28)
        btn_next.setToolTip("Occurrence suivante")
        btn_next.clicked.connect(self.find_next)
        lay.addWidget(btn_next)

        btn_replace = QPushButton("Rempl.")
        btn_replace.setToolTip("Remplacer l'occurrence courante")
        btn_replace.clicked.connect(self.replace_current)
        lay.addWidget(btn_replace)

        btn_replace_all = QPushButton("Tout")
        btn_replace_all.setToolTip("Remplacer toutes les occurrences")
        btn_replace_all.clicked.connect(self.replace_all)
        lay.addWidget(btn_replace_all)

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet("color:#6272a4; font-size:9px; min-width:60px;")
        lay.addWidget(self._lbl_count)

        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(22)
        btn_close.clicked.connect(self._close)
        lay.addWidget(btn_close)

        self.setFixedHeight(34)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_flags(self) -> QTextDocument.FindFlag:
        flags = QTextDocument.FindFlag(0)
        if self.chk_case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        return flags

    def _pattern(self) -> str:
        txt = self.find_edit.text()
        if not self.chk_regex.isChecked():
            txt = re.escape(txt)
        return txt

    def _on_find_changed(self):
        self._highlight_all()
        self._update_count()

    def _highlight_all(self):
        """Surligne toutes les occurrences en jaune pâle."""
        ed = self._editor
        pattern = self._pattern()
        selections: list[QTextEdit.ExtraSelection] = []

        # Garde la sélection d'erreur existante
        for sel in ed.extraSelections():
            if sel.format.background().color() == QColor(ed._theme["error_line"]):
                selections.append(sel)

        if pattern:
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#3d3a20"))
            doc = ed.document()
            flags = self._build_flags()
            cursor = doc.find(QRegularExpression(pattern) if self.chk_regex.isChecked()
                              else pattern, 0, flags)
            while not cursor.isNull():
                sel = QTextEdit.ExtraSelection()
                sel.cursor = cursor
                sel.format = fmt
                selections.append(sel)
                cursor = doc.find(QRegularExpression(pattern) if self.chk_regex.isChecked()
                                  else pattern, cursor, flags)
        ed.setExtraSelections(selections)

    def _update_count(self):
        pattern = self._pattern()
        if not pattern:
            self._lbl_count.setText("")
            return
        try:
            count = len(re.findall(
                pattern,
                self._editor.toPlainText(),
                0 if self.chk_case.isChecked() else re.IGNORECASE,
            ))
            self._lbl_count.setText(f"{count} résultat{'s' if count != 1 else ''}")
        except re.error:
            self._lbl_count.setText("regex ✗")

    def _find(self, backward: bool = False):
        pattern = self._pattern()
        if not pattern:
            return
        ed = self._editor
        flags = self._build_flags()
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
        found = False
        if self.chk_regex.isChecked():
            found = ed.find(QRegularExpression(pattern), flags)
        else:
            found = ed.find(pattern, flags)
        if not found:
            # Wrap around
            cursor = ed.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End if backward
                else QTextCursor.MoveOperation.Start
            )
            ed.setTextCursor(cursor)
            if self.chk_regex.isChecked():
                ed.find(QRegularExpression(pattern), flags)
            else:
                ed.find(pattern, flags)

    def find_next(self):
        self._find(backward=False)

    def find_prev(self):
        self._find(backward=True)

    def replace_current(self):
        ed = self._editor
        cursor = ed.textCursor()
        if cursor.hasSelection():
            pattern = self._pattern()
            selected = cursor.selectedText()
            try:
                if re.fullmatch(pattern, selected,
                                0 if self.chk_case.isChecked() else re.IGNORECASE):
                    cursor.insertText(self.replace_edit.text())
            except re.error:
                pass
        self.find_next()

    def replace_all(self):
        ed = self._editor
        pattern = self._pattern()
        repl = self.replace_edit.text()
        if not pattern:
            return
        text = ed.toPlainText()
        try:
            flags_re = 0 if self.chk_case.isChecked() else re.IGNORECASE
            new_text, n = re.subn(pattern, repl, text, flags=flags_re)
        except re.error:
            return
        if n:
            cursor = ed.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.insertText(new_text)
        self._lbl_count.setText(f"{n} remplacé{'s' if n != 1 else ''}")

    def _close(self):
        self._editor.setExtraSelections([])
        self.hide()
        self.closed.emit()
        self._editor.setFocus()

    def focus_find(self):
        self.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._close()
        else:
            super().keyPressEvent(e)


# ── CodeEditor ───────────────────────────────────────────────────────────────

class CodeEditor(AICompletionMixin, QPlainTextEdit):
    """
    Éditeur GLSL v2.3 avec gutter, find/replace, fold, hover-docs,
    autocomplétion contextuelle, snippets, thèmes,
    color picker inline et complétion IA (Copilot-style).
    """
    code_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._shader_type = "glsl"
        self._header_line_count: int = 0
        self._theme: dict = THEMES[_DEFAULT_THEME]
        self._folded_blocks: set[int] = set()  # numéros de bloc repliés
        self._context_uniforms: list[str] = []  # uniforms extraits du source courant

        # ── Police ───────────────────────────────────────────────────────────
        font = QFont("Cascadia Code", 10)
        if not font.exactMatch():
            font = QFont("Consolas", 10)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # ── Gutter ───────────────────────────────────────────────────────────
        self._gutter = LineNumberGutter(self)
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_gutter_width()

        # ── Highlighter ───────────────────────────────────────────────────────
        self._highlighter = GLSLHighlighter(self.document())

        # ── Autocomplétion ───────────────────────────────────────────────────
        self._completer = QCompleter(GLSL_KEYWORDS, self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.activated.connect(self.insertCompletion)

        # ── Timer code_changed ───────────────────────────────────────────────
        self._change_timer = QTimer(self)
        self._change_timer.setSingleShot(True)
        self._change_timer.setInterval(500)
        self._change_timer.timeout.connect(self._emit_code_changed)
        self.textChanged.connect(self._change_timer.start)
        self.textChanged.connect(self._refresh_context_uniforms)

        # ── Hover tooltip timer ───────────────────────────────────────────────
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(600)
        self._hover_timer.timeout.connect(self._show_hover_doc)
        self._hover_pos: Optional[QPoint] = None
        self.setMouseTracking(True)

        # ── Find/Replace bar ─────────────────────────────────────────────────
        # Sera créée dans le widget parent si besoin ; on stocke une ref
        self._find_bar: Optional[FindReplaceBar] = None

        # ── Raccourcis ───────────────────────────────────────────────────────
        self._setup_shortcuts()

        # ── Complétion IA (Copilot-style) ────────────────────────────────────
        self._init_ai_completion()

        # ── Appliquer le thème ───────────────────────────────────────────────
        self.apply_theme(_DEFAULT_THEME)

    # ── Raccourcis ────────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        from PyQt6.QtGui import QShortcut
        self._sc_find        = QShortcut(QKeySequence("Ctrl+H"),       self, self.toggle_find_replace)
        self._sc_snippet     = QShortcut(QKeySequence("Ctrl+J"),       self, self.show_snippet_menu)
        self._sc_fold        = QShortcut(QKeySequence("Ctrl+Shift+["), self, self.fold_current_block)
        self._sc_unfold      = QShortcut(QKeySequence("Ctrl+Shift+]"), self, self.unfold_current_block)
        self._sc_fold_all    = QShortcut(QKeySequence("Ctrl+Shift+F"), self, self.fold_all)
        self._sc_unfold_all  = QShortcut(QKeySequence("Ctrl+Shift+E"), self, self.unfold_all)

        # Mapping id → QShortcut pour le ShortcutManager
        self._editor_shortcuts: dict[str, object] = {
            "editor_find":       self._sc_find,
            "editor_snippet":    self._sc_snippet,
            "editor_fold":       self._sc_fold,
            "editor_unfold":     self._sc_unfold,
            "editor_fold_all":   self._sc_fold_all,
            "editor_unfold_all": self._sc_unfold_all,
        }

    def register_shortcuts_in_manager(self, mgr):
        """Enregistre tous les QShortcuts de l'éditeur dans le ShortcutManager."""
        for action_id, sc in self._editor_shortcuts.items():
            mgr.register_qshortcut(action_id, sc)

    # ── Thèmes ────────────────────────────────────────────────────────────────

    def apply_theme(self, name: str):
        if name not in THEMES:
            return
        self._theme = THEMES[name]
        t = self._theme
        self.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background-color: {t['bg']};"
            f"  color: {t['fg']};"
            f"  selection-background-color: {t['selection']};"
            f"  border: none;"
            f"}}"
        )
        self._gutter.update()
        self._highlight_current_line()

    @staticmethod
    def theme_names() -> list[str]:
        return list(THEMES.keys())

    # ── Gutter ────────────────────────────────────────────────────────────────

    def _gutter_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 8 + self.fontMetrics().horizontalAdvance("9") * (digits + 1)

    def _update_gutter_width(self):
        self.setViewportMargins(self._gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int):
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self._gutter_width(), cr.height()))

    def _paint_gutter(self, event):
        t = self._theme
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(t["gutter_bg"]))

        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        fm = self.fontMetrics()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                is_current = (block_num == self.textCursor().blockNumber())
                color = QColor(t["fg"]) if is_current else QColor(t["gutter_fg"])
                painter.setPen(color)
                # Fold marker
                text = block.text()
                has_open = "{" in text and "}" not in text
                has_close = "}" in text
                if block_num in self._folded_blocks:
                    painter.setPen(QColor("#89b4fa"))
                    painter.drawText(0, top, self._gutter_width() - 4, fm.height(),
                                     Qt.AlignmentFlag.AlignRight, "▶")
                else:
                    num_str = str(block_num + 1)
                    painter.drawText(0, top, self._gutter_width() - 4, fm.height(),
                                     Qt.AlignmentFlag.AlignRight, num_str)

            # Indicateur IA loading sur la ligne courante
            if block_num == self.textCursor().blockNumber() and self.ai_status_msg:
                painter.setPen(QColor("#89b4fa"))
                painter.drawText(0, top, self._gutter_width() - 4, fm.height(),
                                 Qt.AlignmentFlag.AlignLeft, "⟳")

            block = block.next()
            block_num += 1
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())

    # ── Current line highlight ────────────────────────────────────────────────

    def _highlight_current_line(self):
        extra: list[QTextEdit.ExtraSelection] = []
        # Garde les erreurs existantes
        for sel in self.extraSelections():
            if sel.format.background().color() == QColor(self._theme.get("error_line", "#3c1414")):
                extra.append(sel)

        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor(self._theme["cursor_line"]))
        sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        extra.append(sel)
        self.setExtraSelections(extra)
        self._gutter.update()

    # ── Hover docs ────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        self._hover_pos = e.pos()
        self._hover_timer.start()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._hover_timer.stop()
        QToolTip.hideText()

    def mousePressEvent(self, e):
        """Ctrl+clic → color picker si on est sur un littéral vec3/vec4."""
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.button() == Qt.MouseButton.LeftButton:
            cursor = self.cursorForPosition(e.pos())
            if self._try_open_color_picker(cursor):
                return  # on a absorbé l'événement
        super().mousePressEvent(e)

    # ── Color Picker inline ───────────────────────────────────────────────────

    # Regex pour détecter vec3(r, g, b) ou vec4(r, g, b, a) avec valeurs flottantes
    _VEC_COLOR_RE = re.compile(
        r'\b(vec[34])\s*\(\s*'
        r'([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'  # composante 1
        r'\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'  # composante 2
        r'\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'  # composante 3
        r'(?:\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?))?'  # composante 4 (vec4)
        r'\s*\)'
    )

    def _try_open_color_picker(self, cursor: QTextCursor) -> bool:
        """
        Cherche un littéral vec3/vec4 autour de la position du curseur
        dans la ligne courante. Si trouvé, ouvre un QColorDialog et remplace
        le littéral par les nouvelles valeurs. Retourne True si un picker a été ouvert.
        """
        block = cursor.block()
        line  = block.text()
        col   = cursor.positionInBlock()

        # Recherche tous les matchs vec3/vec4 sur la ligne
        best_match = None
        for m in self._VEC_COLOR_RE.finditer(line):
            if m.start() <= col <= m.end():
                best_match = m
                break

        if best_match is None:
            return False

        vtype = best_match.group(1)  # 'vec3' or 'vec4'
        try:
            r = max(0.0, min(1.0, float(best_match.group(2))))
            g = max(0.0, min(1.0, float(best_match.group(3))))
            b = max(0.0, min(1.0, float(best_match.group(4))))
            a = float(best_match.group(5)) if best_match.group(5) is not None else 1.0
        except (TypeError, ValueError):
            return False

        # Convertit [0,1] → [0,255]
        initial = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))

        dlg = QColorDialog(initial, self)
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, vtype == 'vec4')
        dlg.setWindowTitle(f"Color Picker — {vtype}")

        # Mise à jour en temps réel pendant que l'utilisateur bouge le picker
        block_start = block.position()
        match_start = block_start + best_match.start()
        match_end   = block_start + best_match.end()
        original_text = best_match.group(0)

        def _apply_color(color: QColor):
            nonlocal match_end
            nr = color.redF()
            ng = color.greenF()
            nb = color.blueF()
            na = color.alphaF()
            if vtype == 'vec4':
                new_literal = (f"vec4({nr:.4f}, {ng:.4f}, {nb:.4f}, {na:.4f})")
            else:
                new_literal = (f"vec3({nr:.4f}, {ng:.4f}, {nb:.4f})")
            doc_cursor = QTextCursor(self.document())
            doc_cursor.setPosition(match_start)
            doc_cursor.setPosition(match_end, QTextCursor.MoveMode.KeepAnchor)
            doc_cursor.insertText(new_literal)
            # Recalcule la fin après insertion (longueur peut avoir changé)
            match_end = match_start + len(new_literal)

        dlg.currentColorChanged.connect(_apply_color)

        if dlg.exec() != QColorDialog.DialogCode.Accepted:
            # Annule → restaure le texte original
            doc_cursor = QTextCursor(self.document())
            doc_cursor.setPosition(match_start)
            doc_cursor.setPosition(match_end, QTextCursor.MoveMode.KeepAnchor)
            doc_cursor.insertText(original_text)

        return True

    def _show_hover_doc(self):
        if self._hover_pos is None:
            return
        cursor = self.cursorForPosition(self._hover_pos)
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        word = cursor.selectedText()
        if not word:
            QToolTip.hideText()
            return

        # Essaie d'abord la HelpDatabase (docs complètes)
        try:
            from .help_system import _DB
            entry = _DB.get(word.lower())
            if entry:
                sig   = entry.signature or word
                first_line = next(
                    (l.strip() for l in entry.body_md.splitlines()
                     if l.strip() and not l.startswith('**') and not l.startswith('#')),
                    ""
                )
                doc = f"{sig}\n{first_line}\n\n[Shift+F1 pour la doc complète]"
                global_pos = self.mapToGlobal(self._hover_pos)
                QToolTip.showText(global_pos, doc, self)
                return
        except Exception:
            pass

        # Fallback : GLSL_DOCS dict inline
        doc = GLSL_DOCS.get(word)
        if doc:
            global_pos = self.mapToGlobal(self._hover_pos)
            QToolTip.showText(global_pos, doc, self)
        else:
            QToolTip.hideText()

    # ── Autocomplétion contextuelle ───────────────────────────────────────────

    def _refresh_context_uniforms(self):
        """Extrait les uniforms déclarés dans le shader courant."""
        src = self.toPlainText()
        found = re.findall(r'uniform\s+\w+\s+(\w+)', src)
        self._context_uniforms = found
        all_words = list(dict.fromkeys(found + GLSL_KEYWORDS))
        self._completer.model().setStringList(all_words)  # type: ignore[attr-defined]

    def set_context_uniforms(self, uniforms: list[str]):
        """Injecte des uniforms depuis l'extérieur (ex: depuis main_window)."""
        self._context_uniforms = uniforms
        all_words = list(dict.fromkeys(uniforms + GLSL_KEYWORDS))
        self._completer.model().setStringList(all_words)  # type: ignore[attr-defined]

    def insertCompletion(self, completion: str):
        tc = self.textCursor()
        extra = len(completion) - len(self._completer.completionPrefix())
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)

    def textUnderCursor(self) -> str:
        tc = self.textCursor()
        tc.select(QTextCursor.SelectionType.WordUnderCursor)
        return tc.selectedText()

    def keyPressEvent(self, e):
        # ── Complétion IA : Tab sur commentaire ou acceptation ghost ──────────
        if self._ai_completion_key_press(e):
            return

        # Laisse le completer gérer Enter/Tab
        if self._completer.popup().isVisible():
            if e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return,
                           Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                e.ignore()
                return

        # Auto-indent : Enter → même indentation que la ligne précédente
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cursor = self.textCursor()
            block_text = cursor.block().text()
            indent = len(block_text) - len(block_text.lstrip())
            # Ouvrir accolade → indent + 1
            if block_text.rstrip().endswith("{"):
                indent += 4
            super().keyPressEvent(e)
            self.insertPlainText(" " * indent)
            return

        # Tab → 4 espaces
        if e.key() == Qt.Key.Key_Tab and not e.modifiers():
            self.insertPlainText("    ")
            return

        super().keyPressEvent(e)

        # Autocomplétion
        prefix = self.textUnderCursor()
        if len(prefix) >= 2:
            self._completer.setCompletionPrefix(prefix)
            popup = self._completer.popup()
            popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
            cr = self.cursorRect()
            cr.setWidth(
                popup.sizeHintForColumn(0)
                + popup.verticalScrollBar().sizeHint().width()
            )
            self._completer.complete(cr)
        else:
            self._completer.popup().hide()

    # ── Find / Replace ────────────────────────────────────────────────────────

    def get_find_bar(self) -> FindReplaceBar:
        """Retourne (et crée si nécessaire) la barre Find/Replace."""
        if self._find_bar is None:
            # Cherche un parent QWidget pour y attacher la barre
            parent_widget = self.parentWidget()
            if parent_widget is not None:
                self._find_bar = FindReplaceBar(self)
                self._find_bar.hide()
                # Positionne en bas de l'éditeur
                self._find_bar.move(0, self.height() - self._find_bar.height())
                self._find_bar.resize(self.width(), self._find_bar.height())
            else:
                # Fallback : dialog flottant
                self._find_bar = FindReplaceBar(self)
        return self._find_bar

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self._gutter_width(), cr.height()))
        if self._find_bar is not None:
            self._find_bar.move(0, self.height() - self._find_bar.height())
            self._find_bar.resize(self.width(), self._find_bar.height())
        self._ai_resize_overlay()

    def toggle_find_replace(self):
        bar = self.get_find_bar()
        if bar.isVisible():
            bar._close()
        else:
            bar.focus_find()

    # ── Pliage de blocs ───────────────────────────────────────────────────────

    def _block_at_cursor(self) -> QTextBlock:
        return self.textCursor().block()

    def fold_current_block(self):
        """Replie le bloc { ... } commençant à la ligne courante."""
        block = self._block_at_cursor()
        bn = block.blockNumber()
        if "{" not in block.text():
            return
        depth = 0
        b = block
        while b.isValid():
            text = b.text()
            depth += text.count("{") - text.count("}")
            if b != block:
                b.setVisible(depth > 0)
                if depth <= 0:
                    b.setVisible(True)
                    break
            b = b.next()
        self._folded_blocks.add(bn)
        self.document().markContentsDirty(
            block.position(), self.document().characterCount()
        )
        self._gutter.update()

    def unfold_current_block(self):
        block = self._block_at_cursor()
        bn = block.blockNumber()
        self._folded_blocks.discard(bn)
        b = block.next()
        depth = 0
        while b.isValid():
            text = b.text()
            depth += block.text().count("{") - b.text().count("}")
            b.setVisible(True)
            if "}" in b.text():
                break
            b = b.next()
        self.document().markContentsDirty(
            block.position(), self.document().characterCount()
        )
        self._gutter.update()

    def fold_all(self):
        block = self.document().begin()
        while block.isValid():
            if "{" in block.text() and "}" not in block.text():
                cursor = QTextCursor(block)
                self.setTextCursor(cursor)
                self.fold_current_block()
            block = block.next()

    def unfold_all(self):
        self._folded_blocks.clear()
        block = self.document().begin()
        while block.isValid():
            block.setVisible(True)
            block = block.next()
        self.document().markContentsDirty(0, self.document().characterCount())
        self._gutter.update()

    # ── Snippets ──────────────────────────────────────────────────────────────

    def show_snippet_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background:#12141a; color:#cdd6f4; border:1px solid #2a3060; }"
            "QMenu::item:selected { background:#2a3060; }"
        )
        menu.setTitle("Snippets GLSL")
        for name, code in GLSL_SNIPPETS.items():
            act = menu.addAction(name)
            act.setData(code)
        action = menu.exec(self.mapToGlobal(self.cursorRect().bottomLeft()))
        if action is not None:
            code = action.data()
            cursor = self.textCursor()
            cursor.insertText(code)
            self.setTextCursor(cursor)

    def insert_snippet(self, name: str):
        code = GLSL_SNIPPETS.get(name)
        if code:
            self.textCursor().insertText(code)

    # ── API publique (héritée + nouvelles) ───────────────────────────────────

    def _emit_code_changed(self):
        self.code_changed.emit(self.toPlainText())

    def get_code(self) -> str:
        return self.toPlainText()

    def set_code(self, text: str):
        self.blockSignals(True)
        self.setPlainText(text)
        self.blockSignals(False)
        self._refresh_context_uniforms()

    def set_shader_type(self, stype: str):
        self._shader_type = stype

    def set_header_lines(self, count: int):
        self._header_line_count = count

    def show_error(self, error: str):
        self.clear_error()
        match = re.search(r'0[:\(](\d+)[\):]', error)
        if not match:
            return
        line_num_in_source = int(match.group(1))
        user_line = line_num_in_source - self._header_line_count
        if user_line >= 1:
            self._highlight_error_line(user_line)

    def _highlight_error_line(self, line_num: int):
        doc = self.document()
        if line_num > doc.blockCount():
            return
        cursor = QTextCursor(doc.findBlockByNumber(line_num - 1))
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor(self._theme.get("error_line", "#3c1414")))
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = cursor
        # Garde le surlignage de ligne courante
        existing = [s for s in self.extraSelections()
                    if s.format.background().color() == QColor(self._theme["cursor_line"])]
        self.setExtraSelections(existing + [selection])

    def clear_error(self):
        # Garde uniquement le surlignage de ligne courante
        existing = [s for s in self.extraSelections()
                    if s.format.background().color() == QColor(self._theme["cursor_line"])]
        self.setExtraSelections(existing)


# ── SplitEditorView ───────────────────────────────────────────────────────────

class SplitEditorView(QSplitter):
    """
    Vue Split : deux CodeEditor côte à côte partageant le même QTextDocument.
    Utilisé pour comparer/éditer deux passes simultanément depuis main_window.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.editor_left  = CodeEditor()
        self.editor_right = CodeEditor()
        # Les deux éditeurs partagent le même document
        self.editor_right.setDocument(self.editor_left.document())
        self.addWidget(self.editor_left)
        self.addWidget(self.editor_right)
        self.setSizes([500, 500])

    def set_code(self, text: str):
        self.editor_left.set_code(text)

    def get_code(self) -> str:
        return self.editor_left.get_code()
