"""
glsl_highlighter.py
-------------------
Coloration syntaxique pour le langage GLSL.

Améliorations v1.1 :
  - Support des commentaires multi-lignes /* ... */
  - Types entiers (uint, ivec*, uvec*) ajoutés
  - Précision qualifiers (lowp, mediump, highp)
"""

from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt6.QtCore import QRegularExpression


class GLSLHighlighter(QSyntaxHighlighter):
    """Coloration syntaxique GLSL avec support commentaires multi-lignes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        # ── Mots-clés ──────────────────────────────────────────────────────
        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor(200, 120, 220))
        keyword_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "#version", "#define", "#ifdef", "#ifndef", "#endif", "#else", "#if",
            "uniform", "in", "out", "inout", "attribute", "varying", "const",
            "if", "else", "for", "while", "do", "break", "continue", "return",
            "struct", "void", "true", "false", "discard",
            "lowp", "mediump", "highp", "precision",
        ]
        for word in keywords:
            pat = QRegularExpression(
                f"(?<![\\w#]){word}(?![\\w])" if not word.startswith("#")
                else f"{word}\\b"
            )
            self.highlighting_rules.append((pat, keyword_fmt))

        # ── Types ──────────────────────────────────────────────────────────
        type_fmt = QTextCharFormat()
        type_fmt.setForeground(QColor(70, 180, 220))
        types = [
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
            "sampler2DShadow", "samplerCubeShadow",
        ]
        for word in types:
            self.highlighting_rules.append(
                (QRegularExpression(f"\\b{word}\\b"), type_fmt)
            )

        # ── Fonctions natives ───────────────────────────────────────────────
        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor(120, 200, 150))
        functions = [
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
            "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract",
            "mod", "modf", "min", "max", "clamp", "mix", "step", "smoothstep",
            "isnan", "isinf", "floatBitsToInt", "floatBitsToUint",
            "intBitsToFloat", "uintBitsToFloat",
            "length", "distance", "dot", "cross", "normalize",
            "faceforward", "reflect", "refract",
            "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse",
            "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
            "equal", "notEqual", "any", "all", "not",
            "texture", "texture2D", "textureCube", "textureSize", "texelFetch",
            "dFdx", "dFdy", "fwidth",
            "mainImage",
        ]
        for word in functions:
            self.highlighting_rules.append(
                (QRegularExpression(f"\\b{word}\\b"), func_fmt)
            )

        # ── Nombres (int, float, hex) ───────────────────────────────────────
        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor(210, 165, 100))
        self.highlighting_rules.append(
            (QRegularExpression(r"\b(0x[0-9A-Fa-f]+|\d+\.?\d*([eE][+-]?\d+)?[fFuU]?)\b"),
             number_fmt)
        )

        # ── Commentaires mono-ligne // ──────────────────────────────────────
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor(100, 110, 130))
        comment_fmt.setFontItalic(True)
        self.highlighting_rules.append(
            (QRegularExpression("//[^\n]*"), comment_fmt)
        )

        # Format commentaire multi-lignes (utilisé dans highlightBlock)
        self._ml_comment_fmt = QTextCharFormat()
        self._ml_comment_fmt.setForeground(QColor(100, 110, 130))
        self._ml_comment_fmt.setFontItalic(True)

        self._ml_start = QRegularExpression(r"/\*")
        self._ml_end   = QRegularExpression(r"\*/")

    def highlightBlock(self, text: str):
        # ── Règles mono-ligne ───────────────────────────────────────────────
        for pattern, fmt in self.highlighting_rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)

        # ── Commentaires multi-lignes /* ... */ ─────────────────────────────
        # On utilise les états de bloc Qt pour tracer si on est "dans" un commentaire.
        self.setCurrentBlockState(0)

        start_index = 0
        if self.previousBlockState() != 1:
            # On cherche le début d'un commentaire dans ce bloc
            m = self._ml_start.match(text, start_index)
            start_index = m.capturedStart() if m.hasMatch() else -1

        while start_index >= 0:
            m_end = self._ml_end.match(text, start_index)
            if m_end.hasMatch():
                end_index   = m_end.capturedEnd()
                comment_len = end_index - start_index
                self.setFormat(start_index, comment_len, self._ml_comment_fmt)
                # Cherche un autre /* après la fermeture
                m_next = self._ml_start.match(text, end_index)
                start_index = m_next.capturedStart() if m_next.hasMatch() else -1
            else:
                # Commentaire non fermé : colorie jusqu'à la fin et marque l'état
                self.setCurrentBlockState(1)
                self.setFormat(start_index, len(text) - start_index, self._ml_comment_fmt)
                break
