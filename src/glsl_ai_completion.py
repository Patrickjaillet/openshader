"""
glsl_ai_completion.py
----------------------
Autocomplétion IA façon Copilot pour GLSL — v1.0

Fonctionnement :
  - L'utilisateur tape un commentaire (// ...) et appuie sur Tab
  - Le moteur détecte le commentaire, envoie le contexte au LLM
  - Le LLM génère une suite de code GLSL cohérente
  - Le résultat est affiché en "ghost text" grisé dans l'éditeur
  - Tab → accepte | Échap → refuse

Backends (même config que AIShaderGenerator) :
  OpenAI API / Ollama / llama.cpp / Stub offline

Architecture :
  - GLSLCompletionRequest   : contexte envoyé au LLM
  - GLSLCompletionResult    : texte généré
  - GLSLCompletionEngine    : QObject threading + signaux Qt
  - GhostTextOverlay        : QWidget overlay semi-transparent
  - AICompletionMixin       : à mixer dans CodeEditor
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

from PyQt6.QtCore import (
    QObject, pyqtSignal, QTimer, Qt, QRect, QPoint, QSize,
)
from PyQt6.QtGui import (
    QColor, QPainter, QFont, QTextCursor, QFontMetrics,
    QTextCharFormat, QPalette,
)
from PyQt6.QtWidgets import QWidget, QApplication

from .logger import get_logger

log = get_logger(__name__)

# ── Imports HTTP conditionnels ────────────────────────────────────────────────

try:
    import urllib.request
    import urllib.error
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False

try:
    from openai import OpenAI as _OpenAI  # type: ignore
    _OPENAI_SDK = True
except ImportError:
    _OPENAI_SDK = False

# ═════════════════════════════════════════════════════════════════════════════
#  Prompt système spécialisé complétion inline
# ═════════════════════════════════════════════════════════════════════════════

_COMPLETION_SYSTEM = """\
Tu es un assistant de complétion de code GLSL. Tu complètes du code GLSL \
dans un éditeur de shaders en temps réel (style Shadertoy / OpenShader).

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec du code GLSL brut. Aucune explication, \
aucune balise markdown, aucun texte.
2. Complète naturellement ce qui suit le commentaire ou la ligne partielle fournie.
3. Génère entre 2 et 20 lignes de code pertinent et complet.
4. Respecte le style et l'indentation du contexte fourni.
5. Le code doit être immédiatement valide et compilable.
6. N'ajoute PAS de fonction mainImage — tu complètes du code interne.
7. Si le commentaire décrit une technique (noise, SDF, FBM...), implémente-la.
"""

_COMPLETION_USER = """\
Contexte du shader (lignes précédant le curseur) :
```glsl
{context}
```

Complète le code à partir du commentaire suivant ou de la position du curseur :
// {trigger_comment}
"""

_COMPLETION_USER_PARTIAL = """\
Contexte du shader (lignes précédant le curseur) :
```glsl
{context}
```

La ligne courante (partielle) est :
{partial_line}

Continue cette ligne et les suivantes de façon cohérente.
"""


# ═════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GLSLCompletionRequest:
    context_before: str        # lignes avant le curseur
    trigger_comment: str       # commentaire déclencheur (ou ligne partielle)
    is_comment_trigger: bool   # True = Tab après //..., False = Tab mid-line
    max_tokens: int = 256
    temperature: float = 0.25  # faible pour la complétion (précis)


@dataclass
class GLSLCompletionResult:
    text: str                  # code GLSL à insérer
    backend: str
    duration_s: float
    ok: bool = True
    error: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#  Backends (légers — réutilisent la logique de ai_shader_generator)
# ═════════════════════════════════════════════════════════════════════════════

def _clean_completion(raw: str) -> str:
    """Retire les balises markdown et normalise l'indentation."""
    raw = re.sub(r'```(?:glsl|GLSL)?\n?', '', raw)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    # Retire un éventuel commentaire repété en début de réponse
    lines = raw.strip().splitlines()
    return "\n".join(lines)


def _build_completion_message(req: GLSLCompletionRequest) -> str:
    if req.is_comment_trigger:
        return _COMPLETION_USER.format(
            context=req.context_before[-2000:],  # contexte tronqué
            trigger_comment=req.trigger_comment,
        )
    return _COMPLETION_USER_PARTIAL.format(
        context=req.context_before[-2000:],
        partial_line=req.trigger_comment,
    )


class _CompletionOpenAIBackend:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._key = api_key
        self._model = model

    def is_available(self) -> bool:
        return bool(self._key)

    def complete(self, req: GLSLCompletionRequest) -> GLSLCompletionResult:
        t0 = time.time()
        model = self._model
        msg = _build_completion_message(req)

        if _OPENAI_SDK:
            try:
                client = _OpenAI(api_key=self._key)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _COMPLETION_SYSTEM},
                        {"role": "user",   "content": msg},
                    ],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                raw = resp.choices[0].message.content or ""
                return GLSLCompletionResult(
                    text=_clean_completion(raw), backend=self.name,
                    duration_s=time.time() - t0,
                )
            except Exception as e:
                return GLSLCompletionResult(
                    text="", backend=self.name, duration_s=time.time() - t0,
                    ok=False, error=str(e),
                )

        # Fallback HTTP
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": _COMPLETION_SYSTEM},
                    {"role": "user",   "content": msg},
                ],
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            }).encode()
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type":  "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=15) as r:
                data = json.loads(r.read())
                raw = data["choices"][0]["message"]["content"]
            return GLSLCompletionResult(
                text=_clean_completion(raw), backend=self.name,
                duration_s=time.time() - t0,
            )
        except Exception as e:
            return GLSLCompletionResult(
                text="", backend=self.name, duration_s=time.time() - t0,
                ok=False, error=str(e),
            )


class _CompletionOllamaBackend:
    name = "ollama"
    DEFAULT_MODEL = "codestral"

    def __init__(self, host: str = "http://localhost:11434", model: str = ""):
        self._host = host.rstrip("/")
        self._model = model or self.DEFAULT_MODEL

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def complete(self, req: GLSLCompletionRequest) -> GLSLCompletionResult:
        t0 = time.time()
        full_prompt = f"{_COMPLETION_SYSTEM}\n\n{_build_completion_message(req)}"
        payload = json.dumps({
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }).encode()
        try:
            request = urllib.request.Request(
                f"{self._host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as r:
                data = json.loads(r.read())
                raw = data.get("response", "")
            return GLSLCompletionResult(
                text=_clean_completion(raw), backend=self.name,
                duration_s=time.time() - t0,
            )
        except Exception as e:
            return GLSLCompletionResult(
                text="", backend=self.name, duration_s=time.time() - t0,
                ok=False, error=str(e),
            )


class _CompletionLlamaCppBackend:
    name = "llamacpp"

    def __init__(self, host: str = "http://localhost:8080"):
        self._host = host.rstrip("/")

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._host}/health")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def complete(self, req: GLSLCompletionRequest) -> GLSLCompletionResult:
        t0 = time.time()
        full_prompt = (
            f"<s>[INST] {_COMPLETION_SYSTEM}\n\n"
            f"{_build_completion_message(req)} [/INST]"
        )
        payload = json.dumps({
            "prompt":      full_prompt,
            "n_predict":   req.max_tokens,
            "temperature": req.temperature,
            "stream":      False,
            "stop":        ["</s>", "[INST]", "```"],
        }).encode()
        try:
            request = urllib.request.Request(
                f"{self._host}/completion",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as r:
                data = json.loads(r.read())
                raw = data.get("content", "")
            return GLSLCompletionResult(
                text=_clean_completion(raw), backend=self.name,
                duration_s=time.time() - t0,
            )
        except Exception as e:
            return GLSLCompletionResult(
                text="", backend=self.name, duration_s=time.time() - t0,
                ok=False, error=str(e),
            )


# Stub avec quelques exemples réalistes selon le commentaire
_STUB_EXAMPLES = {
    "noise":
        "float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }\n"
        "float noise(vec2 p) {\n"
        "    vec2 i = floor(p), f = fract(p);\n"
        "    vec2 u = f * f * (3.0 - 2.0 * f);\n"
        "    return mix(mix(hash(i), hash(i+vec2(1,0)), u.x),\n"
        "               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);\n"
        "}\n",
    "fbm":
        "float fbm(vec2 p) {\n"
        "    float v = 0.0, a = 0.5;\n"
        "    for (int i = 0; i < 6; i++) {\n"
        "        v += a * noise(p); p = p * 2.0 + vec2(1.7, 9.2); a *= 0.5;\n"
        "    }\n"
        "    return v;\n"
        "}\n",
    "sdf":
        "float sdCircle(vec2 p, float r) { return length(p) - r; }\n"
        "float sdBox(vec2 p, vec2 b) {\n"
        "    vec2 d = abs(p) - b;\n"
        "    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);\n"
        "}\n",
    "palette":
        "vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {\n"
        "    return a + b * cos(6.28318 * (c * t + d));\n"
        "}\n",
    "raymarch":
        "float map(vec3 p) { return length(p) - 1.0; }\n"
        "float raymarch(vec3 ro, vec3 rd) {\n"
        "    float t = 0.0;\n"
        "    for (int i = 0; i < 100; i++) {\n"
        "        float d = map(ro + rd * t);\n"
        "        if (d < 0.001) return t;\n"
        "        t += d; if (t > 100.0) break;\n"
        "    }\n"
        "    return -1.0;\n"
        "}\n",
}


class _CompletionStubBackend:
    name = "stub"

    def is_available(self) -> bool:
        return True

    def complete(self, req: GLSLCompletionRequest) -> GLSLCompletionResult:
        comment = req.trigger_comment.lower()
        code = ""
        for kw, snippet in _STUB_EXAMPLES.items():
            if kw in comment:
                code = snippet
                break
        if not code:
            code = (
                "// [stub] Connectez OpenAI, Ollama ou llama.cpp pour la complétion IA.\n"
                "float val = sin(iTime) * 0.5 + 0.5;\n"
            )
        return GLSLCompletionResult(text=code, backend=self.name, duration_s=0.01)


# ═════════════════════════════════════════════════════════════════════════════
#  GLSLCompletionEngine — QObject principal
# ═════════════════════════════════════════════════════════════════════════════

class GLSLCompletionEngine(QObject):
    """
    Moteur de complétion IA GLSL configurable.

    Signaux :
      completion_ready(str)   — code à insérer en ghost text
      completion_error(str)   — message d'erreur
      completion_started()    — début de requête (pour le spinner UI)
    """

    completion_ready   = pyqtSignal(str)
    completion_error   = pyqtSignal(str)
    completion_started = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._openai_key    = ""
        self._ollama_host   = "http://localhost:11434"
        self._llamacpp_host = "http://localhost:8080"
        self._preferred_model = ""
        self._backends: list[Any] = []
        self._stub = _CompletionStubBackend()
        self._running = False
        self._enabled = True

    # ── Configuration ─────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v

    def configure(self, openai_key: str = "", ollama_host: str = "",
                  llamacpp_host: str = "", model: str = ""):
        self._openai_key     = openai_key
        self._ollama_host    = ollama_host or self._ollama_host
        self._llamacpp_host  = llamacpp_host or self._llamacpp_host
        self._preferred_model = model
        self._backends = []  # force re-détection

    def sync_from_ai_generator(self, gen):
        """Synchronise la config depuis AIShaderGenerator existant."""
        self._openai_key    = getattr(gen, "_openai_key",    "")
        self._ollama_host   = getattr(gen, "_ollama_host",   self._ollama_host)
        self._llamacpp_host = getattr(gen, "_llamacpp_host", self._llamacpp_host)
        self._backends = []

    # ── Détection ────────────────────────────────────────────────────────────

    def _detect_backends(self) -> list[Any]:
        candidates = []
        if self._openai_key:
            b = _CompletionOpenAIBackend(self._openai_key, self._preferred_model or "gpt-4o")
            if b.is_available():
                candidates.append(b)
        try:
            b = _CompletionOllamaBackend(self._ollama_host, self._preferred_model)
            if b.is_available():
                candidates.append(b)
        except Exception:
            pass
        try:
            b = _CompletionLlamaCppBackend(self._llamacpp_host)
            if b.is_available():
                candidates.append(b)
        except Exception:
            pass
        return candidates

    # ── API publique ──────────────────────────────────────────────────────────

    def request_completion(self, context_before: str, trigger: str,
                           is_comment: bool):
        """Lance une complétion en arrière-plan."""
        if not self._enabled or self._running:
            return
        self._running = True
        self.completion_started.emit()
        req = GLSLCompletionRequest(
            context_before=context_before,
            trigger_comment=trigger,
            is_comment_trigger=is_comment,
        )
        t = threading.Thread(target=self._run, args=(req,), daemon=True,
                             name="GLSLCompletion")
        t.start()

    def _run(self, req: GLSLCompletionRequest):
        try:
            if not self._backends:
                self._backends = self._detect_backends()
            backend = self._backends[0] if self._backends else self._stub
            log.debug("Complétion IA — backend=%s trigger=%r",
                      backend.name, req.trigger_comment[:40])
            result = backend.complete(req)
            if result.ok and result.text.strip():
                self.completion_ready.emit(result.text)
                log.debug("Complétion OK — %.2fs — %d chars",
                          result.duration_s, len(result.text))
            else:
                err = result.error or "Réponse vide"
                log.warning("Complétion IA vide : %s", err)
                self.completion_error.emit(err)
        except Exception as e:
            log.error("Complétion IA exception : %s", e)
            self.completion_error.emit(str(e))
        finally:
            self._running = False


# ═════════════════════════════════════════════════════════════════════════════
#  GhostTextOverlay — widget semi-transparent de ghost text
# ═════════════════════════════════════════════════════════════════════════════

class GhostTextOverlay(QWidget):
    """
    Overlay positionné sur le CodeEditor affichant le ghost text en grisé.
    Suit automatiquement le viewport de l'éditeur parent.
    """

    def __init__(self, editor: QWidget):
        super().__init__(editor)
        self._editor = editor
        self._ghost_text: str = ""
        self._cursor_rect: QRect = QRect()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hide()

    def show_ghost(self, text: str, cursor_rect: QRect):
        self._ghost_text = text
        self._cursor_rect = cursor_rect
        # Taille = toute la surface de l'éditeur
        self.setGeometry(self._editor.rect())
        self.raise_()
        self.show()
        self.update()

    def hide_ghost(self):
        self._ghost_text = ""
        self.hide()

    @property
    def has_ghost(self) -> bool:
        return bool(self._ghost_text)

    @property
    def ghost_text(self) -> str:
        return self._ghost_text

    def paintEvent(self, event):
        if not self._ghost_text or self._cursor_rect.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        font = self._editor.font()
        painter.setFont(font)
        fm = QFontMetrics(font)
        line_h = fm.height()

        # Couleur ghost : gris adapté au thème
        ghost_color = QColor(130, 140, 160, 180)
        painter.setPen(ghost_color)

        lines = self._ghost_text.splitlines(keepends=True)
        x = self._cursor_rect.x()
        y = self._cursor_rect.y()

        for i, line in enumerate(lines):
            line_clean = line.rstrip("\n")
            if i == 0:
                # Première ligne : même y que le curseur
                painter.drawText(x, y + fm.ascent(), line_clean)
            else:
                # Lignes suivantes : indentées depuis le bord gauche de l'éditeur
                left_margin = self._editor.viewportMargins().left() if hasattr(
                    self._editor, 'viewportMargins') else 0
                painter.drawText(left_margin + 4, y + i * line_h + fm.ascent(), line_clean)

        painter.end()


# ═════════════════════════════════════════════════════════════════════════════
#  AICompletionMixin — à mixer dans CodeEditor
# ═════════════════════════════════════════════════════════════════════════════

# Regex : ligne qui ne contient que du blanc + un commentaire (// ...)
_COMMENT_LINE_RE = re.compile(r'^\s*//\s*(.+)$')


class AICompletionMixin:
    """
    Mixin à hériter en plus de CodeEditor pour activer la complétion IA.

    Usage dans CodeEditor.__init__() :
        self._init_ai_completion()

    Dans CodeEditor.keyPressEvent() — avant super() :
        if self._ai_completion_key_press(e):
            return
    """

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_ai_completion(self):
        self._ai_engine = GLSLCompletionEngine(self)
        self._ai_engine.completion_ready.connect(self._on_ai_completion_ready)
        self._ai_engine.completion_error.connect(self._on_ai_completion_error)
        self._ai_engine.completion_started.connect(self._on_ai_completion_started)

        self._ghost_overlay = GhostTextOverlay(self.viewport())  # type: ignore[attr-defined]
        self._pending_ghost: str = ""

        # Spinner / indicateur dans le gutter (optionnel)
        self._ai_loading = False
        self._ai_status_msg = ""

        # Timer debounce : évite de lancer la complétion trop vite
        self._ai_debounce = QTimer(self)
        self._ai_debounce.setSingleShot(True)
        self._ai_debounce.setInterval(80)
        self._ai_debounce.timeout.connect(self._trigger_ai_completion)
        self._ai_pending_trigger: Optional[tuple[str, str, bool]] = None

    # ── Connexion à la config AIShaderGenerator ───────────────────────────────

    def sync_ai_completion_config(self, ai_generator):
        """Synchronise les backends depuis le AIShaderGenerator principal."""
        self._ai_engine.sync_from_ai_generator(ai_generator)

    @property
    def ai_completion_enabled(self) -> bool:
        return self._ai_engine.enabled

    @ai_completion_enabled.setter
    def ai_completion_enabled(self, v: bool):
        self._ai_engine.enabled = v
        if not v:
            self._dismiss_ghost()

    # ── Interception keyPressEvent ────────────────────────────────────────────

    def _ai_completion_key_press(self, e) -> bool:
        """
        Retourne True si l'événement a été consommé par la complétion IA.
        À appeler en tête de keyPressEvent.
        """
        # Tab accepte le ghost text
        if e.key() == Qt.Key.Key_Tab and not e.modifiers():
            if self._ghost_overlay.has_ghost:
                self._accept_ghost()
                return True
            # Pas de ghost visible → tester si on est sur un commentaire
            if self._ai_engine.enabled and self._try_schedule_completion():
                return True  # on a absorbé Tab et planifié une complétion

        # Échap / mouvement → dismiss le ghost
        if self._ghost_overlay.has_ghost:
            if e.key() in (Qt.Key.Key_Escape,
                           Qt.Key.Key_Left, Qt.Key.Key_Right,
                           Qt.Key.Key_Up, Qt.Key.Key_Down,
                           Qt.Key.Key_Return, Qt.Key.Key_Enter,
                           Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                self._dismiss_ghost()
                # On laisse l'événement se propager (pas de return True)

        return False

    # ── Détection du déclencheur ──────────────────────────────────────────────

    def _try_schedule_completion(self) -> bool:
        """
        Vérifie si la ligne courante contient un commentaire déclencheur.
        Si oui, planifie la complétion et retourne True.
        """
        cursor = self.textCursor()  # type: ignore[attr-defined]
        line = cursor.block().text()
        m = _COMMENT_LINE_RE.match(line)
        if m:
            comment = m.group(1).strip()
            context = self._get_context_before(cursor)
            self._ai_pending_trigger = (context, comment, True)
            self._ai_debounce.start()
            return True
        return False

    def _get_context_before(self, cursor: QTextCursor) -> str:
        """Retourne les N lignes avant le curseur comme contexte."""
        doc = self.document()  # type: ignore[attr-defined]
        block_num = cursor.blockNumber()
        lines = []
        block = doc.begin()
        for _ in range(min(block_num, 60)):  # max 60 lignes de contexte
            lines.append(block.text())
            block = block.next()
        return "\n".join(lines)

    def _trigger_ai_completion(self):
        if self._ai_pending_trigger is None:
            return
        context, trigger, is_comment = self._ai_pending_trigger
        self._ai_pending_trigger = None
        self._ai_engine.request_completion(context, trigger, is_comment)

    # ── Callbacks engine ──────────────────────────────────────────────────────

    def _on_ai_completion_started(self):
        self._ai_loading = True
        self._ai_status_msg = "⟳ IA…"
        self._update_ai_status()

    def _on_ai_completion_ready(self, text: str):
        self._ai_loading = False
        self._ai_status_msg = ""
        self._update_ai_status()
        if not text.strip():
            return
        # Affiche le ghost text à la position du curseur
        cursor_rect = self.cursorRect()  # type: ignore[attr-defined]
        # Décale d'une ligne vers le bas
        line_h = self.fontMetrics().height()  # type: ignore[attr-defined]
        offset_rect = QRect(
            cursor_rect.x(),
            cursor_rect.bottom(),
            cursor_rect.width(),
            line_h,
        )
        self._ghost_overlay.setGeometry(self.viewport().rect())  # type: ignore[attr-defined]
        self._ghost_overlay.show_ghost(text, offset_rect)

    def _on_ai_completion_error(self, error: str):
        self._ai_loading = False
        self._ai_status_msg = ""
        self._update_ai_status()
        log.warning("Complétion IA erreur : %s", error)

    def _update_ai_status(self):
        """Demande un repaint du gutter pour afficher le spinner."""
        if hasattr(self, '_gutter'):
            self._gutter.update()  # type: ignore[attr-defined]

    # ── Acceptation / rejet du ghost ─────────────────────────────────────────

    def _accept_ghost(self):
        """Insère le ghost text dans le document."""
        text = self._ghost_overlay.ghost_text
        self._dismiss_ghost()
        if not text:
            return
        cursor = self.textCursor()  # type: ignore[attr-defined]
        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine)
        cursor.insertText("\n" + text)
        self.setTextCursor(cursor)  # type: ignore[attr-defined]
        log.debug("Ghost text accepté — %d chars", len(text))

    def _dismiss_ghost(self):
        self._ghost_overlay.hide_ghost()

    # ── Resize ────────────────────────────────────────────────────────────────

    def _ai_resize_overlay(self):
        """À appeler dans resizeEvent."""
        if hasattr(self, '_ghost_overlay'):
            self._ghost_overlay.setGeometry(self.viewport().rect())  # type: ignore[attr-defined]

    # ── Indicateur dans le gutter ─────────────────────────────────────────────

    @property
    def ai_status_msg(self) -> str:
        return getattr(self, '_ai_status_msg', "")
