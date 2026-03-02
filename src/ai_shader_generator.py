"""
ai_shader_generator.py
-----------------------
v1.0 — Génération de shaders GLSL par prompt LLM pour OpenShader / DemoMaker.

Backends supportés (par ordre de priorité) :
  1. OpenAI API (GPT-4o, GPT-4-turbo) — cloud, clé API requise
  2. Ollama (localhost:11434) — local, modèles : codestral, deepseek-coder, etc.
  3. llama.cpp server (localhost:8080) — local, modèles GGUF (CodeLlama, Mistral…)
  4. Stub offline — retourne un shader de démonstration sans LLM

Architecture :
  - ShaderGenerationRequest  : encapsule le prompt + contexte shader courant
  - ShaderGenerationResult   : résultat + métadonnées (backend, durée, tokens)
  - GenerationHistoryEntry   : un appel complet avec before/after pour diff
  - AIShaderGenerator        : QObject principal, threading, streaming Qt signals
  - ContextualSuggestions    : générateur de suggestions basées sur le shader courant

Signaux Qt :
  token_received(str)          — streaming token par token pour l'UI live
  generation_done(result)      — résultat complet
  generation_error(str)        — message d'erreur
  suggestion_ready(list[str])  — liste de suggestions contextuelles

Usage :
    gen = AIShaderGenerator()
    gen.generation_done.connect(lambda r: editor.set_code(r.glsl))
    gen.generate("Ajouter un effet de vague sinusoïdale audio-réactive",
                 current_shader=editor.get_code())
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)

# ── Imports conditionnels ─────────────────────────────────────────────────────

try:
    import urllib.request
    import urllib.error
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAI   # type: ignore
    _OPENAI_SDK = True
except ImportError:
    _OPENAI_SDK = False

# ═════════════════════════════════════════════════════════════════════════════
#  Prompt système GLSL — condensé des meilleures pratiques Shadertoy
# ═════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
Tu es un expert en shaders GLSL spécialisé dans l'art génératif, la demoscene et les effets visuels en temps réel.
Tu génères exclusivement du code GLSL valide, compatible WebGL2 / OpenGL ES 3.0 et le format Shadertoy.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec du code GLSL brut, sans aucune explication, sans balises markdown, sans ```glsl.
2. Le shader doit contenir exactement : void mainImage(out vec4 fragColor, in vec2 fragCoord)
3. Uniforms disponibles (déjà déclarés par l'hôte, ne pas les redéclarer) :
   uniform vec3  iResolution;   // résolution canvas (px)
   uniform float iTime;         // temps en secondes
   uniform float iTimeDelta;    // durée du frame précédent
   uniform int   iFrame;        // numéro de frame
   uniform vec4  iMouse;        // état souris (x,y,click_x,click_y)
   uniform float iSampleRate;   // 44100.0
   // Uniforms custom OpenShader (audio-réactif) :
   uniform float uRMS;          // RMS audio normalisé [0,1]
   uniform float uBeat;         // position dans la mesure [0,1]
   uniform float uBPM;          // BPM du projet
4. N'utilise jamais de textures externes (iChannel0…) sauf si explicitement demandé.
5. Le code doit être efficace (< 100 lignes sauf si la complexité le justifie).
6. Favorise les techniques procédurales : SDF, raymarching, noise, FBM, palette cosinus.
7. Si le shader existant est fourni, modifie-le en conservant sa structure générale.
8. Assure-toi que le code compile sans erreur (pas de divisions par zéro, arrays bien déclarés).

FORMAT DE RÉPONSE : code GLSL uniquement, première ligne = commentaire décrivant l'effet.
"""

_MODIFY_PREFIX = """\
Shader existant à modifier (conserve la structure, applique uniquement la modification demandée) :
```glsl
{current_shader}
```

Modification demandée : {prompt}
"""

_CREATE_PREFIX = "Crée un shader GLSL avec l'effet suivant : {prompt}"

# ═════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ShaderGenerationRequest:
    prompt:         str
    current_shader: str = ""         # shader courant (vide = génération from scratch)
    model:          str = ""         # vide = auto-sélection
    temperature:    float = 0.7
    max_tokens:     int   = 2048


@dataclass
class ShaderGenerationResult:
    glsl:           str
    prompt:         str
    backend:        str              # 'openai' | 'ollama' | 'llamacpp' | 'stub'
    model:          str
    duration_s:     float
    tokens_used:    int = 0
    ok:             bool = True
    error:          str = ""


@dataclass
class GenerationHistoryEntry:
    """Un appel complet — conservé pour le diff avant/après."""
    timestamp:      float = field(default_factory=time.time)
    prompt:         str   = ""
    glsl_before:    str   = ""       # shader avant modification
    glsl_after:     str   = ""       # shader généré
    backend:        str   = ""
    model:          str   = ""
    duration_s:     float = 0.0
    ok:             bool  = True

    @property
    def time_label(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')

    @property
    def prompt_short(self) -> str:
        return self.prompt[:60] + ('…' if len(self.prompt) > 60 else '')


# ═════════════════════════════════════════════════════════════════════════════
#  Suggestions contextuelles
# ═════════════════════════════════════════════════════════════════════════════

# Patterns → suggestions associées
_SUGGESTION_RULES: list[tuple[str, list[str]]] = [
    # Contenu du shader → suggestions proposées
    (r'sin\(|cos\(|wave',          [
        "Ajouter une distorsion de domaine FBM",
        "Rendre les vagues audio-réactives avec uRMS",
        "Ajouter un deuxième plan d'ondes déphasé",
    ]),
    (r'raymar|map\(|SDF|sdCircle|sdBox', [
        "Ajouter de l'ambient occlusion (AO) approximé",
        "Ajouter des réflexions sur la surface",
        "Animer la scène au rythme de uBeat",
        "Ajouter du brouillard exponentiel",
    ]),
    (r'noise|fbm|hash',            [
        "Ajouter une turbulence de couleur HSV",
        "Animer le noise avec iTime * uBPM",
        "Combiner plusieurs octaves de FBM (6 → 8)",
    ]),
    (r'palette|hueShift|hue',      [
        "Synchroniser la rotation de teinte sur uBeat",
        "Ajouter une vignette sur les bords",
        "Créer un dégradé radial avec la palette",
    ]),
    (r'gl_FragCoord|fragCoord|uv', [
        "Passer en coordonnées polaires",
        "Ajouter un effet miroir (abs(uv))",
        "Appliquer une distorsion fisheye",
        "Ajouter un fond étoilé procédural",
    ]),
    (r'',                          [
        "Générer un tunnel raymarché psychédélique",
        "Créer un plasma couleur audio-réactif",
        "Faire un effet de particules CPU (200 pts)",
        "Générer un fond de Voronoï animé",
        "Créer un shader de terrain fractal 2D",
        "Faire un effet de glitch VHS audio-réactif",
    ]),
]

_CONTEXTUAL_MODIFIERS = [
    "Rendre cet effet audio-réactif avec uRMS et uBeat",
    "Optimiser pour 60 fps (réduire les itérations)",
    "Ajouter une vignette sombre sur les bords",
    "Ajouter du bruit de grain filmique animé",
    "Passer en palette monochrome avec accent coloré",
    "Ajouter un effet de blur radial au centre",
    "Inverser les couleurs de façon périodique",
    "Ajouter des reflets spéculaires",
]


class ContextualSuggestions:
    """Génère des suggestions de prompt basées sur le contenu du shader courant."""

    @staticmethod
    def get(current_shader: str, n: int = 6) -> list[str]:
        suggestions: list[str] = []
        for pattern, items in _SUGGESTION_RULES:
            if pattern and re.search(pattern, current_shader, re.IGNORECASE):
                suggestions.extend(items)
                if len(suggestions) >= n:
                    break
        # Complète avec des modificateurs génériques si nécessaire
        if len(suggestions) < n:
            suggestions.extend(_CONTEXTUAL_MODIFIERS)
        # Déduplique et tronque
        seen: set[str] = set()
        result: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                result.append(s)
            if len(result) >= n:
                break
        return result


# ═════════════════════════════════════════════════════════════════════════════
#  Backends LLM
# ═════════════════════════════════════════════════════════════════════════════

def _extract_glsl(raw: str) -> str:
    """Extrait le code GLSL brut d'une réponse LLM (retire les balises markdown)."""
    # Retire les blocs ```glsl … ``` ou ``` … ```
    raw = re.sub(r'```(?:glsl|GLSL)?\n?', '', raw)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    # Vérifie qu'il y a une fonction mainImage
    if 'void mainImage' not in raw and 'void main()' not in raw:
        log.warning("Réponse LLM sans fonction principale GLSL détectée")
    return raw


def _build_user_message(req: ShaderGenerationRequest) -> str:
    if req.current_shader.strip():
        return _MODIFY_PREFIX.format(
            current_shader=req.current_shader,
            prompt=req.prompt,
        )
    return _CREATE_PREFIX.format(prompt=req.prompt)


class _OpenAIBackend:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._key   = api_key
        self._model = model

    def is_available(self) -> bool:
        return bool(self._key)

    def generate(self, req: ShaderGenerationRequest,
                 on_token=None) -> ShaderGenerationResult:
        t0 = time.time()
        model = req.model or self._model

        if _OPENAI_SDK:
            return self._via_sdk(req, model, on_token, t0)
        return self._via_http(req, model, on_token, t0)

    def _via_sdk(self, req, model, on_token, t0):
        client = _OpenAI(api_key=self._key)
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(req)},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                if on_token:
                    on_token(delta)
        raw = "".join(chunks)
        return ShaderGenerationResult(
            glsl=_extract_glsl(raw), prompt=req.prompt,
            backend=self.name, model=model,
            duration_s=time.time()-t0,
        )

    def _via_http(self, req, model, on_token, t0):
        """Fallback HTTP sans SDK openai."""
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(req)},
            ],
            "temperature": req.temperature,
            "max_tokens":  req.max_tokens,
            "stream": bool(on_token),
        }).encode()

        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                if on_token:
                    # SSE streaming
                    raw_chunks = []
                    for line in resp:
                        line = line.decode().strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0]["delta"].get("content", "")
                                if delta:
                                    raw_chunks.append(delta)
                                    on_token(delta)
                            except (json.JSONDecodeError, KeyError):
                                pass
                    raw = "".join(raw_chunks)
                else:
                    data = json.loads(resp.read())
                    raw  = data["choices"][0]["message"]["content"]

            return ShaderGenerationResult(
                glsl=_extract_glsl(raw), prompt=req.prompt,
                backend=self.name, model=model,
                duration_s=time.time()-t0,
            )
        except Exception as e:
            return ShaderGenerationResult(
                glsl="", prompt=req.prompt, backend=self.name, model=model,
                duration_s=time.time()-t0, ok=False, error=str(e),
            )


class _OllamaBackend:
    """Backend Ollama (http://localhost:11434) — modèles locaux."""
    name = "ollama"
    DEFAULT_MODELS = ["codestral", "deepseek-coder:6.7b", "mistral", "llama3.1"]

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "codestral"):
        self._host  = host.rstrip("/")
        self._model = model

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self._host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, req: ShaderGenerationRequest,
                 on_token=None) -> ShaderGenerationResult:
        t0    = time.time()
        model = req.model or self._model

        payload = json.dumps({
            "model":  model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{_build_user_message(req)}",
            "stream": True,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }).encode()

        request = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            chunks: list[str] = []
            with urllib.request.urlopen(request, timeout=120) as resp:
                for line in resp:
                    try:
                        data = json.loads(line.decode())
                        tok  = data.get("response", "")
                        if tok:
                            chunks.append(tok)
                            if on_token:
                                on_token(tok)
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        pass
            raw = "".join(chunks)
            return ShaderGenerationResult(
                glsl=_extract_glsl(raw), prompt=req.prompt,
                backend=self.name, model=model,
                duration_s=time.time()-t0,
            )
        except Exception as e:
            return ShaderGenerationResult(
                glsl="", prompt=req.prompt, backend=self.name, model=model,
                duration_s=time.time()-t0, ok=False, error=str(e),
            )


class _LlamaCppBackend:
    """Backend llama.cpp server (http://localhost:8080) — modèles GGUF locaux."""
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

    def generate(self, req: ShaderGenerationRequest,
                 on_token=None) -> ShaderGenerationResult:
        t0 = time.time()
        payload = json.dumps({
            "prompt":      f"<s>[INST] {_SYSTEM_PROMPT}\n\n{_build_user_message(req)} [/INST]",
            "n_predict":   req.max_tokens,
            "temperature": req.temperature,
            "stream":      True,
            "stop":        ["</s>", "[INST]"],
        }).encode()

        request = urllib.request.Request(
            f"{self._host}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            chunks: list[str] = []
            with urllib.request.urlopen(request, timeout=180) as resp:
                for line in resp:
                    line_s = line.decode().strip()
                    if line_s.startswith("data: "):
                        try:
                            data = json.loads(line_s[6:])
                            tok  = data.get("content", "")
                            if tok:
                                chunks.append(tok)
                                if on_token:
                                    on_token(tok)
                            if data.get("stop"):
                                break
                        except json.JSONDecodeError:
                            pass
            raw = "".join(chunks)
            return ShaderGenerationResult(
                glsl=_extract_glsl(raw), prompt=req.prompt,
                backend=self.name, model="gguf",
                duration_s=time.time()-t0,
            )
        except Exception as e:
            return ShaderGenerationResult(
                glsl="", prompt=req.prompt, backend=self.name, model="gguf",
                duration_s=time.time()-t0, ok=False, error=str(e),
            )


class _StubBackend:
    """Backend stub offline — retourne un shader de démonstration."""
    name = "stub"

    def is_available(self) -> bool:
        return True

    def generate(self, req: ShaderGenerationRequest,
                 on_token=None) -> ShaderGenerationResult:
        # Shader de démonstration générique audio-réactif
        glsl = f"""\
// Généré (stub offline) — {req.prompt[:60]}
// Connectez OpenAI, Ollama ou llama.cpp pour une vraie génération IA.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float t  = iTime * 0.5;
    float rms = uRMS;

    // Plasma audio-réactif de démonstration
    float d = length(uv);
    float wave = sin(d * 10.0 - t * 3.0 + rms * 6.28)
               + sin(uv.x * 8.0 + t * 2.1)
               + sin(uv.y * 7.0 - t * 1.7);
    wave = wave / 3.0 * 0.5 + 0.5;

    // Palette cosinus
    vec3 col = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0,0.33,0.67) + wave + uBeat * 0.3));
    col *= 1.0 - smoothstep(0.4, 0.7, d);  // vignette

    fragColor = vec4(col, 1.0);
}}"""
        if on_token:
            # Simule un streaming token par token
            for ch in glsl:
                on_token(ch)
                time.sleep(0.004)
        return ShaderGenerationResult(
            glsl=glsl, prompt=req.prompt,
            backend=self.name, model="stub",
            duration_s=0.1,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  AIShaderGenerator — moteur principal
# ═════════════════════════════════════════════════════════════════════════════

class AIShaderGenerator(QObject):
    """
    Génère des shaders GLSL depuis un prompt LLM en arrière-plan.

    Signaux
    -------
    token_received    (str)          — streaming token par token
    generation_done   (object)       — ShaderGenerationResult complet
    generation_error  (str)          — message d'erreur lisible
    suggestion_ready  (list)         — suggestions contextuelles (list[str])
    backend_changed   (str)          — backend actif (après auto-détection)
    """

    token_received   = pyqtSignal(str)
    generation_done  = pyqtSignal(object)
    generation_error = pyqtSignal(str)
    suggestion_ready = pyqtSignal(list)
    backend_changed  = pyqtSignal(str)

    MAX_HISTORY = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[GenerationHistoryEntry] = []
        self._generating = False

        # Backends — instanciés lazily
        self._openai_key: str  = ""
        self._ollama_host: str = "http://localhost:11434"
        self._llamacpp_host: str = "http://localhost:8080"
        self._preferred_model: str = ""
        self._active_backend_name: str = ""

        self._backends: list[Any] = []
        self._stub = _StubBackend()

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_openai_key(self, key: str):
        self._openai_key = key.strip()
        self._backends = []  # force re-détection

    def set_ollama_host(self, host: str):
        self._ollama_host = host
        self._backends = []

    def set_llamacpp_host(self, host: str):
        self._llamacpp_host = host
        self._backends = []

    def set_preferred_model(self, model: str):
        self._preferred_model = model

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def history(self) -> list[GenerationHistoryEntry]:
        return list(self._history)

    @property
    def active_backend(self) -> str:
        return self._active_backend_name

    # ── Auto-détection du backend ─────────────────────────────────────────────

    def detect_backends(self) -> list[str]:
        """
        Sonde les backends disponibles en parallèle.
        Retourne la liste des noms disponibles.
        """
        available: list[str] = []
        candidates: list[Any] = []

        if self._openai_key:
            b = _OpenAIBackend(self._openai_key)
            if b.is_available():
                candidates.append(b)
                available.append("openai")

        try:
            b = _OllamaBackend(self._ollama_host)
            if b.is_available():
                candidates.append(b)
                available.append("ollama")
        except Exception:
            pass

        try:
            b = _LlamaCppBackend(self._llamacpp_host)
            if b.is_available():
                candidates.append(b)
                available.append("llamacpp")
        except Exception:
            pass

        self._backends = candidates
        if candidates:
            self._active_backend_name = candidates[0].name
            self.backend_changed.emit(self._active_backend_name)
            log.info("Backend IA actif : %s", self._active_backend_name)
        else:
            self._active_backend_name = "stub"
            self.backend_changed.emit("stub")
            log.info("Aucun backend IA disponible — mode stub")

        return available

    def list_ollama_models(self) -> list[str]:
        try:
            b = _OllamaBackend(self._ollama_host)
            return b.list_models()
        except Exception:
            return []

    # ── Génération ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, current_shader: str = "",
                 model: str = "", temperature: float = 0.7):
        """
        Lance la génération en arrière-plan.
        Émet token_received() en streaming, puis generation_done() ou generation_error().
        """
        if self._generating:
            log.warning("Génération déjà en cours — ignorée.")
            return

        req = ShaderGenerationRequest(
            prompt        = prompt,
            current_shader= current_shader,
            model         = model or self._preferred_model,
            temperature   = temperature,
        )
        self._generating = True

        t = threading.Thread(
            target=self._run_generation,
            args=(req,),
            daemon=True,
            name="AIShaderGen",
        )
        t.start()

    def _run_generation(self, req: ShaderGenerationRequest):
        # Sélectionne le backend
        if not self._backends:
            self.detect_backends()

        backend = self._backends[0] if self._backends else self._stub

        log.info("Génération IA — backend=%s prompt=%r", backend.name, req.prompt[:60])

        # Streaming callback thread-safe via signal Qt
        def on_token(tok: str):
            self.token_received.emit(tok)

        result = backend.generate(req, on_token=on_token)
        self._generating = False

        if result.ok and result.glsl:
            # Enregistre dans l'historique
            entry = GenerationHistoryEntry(
                prompt      = req.prompt,
                glsl_before = req.current_shader,
                glsl_after  = result.glsl,
                backend     = result.backend,
                model       = result.model,
                duration_s  = result.duration_s,
                ok          = True,
            )
            self._history.append(entry)
            if len(self._history) > self.MAX_HISTORY:
                self._history.pop(0)

            log.info("Génération OK — %s en %.1fs", backend.name, result.duration_s)
            self.generation_done.emit(result)
        else:
            err_msg = result.error or "Réponse vide du LLM."
            entry = GenerationHistoryEntry(
                prompt     = req.prompt,
                glsl_before= req.current_shader,
                glsl_after = "",
                backend    = result.backend,
                model      = result.model,
                ok         = False,
            )
            self._history.append(entry)
            log.error("Génération IA échouée : %s", err_msg)
            self.generation_error.emit(err_msg)

    # ── Suggestions contextuelles ─────────────────────────────────────────────

    def suggest(self, current_shader: str):
        """
        Génère des suggestions basées sur le shader courant.
        Émet suggestion_ready(list[str]) de façon synchrone (très rapide).
        """
        suggestions = ContextualSuggestions.get(current_shader, n=6)
        self.suggestion_ready.emit(suggestions)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Config sérialisable (sans la clé API — elle n'est pas stockée dans le projet)."""
        return {
            "ollama_host":   self._ollama_host,
            "llamacpp_host": self._llamacpp_host,
            "preferred_model": self._preferred_model,
        }

    def from_dict(self, data: dict):
        self._ollama_host    = data.get("ollama_host",     self._ollama_host)
        self._llamacpp_host  = data.get("llamacpp_host",   self._llamacpp_host)
        self._preferred_model= data.get("preferred_model", "")
        self._backends = []  # force re-détection au prochain generate()
