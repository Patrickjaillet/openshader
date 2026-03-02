"""
rest_api.py
-----------
Serveur REST local pour OpenShader — v1.0

Expose l'état et les commandes de la MainWindow via HTTP/JSON (FastAPI + uvicorn).
Tourne dans un thread daemon distinct du thread PyQt6.
Les appels mutants utilisent QMetaObject.invokeMethod pour rester thread-safe.

Endpoints disponibles :
  GET  /status                    → état courant (play, time, uniforms, projet)
  POST /play                      → lecture
  POST /pause                     → pause
  POST /stop                      → stop (retour t=0)
  POST /seek                      → seek { "time": float }
  POST /load                      → charge un projet { "path": str }
  POST /render                    → lance un rendu headless { ... }
  GET  /uniforms                  → dict des uniforms courants
  POST /uniforms                  → set uniforms { "name": value, ... }
  GET  /timeline                  → données de la timeline (durée, pistes, keyframes)
  POST /timeline                  → remplace la duration { "duration": float }
  GET  /shader/{pass_name}        → source GLSL d'une passe
  POST /shader/{pass_name}        → remplace la source GLSL d'une passe
  GET  /health                    → { "ok": true }

Auth : aucune (localhost uniquement par défaut)
Thread-safety : toutes les mutations passent par invokeMethod Qt.QueuedConnection

Démarrage depuis main_window.py :
    from .rest_api import OpenShaderRESTServer
    self._rest_server = OpenShaderRESTServer(self)
    self._rest_server.start(host="127.0.0.1", port=8765)
"""

from __future__ import annotations

import json
import threading
import time
import logging
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QMetaObject, Qt, Q_ARG

log = logging.getLogger(__name__)

# ── Imports conditionnels FastAPI / uvicorn ───────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    log.warning("fastapi/uvicorn non installé — REST API indisponible. "
                "pip install fastapi uvicorn")


# ═════════════════════════════════════════════════════════════════════════════
#  Bridge thread-safe Qt ↔ REST
# ═════════════════════════════════════════════════════════════════════════════

class _QtBridge(QObject):
    """
    QObject vivant dans le thread Qt principal.
    Les méthodes sont appelées via invokeMethod depuis le thread REST.
    Retourne les résultats via des Events Python (threading.Event).
    """

    def __init__(self, window):
        super().__init__(window)
        self._win = window

    # ── Transport ─────────────────────────────────────────────────────────────

    def play(self):
        self._win._play()

    def pause(self):
        self._win._pause()

    def stop(self):
        self._win._on_stop()

    def seek(self, t: float):
        self._win._on_timeline_seek(t)

    # ── Projet ────────────────────────────────────────────────────────────────

    def load_project(self, path: str, result_event: threading.Event,
                     result_box: list):
        try:
            self._win._load_project_from_path(path)
            result_box.append({"ok": True})
        except Exception as e:
            result_box.append({"ok": False, "error": str(e)})
        result_event.set()

    # ── Uniforms ──────────────────────────────────────────────────────────────

    def set_uniforms(self, values: dict):
        for name, val in values.items():
            try:
                self._win.shader_engine.set_uniform(name, val)
            except Exception as e:
                log.warning("set_uniform(%s) : %s", name, e)

    def get_uniforms(self, result_event: threading.Event, result_box: list):
        try:
            engine = self._win.shader_engine
            result_box.append(dict(getattr(engine, '_custom_uniforms', {})))
        except Exception:
            result_box.append({})
        result_event.set()

    # ── Shader source ─────────────────────────────────────────────────────────

    def get_shader_source(self, pass_name: str,
                          result_event: threading.Event, result_box: list):
        editor = self._win.editors.get(pass_name)
        if editor:
            result_box.append(editor.get_code())
        else:
            result_box.append(None)
        result_event.set()

    def set_shader_source(self, pass_name: str, source: str,
                          result_event: threading.Event, result_box: list):
        editor = self._win.editors.get(pass_name)
        if editor is None:
            result_box.append({"ok": False, "error": f"Pass '{pass_name}' introuvable"})
        else:
            editor.set_code(source)
            result_box.append({"ok": True})
        result_event.set()

    # ── Status snapshot ───────────────────────────────────────────────────────

    def get_status(self, result_event: threading.Event, result_box: list):
        try:
            win = self._win
            result_box.append({
                "playing":      win._is_playing,
                "time":         win._current_time,
                "duration":     win.timeline.duration,
                "pass_names":   list(win.editors.keys()),
                "active_pass":  win.editor_tabs.tabText(
                                    win.editor_tabs.currentIndex()
                                ).lstrip("○●✕ ").strip(),
            })
        except Exception as e:
            result_box.append({"error": str(e)})
        result_event.set()

    # ── Timeline info ─────────────────────────────────────────────────────────

    def get_timeline(self, result_event: threading.Event, result_box: list):
        try:
            tl = self._win.timeline
            tracks_data = []
            for track in getattr(tl, 'tracks', []):
                kfs = []
                for kf in getattr(track, 'keyframes', []):
                    kfs.append({"time": kf.time, "value": kf.value})
                tracks_data.append({
                    "name":         track.name,
                    "uniform_name": getattr(track, 'uniform_name', ''),
                    "kind":         getattr(track, 'kind', 'float'),
                    "keyframes":    kfs,
                })
            result_box.append({
                "duration": tl.duration,
                "tracks":   tracks_data,
            })
        except Exception as e:
            result_box.append({"error": str(e)})
        result_event.set()

    def set_timeline_duration(self, duration: float):
        try:
            self._win.timeline.duration = duration
            self._win.timeline_widget.set_duration(duration)
        except Exception as e:
            log.warning("set_timeline_duration : %s", e)


def _invoke_sync(bridge: _QtBridge, method_name: str, *args,
                 timeout: float = 5.0) -> Any:
    """
    Appelle bridge.<method_name>(*args, result_event, result_box) via
    QMetaObject.invokeMethod (thread-safe) et attend le résultat.
    """
    event = threading.Event()
    box: list = []
    QMetaObject.invokeMethod(
        bridge,
        method_name,
        Qt.ConnectionType.QueuedConnection,
        *[Q_ARG(type(a), a) for a in args],
        Q_ARG(object, event),
        Q_ARG(object, box),
    )
    event.wait(timeout)
    return box[0] if box else None


def _invoke_fire(bridge: _QtBridge, method_name: str, *args):
    """Fire-and-forget via QueuedConnection."""
    QMetaObject.invokeMethod(
        bridge,
        method_name,
        Qt.ConnectionType.QueuedConnection,
        *[Q_ARG(type(a), a) for a in args],
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Construction de l'app FastAPI
# ═════════════════════════════════════════════════════════════════════════════

def _build_app(bridge: _QtBridge) -> "FastAPI":
    app = FastAPI(
        title="OpenShader REST API",
        version="1.0.0",
        description="Contrôle local de OpenShader via HTTP/JSON",
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"ok": True, "timestamp": time.time()}

    # ── Status ────────────────────────────────────────────────────────────────

    @app.get("/status")
    def get_status():
        result = _invoke_sync(bridge, "get_status")
        if result is None:
            raise HTTPException(503, "Timeout Qt")
        return result

    # ── Transport ─────────────────────────────────────────────────────────────

    @app.post("/play")
    def play():
        _invoke_fire(bridge, "play")
        return {"ok": True}

    @app.post("/pause")
    def pause():
        _invoke_fire(bridge, "pause")
        return {"ok": True}

    @app.post("/stop")
    def stop():
        _invoke_fire(bridge, "stop")
        return {"ok": True}

    @app.post("/seek")
    async def seek(request: Request):
        body = await request.json()
        t = float(body.get("time", 0.0))
        _invoke_fire(bridge, "seek", t)
        return {"ok": True, "time": t}

    # ── Projet ────────────────────────────────────────────────────────────────

    @app.post("/load")
    async def load(request: Request):
        body = await request.json()
        path = str(body.get("path", ""))
        if not path:
            raise HTTPException(400, "'path' requis")
        result = _invoke_sync(bridge, "load_project", path)
        if result is None:
            raise HTTPException(503, "Timeout Qt")
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "Erreur inconnue"))
        return result

    # ── Rendu headless ────────────────────────────────────────────────────────

    @app.post("/render")
    async def render(request: Request):
        """
        Lance un rendu headless en arrière-plan.
        Body JSON : { shader, output, width?, height?, fps?, duration? }
        """
        body = await request.json()
        shader = body.get("shader", "")
        output = body.get("output", "")
        if not shader or not output:
            raise HTTPException(400, "'shader' et 'output' requis")

        import threading as _thr
        from .headless import render_headless

        def _do_render():
            render_headless(
                shader_path=shader,
                output=output,
                width=int(body.get("width", 800)),
                height=int(body.get("height", 450)),
                fps=float(body.get("fps", 60.0)),
                duration=float(body.get("duration", 10.0)),
                pass_name=body.get("pass", "Image"),
            )

        t = _thr.Thread(target=_do_render, daemon=True, name="RESTRender")
        t.start()
        return {"ok": True, "message": "Rendu démarré en arrière-plan"}

    # ── Uniforms ──────────────────────────────────────────────────────────────

    @app.get("/uniforms")
    def get_uniforms():
        result = _invoke_sync(bridge, "get_uniforms")
        return result if result is not None else {}

    @app.post("/uniforms")
    async def set_uniforms(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "Body doit être un objet JSON { name: value }")
        _invoke_fire(bridge, "set_uniforms", body)
        return {"ok": True, "count": len(body)}

    # ── Timeline ──────────────────────────────────────────────────────────────

    @app.get("/timeline")
    def get_timeline():
        result = _invoke_sync(bridge, "get_timeline")
        if result is None:
            raise HTTPException(503, "Timeout Qt")
        return result

    @app.post("/timeline")
    async def set_timeline(request: Request):
        body = await request.json()
        if "duration" in body:
            _invoke_fire(bridge, "set_timeline_duration", float(body["duration"]))
        return {"ok": True}

    # ── Shader source ─────────────────────────────────────────────────────────

    @app.get("/shader/{pass_name}")
    def get_shader(pass_name: str):
        result = _invoke_sync(bridge, "get_shader_source", pass_name)
        if result is None:
            raise HTTPException(404, f"Pass '{pass_name}' introuvable")
        return {"pass": pass_name, "source": result}

    @app.post("/shader/{pass_name}")
    async def set_shader(pass_name: str, request: Request):
        body = await request.json()
        source = body.get("source", "")
        if not source:
            raise HTTPException(400, "'source' requis")
        result = _invoke_sync(bridge, "set_shader_source", pass_name, source)
        if result is None:
            raise HTTPException(503, "Timeout Qt")
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "Erreur inconnue"))
        return result

    return app


# ═════════════════════════════════════════════════════════════════════════════
#  OpenShaderRESTServer — façade publique
# ═════════════════════════════════════════════════════════════════════════════

class OpenShaderRESTServer:
    """
    Serveur REST FastAPI pour OpenShader.
    Tournez dans un thread daemon séparé du thread Qt.

    Usage :
        server = OpenShaderRESTServer(main_window)
        server.start(host="127.0.0.1", port=8765)
        # ...
        server.stop()
    """

    def __init__(self, window):
        self._win    = window
        self._bridge = _QtBridge(window)
        self._server: Optional["uvicorn.Server"] = None
        self._thread: Optional[threading.Thread] = None
        self._host   = "127.0.0.1"
        self._port   = 8765
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self, host: str = "127.0.0.1", port: int = 8765):
        if not _FASTAPI_OK:
            log.error("FastAPI/uvicorn non disponible — "
                      "pip install fastapi uvicorn")
            return
        if self._running:
            log.warning("REST server déjà démarré")
            return

        self._host = host
        self._port = port

        app = _build_app(self._bridge)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",  # silencieux dans les logs OpenShader
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True,
                                        name="OpenShaderREST")
        self._thread.start()
        self._running = True
        log.info("REST API démarrée → %s", self.base_url)

    def stop(self):
        if self._server and self._running:
            self._server.should_exit = True
            self._running = False
            log.info("REST API arrêtée")
