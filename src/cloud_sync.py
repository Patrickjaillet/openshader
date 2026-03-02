"""
cloud_sync.py
-------------
Cloud Sync des projets OpenShader  (v5.0)

Fonctionnalités :
  • OAuth 2.0 PKCE via GitHub  (ou Google — même flow)
  • Sauvegarde automatique dans le cloud (configurable : à chaque save, ou toutes les N min)
  • Historique de 30 révisions par projet, restauration one-click
  • Partage par lien en lecture seule (token JWT signé côté serveur)
  • Aucune dépendance externe : stdlib pure (urllib, http.server, threading, hashlib, secrets)

API fictive attendue côté serveur (openshader.io)
====================================================

  POST   /auth/github/exchange   { code, verifier } → { access_token, user: {id,login,avatar} }
  GET    /auth/me                                    → { id, login, avatar_url, plan }

  GET    /projects                                   → [ ProjectMeta, … ]
  POST   /projects                  { name, data_b64, size } → ProjectMeta
  GET    /projects/{id}                              → ProjectMeta + data_b64
  PUT    /projects/{id}             { name, data_b64 }       → ProjectMeta
  DELETE /projects/{id}                              → { ok }

  GET    /projects/{id}/revisions                    → [ RevisionMeta, … ]
  GET    /projects/{id}/revisions/{rev_id}           → RevisionMeta + data_b64
  POST   /projects/{id}/share                        → { share_url, token }

Stockage local :
  ~/.openshader/cloud/token.json      — access_token + user info
  ~/.openshader/cloud/projects.json   — cache index des projets distants
  ~/.openshader/cloud/revisions/      — cache révisions téléchargées
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt, QSettings
from PyQt6.QtGui import QColor, QPixmap, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QScrollArea, QListWidget, QListWidgetItem,
    QDialog, QTabWidget, QTextEdit, QComboBox, QCheckBox,
    QSpinBox, QProgressBar, QSizePolicy, QMessageBox,
    QSplitter, QGridLayout, QMenu, QInputDialog,
)

from .logger import get_logger

log = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

CLOUD_API_BASE    = "https://api.openshader.io/v1"
OAUTH_GITHUB_AUTH = "https://github.com/login/oauth/authorize"
OAUTH_GITHUB_APP  = "Ov23liOpenShaderDev"   # GitHub OAuth App client_id
OAUTH_REDIRECT    = "http://localhost:9877/callback"
OAUTH_SCOPES      = "read:user"
CLOUD_DIR         = os.path.join(os.path.expanduser("~"), ".openshader", "cloud")
REVISIONS_DIR     = os.path.join(CLOUD_DIR, "revisions")
MAX_REVISIONS     = 30
AUTO_SYNC_INTERVAL_DEFAULT = 5   # minutes


# ══════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserInfo:
    user_id:    str
    login:      str
    avatar_url: str = ""
    plan:       str = "free"   # free | pro

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "login": self.login,
                "avatar_url": self.avatar_url, "plan": self.plan}

    @classmethod
    def from_dict(cls, d: dict) -> "UserInfo":
        return cls(user_id=d.get("user_id", d.get("id", "")),
                   login=d.get("login", ""),
                   avatar_url=d.get("avatar_url", ""),
                   plan=d.get("plan", "free"))


@dataclass
class ProjectMeta:
    project_id:   str
    name:         str
    updated_at:   str
    size_bytes:   int          = 0
    revision_count: int        = 0
    share_url:    str          = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMeta":
        return cls(
            project_id=d.get("id", d.get("project_id", "")),
            name=d.get("name", "Sans titre"),
            updated_at=d.get("updated_at", ""),
            size_bytes=int(d.get("size_bytes", 0)),
            revision_count=int(d.get("revision_count", 0)),
            share_url=d.get("share_url", ""),
        )


@dataclass
class RevisionMeta:
    revision_id: str
    project_id:  str
    created_at:  str
    message:     str  = ""
    size_bytes:  int  = 0

    @classmethod
    def from_dict(cls, d: dict) -> "RevisionMeta":
        return cls(
            revision_id=d.get("id", d.get("revision_id", "")),
            project_id=d.get("project_id", ""),
            created_at=d.get("created_at", ""),
            message=d.get("message", ""),
            size_bytes=int(d.get("size_bytes", 0)),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  OAuth 2.0 PKCE  —  stdlib pure, serveur de callback local
# ══════════════════════════════════════════════════════════════════════════════

class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Gestionnaire HTTP minimaliste pour recevoir le code OAuth."""

    def do_GET(self):   # noqa: N802
        parsed   = urllib.parse.urlparse(self.path)
        params   = urllib.parse.parse_qs(parsed.query)
        code     = params.get("code",  [""])[0]
        error    = params.get("error", [""])[0]

        if code:
            self.server.oauth_result = {"code": code, "error": None}
            body = b"<html><body style='font-family:monospace;background:#0d0f16;color:#50e0a0;'>" \
                   b"<h2>Connexion r&eacute;ussie !</h2>" \
                   b"<p>Vous pouvez fermer cet onglet et revenir dans OpenShader.</p></body></html>"
        else:
            self.server.oauth_result = {"code": None, "error": error or "unknown"}
            body = b"<html><body style='font-family:monospace;background:#0d0f16;color:#e07850;'>" \
                   b"<h2>Erreur d'autorisation</h2>" \
                   b"<p>Veuillez r&eacute;essayer depuis OpenShader.</p></body></html>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass   # silence stdlib logging


class OAuthFlow:
    """
    Orchestre le flow OAuth 2.0 PKCE.
    Lance un mini-serveur HTTP sur localhost:9877 pour récupérer le code,
    puis l'échange contre un access_token via l'API OpenShader.
    """

    def __init__(self, on_done: Callable[[Optional[str], Optional[UserInfo]], None]):
        """on_done(access_token_or_None, user_or_None)"""
        self._on_done = on_done
        self._thread: Optional[threading.Thread] = None

    def start(self, provider: str = "github"):
        """Lance le flow dans un thread daemon."""
        self._thread = threading.Thread(
            target=self._run, args=(provider,),
            daemon=True, name="OAuthFlow",
        )
        self._thread.start()

    def _run(self, provider: str):
        # ── PKCE ─────────────────────────────────────────────────────────────
        verifier  = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)

        # ── URL d'autorisation ────────────────────────────────────────────────
        if provider == "github":
            auth_url = (
                f"{OAUTH_GITHUB_AUTH}"
                f"?client_id={OAUTH_GITHUB_APP}"
                f"&redirect_uri={urllib.parse.quote(OAUTH_REDIRECT, safe='')}"
                f"&scope={OAUTH_SCOPES}"
                f"&state={state}"
            )
        else:
            log.error("OAuth: provider inconnu %s", provider)
            QTimer.singleShot(0, lambda: self._on_done(None, None))
            return

        # ── Ouvre le navigateur ───────────────────────────────────────────────
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            log.error("OAuth: webbrowser.open failed: %s", e)
            QTimer.singleShot(0, lambda: self._on_done(None, None))
            return

        # ── Serveur de callback ───────────────────────────────────────────────
        try:
            server = http.server.HTTPServer(("127.0.0.1", 9877), _OAuthCallbackHandler)
            server.oauth_result = None
            server.timeout      = 120   # 2 min pour que l'utilisateur se connecte
            server.handle_request()
        except OSError as e:
            log.error("OAuth callback server: %s", e)
            QTimer.singleShot(0, lambda: self._on_done(None, None))
            return

        result = getattr(server, "oauth_result", None)
        if not result or not result.get("code"):
            log.warning("OAuth: pas de code reçu (erreur: %s)", result)
            QTimer.singleShot(0, lambda: self._on_done(None, None))
            return

        # ── Échange du code contre un token ───────────────────────────────────
        try:
            payload = json.dumps({
                "code":     result["code"],
                "verifier": verifier,
                "state":    state,
            }).encode()
            req = urllib.request.Request(
                f"{CLOUD_API_BASE}/auth/{provider}/exchange",
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json",
                         "User-Agent": "OpenShader/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            token = data.get("access_token", "")
            user  = UserInfo.from_dict(data.get("user", {}))
            log.info("OAuth: connecté en tant que %s", user.login)
            QTimer.singleShot(0, lambda t=token, u=user: self._on_done(t, u))
        except Exception as e:
            log.error("OAuth token exchange: %s", e)
            QTimer.singleShot(0, lambda: self._on_done(None, None))


# ══════════════════════════════════════════════════════════════════════════════
#  CloudClient  —  appels HTTP vers l'API OpenShader
# ══════════════════════════════════════════════════════════════════════════════

class CloudClient:
    """Wrapper HTTP synchrone + méthodes métier cloud."""

    def __init__(self, token: str):
        self._token = token

    def _req(self, method: str, path: str,
             body: Optional[dict] = None,
             timeout: int = 20) -> dict:
        url = f"{CLOUD_API_BASE}{path}"
        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type":  "application/json",
                "User-Agent":    "OpenShader/5.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body_str = ""
            try:
                body_str = e.read().decode()[:300]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {e.reason} — {body_str}") from e
        except Exception as e:
            raise RuntimeError(str(e)) from e

    # ── Compte ────────────────────────────────────────────────────────────────

    def get_me(self) -> UserInfo:
        return UserInfo.from_dict(self._req("GET", "/auth/me"))

    # ── Projets ───────────────────────────────────────────────────────────────

    def list_projects(self) -> list[ProjectMeta]:
        data = self._req("GET", "/projects")
        return [ProjectMeta.from_dict(d) for d in data.get("projects", [])]

    def create_project(self, name: str, project_bytes: bytes) -> ProjectMeta:
        b64 = base64.b64encode(project_bytes).decode()
        data = self._req("POST", "/projects",
                         {"name": name, "data_b64": b64,
                          "size": len(project_bytes)})
        return ProjectMeta.from_dict(data)

    def update_project(self, project_id: str, name: str,
                       project_bytes: bytes) -> ProjectMeta:
        b64 = base64.b64encode(project_bytes).decode()
        data = self._req("PUT", f"/projects/{project_id}",
                         {"name": name, "data_b64": b64})
        return ProjectMeta.from_dict(data)

    def get_project_data(self, project_id: str) -> bytes:
        data = self._req("GET", f"/projects/{project_id}")
        return base64.b64decode(data["data_b64"])

    def delete_project(self, project_id: str) -> bool:
        self._req("DELETE", f"/projects/{project_id}")
        return True

    # ── Révisions ─────────────────────────────────────────────────────────────

    def list_revisions(self, project_id: str) -> list[RevisionMeta]:
        data = self._req("GET", f"/projects/{project_id}/revisions")
        return [RevisionMeta.from_dict(d) for d in data.get("revisions", [])]

    def get_revision_data(self, project_id: str, revision_id: str) -> bytes:
        data = self._req("GET",
                         f"/projects/{project_id}/revisions/{revision_id}")
        return base64.b64decode(data["data_b64"])

    # ── Partage ───────────────────────────────────────────────────────────────

    def share_project(self, project_id: str) -> str:
        """Crée / renouvelle le lien de partage. Retourne l'URL."""
        data = self._req("POST", f"/projects/{project_id}/share")
        return data.get("share_url", "")


# ══════════════════════════════════════════════════════════════════════════════
#  CloudSyncManager  —  QObject orchestrateur
# ══════════════════════════════════════════════════════════════════════════════

class CloudSyncManager(QObject):
    """
    Signaux :
      auth_changed(bool, UserInfo|None)   — connexion / déconnexion
      sync_started()
      sync_done(bool, str)                — (ok, message)
      projects_refreshed(list)            — list[ProjectMeta]
      revisions_refreshed(list)           — list[RevisionMeta]
      restore_ready(bytes)                — données brutes prêtes à charger
      share_url_ready(str)                — URL de partage
      progress(int, str)                  — (%, label)
    """

    auth_changed        = pyqtSignal(bool, object)
    sync_started        = pyqtSignal()
    sync_done           = pyqtSignal(bool, str)
    projects_refreshed  = pyqtSignal(list)
    revisions_refreshed = pyqtSignal(list)
    restore_ready       = pyqtSignal(bytes)
    share_url_ready     = pyqtSignal(str)
    progress            = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        os.makedirs(CLOUD_DIR,      exist_ok=True)
        os.makedirs(REVISIONS_DIR,  exist_ok=True)

        self._token:       str           = ""
        self._user:        Optional[UserInfo] = None
        self._client:      Optional[CloudClient] = None
        self._projects:    list[ProjectMeta]  = []
        self._current_project_id: str    = ""

        # Auto-sync
        self._auto_sync_enabled:  bool   = False
        self._auto_sync_interval: int    = AUTO_SYNC_INTERVAL_DEFAULT
        self._auto_sync_timer             = QTimer(self)
        self._auto_sync_timer.timeout.connect(self._on_autosync_tick)
        self._get_project_data: Optional[Callable[[], bytes]] = None
        self._get_project_name: Optional[Callable[[], str]]   = None

        self._load_token()

    # ── Token persistant ──────────────────────────────────────────────────────

    _TOKEN_FILE = os.path.join(CLOUD_DIR, "token.json")

    def _load_token(self):
        if not os.path.isfile(self._TOKEN_FILE):
            return
        try:
            d = json.loads(open(self._TOKEN_FILE, encoding="utf-8").read())
            self._token = d.get("token", "")
            if d.get("user"):
                self._user   = UserInfo.from_dict(d["user"])
                self._client = CloudClient(self._token)
            self._current_project_id = d.get("current_project_id", "")
            self._auto_sync_interval = d.get("auto_sync_interval",
                                              AUTO_SYNC_INTERVAL_DEFAULT)
            self._auto_sync_enabled  = d.get("auto_sync_enabled", False)
            if self._auto_sync_enabled and self._token:
                self._start_autosync_timer()
            log.info("Cloud: token chargé pour %s",
                     self._user.login if self._user else "?")
        except Exception as e:
            log.warning("Cloud: erreur chargement token: %s", e)

    def _save_token(self):
        d = {
            "token":              self._token,
            "user":               self._user.to_dict() if self._user else None,
            "current_project_id": self._current_project_id,
            "auto_sync_interval": self._auto_sync_interval,
            "auto_sync_enabled":  self._auto_sync_enabled,
        }
        with open(self._TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    # ── Propriétés publiques ──────────────────────────────────────────────────

    @property
    def is_logged_in(self) -> bool:
        return bool(self._token and self._user)

    @property
    def user(self) -> Optional[UserInfo]:
        return self._user

    @property
    def current_project_id(self) -> str:
        return self._current_project_id

    @current_project_id.setter
    def current_project_id(self, pid: str):
        self._current_project_id = pid
        self._save_token()

    # ── Auth ──────────────────────────────────────────────────────────────────

    def login(self, provider: str = "github"):
        """Lance le flow OAuth PKCE dans un thread daemon."""
        flow = OAuthFlow(on_done=self._on_oauth_done)
        flow.start(provider)

    def _on_oauth_done(self, token: Optional[str], user: Optional[UserInfo]):
        if token and user:
            self._token  = token
            self._user   = user
            self._client = CloudClient(token)
            self._save_token()
            self.auth_changed.emit(True, user)
            log.info("Cloud: authentifié en tant que %s", user.login)
            self.refresh_projects()
        else:
            self.auth_changed.emit(False, None)

    def logout(self):
        self._token   = ""
        self._user    = None
        self._client  = None
        self._projects = []
        self._current_project_id = ""
        self._auto_sync_timer.stop()
        try:
            os.remove(self._TOKEN_FILE)
        except OSError:
            pass
        self.auth_changed.emit(False, None)
        log.info("Cloud: déconnecté")

    # ── Projets ───────────────────────────────────────────────────────────────

    def set_data_callbacks(self, get_data: Callable[[], bytes],
                           get_name: Callable[[], str]):
        """Injecte les callbacks MainWindow pour accéder aux données du projet."""
        self._get_project_data = get_data
        self._get_project_name = get_name

    def refresh_projects(self):
        def _run():
            try:
                projects = self._client.list_projects()
                self._projects = projects
                self._cache_projects(projects)
                self.projects_refreshed.emit(projects)
            except Exception as e:
                log.warning("Cloud: list_projects: %s", e)
                # Fallback: cache local
                cached = self._load_cached_projects()
                if cached:
                    self.projects_refreshed.emit(cached)
        threading.Thread(target=_run, daemon=True, name="CloudListProjects").start()

    def save_to_cloud(self, force_new: bool = False):
        """Sauvegarde le projet courant vers le cloud."""
        if not self.is_logged_in or not self._get_project_data:
            self.sync_done.emit(False, "Non connecté au cloud")
            return

        self.sync_started.emit()
        self.progress.emit(5, "Collecte des données…")

        def _run():
            try:
                data  = self._get_project_data()
                name  = self._get_project_name() if self._get_project_name else "Sans titre"
                self.progress.emit(30, "Envoi vers le cloud…")

                if self._current_project_id and not force_new:
                    meta = self._client.update_project(
                        self._current_project_id, name, data)
                else:
                    meta = self._client.create_project(name, data)
                    self._current_project_id = meta.project_id
                    self._save_token()

                self.progress.emit(100, "Sauvegardé ✓")
                self._projects = self._client.list_projects()
                self._cache_projects(self._projects)
                self.projects_refreshed.emit(self._projects)
                self.sync_done.emit(True, f"✓ Sauvegardé dans le cloud — {name}")
                log.info("Cloud: projet sauvegardé (%s)", meta.project_id)
            except Exception as e:
                self.progress.emit(0, "Erreur")
                self.sync_done.emit(False, f"Erreur de sauvegarde cloud : {e}")
                log.error("Cloud: save_to_cloud: %s", e)

        threading.Thread(target=_run, daemon=True, name="CloudSave").start()

    def load_from_cloud(self, project_id: str):
        """Télécharge et émet restore_ready() avec les données brutes."""
        if not self.is_logged_in:
            return
        self.progress.emit(10, "Téléchargement…")

        def _run():
            try:
                data = self._client.get_project_data(project_id)
                self._current_project_id = project_id
                self._save_token()
                self.progress.emit(100, "Chargé ✓")
                self.restore_ready.emit(data)
                log.info("Cloud: projet chargé (%s)", project_id)
            except Exception as e:
                self.progress.emit(0, "Erreur")
                self.sync_done.emit(False, f"Erreur de chargement cloud : {e}")
                log.error("Cloud: load_from_cloud: %s", e)

        threading.Thread(target=_run, daemon=True, name="CloudLoad").start()

    def delete_cloud_project(self, project_id: str):
        if not self.is_logged_in:
            return
        def _run():
            try:
                self._client.delete_project(project_id)
                if self._current_project_id == project_id:
                    self._current_project_id = ""
                    self._save_token()
                self.refresh_projects()
                self.sync_done.emit(True, "Projet supprimé du cloud")
            except Exception as e:
                self.sync_done.emit(False, f"Erreur suppression : {e}")
        threading.Thread(target=_run, daemon=True, name="CloudDelete").start()

    # ── Révisions ─────────────────────────────────────────────────────────────

    def load_revisions(self, project_id: str):
        if not self.is_logged_in:
            return
        def _run():
            try:
                revs = self._client.list_revisions(project_id)
                self.revisions_refreshed.emit(revs)
            except Exception as e:
                self.sync_done.emit(False, f"Erreur révisions : {e}")
                log.error("Cloud: list_revisions: %s", e)
        threading.Thread(target=_run, daemon=True, name="CloudRevisions").start()

    def restore_revision(self, project_id: str, revision_id: str):
        if not self.is_logged_in:
            return
        self.progress.emit(10, "Téléchargement révision…")
        def _run():
            try:
                # Cache local de la révision
                cache_path = os.path.join(REVISIONS_DIR,
                                          f"{project_id}_{revision_id}.demomaker")
                if os.path.isfile(cache_path):
                    with open(cache_path, "rb") as f:
                        data = f.read()
                else:
                    data = self._client.get_revision_data(project_id, revision_id)
                    with open(cache_path, "wb") as f:
                        f.write(data)
                self.progress.emit(100, "Révision prête")
                self.restore_ready.emit(data)
                log.info("Cloud: révision restaurée (%s / %s)",
                         project_id, revision_id)
            except Exception as e:
                self.progress.emit(0, "Erreur")
                self.sync_done.emit(False, f"Erreur restauration : {e}")
        threading.Thread(target=_run, daemon=True, name="CloudRestore").start()

    # ── Partage ───────────────────────────────────────────────────────────────

    def share_project(self, project_id: str):
        if not self.is_logged_in:
            return
        def _run():
            try:
                url = self._client.share_project(project_id)
                self.share_url_ready.emit(url)
                log.info("Cloud: lien partagé créé : %s", url)
            except Exception as e:
                self.sync_done.emit(False, f"Erreur partage : {e}")
        threading.Thread(target=_run, daemon=True, name="CloudShare").start()

    # ── Auto-sync ─────────────────────────────────────────────────────────────

    def set_auto_sync(self, enabled: bool, interval_minutes: int = 0):
        self._auto_sync_enabled  = enabled
        if interval_minutes > 0:
            self._auto_sync_interval = interval_minutes
        self._save_token()
        if enabled and self.is_logged_in:
            self._start_autosync_timer()
        else:
            self._auto_sync_timer.stop()

    def _start_autosync_timer(self):
        ms = self._auto_sync_interval * 60 * 1000
        self._auto_sync_timer.start(ms)
        log.debug("Cloud: auto-sync activé (%d min)", self._auto_sync_interval)

    def _on_autosync_tick(self):
        if self.is_logged_in and self._current_project_id:
            log.debug("Cloud: auto-sync déclenchée")
            self.save_to_cloud()

    def notify_manual_save(self):
        """Appelé après chaque Ctrl+S : déclenche une sync si activée."""
        if self._auto_sync_enabled and self.is_logged_in:
            self.save_to_cloud()

    # ── Cache local ───────────────────────────────────────────────────────────

    _CACHE_FILE = os.path.join(CLOUD_DIR, "projects.json")

    def _cache_projects(self, projects: list[ProjectMeta]):
        data = [{"id": p.project_id, "name": p.name,
                 "updated_at": p.updated_at, "size_bytes": p.size_bytes,
                 "revision_count": p.revision_count}
                for p in projects]
        with open(self._CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_cached_projects(self) -> list[ProjectMeta]:
        if not os.path.isfile(self._CACHE_FILE):
            return []
        try:
            data = json.loads(open(self._CACHE_FILE, encoding="utf-8").read())
            return [ProjectMeta.from_dict(d) for d in data]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
#  UI helpers
# ══════════════════════════════════════════════════════════════════════════════

_BG    = "#0d0f16"
_SURF  = "#12141e"
_SURF2 = "#181a26"
_BORD  = "#1e2235"
_TEXT  = "#c0c8e0"
_MUTED = "#5a6080"
_ACC   = "#4e7fff"
_ACC2  = "#50e0a0"
_WARN  = "#e07850"
_SANS  = "Segoe UI, Arial, sans-serif"
_MONO  = "Cascadia Code, Consolas, monospace"

_SS_BASE = f"""
QWidget           {{ background:{_BG}; color:{_TEXT}; font:9px '{_SANS}'; }}
QFrame            {{ background:{_SURF}; }}
QScrollBar:vertical   {{ background:{_SURF}; width:6px; border:none; }}
QScrollBar::handle:vertical {{ background:{_BORD}; border-radius:3px; min-height:16px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""


def _btn(label: str, color: str = _ACC, small: bool = False) -> QPushButton:
    fs, pad = ("8px", "2px 8px") if small else ("9px", "4px 14px")
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{_SURF};color:{color};border:1px solid {color}44;"
        f"border-radius:3px;padding:{pad};font:{fs} '{_SANS}';}}"
        f"QPushButton:hover{{background:{color}22;border-color:{color}88;}}"
        f"QPushButton:pressed{{background:{color}33;}}"
        f"QPushButton:disabled{{color:{_MUTED};border-color:{_BORD};}}"
    )
    return b


def _lbl(text: str, color: str = _TEXT, size: str = "9px",
         bold: bool = False, wrap: bool = False) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet(
        f"color:{color};font:{'bold ' if bold else ''}{size} '{_SANS}';"
        f"background:transparent;")
    if wrap:
        w.setWordWrap(True)
    return w


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:{_BORD};max-height:1px;border:none;")
    return f


def _tag(text: str, color: str = _ACC) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet(
        f"color:{color};background:{color}18;border-radius:3px;"
        f"padding:1px 6px;font:7px '{_SANS}';")
    return w


# ══════════════════════════════════════════════════════════════════════════════
#  _ProjectCard  —  carte dans la liste de projets
# ══════════════════════════════════════════════════════════════════════════════

class _ProjectCard(QFrame):
    open_requested   = pyqtSignal(str)    # project_id
    delete_requested = pyqtSignal(str)
    share_requested  = pyqtSignal(str)
    revisions_requested = pyqtSignal(str)

    def __init__(self, meta: ProjectMeta, is_current: bool = False, parent=None):
        super().__init__(parent)
        self._id = meta.project_id
        self.setStyleSheet(
            f"QFrame{{background:{_SURF};border:1px solid "
            f"{'#4e7fff' if is_current else _BORD};"
            f"border-radius:5px;padding:4px;}}"
            f"QFrame:hover{{border-color:{_ACC}55;}}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        # Titre + badge courant
        row1 = QHBoxLayout()
        name_lbl = _lbl(meta.name or "Sans titre", _TEXT, "10px", bold=True)
        row1.addWidget(name_lbl, 1)
        if is_current:
            row1.addWidget(_tag("● actif", _ACC2))
        lay.addLayout(row1)

        # Meta
        size_kb  = f"{meta.size_bytes // 1024} Ko" if meta.size_bytes else "—"
        rev_str  = f"{meta.revision_count} rév." if meta.revision_count else "0 rév."
        date_str = meta.updated_at[:10] if meta.updated_at else "—"
        row2 = QHBoxLayout()
        row2.addWidget(_lbl(f"📅 {date_str}", _MUTED, "8px"))
        row2.addWidget(_lbl(f"💾 {size_kb}", _MUTED, "8px"))
        row2.addWidget(_lbl(f"🔁 {rev_str}", _MUTED, "8px"))
        row2.addStretch()
        lay.addLayout(row2)

        # Boutons
        row3 = QHBoxLayout()
        btn_open = _btn("📂 Ouvrir", _ACC, small=True)
        btn_open.clicked.connect(lambda: self.open_requested.emit(self._id))
        row3.addWidget(btn_open)

        btn_rev = _btn("🔁 Révisions", _MUTED, small=True)
        btn_rev.clicked.connect(lambda: self.revisions_requested.emit(self._id))
        row3.addWidget(btn_rev)

        btn_share = _btn("🔗 Partager", _ACC2, small=True)
        btn_share.clicked.connect(lambda: self.share_requested.emit(self._id))
        row3.addWidget(btn_share)

        row3.addStretch()

        btn_del = _btn("✕", _WARN, small=True)
        btn_del.setFixedWidth(24)
        btn_del.setToolTip("Supprimer ce projet du cloud")
        btn_del.clicked.connect(lambda: self.delete_requested.emit(self._id))
        row3.addWidget(btn_del)

        lay.addLayout(row3)


# ══════════════════════════════════════════════════════════════════════════════
#  _RevisionList  —  liste des révisions d'un projet
# ══════════════════════════════════════════════════════════════════════════════

class _RevisionList(QWidget):
    restore_requested = pyqtSignal(str, str)   # project_id, revision_id

    def __init__(self, manager: "CloudSyncManager", parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._project_id = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        header = QHBoxLayout()
        self._title_lbl = _lbl("Révisions", _TEXT, "10px", bold=True)
        header.addWidget(self._title_lbl)
        header.addStretch()
        btn_back = _btn("← Retour", _MUTED, small=True)
        btn_back.clicked.connect(self._on_back)
        header.addWidget(btn_back)
        lay.addLayout(header)

        lay.addWidget(_sep())

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget{{background:{_BG};border:none;color:{_TEXT};}}"
            f"QListWidget::item{{padding:6px;border-bottom:1px solid {_BORD};}}"
            f"QListWidget::item:selected{{background:{_ACC}22;color:{_TEXT};}}"
        )
        lay.addWidget(self._list, 1)

        self._hint = _lbl(
            f"Les {MAX_REVISIONS} dernières révisions sont conservées.", _MUTED, "8px")
        lay.addWidget(self._hint)

        manager.revisions_refreshed.connect(self._on_revisions)

    def load(self, project_id: str, project_name: str):
        self._project_id = project_id
        self._title_lbl.setText(f"Révisions — {project_name}")
        self._list.clear()
        self._list.addItem("Chargement…")
        self._mgr.load_revisions(project_id)

    def _on_revisions(self, revs: list):
        self._list.clear()
        if not revs:
            self._list.addItem("Aucune révision disponible")
            return
        for rev in revs:
            date = rev.created_at[:16].replace("T", " ") if rev.created_at else "?"
            size = f"{rev.size_bytes // 1024} Ko" if rev.size_bytes else ""
            msg  = rev.message or ""
            text = f"{date}   {size}   {msg}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, rev.revision_id)
            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._on_item_activate)

        # Bouton de restauration contextuel
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)

    def _on_item_activate(self, item: QListWidgetItem):
        rev_id = item.data(Qt.ItemDataRole.UserRole)
        if rev_id:
            self._ask_restore(rev_id)

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if not item:
            return
        rev_id = item.data(Qt.ItemDataRole.UserRole)
        if not rev_id:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};}}"
            f"QMenu::item:selected{{background:{_ACC}33;}}"
        )
        act = menu.addAction("⏮ Restaurer cette révision")
        act.triggered.connect(lambda: self._ask_restore(rev_id))
        menu.exec(self._list.viewport().mapToGlobal(pos))

    def _ask_restore(self, rev_id: str):
        r = QMessageBox.question(
            self, "Restaurer",
            "Restaurer cette révision ?\n\n"
            "Le projet courant sera remplacé.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Yes:
            self.restore_requested.emit(self._project_id, rev_id)

    def _on_back(self):
        # Remonte au parent QStackedWidget
        if self.parent() and hasattr(self.parent(), "setCurrentIndex"):
            self.parent().setCurrentIndex(0)


# ══════════════════════════════════════════════════════════════════════════════
#  CloudSyncPanel  —  widget principal
# ══════════════════════════════════════════════════════════════════════════════

class CloudSyncPanel(QWidget):
    """
    Widget complet Cloud Sync.
    S'intègre comme QDialog ou DockWidget dans MainWindow.
    """

    def __init__(self, manager: CloudSyncManager, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self.setStyleSheet(_SS_BASE)
        self._build_ui()
        self._connect_signals()
        # Affiche l'état initial
        if manager.is_logged_in:
            self._show_logged_in(manager.user)
        else:
            self._show_logged_out()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── En-tête compte ────────────────────────────────────────────────────
        self._header = QFrame()
        self._header.setStyleSheet(
            f"QFrame{{background:{_SURF2};border-bottom:1px solid {_BORD};"
            f"border-radius:0;padding:0;}}"
        )
        hlay = QHBoxLayout(self._header)
        hlay.setContentsMargins(12, 10, 12, 10)
        hlay.setSpacing(10)

        self._avatar_lbl = QLabel("☁")
        self._avatar_lbl.setFixedSize(32, 32)
        self._avatar_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._avatar_lbl.setStyleSheet(
            f"font:18px;background:{_BORD};border-radius:16px;color:{_ACC};")
        hlay.addWidget(self._avatar_lbl)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self._user_lbl   = _lbl("Non connecté", _MUTED, "9px", bold=True)
        self._plan_lbl   = _lbl("", _MUTED, "8px")
        info_col.addWidget(self._user_lbl)
        info_col.addWidget(self._plan_lbl)
        hlay.addLayout(info_col, 1)

        self._login_btn  = _btn("Connexion GitHub", _ACC, small=True)
        self._login_btn.clicked.connect(lambda: self._mgr.login("github"))
        hlay.addWidget(self._login_btn)

        self._logout_btn = _btn("Déconnexion", _WARN, small=True)
        self._logout_btn.hide()
        self._logout_btn.clicked.connect(self._mgr.logout)
        hlay.addWidget(self._logout_btn)

        root.addWidget(self._header)

        # ── Corps — QStackedWidget ────────────────────────────────────────────
        from PyQt6.QtWidgets import QStackedWidget
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background:{_BG};")

        # Page 0 : liste projets
        self._projects_page = self._build_projects_page()
        self._stack.addWidget(self._projects_page)

        # Page 1 : révisions
        self._rev_page = _RevisionList(self._mgr)
        self._rev_page.restore_requested.connect(self._on_restore_revision)
        self._stack.addWidget(self._rev_page)

        root.addWidget(self._stack, 1)

        # ── Barre de statut ───────────────────────────────────────────────────
        status_bar = QFrame()
        status_bar.setStyleSheet(
            f"QFrame{{background:{_SURF2};border-top:1px solid {_BORD};padding:0;}}")
        sb_lay = QHBoxLayout(status_bar)
        sb_lay.setContentsMargins(10, 6, 10, 6)
        sb_lay.setSpacing(8)

        self._status_lbl  = _lbl("", _MUTED, "8px")
        sb_lay.addWidget(self._status_lbl, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(100)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setFixedWidth(80)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar{{background:{_BORD};border-radius:2px;border:none;}}"
            f"QProgressBar::chunk{{background:{_ACC2};border-radius:2px;}}")
        self._progress_bar.hide()
        sb_lay.addWidget(self._progress_bar)

        root.addWidget(status_bar)

    def _build_projects_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(_lbl("Projets cloud", _TEXT, "10px", bold=True))
        toolbar.addStretch()

        self._refresh_btn = _btn("↻ Rafraîchir", _MUTED, small=True)
        self._refresh_btn.clicked.connect(self._mgr.refresh_projects)
        toolbar.addWidget(self._refresh_btn)

        self._save_cloud_btn = _btn("☁ Sauvegarder", _ACC2)
        self._save_cloud_btn.clicked.connect(lambda: self._mgr.save_to_cloud())
        toolbar.addWidget(self._save_cloud_btn)

        self._new_cloud_btn = _btn("+ Nouveau", _ACC, small=True)
        self._new_cloud_btn.clicked.connect(
            lambda: self._mgr.save_to_cloud(force_new=True))
        toolbar.addWidget(self._new_cloud_btn)
        lay.addLayout(toolbar)

        # Auto-sync row
        autosync_row = QHBoxLayout()
        self._autosync_chk = QCheckBox("Auto-sync")
        self._autosync_chk.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        self._autosync_chk.setChecked(self._mgr._auto_sync_enabled)
        self._autosync_chk.toggled.connect(self._on_autosync_toggle)
        autosync_row.addWidget(self._autosync_chk)
        autosync_row.addWidget(_lbl("toutes les", _MUTED, "8px"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 60)
        self._interval_spin.setValue(self._mgr._auto_sync_interval)
        self._interval_spin.setSuffix(" min")
        self._interval_spin.setFixedWidth(68)
        self._interval_spin.setStyleSheet(
            f"QSpinBox{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:1px 4px;font:8px '{_SANS}';}}")
        self._interval_spin.valueChanged.connect(
            lambda v: self._mgr.set_auto_sync(self._autosync_chk.isChecked(), v))
        autosync_row.addWidget(self._interval_spin)
        autosync_row.addStretch()
        lay.addLayout(autosync_row)

        lay.addWidget(_sep())

        # Scroll liste projets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_BG};}}")
        self._list_widget = QWidget()
        self._list_widget.setStyleSheet(f"background:{_BG};")
        self._list_lay = QVBoxLayout(self._list_widget)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(6)
        self._list_lay.addStretch()
        scroll.setWidget(self._list_widget)
        lay.addWidget(scroll, 1)

        self._empty_lbl = _lbl(
            "Aucun projet dans le cloud.\nCliquez sur ☁ Sauvegarder pour commencer.",
            _MUTED, wrap=True)
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._empty_lbl)

        return page

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self):
        m = self._mgr
        m.auth_changed.connect(self._on_auth_changed)
        m.sync_done.connect(self._on_sync_done)
        m.sync_started.connect(self._on_sync_started)
        m.projects_refreshed.connect(self._on_projects_refreshed)
        m.share_url_ready.connect(self._on_share_url)
        m.progress.connect(self._on_progress)
        m.restore_ready.connect(self._on_restore_ready)

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _on_auth_changed(self, logged_in: bool, user):
        if logged_in and user:
            self._show_logged_in(user)
            self._mgr.refresh_projects()
        else:
            self._show_logged_out()

    def _show_logged_in(self, user):
        self._user_lbl.setText(f"@{user.login}")
        plan_text = "✦ Pro" if user.plan == "pro" else "Free"
        self._plan_lbl.setText(plan_text)
        self._plan_lbl.setStyleSheet(
            f"color:{'#f0c040' if user.plan=='pro' else _MUTED};"
            f"font:8px '{_SANS}';background:transparent;")
        self._login_btn.hide()
        self._logout_btn.show()
        self._save_cloud_btn.setEnabled(True)
        self._new_cloud_btn.setEnabled(True)
        self._empty_lbl.hide()

    def _show_logged_out(self):
        self._user_lbl.setText("Non connecté")
        self._plan_lbl.setText("")
        self._login_btn.show()
        self._logout_btn.hide()
        self._save_cloud_btn.setEnabled(False)
        self._new_cloud_btn.setEnabled(False)
        self._empty_lbl.show()
        self._empty_lbl.setText(
            "Connectez-vous pour synchroniser vos projets.")
        self._clear_project_list()

    # ── Projets ───────────────────────────────────────────────────────────────

    def _on_projects_refreshed(self, projects: list):
        self._clear_project_list()
        if not projects:
            self._empty_lbl.show()
            self._empty_lbl.setText(
                "Aucun projet dans le cloud.\n"
                "Cliquez sur ☁ Sauvegarder pour commencer.")
            return
        self._empty_lbl.hide()
        current_id = self._mgr.current_project_id
        for meta in projects:
            card = _ProjectCard(meta, is_current=(meta.project_id == current_id))
            card.open_requested.connect(self._mgr.load_from_cloud)
            card.delete_requested.connect(self._on_delete_project)
            card.share_requested.connect(self._mgr.share_project)
            card.revisions_requested.connect(self._open_revisions)
            self._list_lay.insertWidget(self._list_lay.count() - 1, card)
        self._status_lbl.setText(f"{len(projects)} projet(s) dans le cloud")

    def _clear_project_list(self):
        while self._list_lay.count() > 1:
            item = self._list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_delete_project(self, project_id: str):
        r = QMessageBox.question(
            self, "Supprimer",
            "Supprimer ce projet du cloud ?\n\n"
            "Cette action est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Yes:
            self._mgr.delete_cloud_project(project_id)

    def _open_revisions(self, project_id: str):
        # Trouve le nom
        name = next((p.name for p in self._mgr._projects
                     if p.project_id == project_id), project_id)
        self._rev_page.load(project_id, name)
        self._stack.setCurrentIndex(1)

    def _on_restore_revision(self, project_id: str, revision_id: str):
        self._mgr.restore_revision(project_id, revision_id)
        self._stack.setCurrentIndex(0)

    # ── Statut / Progress ─────────────────────────────────────────────────────

    def _on_sync_started(self):
        self._save_cloud_btn.setEnabled(False)
        self._status_lbl.setText("Synchronisation…")
        self._progress_bar.setValue(0)
        self._progress_bar.show()

    def _on_sync_done(self, ok: bool, message: str):
        self._save_cloud_btn.setEnabled(self._mgr.is_logged_in)
        color = _ACC2 if ok else _WARN
        self._status_lbl.setText(message.split("\n")[0][:60])
        self._status_lbl.setStyleSheet(
            f"color:{color};font:8px '{_SANS}';background:transparent;")
        QTimer.singleShot(4000, lambda: self._status_lbl.setStyleSheet(
            f"color:{_MUTED};font:8px '{_SANS}';background:transparent;"))
        self._progress_bar.hide()

    def _on_progress(self, pct: int, label: str):
        self._progress_bar.show()
        self._progress_bar.setValue(pct)
        self._status_lbl.setText(label)

    def _on_share_url(self, url: str):
        dlg = _ShareDialog(url, self)
        dlg.exec()

    def _on_restore_ready(self, data: bytes):
        # Remonté vers MainWindow via un signal re-émis (voir intégration MW)
        pass   # surcharge dans MainWindow._on_cloud_restore_ready

    def _on_autosync_toggle(self, checked: bool):
        self._mgr.set_auto_sync(checked, self._interval_spin.value())


# ══════════════════════════════════════════════════════════════════════════════
#  _ShareDialog  —  dialog affichant le lien de partage
# ══════════════════════════════════════════════════════════════════════════════

class _ShareDialog(QDialog):
    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔗 Lien de partage")
        self.setFixedWidth(500)
        self.setStyleSheet(
            f"QDialog{{background:{_BG};color:{_TEXT};}}" + _SS_BASE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)

        lay.addWidget(_lbl("Lien de partage (lecture seule)", _TEXT, "10px", bold=True))
        lay.addWidget(_lbl(
            "Partagez ce lien pour permettre à n'importe qui de visualiser "
            "votre projet sans pouvoir le modifier.",
            _MUTED, wrap=True))

        url_edit = QLineEdit(url)
        url_edit.setReadOnly(True)
        url_edit.setStyleSheet(
            f"QLineEdit{{background:{_SURF2};color:{_ACC2};"
            f"border:1px solid {_BORD};border-radius:3px;"
            f"padding:5px 8px;font:9px '{_MONO}';}}")
        lay.addWidget(url_edit)

        btns = QHBoxLayout()
        btn_copy = _btn("📋 Copier", _ACC)
        btn_copy.clicked.connect(lambda: self._copy(url, btn_copy))
        btns.addWidget(btn_copy)
        btn_open = _btn("🌐 Ouvrir dans le navigateur", _ACC2)
        btn_open.clicked.connect(lambda: webbrowser.open(url))
        btns.addWidget(btn_open)
        btns.addStretch()
        btn_close = _btn("Fermer", _MUTED, small=True)
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        lay.addLayout(btns)

    def _copy(self, url: str, btn: QPushButton):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(url)
        btn.setText("✓ Copié !")
        QTimer.singleShot(2000, lambda: btn.setText("📋 Copier"))


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_cloud_sync(parent=None) -> tuple["CloudSyncManager", "CloudSyncPanel"]:
    """Crée et retourne (manager, panel)."""
    mgr   = CloudSyncManager(parent)
    panel = CloudSyncPanel(mgr, parent)
    return mgr, panel
