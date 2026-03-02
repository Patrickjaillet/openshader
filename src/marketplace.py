"""
marketplace.py
--------------
Marketplace de plugins OpenShader — v1.0

Architecture :
  - Index JSON hébergé sur GitHub Pages (signé SHA-256)
  - MarketplaceIndex     : fetch + parse + cache de l'index
  - MarketplaceClient   : download + install + update + scan statique
  - MarketplaceManager  : QObject principal, threading, signaux Qt
  - MarketplaceBrowser  : widget PyQt6 Browse / Install / Update / Rate

Format de l'index (marketplace_index.json) :
{
  "version": 1,
  "generated": "2025-01-01T00:00:00Z",
  "plugins": [
    {
      "id":          "chroma-shift",
      "name":        "Chroma Shift",
      "description": "Aberration chromatique paramétrable",
      "author":      "OpenShader Team",
      "version":     "1.2.0",
      "tags":        ["post-process", "color"],
      "url":         "https://raw.githubusercontent.com/.../chroma_shift.py",
      "sha256":      "abc123...",
      "rating":      4.2,
      "ratings_count": 18,
      "downloads":   342,
      "updated":     "2025-03-15T12:00:00Z",
      "min_version": "3.0"
    }
  ]
}

Stockage local :
  ~/.openshader/marketplace/cache.json   — cache index
  ~/.openshader/marketplace/ratings.json — notes locales
  ~/.openshader/plugins/                 — plugins installés
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from PyQt6.QtCore import (
    QObject, pyqtSignal, Qt, QTimer, QSize,
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QScrollArea,
    QGroupBox, QTabWidget, QTextEdit, QComboBox, QProgressBar,
    QFrame, QSizePolicy, QDialog, QDialogButtonBox,
)

from .logger import get_logger

log = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_URL    = "https://raw.githubusercontent.com/openshader-org/marketplace/main/index.json"
CACHE_TTL_S  = 3600          # 1h avant re-fetch de l'index
_FETCH_TIMEOUT_S = 8         # timeout réseau (secondes)
OPENSHADER_DIR = os.path.join(os.path.expanduser("~"), ".openshader")
MARKETPLACE_DIR = os.path.join(OPENSHADER_DIR, "marketplace")
PLUGINS_DIR     = os.path.join(OPENSHADER_DIR, "plugins")

# Imports conditionnels HTTP
try:
    import urllib.request
    import urllib.error
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False


# ═════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PluginEntry:
    id:            str
    name:          str
    description:   str
    author:        str
    version:       str
    tags:          list[str]
    url:           str
    sha256:        str
    rating:        float       = 0.0
    ratings_count: int         = 0
    downloads:     int         = 0
    updated:       str         = ""
    min_version:   str         = "1.0"
    installed:     bool        = False
    installed_ver: str         = ""
    update_available: bool     = False

    @property
    def tags_str(self) -> str:
        return "  ".join(f"#{t}" for t in self.tags)

    @property
    def stars(self) -> str:
        full  = int(self.rating)
        half  = 1 if (self.rating - full) >= 0.5 else 0
        empty = 5 - full - half
        return "★" * full + "½" * half + "☆" * empty

    @classmethod
    def from_dict(cls, d: dict) -> "PluginEntry":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            author=d.get("author", ""),
            version=d.get("version", "1.0.0"),
            tags=d.get("tags", []),
            url=d.get("url", ""),
            sha256=d.get("sha256", ""),
            rating=float(d.get("rating", 0.0)),
            ratings_count=int(d.get("ratings_count", 0)),
            downloads=int(d.get("downloads", 0)),
            updated=d.get("updated", ""),
            min_version=d.get("min_version", "1.0"),
        )


@dataclass
class LocalRating:
    plugin_id: str
    stars: int       # 1-5
    comment: str = ""
    timestamp: float = field(default_factory=time.time)


# ═════════════════════════════════════════════════════════════════════════════
#  Scan de sécurité statique (AST)
# ═════════════════════════════════════════════════════════════════════════════

# Patterns dangereux — imports / appels interdits
_DANGEROUS_IMPORTS = {
    "subprocess", "os.system", "pty", "socket",
    "ctypes", "cffi", "winreg", "signal",
}
_DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__",
    "open",          # autorisé dans la whitelist seulement
    "os.remove", "os.rmdir", "shutil.rmtree",
    "urllib.request.urlopen",   # réseau non autorisé dans plugins
}
_WHITELIST_OPEN_MODES = {"r", "rb"}  # open() en lecture seule toléré


class SecurityScanner:
    """
    Analyse statique AST d'un plugin Python.
    Retourne (ok: bool, issues: list[str]).
    """

    @staticmethod
    def scan(source: str, filename: str = "<plugin>") -> tuple[bool, list[str]]:
        issues: list[str] = []

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return False, [f"Erreur de syntaxe : {e}"]

        for node in ast.walk(tree):

            # Import suspects
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                        else ([node.module or ""] + [a.name for a in node.names])
                for name in names:
                    if any(name == d or name.startswith(d + ".") for d in _DANGEROUS_IMPORTS):
                        issues.append(f"L{node.lineno}: import interdit : {name!r}")

            # Appels suspects
            if isinstance(node, ast.Call):
                call_str = ""
                if isinstance(node.func, ast.Name):
                    call_str = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # ex: os.system
                    parts = []
                    n = node.func
                    while isinstance(n, ast.Attribute):
                        parts.insert(0, n.attr)
                        n = n.value
                    if isinstance(n, ast.Name):
                        parts.insert(0, n.id)
                    call_str = ".".join(parts)

                if call_str in _DANGEROUS_CALLS:
                    # Cas spécial open() en lecture
                    if call_str == "open":
                        mode = "r"
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                            mode = node.args[1].value
                        if mode not in _WHITELIST_OPEN_MODES:
                            issues.append(
                                f"L{node.lineno}: open() en écriture interdit (mode={mode!r})"
                            )
                    else:
                        issues.append(f"L{node.lineno}: appel interdit : {call_str}()")

        ok = len(issues) == 0
        return ok, issues


# ═════════════════════════════════════════════════════════════════════════════
#  MarketplaceIndex — fetch + cache de l'index JSON
# ═════════════════════════════════════════════════════════════════════════════

class MarketplaceIndex:
    """Gère le fetch, le cache local et le parse de l'index marketplace."""

    def __init__(self):
        os.makedirs(MARKETPLACE_DIR, exist_ok=True)
        os.makedirs(PLUGINS_DIR, exist_ok=True)
        self._cache_path    = os.path.join(MARKETPLACE_DIR, "cache.json")
        self._ratings_path  = os.path.join(MARKETPLACE_DIR, "ratings.json")
        self._installed_path = os.path.join(MARKETPLACE_DIR, "installed.json")
        self._plugins: list[PluginEntry] = []
        self._local_ratings: dict[str, LocalRating] = {}
        self._installed: dict[str, str] = {}   # id → version installée
        self._load_local_data()

    # ── Données locales ───────────────────────────────────────────────────────

    def _load_local_data(self):
        # Ratings locaux
        if os.path.isfile(self._ratings_path):
            try:
                data = json.loads(open(self._ratings_path).read())
                for pid, rd in data.items():
                    self._local_ratings[pid] = LocalRating(
                        plugin_id=pid, stars=rd.get("stars", 0),
                        comment=rd.get("comment", ""),
                        timestamp=rd.get("timestamp", 0),
                    )
            except Exception:
                pass

        # Plugins installés
        if os.path.isfile(self._installed_path):
            try:
                self._installed = json.loads(open(self._installed_path).read())
            except Exception:
                pass

    def _save_ratings(self):
        data = {pid: {"stars": r.stars, "comment": r.comment, "timestamp": r.timestamp}
                for pid, r in self._local_ratings.items()}
        with open(self._ratings_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_installed(self):
        with open(self._installed_path, "w") as f:
            json.dump(self._installed, f, indent=2)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        if not os.path.isfile(self._cache_path):
            return False
        age = time.time() - os.path.getmtime(self._cache_path)
        return age < CACHE_TTL_S

    def _load_from_cache(self) -> bool:
        if not os.path.isfile(self._cache_path):
            return False
        try:
            data = json.loads(open(self._cache_path).read())
            self._parse_index(data)
            return True
        except Exception:
            return False

    def _save_cache(self, data: dict):
        with open(self._cache_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch(self, force: bool = False) -> list[PluginEntry]:
        """
        Retourne la liste des plugins de la marketplace.
        Utilise le cache si frais, sinon re-fetch l'index distant.
        """
        if not force and self._is_cache_fresh():
            if self._load_from_cache():
                log.debug("Marketplace : index chargé depuis le cache")
                return self._plugins

        if not _HTTP_OK:
            self._load_from_cache()
            return self._plugins

        try:
            req = urllib.request.Request(
                INDEX_URL,
                headers={"User-Agent": "OpenShader/6.0"},
            )
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            self._save_cache(data)
            self._parse_index(data)
            log.info("Marketplace : index mis à jour (%d plugins)", len(self._plugins))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Index pas encore publié — silencieux, pas d'alarme
                log.debug("Marketplace : index distant introuvable (404) — "
                          "la marketplace sera disponible lors d'une prochaine version")
            else:
                log.warning("Marketplace : fetch échoué (HTTP %s), fallback cache", e.code)
            self._load_from_cache()
        except urllib.error.URLError as e:
            # Pas de réseau, timeout, etc. — avertissement discret
            log.debug("Marketplace : pas d'accès réseau (%s), fallback cache", e.reason)
            self._load_from_cache()
        except Exception as e:
            log.warning("Marketplace : fetch inattendu (%s), fallback cache", e)
            self._load_from_cache()

        return self._plugins

    def _parse_index(self, data: dict):
        entries = []
        for d in data.get("plugins", []):
            entry = PluginEntry.from_dict(d)
            # Applique les données locales
            if entry.id in self._installed:
                entry.installed = True
                entry.installed_ver = self._installed[entry.id]
                # Comparaison simplifiée des versions
                entry.update_available = (
                    entry.installed_ver != entry.version
                )
            if entry.id in self._local_ratings:
                entry.rating = float(self._local_ratings[entry.id].stars)
            entries.append(entry)
        self._plugins = entries

    # ── Actions ───────────────────────────────────────────────────────────────

    def get_plugins(self, query: str = "", tag: str = "") -> list[PluginEntry]:
        q = query.lower()
        result = self._plugins
        if q:
            result = [p for p in result
                      if q in p.name.lower() or q in p.description.lower()
                      or q in p.author.lower() or any(q in t for t in p.tags)]
        if tag and tag != "Tous":
            result = [p for p in result if tag in p.tags]
        return result

    def all_tags(self) -> list[str]:
        tags: set[str] = set()
        for p in self._plugins:
            tags.update(p.tags)
        return sorted(tags)

    def get_updates(self) -> list[PluginEntry]:
        return [p for p in self._plugins if p.update_available]

    def mark_installed(self, plugin_id: str, version: str):
        self._installed[plugin_id] = version
        self._save_installed()
        for p in self._plugins:
            if p.id == plugin_id:
                p.installed = True
                p.installed_ver = version
                p.update_available = False

    def mark_uninstalled(self, plugin_id: str):
        self._installed.pop(plugin_id, None)
        self._save_installed()
        for p in self._plugins:
            if p.id == plugin_id:
                p.installed = False
                p.installed_ver = ""
                p.update_available = False

    def save_rating(self, plugin_id: str, stars: int, comment: str = ""):
        self._local_ratings[plugin_id] = LocalRating(
            plugin_id=plugin_id, stars=stars, comment=comment
        )
        self._save_ratings()

    def get_local_rating(self, plugin_id: str) -> Optional[LocalRating]:
        return self._local_ratings.get(plugin_id)


# ═════════════════════════════════════════════════════════════════════════════
#  MarketplaceManager — QObject principal
# ═════════════════════════════════════════════════════════════════════════════

class MarketplaceManager(QObject):
    """
    Moteur de la marketplace.

    Signaux :
      index_ready(list)       — index chargé/rafraîchi
      install_progress(str, int) — (plugin_id, %)
      install_done(str, bool, str) — (plugin_id, ok, message)
      update_available(int)   — nb de mises à jour disponibles
    """

    index_ready       = pyqtSignal(list)
    install_progress  = pyqtSignal(str, int)
    install_done      = pyqtSignal(str, bool, str)
    update_available  = pyqtSignal(int)

    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self._pm    = plugin_manager   # PluginManager existant
        self._index = MarketplaceIndex()
        self._lock  = threading.Lock()

    # ── Index ─────────────────────────────────────────────────────────────────

    def refresh(self, force: bool = False):
        """Lance un fetch de l'index en arrière-plan.
        Au premier lancement (pas de cache), attend quelques secondes
        pour ne pas ralentir le démarrage ni afficher un warning immédiat.
        """
        def _run():
            # Si pas de cache et pas forcé, laisse l'app démarrer d'abord
            if not force and not os.path.isfile(self._index._cache_path):
                time.sleep(5)
            plugins = self._index.fetch(force=force)
            self.index_ready.emit(plugins)
            updates = self._index.get_updates()
            if updates:
                self.update_available.emit(len(updates))

        threading.Thread(target=_run, daemon=True, name="MarketplaceFetch").start()

    def get_plugins(self, query: str = "", tag: str = "") -> list[PluginEntry]:
        return self._index.get_plugins(query, tag)

    def all_tags(self) -> list[str]:
        return self._index.all_tags()

    # ── Installation ──────────────────────────────────────────────────────────

    def install(self, entry: PluginEntry):
        """Télécharge, scanne et installe un plugin en arrière-plan."""
        threading.Thread(
            target=self._do_install, args=(entry,),
            daemon=True, name=f"MarketplaceInstall-{entry.id}",
        ).start()

    def _do_install(self, entry: PluginEntry):
        pid = entry.id
        self.install_progress.emit(pid, 5)

        # 1. Téléchargement
        try:
            self.install_progress.emit(pid, 20)
            req = urllib.request.Request(
                entry.url,
                headers={"User-Agent": "OpenShader/6.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                source_bytes = r.read()
            self.install_progress.emit(pid, 50)
        except Exception as e:
            self.install_done.emit(pid, False, f"Téléchargement échoué : {e}")
            return

        # 2. Vérification SHA-256
        if entry.sha256:
            digest = hashlib.sha256(source_bytes).hexdigest()
            if digest != entry.sha256:
                self.install_done.emit(
                    pid, False,
                    f"Vérification SHA-256 échouée.\n"
                    f"Attendu : {entry.sha256}\n"
                    f"Reçu    : {digest}"
                )
                return
        self.install_progress.emit(pid, 65)

        # 3. Scan statique AST
        try:
            source_str = source_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            self.install_done.emit(pid, False, f"Encodage invalide : {e}")
            return

        ok, issues = SecurityScanner.scan(source_str, filename=f"{pid}.py")
        self.install_progress.emit(pid, 80)
        if not ok:
            msg = "Scan de sécurité échoué :\n" + "\n".join(f"  • {i}" for i in issues)
            self.install_done.emit(pid, False, msg)
            return

        # 4. Écriture sur disque
        os.makedirs(PLUGINS_DIR, exist_ok=True)
        dest = os.path.join(PLUGINS_DIR, f"{pid}.py")
        try:
            with open(dest, "wb") as f:
                f.write(source_bytes)
        except OSError as e:
            self.install_done.emit(pid, False, f"Écriture échouée : {e}")
            return
        self.install_progress.emit(pid, 90)

        # 5. Chargement dans le PluginManager
        try:
            self._pm.add_search_dir(PLUGINS_DIR)
            self._pm.scan_and_load()
        except Exception as e:
            log.warning("Chargement plugin post-install : %s", e)

        self._index.mark_installed(pid, entry.version)
        self.install_progress.emit(pid, 100)
        self.install_done.emit(pid, True, f"✓ {entry.name} v{entry.version} installé")
        log.info("Plugin marketplace installé : %s v%s", entry.name, entry.version)

    # ── Désinstallation ───────────────────────────────────────────────────────

    def uninstall(self, entry: PluginEntry) -> tuple[bool, str]:
        dest = os.path.join(PLUGINS_DIR, f"{entry.id}.py")
        try:
            if os.path.isfile(dest):
                os.remove(dest)
            self._index.mark_uninstalled(entry.id)
            log.info("Plugin désinstallé : %s", entry.name)
            return True, f"✓ {entry.name} désinstallé"
        except OSError as e:
            return False, f"Erreur : {e}"

    # ── Mises à jour ──────────────────────────────────────────────────────────

    def update_all(self):
        """Installe toutes les mises à jour disponibles."""
        for entry in self._index.get_updates():
            self.install(entry)

    # ── Notation ─────────────────────────────────────────────────────────────

    def rate(self, plugin_id: str, stars: int, comment: str = ""):
        stars = max(1, min(5, stars))
        self._index.save_rating(plugin_id, stars, comment)
        log.debug("Note sauvegardée : %s → %d★", plugin_id, stars)

    def get_local_rating(self, plugin_id: str) -> Optional[LocalRating]:
        return self._index.get_local_rating(plugin_id)


# ═════════════════════════════════════════════════════════════════════════════
#  MarketplaceBrowser — widget PyQt6
# ═════════════════════════════════════════════════════════════════════════════

_BG    = "#0d0f16"
_SURF  = "#12141e"
_BORD  = "#1e2235"
_TEXT  = "#c0c8e0"
_MUTED = "#5a6080"
_ACC   = "#4e7fff"
_ACC2  = "#50e0a0"
_WARN  = "#e0804e"
_SANS  = "Segoe UI, Arial, sans-serif"
_MONO  = "Cascadia Code, Consolas, monospace"


def _btn(label: str, color: str = _ACC) -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton {{background:{_SURF};color:{color};border:1px solid {color}44;"
        f"border-radius:3px;padding:3px 10px;font:9px '{_SANS}';}}"
        f"QPushButton:hover {{background:{color}22;}}"
        f"QPushButton:disabled {{color:{_MUTED};border-color:{_BORD};}}"
    )
    return b


class _StarWidget(QWidget):
    """Sélecteur d'étoiles cliquable (1–5)."""
    rating_changed = pyqtSignal(int)

    def __init__(self, value: int = 0, interactive: bool = True, parent=None):
        super().__init__(parent)
        self._value = value
        self._interactive = interactive
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._btns: list[QPushButton] = []
        for i in range(1, 6):
            b = QPushButton("★")
            b.setFixedSize(22, 22)
            b.setFlat(True)
            b.setStyleSheet("QPushButton{border:none;font:14px;padding:0;}")
            if interactive:
                b.clicked.connect(lambda _, v=i: self._set(v))
            self._btns.append(b)
            layout.addWidget(b)
        self._refresh()

    def _set(self, v: int):
        self._value = v
        self._refresh()
        self.rating_changed.emit(v)

    def _refresh(self):
        for i, b in enumerate(self._btns):
            b.setStyleSheet(
                f"QPushButton{{border:none;font:14px;padding:0;"
                f"color:{'#f0c040' if i < self._value else _MUTED};}}"
            )

    @property
    def value(self) -> int:
        return self._value


class _PluginCard(QFrame):
    """Carte compacte d'un plugin dans la liste."""

    install_requested   = pyqtSignal(object)   # PluginEntry
    uninstall_requested = pyqtSignal(object)

    def __init__(self, entry: PluginEntry, parent=None):
        super().__init__(parent)
        self._entry = entry
        self.setStyleSheet(
            f"QFrame{{background:{_SURF};border:1px solid {_BORD};"
            f"border-radius:5px;padding:6px;}}"
            f"QFrame:hover{{border-color:{_ACC}55;}}"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(3)

        # Ligne titre + badge installé
        row1 = QHBoxLayout()
        name_lbl = QLabel(entry.name)
        name_lbl.setStyleSheet(f"color:{_TEXT};font:bold 10px '{_SANS}';")
        row1.addWidget(name_lbl)
        if entry.installed:
            badge = QLabel("✓ Installé")
            badge.setStyleSheet(
                f"color:{_ACC2};background:{_ACC2}18;border:1px solid {_ACC2}44;"
                f"border-radius:3px;padding:1px 5px;font:8px '{_SANS}';"
            )
            row1.addWidget(badge)
        if entry.update_available:
            upd = QLabel("↑ MàJ")
            upd.setStyleSheet(
                f"color:{_WARN};background:{_WARN}18;border:1px solid {_WARN}44;"
                f"border-radius:3px;padding:1px 5px;font:8px '{_SANS}';"
            )
            row1.addWidget(upd)
        row1.addStretch()
        ver_lbl = QLabel(f"v{entry.version}")
        ver_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_MONO}';")
        row1.addWidget(ver_lbl)
        lay.addLayout(row1)

        # Description
        desc = QLabel(entry.description[:90] + ("…" if len(entry.description) > 90 else ""))
        desc.setStyleSheet(f"color:{_MUTED};font:9px '{_SANS}';")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        # Ligne auteur + étoiles + tags
        row2 = QHBoxLayout()
        author_lbl = QLabel(f"par {entry.author}")
        author_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        row2.addWidget(author_lbl)
        stars = _StarWidget(round(entry.rating), interactive=False)
        stars.setFixedHeight(18)
        row2.addWidget(stars)
        cnt_lbl = QLabel(f"({entry.ratings_count})")
        cnt_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        row2.addWidget(cnt_lbl)
        row2.addStretch()
        for tag in entry.tags[:3]:
            t = QLabel(f"#{tag}")
            t.setStyleSheet(
                f"color:{_ACC};background:{_ACC}18;border-radius:3px;"
                f"padding:1px 4px;font:7px '{_SANS}';"
            )
            row2.addWidget(t)
        lay.addLayout(row2)

    @property
    def entry(self) -> PluginEntry:
        return self._entry


class MarketplaceBrowser(QWidget):
    """
    Widget complet Marketplace : Browse / Updates / Installed.
    S'intègre comme onglet dans le Plugin Manager existant.
    """

    def __init__(self, manager: MarketplaceManager, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._all_plugins: list[PluginEntry] = []
        self._selected: Optional[PluginEntry] = None
        self._install_buttons: dict[str, QPushButton] = {}
        self._progress_bars: dict[str, QProgressBar] = {}

        self._build_ui()

        # Connexions
        manager.index_ready.connect(self._on_index_ready)
        manager.install_progress.connect(self._on_install_progress)
        manager.install_done.connect(self._on_install_done)

        # Refresh initial
        manager.refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre de recherche + filtre tag ──────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 8, 8, 6)
        top_bar.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Rechercher un plugin…")
        self._search.setStyleSheet(
            f"QLineEdit{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:4px;padding:4px 8px;font:9px '{_SANS}';}}"
            f"QLineEdit:focus{{border-color:{_ACC}88;}}"
        )
        self._search.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self._search, 1)

        self._tag_combo = QComboBox()
        self._tag_combo.addItem("Tous")
        self._tag_combo.setStyleSheet(
            f"QComboBox{{background:{_SURF};color:{_MUTED};border:1px solid {_BORD};"
            f"border-radius:4px;padding:3px 8px;font:9px '{_SANS}';}}"
        )
        self._tag_combo.currentTextChanged.connect(self._apply_filter)
        top_bar.addWidget(self._tag_combo)

        btn_refresh = _btn("↻ Rafraîchir", _MUTED)
        btn_refresh.clicked.connect(lambda: self._mgr.refresh(force=True))
        top_bar.addWidget(btn_refresh)

        root.addLayout(top_bar)

        # ── Tabs ──────────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {_BORD};background:{_BG};}}"
            f"QTabBar::tab{{background:{_SURF};color:{_MUTED};padding:5px 14px;"
            f"border:1px solid {_BORD};margin-right:2px;font:9px '{_SANS}';}}"
            f"QTabBar::tab:selected{{color:{_TEXT};border-bottom:2px solid {_ACC};}}"
        )

        # Tab Browse
        self._browse_tab = QWidget()
        self._tabs.addTab(self._browse_tab, "📦 Browse")
        self._build_browse_tab()

        # Tab Mises à jour
        self._updates_tab = QWidget()
        self._tabs.addTab(self._updates_tab, "↑ Mises à jour")
        self._build_updates_tab()

        # Tab Installés
        self._installed_tab = QWidget()
        self._tabs.addTab(self._installed_tab, "✓ Installés")
        self._build_installed_tab()

        root.addWidget(self._tabs, 1)

    def _build_browse_tab(self):
        lay = QHBoxLayout(self._browse_tab)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # Liste des plugins (scroll)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_BG};}}")
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(5)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        lay.addWidget(scroll, 1)

        # Panneau de détail
        self._detail_panel = self._build_detail_panel()
        lay.addWidget(self._detail_panel, 1)

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"QWidget{{background:{_SURF};border-radius:5px;}}")
        panel.setMinimumWidth(260)
        self._detail_layout = QVBoxLayout(panel)
        self._detail_layout.setContentsMargins(12, 12, 12, 12)
        self._detail_layout.setSpacing(8)

        placeholder = QLabel("Sélectionnez un plugin\npour voir les détails")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color:{_MUTED};font:10px '{_SANS}';")
        self._detail_layout.addWidget(placeholder)
        self._detail_layout.addStretch()
        return panel

    def _build_updates_tab(self):
        lay = QVBoxLayout(self._updates_tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self._updates_label = QLabel("Vérification des mises à jour…")
        self._updates_label.setStyleSheet(f"color:{_MUTED};font:9px '{_SANS}';")
        top.addWidget(self._updates_label)
        top.addStretch()
        btn_all = _btn("↑ Tout mettre à jour", _WARN)
        btn_all.clicked.connect(self._mgr.update_all)
        top.addWidget(btn_all)
        lay.addLayout(top)

        self._updates_scroll = QScrollArea()
        self._updates_scroll.setWidgetResizable(True)
        self._updates_scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_BG};}}")
        self._updates_inner = QWidget()
        self._updates_inner_lay = QVBoxLayout(self._updates_inner)
        self._updates_inner_lay.addStretch()
        self._updates_scroll.setWidget(self._updates_inner)
        lay.addWidget(self._updates_scroll, 1)

    def _build_installed_tab(self):
        lay = QVBoxLayout(self._installed_tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._installed_scroll = QScrollArea()
        self._installed_scroll.setWidgetResizable(True)
        self._installed_scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_BG};}}")
        self._installed_inner = QWidget()
        self._installed_inner_lay = QVBoxLayout(self._installed_inner)
        self._installed_inner_lay.addStretch()
        self._installed_scroll.setWidget(self._installed_inner)
        lay.addWidget(self._installed_scroll, 1)

    # ── Rafraîchissement de la liste ──────────────────────────────────────────

    def _on_index_ready(self, plugins: list):
        self._all_plugins = plugins

        # Tags
        current_tag = self._tag_combo.currentText()
        self._tag_combo.blockSignals(True)
        self._tag_combo.clear()
        self._tag_combo.addItem("Tous")
        for tag in self._mgr.all_tags():
            self._tag_combo.addItem(tag)
        idx = self._tag_combo.findText(current_tag)
        self._tag_combo.setCurrentIndex(max(0, idx))
        self._tag_combo.blockSignals(False)

        self._apply_filter()
        self._refresh_updates_tab()
        self._refresh_installed_tab()

    def _apply_filter(self):
        query = self._search.text().strip()
        tag   = self._tag_combo.currentText()
        visible = self._mgr.get_plugins(query, tag)
        self._repopulate_list(visible)

    def _repopulate_list(self, plugins: list[PluginEntry]):
        # Vide la liste
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for entry in plugins:
            card = _PluginCard(entry)
            card.mousePressEvent = lambda ev, e=entry: self._show_detail(e)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

    def _refresh_updates_tab(self):
        updates = [p for p in self._all_plugins if p.update_available]
        self._updates_label.setText(
            f"{len(updates)} mise{'s' if len(updates) != 1 else ''} à jour disponible{'s' if len(updates) != 1 else ''}"
            if updates else "Tous les plugins sont à jour ✓"
        )
        while self._updates_inner_lay.count() > 1:
            item = self._updates_inner_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for entry in updates:
            row = QHBoxLayout()
            lbl = QLabel(f"{entry.name}  {entry.installed_ver} → {entry.version}")
            lbl.setStyleSheet(f"color:{_TEXT};font:9px '{_SANS}';")
            row.addWidget(lbl, 1)
            pb = QProgressBar()
            pb.setMaximum(100); pb.setValue(0)
            pb.setFixedWidth(80); pb.setFixedHeight(8)
            pb.setStyleSheet(f"QProgressBar{{background:{_BORD};border-radius:4px;}}"
                             f"QProgressBar::chunk{{background:{_ACC2};border-radius:4px;}}")
            pb.hide()
            row.addWidget(pb)
            self._progress_bars[entry.id] = pb
            b = _btn("↑ Mettre à jour", _WARN)
            b.clicked.connect(lambda _, e=entry: self._mgr.install(e))
            row.addWidget(b)
            self._updates_inner_lay.insertLayout(
                self._updates_inner_lay.count() - 1, row)

    def _refresh_installed_tab(self):
        installed = [p for p in self._all_plugins if p.installed]
        while self._installed_inner_lay.count() > 1:
            item = self._installed_inner_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for entry in installed:
            row = QHBoxLayout()
            lbl = QLabel(f"{entry.name}  v{entry.installed_ver}")
            lbl.setStyleSheet(f"color:{_TEXT};font:9px '{_SANS}';")
            row.addWidget(lbl, 1)
            b = _btn("✕ Désinstaller", _WARN)
            b.clicked.connect(lambda _, e=entry: self._do_uninstall(e))
            row.addWidget(b)
            self._installed_inner_lay.insertLayout(
                self._installed_inner_lay.count() - 1, row)

    # ── Panneau de détail ─────────────────────────────────────────────────────

    def _show_detail(self, entry: PluginEntry):
        self._selected = entry

        # Vide le panneau
        while self._detail_layout.count():
            item = self._detail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # récursif minimal
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        # ── En-tête ────────────────────────────────────────────────────────
        name_lbl = QLabel(entry.name)
        name_lbl.setStyleSheet(f"color:{_TEXT};font:bold 12px '{_SANS}';")
        self._detail_layout.addWidget(name_lbl)

        meta_lbl = QLabel(
            f"v{entry.version}  ·  par {entry.author}  ·  "
            f"{entry.downloads} télécharg."
        )
        meta_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        self._detail_layout.addWidget(meta_lbl)

        # Tags
        tags_row = QHBoxLayout()
        for tag in entry.tags:
            t = QLabel(f"#{tag}")
            t.setStyleSheet(
                f"color:{_ACC};background:{_ACC}18;border-radius:3px;"
                f"padding:1px 5px;font:8px '{_SANS}';"
            )
            tags_row.addWidget(t)
        tags_row.addStretch()
        self._detail_layout.addLayout(tags_row)

        # Description
        desc = QLabel(entry.description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{_TEXT};font:9px '{_SANS}';margin:6px 0;")
        self._detail_layout.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{_BORD};")
        self._detail_layout.addWidget(sep)

        # Étoiles globales
        stars_row = QHBoxLayout()
        stars_row.addWidget(QLabel("Note globale :"))
        sw = _StarWidget(round(entry.rating), interactive=False)
        stars_row.addWidget(sw)
        cnt = QLabel(f"{entry.rating:.1f}/5  ({entry.ratings_count} avis)")
        cnt.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        stars_row.addWidget(cnt)
        stars_row.addStretch()
        self._detail_layout.addLayout(stars_row)

        # Notation locale
        local = self._mgr.get_local_rating(entry.id)
        rate_lbl = QLabel("Votre note :")
        rate_lbl.setStyleSheet(f"color:{_MUTED};font:9px '{_SANS}';")
        self._detail_layout.addWidget(rate_lbl)
        my_stars = _StarWidget(local.stars if local else 0, interactive=True)
        self._detail_layout.addWidget(my_stars)

        comment_edit = QLineEdit(local.comment if local else "")
        comment_edit.setPlaceholderText("Commentaire (optionnel)…")
        comment_edit.setStyleSheet(
            f"QLineEdit{{background:{_BG};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 6px;font:8px '{_SANS}';}}"
        )
        self._detail_layout.addWidget(comment_edit)

        btn_rate = _btn("Enregistrer la note", _ACC)
        btn_rate.clicked.connect(
            lambda: self._mgr.rate(entry.id, my_stars.value, comment_edit.text())
        )
        self._detail_layout.addWidget(btn_rate)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{_BORD};")
        self._detail_layout.addWidget(sep2)

        # Bouton install / désinstall
        pb = QProgressBar()
        pb.setMaximum(100); pb.setValue(0)
        pb.setFixedHeight(6)
        pb.setStyleSheet(
            f"QProgressBar{{background:{_BORD};border-radius:3px;border:none;}}"
            f"QProgressBar::chunk{{background:{_ACC2};border-radius:3px;}}"
        )
        pb.hide()
        self._detail_layout.addWidget(pb)
        self._progress_bars[entry.id] = pb

        if entry.installed:
            row_btns = QHBoxLayout()
            if entry.update_available:
                btn_upd = _btn("↑ Mettre à jour", _WARN)
                btn_upd.clicked.connect(lambda: self._mgr.install(entry))
                row_btns.addWidget(btn_upd)
            btn_rm = _btn("✕ Désinstaller", _WARN)
            btn_rm.clicked.connect(lambda: self._do_uninstall(entry))
            row_btns.addWidget(btn_rm)
            row_btns.addStretch()
            self._detail_layout.addLayout(row_btns)
        else:
            btn_inst = _btn("⬇ Installer", _ACC2)
            btn_inst.clicked.connect(lambda: self._mgr.install(entry))
            self._install_buttons[entry.id] = btn_inst
            self._detail_layout.addWidget(btn_inst)

        # Mis à jour le
        if entry.updated:
            upd_lbl = QLabel(f"Dernière mise à jour : {entry.updated[:10]}")
            upd_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
            self._detail_layout.addWidget(upd_lbl)

        self._detail_layout.addStretch()

    # ── Callbacks d'installation ──────────────────────────────────────────────

    def _on_install_progress(self, plugin_id: str, pct: int):
        pb = self._progress_bars.get(plugin_id)
        if pb:
            pb.show()
            pb.setValue(pct)
        btn = self._install_buttons.get(plugin_id)
        if btn:
            btn.setEnabled(False)
            btn.setText(f"Installation… {pct}%")

    def _on_install_done(self, plugin_id: str, ok: bool, message: str):
        pb = self._progress_bars.get(plugin_id)
        if pb:
            if ok:
                pb.setValue(100)
                QTimer.singleShot(1500, pb.hide)
            else:
                pb.hide()

        btn = self._install_buttons.get(plugin_id)
        if btn:
            btn.setEnabled(True)
            btn.setText("⬇ Installer" if not ok else "✓ Installé")

        # Recharge la vue
        self._mgr.refresh()

        # Popup erreur si échec
        if not ok:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Erreur d'installation", message)

    def _do_uninstall(self, entry: PluginEntry):
        from PyQt6.QtWidgets import QMessageBox
        r = QMessageBox.question(
            self, "Désinstaller",
            f"Désinstaller « {entry.name} » ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r == QMessageBox.StandardButton.Yes:
            ok, msg = self._mgr.uninstall(entry)
            if not ok:
                QMessageBox.warning(self, "Erreur", msg)
            self._mgr.refresh()
