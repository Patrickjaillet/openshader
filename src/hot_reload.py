"""
hot_reload.py
-------------
v2.1 — Hot-Reload de shaders via watchdog.

Surveille les fichiers .glsl / .st sur disque et notifie MainWindow
dès qu'un fichier est modifié par un éditeur externe.

Dépendance optionnelle : watchdog >= 3.0
Si watchdog n'est pas installé, le module se dégrade gracieusement
(aucune erreur — le hot-reload est simplement désactivé).

Usage :
    from .hot_reload import HotReloadManager
    mgr = HotReloadManager(parent_qobject)
    mgr.file_changed.connect(lambda path: self._reload_shader_from_path(path))
    mgr.watch(path_to_shader_file)
    mgr.unwatch(path_to_shader_file)
    mgr.stop()
"""

import os
import time
import threading
from PyQt6.QtCore import QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)

# ── Tentative d'import watchdog ──────────────────────────────────────────────
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    log.warning(
        "watchdog non installé — hot-reload désactivé. "
        "Installez-le avec : pip install watchdog>=3.0"
    )


# ── Debounce helper ───────────────────────────────────────────────────────────

class _Debouncer:
    """
    Retarde l'émission d'un signal de DEBOUNCE_MS millisecondes
    après le dernier événement pour un même chemin.
    """
    DEBOUNCE_MS = 150  # ms

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(self, path: str, callback):
        """Annule le timer précédent pour ce chemin et en crée un nouveau."""
        with self._lock:
            old = self._timers.get(path)
            if old:
                old.cancel()
            t = threading.Timer(self.DEBOUNCE_MS / 1000.0, callback)
            t.daemon = True
            t.start()
            self._timers[path] = t

    def cancel_all(self):
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()


# ── Gestionnaire watchdog ──────────────────────────────────────────────────────

if _WATCHDOG_AVAILABLE:
    class _ShaderEventHandler(FileSystemEventHandler):
        """
        Écouteur watchdog : filtre les événements sur les fichiers shader
        et déclenche le debouncer.
        """
        SHADER_EXTENSIONS = {'.glsl', '.st', '.frag', '.vert', '.trans'}
        IGNORED_SUFFIXES  = ('.swp', '.swo', '.bak', '~', '.tmp')

        def __init__(self, debouncer: _Debouncer, on_change):
            super().__init__()
            self._debouncer = debouncer
            self._on_change = on_change

        def _is_shader(self, path: str) -> bool:
            lower = path.lower()
            if any(lower.endswith(s) for s in self.IGNORED_SUFFIXES):
                return False
            _, ext = os.path.splitext(lower)
            return ext in self.SHADER_EXTENSIONS

        def on_modified(self, event):
            if not event.is_directory and self._is_shader(event.src_path):
                path = os.path.abspath(event.src_path)
                log.debug("watchdog: modified → %s", path)
                self._debouncer.schedule(path, lambda p=path: self._on_change(p))

        def on_moved(self, event):
            # Certains éditeurs (vim, emacs) sauvegardent via rename
            if not event.is_directory and self._is_shader(event.dest_path):
                path = os.path.abspath(event.dest_path)
                log.debug("watchdog: moved → %s", path)
                self._debouncer.schedule(path, lambda p=path: self._on_change(p))


class HotReloadManager(QObject):
    """
    Gère le hot-reload des shaders surveillés.

    Signaux
    -------
    file_changed(str)
        Émis (thread-safe via Qt) quand un fichier shader surveillé est modifié.
        Le paramètre est le chemin absolu du fichier.
    status_changed(bool)
        Émis quand le hot-reload est activé/désactivé.
    """

    file_changed   = pyqtSignal(str)   # chemin absolu du fichier modifié
    status_changed = pyqtSignal(bool)  # True = actif

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._enabled   = False
        self._debouncer = _Debouncer()
        self._watched_dirs: dict[str, set[str]] = {}   # dir → set(abs_paths)

        if _WATCHDOG_AVAILABLE:
            self._observer = Observer()
            self._handler  = _ShaderEventHandler(self._debouncer, self._on_file_changed)
            self._observer.start()
            log.info("HotReloadManager : watchdog démarré")
        else:
            self._observer = None
            self._handler  = None

    # ── API publique ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True si watchdog est installé et fonctionnel."""
        return _WATCHDOG_AVAILABLE

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool):
        """Active ou désactive le hot-reload sans modifier les watchers."""
        if value != self._enabled:
            self._enabled = value
            self.status_changed.emit(value)
            log.info("HotReloadManager : %s", "activé" if value else "désactivé")

    def watch(self, path: str):
        """
        Commence à surveiller *path* (fichier shader).
        Crée automatiquement un watcher sur le dossier parent.
        """
        if not _WATCHDOG_AVAILABLE or not path:
            return
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if not directory:
            return

        if directory not in self._watched_dirs:
            self._watched_dirs[directory] = set()
            try:
                self._observer.schedule(self._handler, directory, recursive=False)
                log.debug("HotReloadManager : surveillance de %s", directory)
            except Exception as e:
                log.warning("HotReloadManager : impossible de surveiller %s — %s", directory, e)

        self._watched_dirs[directory].add(path)

    def unwatch(self, path: str):
        """Retire *path* des chemins surveillés."""
        if not _WATCHDOG_AVAILABLE or not path:
            return
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        paths_in_dir = self._watched_dirs.get(directory, set())
        paths_in_dir.discard(path)
        # Note : on ne retire pas le watcher de dossier tant qu'il reste
        # d'autres fichiers dans ce dossier — watchdog ne supporte pas
        # le unschedule sélectif facilement ; on filtre dans _on_file_changed.

    def watch_project(self, shader_paths: list[str]):
        """
        Met à jour la liste complète des shaders surveillés.
        Ajoute les nouveaux, ne retire pas les anciens (peu coûteux).
        """
        for p in shader_paths:
            if p:
                self.watch(p)

    def stop(self):
        """Arrête proprement le thread watchdog."""
        self._debouncer.cancel_all()
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                log.warning("HotReloadManager stop : %s", e)
        log.info("HotReloadManager : arrêté")

    # ── Interne ──────────────────────────────────────────────────────────────

    def _on_file_changed(self, path: str):
        """
        Appelé depuis le thread debouncer (pas le thread Qt).
        Vérifie que le chemin est bien surveillé et actif,
        puis émet file_changed via le mécanisme Qt thread-safe.
        """
        if not self._enabled:
            return
        # Vérifie que ce chemin est dans la liste des fichiers surveillés
        directory = os.path.dirname(path)
        tracked   = self._watched_dirs.get(directory, set())
        if path not in tracked:
            return

        # Attend que le fichier soit stable (write terminé)
        try:
            _wait_stable(path, timeout_ms=500)
        except OSError:
            return

        log.info("Hot-reload : %s", os.path.basename(path))
        # Émission thread-safe via Qt (queued connection)
        self.file_changed.emit(path)


# ── Utilitaire : attente stabilisation fichier ────────────────────────────────

def _wait_stable(path: str, timeout_ms: int = 500, interval_ms: int = 50):
    """
    Attend que la taille du fichier soit stable (écriture terminée).
    Lève OSError si le fichier disparaît.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    last_size = -1
    while time.monotonic() < deadline:
        if not os.path.exists(path):
            raise OSError(f"Fichier disparu : {path}")
        sz = os.path.getsize(path)
        if sz == last_size and sz > 0:
            return
        last_size = sz
        time.sleep(interval_ms / 1000.0)
