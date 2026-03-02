"""
shortcut_manager.py
-------------------
Gestionnaire de raccourcis clavier entièrement configurables — OpenShader v2.6.

Architecture :
  • ACTION_REGISTRY  : catalogue de toutes les actions avec leur raccourci par défaut
  • ShortcutManager  : charge/sauvegarde les bindings (QSettings + JSON),
                       applique les QKeySequence aux QAction et QShortcut,
                       détecte les conflits
  • ShortcutEditor   : dialogue éditeur style VS Code (filtrage, capture de touche,
                       highlighting des conflits, reset individuel/global)
  • BUILTIN_PROFILES : profils prédéfinis "Default", "Blender-like", "Premiere-like"

Intégration dans MainWindow :
    self.shortcut_mgr = ShortcutManager(self)
    self.shortcut_mgr.register_action("play_pause", action_obj)
    self.shortcut_mgr.apply_all()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFrame, QComboBox, QFileDialog,
    QMessageBox, QSizePolicy, QWidget, QApplication
)
from PyQt6.QtCore  import Qt, QSettings, pyqtSignal, QObject, QTimer
from PyQt6.QtGui   import QAction, QKeySequence, QColor, QFont, QKeyEvent

from .logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Catalogue des actions (id → métadonnées + raccourci par défaut)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionDef:
    """Définition d'une action dans le catalogue."""
    id:          str
    label:       str
    category:    str
    default_key: str          # chaîne QKeySequence ("Ctrl+S", "F5", …)
    description: str = ""


# Catalogue complet — toutes les actions configurables de l'application
ACTION_REGISTRY: dict[str, ActionDef] = {a.id: a for a in [

    # ── Fichier ──────────────────────────────────────────────────────────────
    ActionDef("new_project",       "Nouveau projet…",           "Fichier",   "Ctrl+N"),
    ActionDef("open_project",      "Ouvrir projet…",            "Fichier",   "Ctrl+O"),
    ActionDef("save_project",      "Enregistrer",               "Fichier",   "Ctrl+S"),
    ActionDef("save_project_as",   "Enregistrer sous…",         "Fichier",   "Ctrl+Shift+S"),
    ActionDef("export_video",      "Export vidéo haute qualité…","Fichier",  "Ctrl+Shift+E"),
    ActionDef("screenshot",        "Capture d'écran",           "Fichier",   "F12"),
    ActionDef("quit",              "Quitter",                   "Fichier",   "Ctrl+Q"),

    # ── Édition ──────────────────────────────────────────────────────────────
    ActionDef("undo",              "Annuler",                   "Édition",   "Ctrl+Z",
              "Annuler la dernière action"),
    ActionDef("redo",              "Rétablir",                  "Édition",   "Ctrl+Y",
              "Rétablir l'action annulée"),

    # ── Lecture ──────────────────────────────────────────────────────────────
    ActionDef("play_pause",        "Play / Pause",              "Lecture",   "Space"),
    ActionDef("stop",              "Stop",                      "Lecture",   "Escape"),
    ActionDef("rewind",            "Retour au début",           "Lecture",   "Home"),

    # ── Shader ───────────────────────────────────────────────────────────────
    ActionDef("recompile",         "Recompiler le shader",      "Shader",    "F5"),

    # ── Éditeur de code ──────────────────────────────────────────────────────
    ActionDef("editor_find",       "Rechercher / Remplacer",    "Éditeur",   "Ctrl+H"),
    ActionDef("editor_snippet",    "Insérer un snippet",        "Éditeur",   "Ctrl+J"),
    ActionDef("editor_fold",       "Replier le bloc courant",   "Éditeur",   "Ctrl+Shift+["),
    ActionDef("editor_unfold",     "Déplier le bloc courant",   "Éditeur",   "Ctrl+Shift+]"),
    ActionDef("editor_fold_all",   "Replier tout",              "Éditeur",   "Ctrl+Shift+F"),
    ActionDef("editor_unfold_all", "Déplier tout",              "Éditeur",   "Ctrl+Shift+E"),
    ActionDef("editor_split",      "Vue partagée (split)",      "Éditeur",   ""),

    # ── Onglets éditeur ──────────────────────────────────────────────────────
    ActionDef("tab_1",             "Onglet éditeur 1",          "Onglets",   "Ctrl+1"),
    ActionDef("tab_2",             "Onglet éditeur 2",          "Onglets",   "Ctrl+2"),
    ActionDef("tab_3",             "Onglet éditeur 3",          "Onglets",   "Ctrl+3"),
    ActionDef("tab_4",             "Onglet éditeur 4",          "Onglets",   "Ctrl+4"),
    ActionDef("tab_5",             "Onglet éditeur 5",          "Onglets",   "Ctrl+5"),
    ActionDef("tab_6",             "Onglet éditeur 6",          "Onglets",   "Ctrl+6"),

    # ── Affichage / Panneaux ─────────────────────────────────────────────────
    ActionDef("show_node_graph",   "Node Graph",                "Panneaux",  "Ctrl+G"),
    ActionDef("show_script",       "Script Python",             "Panneaux",  "Ctrl+P"),
    ActionDef("hotreload",         "Hot-Reload (watchdog)",     "Panneaux",  "Ctrl+Shift+R"),

    # ── VJing ────────────────────────────────────────────────────────────────
    ActionDef("vj_start",          "Démarrer le mode VJ",       "VJing",     "F11"),
    ActionDef("vj_stop",           "Quitter le mode VJ",        "VJing",     ""),

    # ── Raccourcis eux-mêmes ─────────────────────────────────────────────────
    ActionDef("open_shortcut_editor", "Éditeur de raccourcis…", "Raccourcis","Ctrl+K, Ctrl+S",
              "Ouvre l'éditeur de raccourcis clavier"),
]}


# ══════════════════════════════════════════════════════════════════════════════
#  Profils prédéfinis
# ══════════════════════════════════════════════════════════════════════════════

BUILTIN_PROFILES: dict[str, dict[str, str]] = {

    "Default": {a.id: a.default_key for a in ACTION_REGISTRY.values()},

    "Blender-like": {
        "play_pause":    "Space",
        "rewind":        "Shift+Left",
        "stop":          "Escape",
        "recompile":     "F5",
        "new_project":   "Ctrl+N",
        "open_project":  "Ctrl+O",
        "save_project":  "Ctrl+S",
        "save_project_as": "Ctrl+Shift+S",
        "undo":          "Ctrl+Z",
        "redo":          "Ctrl+Shift+Z",
        "editor_find":   "Ctrl+F",
        "show_node_graph": "Ctrl+G",
        "show_script":   "Ctrl+P",
        "vj_start":      "F11",
        "screenshot":    "F12",
        "quit":          "Ctrl+Q",
        "export_video":  "Ctrl+Shift+E",
        "hotreload":     "Ctrl+Shift+R",
        # tabs inchangés
        "tab_1": "Ctrl+1", "tab_2": "Ctrl+2", "tab_3": "Ctrl+3",
        "tab_4": "Ctrl+4", "tab_5": "Ctrl+5", "tab_6": "Ctrl+6",
        "editor_snippet":    "Ctrl+J",
        "editor_fold":       "Ctrl+Shift+[",
        "editor_unfold":     "Ctrl+Shift+]",
        "editor_fold_all":   "Ctrl+Shift+F",
        "editor_unfold_all": "Ctrl+Shift+E",
        "editor_split":      "",
        "vj_stop":           "",
        "open_shortcut_editor": "Ctrl+K, Ctrl+S",
    },

    "Premiere-like": {
        "play_pause":    "Space",
        "rewind":        "Home",
        "stop":          "Escape",
        "recompile":     "F5",
        "new_project":   "Ctrl+N",
        "open_project":  "Ctrl+O",
        "save_project":  "Ctrl+S",
        "save_project_as": "Ctrl+Shift+S",
        "undo":          "Ctrl+Z",
        "redo":          "Ctrl+Shift+Z",
        "editor_find":   "Ctrl+F",
        "export_video":  "Ctrl+M",
        "screenshot":    "F12",
        "quit":          "Ctrl+Q",
        "show_node_graph": "Ctrl+G",
        "show_script":   "Ctrl+P",
        "vj_start":      "F11",
        "hotreload":     "Ctrl+Shift+R",
        "tab_1": "Ctrl+1", "tab_2": "Ctrl+2", "tab_3": "Ctrl+3",
        "tab_4": "Ctrl+4", "tab_5": "Ctrl+5", "tab_6": "Ctrl+6",
        "editor_snippet":    "Ctrl+J",
        "editor_fold":       "Ctrl+Shift+[",
        "editor_unfold":     "Ctrl+Shift+]",
        "editor_fold_all":   "Ctrl+Shift+F",
        "editor_unfold_all": "Ctrl+Shift+E",
        "editor_split":      "",
        "vj_stop":           "",
        "open_shortcut_editor": "Ctrl+K, Ctrl+S",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  ShortcutManager
# ══════════════════════════════════════════════════════════════════════════════

class ShortcutManager(QObject):
    """Gestionnaire central des raccourcis.

    Cycle de vie :
      1. MainWindow instancie ShortcutManager(self)
      2. Chaque QAction / QShortcut enregistré via register_action() / register_shortcut()
      3. apply_all() appelé après _setup_menu() pour appliquer les bindings sauvegardés
      4. L'utilisateur ouvre ShortcutEditor → les modifications sont persistées et appliquées
    """

    shortcuts_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Bindings actifs : action_id → str QKeySequence
        self._bindings: dict[str, str] = dict(BUILTIN_PROFILES["Default"])
        # Objets enregistrés : action_id → QAction ou (QShortcut, widget)
        self._actions:   dict[str, QAction]   = {}
        self._qshortcuts: dict[str, list]     = {}  # id → [QShortcut, ...]
        self._load_from_settings()

    # ── Enregistrement ─────────────────────────────────────────────────────

    def register_action(self, action_id: str, action: QAction):
        """Enregistre un QAction existant sous un identifiant d'action."""
        self._actions[action_id] = action

    def register_qshortcut(self, action_id: str, shortcut):
        """Enregistre un QShortcut existant (pour les raccourcis hors menu)."""
        self._qshortcuts.setdefault(action_id, []).append(shortcut)

    # ── Application ────────────────────────────────────────────────────────

    def apply_all(self):
        """Applique les bindings actifs à tous les objets enregistrés."""
        for action_id, key_str in self._bindings.items():
            seq = QKeySequence(key_str) if key_str else QKeySequence()
            action = self._actions.get(action_id)
            if action:
                action.setShortcut(seq)
            for sc in self._qshortcuts.get(action_id, []):
                sc.setKey(seq)
        self.shortcuts_changed.emit()

    def apply_profile(self, profile_name: str):
        """Charge et applique un profil (builtin ou sauvegardé dans QSettings)."""
        if profile_name in BUILTIN_PROFILES:
            self._bindings = dict(BUILTIN_PROFILES[profile_name])
        else:
            saved = self._load_user_profile(profile_name)
            if saved:
                self._bindings = saved
        self._save_to_settings()
        self.apply_all()

    # ── Getters / setters ──────────────────────────────────────────────────

    def get_key(self, action_id: str) -> str:
        return self._bindings.get(action_id, "")

    def set_key(self, action_id: str, key_str: str):
        self._bindings[action_id] = key_str

    def reset_key(self, action_id: str):
        default = ACTION_REGISTRY.get(action_id)
        self._bindings[action_id] = default.default_key if default else ""

    def reset_all(self):
        self._bindings = dict(BUILTIN_PROFILES["Default"])

    # ── Détection des conflits ─────────────────────────────────────────────

    def find_conflicts(self, key_str: str, exclude_id: str = "") -> list[str]:
        """Retourne la liste des action_ids utilisant déjà cette touche."""
        if not key_str:
            return []
        conflicts = []
        for aid, kstr in self._bindings.items():
            if aid == exclude_id:
                continue
            if kstr and QKeySequence(kstr) == QKeySequence(key_str):
                conflicts.append(aid)
        return conflicts

    # ── Persistance ────────────────────────────────────────────────────────

    def _save_to_settings(self):
        s = QSettings("OpenShader", "OpenShader")
        s.beginGroup("shortcuts")
        for aid, key in self._bindings.items():
            s.setValue(aid, key)
        s.endGroup()

    def _load_from_settings(self):
        s = QSettings("OpenShader", "OpenShader")
        s.beginGroup("shortcuts")
        for aid in s.childKeys():
            self._bindings[aid] = s.value(aid, "")
        s.endGroup()

    def save_user_profile(self, name: str):
        """Sauvegarde les bindings actuels comme profil utilisateur nommé."""
        s = QSettings("OpenShader", "OpenShader")
        s.beginGroup(f"shortcut_profiles/{name}")
        for aid, key in self._bindings.items():
            s.setValue(aid, key)
        s.endGroup()
        log.info("Profil de raccourcis sauvegardé : %s", name)

    def _load_user_profile(self, name: str) -> dict | None:
        s = QSettings("OpenShader", "OpenShader")
        s.beginGroup(f"shortcut_profiles/{name}")
        if not s.childKeys():
            s.endGroup()
            return None
        bindings = {k: s.value(k, "") for k in s.childKeys()}
        s.endGroup()
        return bindings

    def list_user_profiles(self) -> list[str]:
        s = QSettings("OpenShader", "OpenShader")
        s.beginGroup("shortcut_profiles")
        names = s.childGroups()
        s.endGroup()
        return names

    # ── Export / Import JSON ──────────────────────────────────────────────

    def export_json(self, path: str):
        data = {
            "profile": "custom",
            "bindings": self._bindings,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Raccourcis exportés vers %s", path)

    def import_json(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bindings = data.get("bindings", {})
            if not bindings:
                return False
            for aid, key in bindings.items():
                if aid in ACTION_REGISTRY:
                    self._bindings[aid] = key
            self._save_to_settings()
            self.apply_all()
            log.info("Raccourcis importés depuis %s", path)
            return True
        except (OSError, json.JSONDecodeError, KeyError) as e:
            log.warning("Import raccourcis échoué : %s", e)
            return False

    def commit(self):
        """Sauvegarde + applique — appelé après validation dans ShortcutEditor."""
        self._save_to_settings()
        self.apply_all()


# ══════════════════════════════════════════════════════════════════════════════
#  KeyCaptureWidget — champ de saisie de touche
# ══════════════════════════════════════════════════════════════════════════════

class KeyCaptureWidget(QLineEdit):
    """Champ de saisie qui capture la prochaine combinaison de touches pressée."""

    key_captured = pyqtSignal(str)   # émet la chaîne QKeySequence

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capturing = False
        self.setReadOnly(True)
        self.setPlaceholderText("Cliquez pour saisir un raccourci…")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLineEdit {
                background: #0e1018; color: #c0d8ff;
                border: 1px solid #2a2d3a; border-radius: 4px;
                font: bold 11px 'Segoe UI'; padding: 4px 12px;
            }
            QLineEdit:focus {
                border: 1px solid #3a70c0;
                background: #141828;
            }
        """)

    def mousePressEvent(self, event):
        self._capturing = True
        self.setText("Appuyez sur une combinaison…")
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self._capturing:
            return
        # Ignorer les touches modificatrices seules
        mod_only = {
            Qt.Key.Key_Control, Qt.Key.Key_Shift,
            Qt.Key.Key_Alt, Qt.Key.Key_Meta,
        }
        if event.key() in mod_only:
            return

        # Construire le QKeySequence
        key_combo = event.keyCombination()
        seq = QKeySequence(key_combo)
        key_str = seq.toString(QKeySequence.SequenceFormat.NativeText)

        # Escape = effacer le raccourci
        if event.key() == Qt.Key.Key_Escape:
            key_str = ""
            self.setText("(aucun)")
        else:
            self.setText(key_str)

        self._capturing = False
        self.key_captured.emit(key_str)
        event.accept()

    def set_key(self, key_str: str):
        self.setText(key_str if key_str else "(aucun)")
        self._capturing = False


# ══════════════════════════════════════════════════════════════════════════════
#  ShortcutEditor — dialogue principal style VS Code
# ══════════════════════════════════════════════════════════════════════════════

_DIALOG_STYLE = """
QDialog, QWidget {
    background: #0e1016;
    color: #c0c4d0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10px;
}
QTableWidget {
    background: #0e1016;
    border: none;
    gridline-color: #1a1c24;
    outline: none;
}
QTableWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1a1c24;
}
QTableWidget::item:selected {
    background: #1a2840;
    color: #c0d8ff;
}
QHeaderView::section {
    background: #12141a;
    color: #6a7090;
    font: bold 9px 'Segoe UI';
    border: none;
    border-bottom: 1px solid #1e2030;
    padding: 5px 8px;
}
QLineEdit {
    background: #12141a;
    color: #c0c4d0;
    border: 1px solid #2a2d3a;
    border-radius: 3px;
    padding: 3px 8px;
}
QLineEdit:focus { border-color: #3a5888; }
QComboBox {
    background: #1e2030; color: #c0c4d0;
    border: 1px solid #2a2d3a; border-radius: 3px;
    padding: 3px 8px;
}
QComboBox QAbstractItemView {
    background: #12141a; color: #c0c4d0;
    border: 1px solid #2a2d3a;
    selection-background-color: #3a5888;
}
QPushButton {
    background: #1e2030; color: #8090b0;
    border: 1px solid #2a2d3a; border-radius: 3px;
    padding: 3px 12px; font: 9px 'Segoe UI';
}
QPushButton:hover  { background: #2a2d3a; color: #c0c8e0; }
QPushButton:pressed{ background: #3a4060; }
QPushButton:disabled { color: #3a3d4d; border-color: #1e2030; }
QPushButton#btn_apply {
    background: #1a3a5a; color: #80c8ff;
    border-color: #2a5888;
}
QPushButton#btn_apply:hover { background: #1e4870; }
"""

# Colonnes
_COL_LABEL    = 0
_COL_CATEGORY = 1
_COL_KEY      = 2
_COL_DEFAULT  = 3

_COL_COUNT = 4


class ShortcutEditor(QDialog):
    """Éditeur de raccourcis clavier style VS Code.

    Fonctionnalités :
      - Filtrage en temps réel par label ou touche
      - Sélection d'une ligne → capture de la nouvelle touche dans KeyCaptureWidget
      - Détection des conflits en rouge avec tooltip
      - Reset individuel (icône ↺) ou global
      - Profils : builtin (Default / Blender / Premiere) + profils utilisateur
      - Export / Import JSON
      - Sauvegarde des profils utilisateur nommés
    """

    def __init__(self, shortcut_mgr: ShortcutManager, parent=None):
        super().__init__(parent)
        self._mgr     = shortcut_mgr
        self._pending: dict[str, str] = dict(shortcut_mgr._bindings)  # copie de travail
        self._selected_id: str | None = None

        self.setWindowTitle("Éditeur de raccourcis clavier")
        self.setMinimumSize(780, 560)
        self.setStyleSheet(_DIALOG_STYLE)
        self._build_ui()
        self._populate_table()

    # ── Construction UI ───────────────────────────────────────────────────

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(12, 12, 12, 10)
        vl.setSpacing(8)

        # ── Barre du haut : filtre + profil ──────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        # Filtre
        lbl_filter = QLabel("🔍")
        lbl_filter.setStyleSheet("font:14px; color:#5a7090;")
        top.addWidget(lbl_filter)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(
            "Filtrer par action, catégorie ou touche…"
        )
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        self._filter_edit.setFixedHeight(28)
        top.addWidget(self._filter_edit, 1)

        # Sélecteur de profil
        lbl_profile = QLabel("Profil :")
        lbl_profile.setStyleSheet("color:#6a7090;")
        top.addWidget(lbl_profile)

        self._profile_combo = QComboBox()
        self._profile_combo.setFixedWidth(160)
        self._profile_combo.setFixedHeight(28)
        self._refresh_profile_combo()
        self._profile_combo.currentTextChanged.connect(self._on_profile_selected)
        top.addWidget(self._profile_combo)

        vl.addLayout(top)

        # ── Table centrale ────────────────────────────────────────────────────
        self._table = QTableWidget(0, _COL_COUNT)
        self._table.setHorizontalHeaderLabels(
            ["Action", "Catégorie", "Raccourci actuel", "Par défaut"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setAlternatingRowColors(False)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        self._table.setRowHeight(0, 30)
        vl.addWidget(self._table, 1)

        # ── Zone de capture de touche ─────────────────────────────────────────
        capture_frame = QFrame()
        capture_frame.setStyleSheet(
            "QFrame { background:#12141a; border:1px solid #1e2030; border-radius:4px; }"
        )
        cl = QHBoxLayout(capture_frame)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(12)

        self._lbl_action_name = QLabel("← Sélectionnez une action")
        self._lbl_action_name.setStyleSheet(
            "color:#6a7090; font:10px 'Segoe UI'; min-width:220px;"
        )
        cl.addWidget(self._lbl_action_name)

        self._key_capture = KeyCaptureWidget()
        self._key_capture.setFixedWidth(220)
        self._key_capture.setFixedHeight(30)
        self._key_capture.key_captured.connect(self._on_key_captured)
        cl.addWidget(self._key_capture)

        self._lbl_conflict = QLabel("")
        self._lbl_conflict.setStyleSheet(
            "color:#e05050; font:9px 'Segoe UI'; min-width:180px;"
        )
        cl.addWidget(self._lbl_conflict, 1)

        self._btn_reset_one = QPushButton("↺ Défaut")
        self._btn_reset_one.setToolTip("Restaurer le raccourci par défaut pour cette action")
        self._btn_reset_one.setEnabled(False)
        self._btn_reset_one.clicked.connect(self._on_reset_one)
        cl.addWidget(self._btn_reset_one)

        vl.addWidget(capture_frame)

        # ── Barre du bas ──────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(6)

        btn_export = QPushButton("⬆ Exporter JSON…")
        btn_export.clicked.connect(self._on_export)

        btn_import = QPushButton("⬇ Importer JSON…")
        btn_import.clicked.connect(self._on_import)

        btn_save_profile = QPushButton("💾 Sauvegarder profil…")
        btn_save_profile.clicked.connect(self._on_save_profile)

        btn_reset_all = QPushButton("↺ Tout réinitialiser")
        btn_reset_all.setToolTip("Restaure tous les raccourcis aux valeurs par défaut")
        btn_reset_all.clicked.connect(self._on_reset_all)

        bottom.addWidget(btn_export)
        bottom.addWidget(btn_import)
        bottom.addWidget(btn_save_profile)
        bottom.addStretch()
        bottom.addWidget(btn_reset_all)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#1e2030; max-width:1px;")
        bottom.addWidget(sep)

        btn_cancel = QPushButton("Annuler")
        btn_cancel.clicked.connect(self.reject)

        btn_apply = QPushButton("✔ Appliquer")
        btn_apply.setObjectName("btn_apply")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self._on_apply)

        bottom.addWidget(btn_cancel)
        bottom.addWidget(btn_apply)
        vl.addLayout(bottom)

    # ── Table ─────────────────────────────────────────────────────────────

    def _populate_table(self, filter_text: str = ""):
        self._table.setRowCount(0)
        ft = filter_text.lower()

        # Grouper par catégorie pour un affichage ordonné
        categories: dict[str, list[ActionDef]] = {}
        for adef in ACTION_REGISTRY.values():
            categories.setdefault(adef.category, []).append(adef)

        row = 0
        for cat_name, actions in categories.items():
            for adef in sorted(actions, key=lambda a: a.label):
                key_str = self._pending.get(adef.id, "")
                default_key = adef.default_key

                # Filtrage
                if ft and ft not in adef.label.lower() \
                        and ft not in cat_name.lower() \
                        and ft not in key_str.lower():
                    continue

                self._table.insertRow(row)
                self._table.setRowHeight(row, 28)

                # Colonne 0 — Label
                item_label = QTableWidgetItem(adef.label)
                item_label.setData(Qt.ItemDataRole.UserRole, adef.id)
                if adef.description:
                    item_label.setToolTip(adef.description)
                self._table.setItem(row, _COL_LABEL, item_label)

                # Colonne 1 — Catégorie
                item_cat = QTableWidgetItem(cat_name)
                item_cat.setForeground(QColor("#6a7090"))
                self._table.setItem(row, _COL_CATEGORY, item_cat)

                # Colonne 2 — Raccourci actuel
                item_key = QTableWidgetItem(key_str or "(aucun)")
                conflicts = self._find_conflicts(key_str, adef.id)
                if conflicts:
                    item_key.setForeground(QColor("#e05050"))
                    names = ", ".join(
                        ACTION_REGISTRY[c].label for c in conflicts
                        if c in ACTION_REGISTRY
                    )
                    item_key.setToolTip(f"⚠ Conflit avec : {names}")
                elif key_str != default_key:
                    item_key.setForeground(QColor("#80c8ff"))  # modifié
                else:
                    item_key.setForeground(QColor("#c0c4d0"))
                self._table.setItem(row, _COL_KEY, item_key)

                # Colonne 3 — Par défaut
                item_def = QTableWidgetItem(default_key or "(aucun)")
                item_def.setForeground(QColor("#404860"))
                self._table.setItem(row, _COL_DEFAULT, item_def)

                row += 1

    def _find_conflicts(self, key_str: str, exclude_id: str) -> list[str]:
        if not key_str:
            return []
        return [
            aid for aid, k in self._pending.items()
            if aid != exclude_id and k and
            QKeySequence(k) == QKeySequence(key_str)
        ]

    # ── Événements de sélection ───────────────────────────────────────────

    def _on_row_selected(self):
        rows = self._table.selectedItems()
        if not rows:
            self._selected_id = None
            self._lbl_action_name.setText("← Sélectionnez une action")
            self._key_capture.set_key("")
            self._lbl_conflict.setText("")
            self._btn_reset_one.setEnabled(False)
            return

        action_id = self._table.item(rows[0].row(), _COL_LABEL) \
                               .data(Qt.ItemDataRole.UserRole)
        self._selected_id = action_id
        adef = ACTION_REGISTRY.get(action_id)
        self._lbl_action_name.setText(adef.label if adef else action_id)
        current_key = self._pending.get(action_id, "")
        self._key_capture.set_key(current_key)
        self._update_conflict_label(current_key, action_id)
        self._btn_reset_one.setEnabled(True)

    def _on_key_captured(self, key_str: str):
        if not self._selected_id:
            return
        self._pending[self._selected_id] = key_str
        self._update_conflict_label(key_str, self._selected_id)
        self._populate_table(self._filter_edit.text())
        # Re-sélectionner la même ligne
        self._reselect(self._selected_id)

    def _update_conflict_label(self, key_str: str, action_id: str):
        conflicts = self._find_conflicts(key_str, action_id)
        if conflicts:
            names = ", ".join(
                ACTION_REGISTRY[c].label for c in conflicts if c in ACTION_REGISTRY
            )
            self._lbl_conflict.setText(f"⚠ Conflit : {names}")
        else:
            self._lbl_conflict.setText("")

    def _reselect(self, action_id: str):
        for r in range(self._table.rowCount()):
            item = self._table.item(r, _COL_LABEL)
            if item and item.data(Qt.ItemDataRole.UserRole) == action_id:
                self._table.selectRow(r)
                self._table.scrollToItem(item)
                break

    # ── Filtrage ──────────────────────────────────────────────────────────

    def _on_filter_changed(self, text: str):
        self._populate_table(text)

    # ── Profils ───────────────────────────────────────────────────────────

    def _refresh_profile_combo(self):
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItem("— Choisir un profil —")
        for name in BUILTIN_PROFILES:
            self._profile_combo.addItem(f"⭐ {name}")
        for name in self._mgr.list_user_profiles():
            self._profile_combo.addItem(f"👤 {name}")
        self._profile_combo.blockSignals(False)

    def _on_profile_selected(self, text: str):
        if text.startswith("—"):
            return
        name = text.lstrip("⭐👤 ")
        if name in BUILTIN_PROFILES:
            self._pending = dict(BUILTIN_PROFILES[name])
        else:
            loaded = self._mgr._load_user_profile(name)
            if loaded:
                self._pending = loaded
        self._populate_table(self._filter_edit.text())

    # ── Boutons d'action ──────────────────────────────────────────────────

    def _on_reset_one(self):
        if not self._selected_id:
            return
        adef = ACTION_REGISTRY.get(self._selected_id)
        default = adef.default_key if adef else ""
        self._pending[self._selected_id] = default
        self._key_capture.set_key(default)
        self._update_conflict_label(default, self._selected_id)
        self._populate_table(self._filter_edit.text())
        self._reselect(self._selected_id)

    def _on_reset_all(self):
        rep = QMessageBox.question(
            self, "Réinitialiser tous les raccourcis",
            "Restaurer tous les raccourcis aux valeurs par défaut ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if rep == QMessageBox.StandardButton.Yes:
            self._pending = dict(BUILTIN_PROFILES["Default"])
            self._populate_table(self._filter_edit.text())

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les raccourcis",
            os.path.expanduser("~/demomaker_shortcuts.json"),
            "JSON (*.json)"
        )
        if path:
            try:
                data = {"profile": "custom", "bindings": self._pending}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Export", f"Raccourcis exportés :\n{path}")
            except OSError as e:
                QMessageBox.warning(self, "Erreur", str(e))

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Importer des raccourcis",
            os.path.expanduser("~"),
            "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bindings = data.get("bindings", {})
            imported = 0
            for aid, key in bindings.items():
                if aid in ACTION_REGISTRY:
                    self._pending[aid] = key
                    imported += 1
            self._populate_table(self._filter_edit.text())
            QMessageBox.information(
                self, "Import", f"{imported} raccourci(s) importé(s)."
            )
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Erreur d'import", str(e))

    def _on_save_profile(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Sauvegarder le profil",
            "Nom du profil :", text="Mon profil"
        )
        if ok and name.strip():
            self._mgr._bindings = dict(self._pending)
            self._mgr.save_user_profile(name.strip())
            self._refresh_profile_combo()
            QMessageBox.information(
                self, "Profil sauvegardé",
                f"Profil « {name.strip()} » sauvegardé."
            )

    def _on_apply(self):
        self._mgr._bindings = dict(self._pending)
        self._mgr.commit()
        self.accept()
