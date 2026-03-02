"""
help_system.py
--------------
Système de documentation interactive intégrée — v1.0

Fonctionnalités :
  1. F1 contextuel     — aide selon le widget sous le curseur / le mot sous le curseur GLSL
  2. Référence GLSL    — 300+ fonctions documentées depuis docs/glsl/*.md
  3. Tutoriels         — guides step-by-step interactifs depuis docs/tutorials/
  4. GIF de démo       — aperçus animés locaux depuis docs/gifs/

Architecture :
  - HelpEntry          : une entrée de documentation parsée depuis Markdown
  - TutorialStep       : une étape de tutoriel
  - HelpDatabase       : charge et indexe tous les docs Markdown
  - ContextResolver    : résout l'entrée d'aide depuis un widget Qt
  - HelpPanel          : fenêtre principale de documentation (QDialog)
  - TutorialOverlay    : overlay step-by-step sur la MainWindow
  - HelpSystem         : façade singleton, installEventFilter F1
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import (
    Qt, QObject, QEvent, QTimer, QSize, QPoint, QRect,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QFont, QKeySequence, QShortcut, QMovie,
    QPainter, QPen, QBrush,
)
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextBrowser, QLineEdit, QTabWidget, QScrollArea,
    QFrame, QApplication, QSizePolicy, QProgressBar,
)

from .logger import get_logger

log = get_logger(__name__)

# ── Chemins docs ──────────────────────────────────────────────────────────────

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
DOCS_DIR  = os.path.join(_ROOT_DIR, "docs")
GLSL_DIR  = os.path.join(DOCS_DIR, "glsl")
TUTO_DIR  = os.path.join(DOCS_DIR, "tutorials")
GIFS_DIR  = os.path.join(DOCS_DIR, "gifs")
UI_HELP_FILE = os.path.join(DOCS_DIR, "ui_help.md")


# ═════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HelpEntry:
    id:          str           # identifiant unique (ex: "sin", "viewport")
    title:       str
    category:    str           # "GLSL Math", "Texture", "Interface", …
    tags:        list[str]
    body_md:     str           # contenu Markdown complet
    signature:   str = ""      # signature de fonction si applicable
    gif_path:    str = ""      # chemin GIF local optionnel

    def matches(self, query: str) -> bool:
        q = query.lower()
        return (q in self.id.lower() or q in self.title.lower()
                or any(q in t for t in self.tags)
                or q in self.body_md.lower())


@dataclass
class TutorialStep:
    number:  int
    title:   str
    body_md: str
    action:  str = ""    # instruction d'action (texte court)
    gif_path: str = ""


@dataclass
class Tutorial:
    id:    str
    title: str
    desc:  str
    tags:  list[str]
    steps: list[TutorialStep]
    duration: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#  Parseur Markdown → HelpEntry / Tutorial
# ═════════════════════════════════════════════════════════════════════════════

def _md_to_html(md: str) -> str:
    """Convertit du Markdown simple en HTML pour QTextBrowser."""
    html = md

    # Code blocks
    html = re.sub(
        r'```(?:glsl|python|bash|json)?\n(.*?)```',
        lambda m: (
            f'<pre style="background:#0d0f16;color:#a0d0ff;'
            f'padding:10px;border-radius:4px;border-left:3px solid #4e7fff;'
            f'font-family:Cascadia Code,Consolas,monospace;font-size:10px;'
            f'overflow-x:auto;white-space:pre;">'
            f'{m.group(1).replace("<","&lt;").replace(">","&gt;")}</pre>'
        ),
        html, flags=re.DOTALL,
    )

    # Inline code
    html = re.sub(
        r'`([^`]+)`',
        r'<code style="background:#1a1d2e;color:#89dceb;padding:1px 4px;'
        r'border-radius:3px;font-family:Cascadia Code,Consolas,monospace;">\1</code>',
        html,
    )

    # Headers
    html = re.sub(r'^#### (.+)$', r'<h4 style="color:#cdd6f4;">\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$',  r'<h3 style="color:#89b4fa;">\1</h3>',  html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$',   r'<h2 style="color:#cba6f7;">\1</h2>',  html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$',    r'<h1 style="color:#f5c2e7;">\1</h1>',  html, flags=re.MULTILINE)

    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<b style="color:#f5c2e7;">\1</b>', html)

    # Italic
    html = re.sub(r'\*(.+?)\*', r'<i style="color:#a6adc8;">\1</i>', html)

    # HR
    html = re.sub(r'^---$', '<hr style="border-color:#2a2d3a;">', html, flags=re.MULTILINE)

    # Paragraphes : double newline → <p>
    html = re.sub(r'\n\n', '</p><p style="color:#cdd6f4;line-height:1.6;">', html)
    html = f'<p style="color:#cdd6f4;line-height:1.6;">{html}</p>'

    return html


def _parse_glsl_md(filepath: str) -> list[HelpEntry]:
    """Parse un fichier docs/glsl/*.md en liste de HelpEntry."""
    entries: list[HelpEntry] = []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Détecte la catégorie depuis la première ligne H1
    cat_match = re.match(r'^# (.+)', content)
    category = cat_match.group(1) if cat_match else os.path.basename(filepath)[:-3].title()

    # Découpe par ## (chaque ## = une fonction/entrée)
    sections = re.split(r'\n(?=## )', content)
    for sec in sections:
        if not sec.strip() or not sec.startswith('## '):
            continue

        lines = sec.strip().splitlines()
        title = lines[0][3:].strip()  # retire "## "
        body  = "\n".join(lines[1:]).strip()

        # Signature
        sig_match = re.search(r'\*\*Signature\*\* : `([^`]+)`', body)
        signature = sig_match.group(1) if sig_match else ""

        # Tags
        tags_match = re.search(r'\*\*Tags\*\* : (.+)', body)
        tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []

        # ID = titre en minuscules, espaces → _
        entry_id = re.sub(r'[^a-zA-Z0-9_]', '_', title.lower()).strip('_')

        # GIF associé ?
        gif_path = os.path.join(GIFS_DIR, f"{entry_id}.gif")
        if not os.path.isfile(gif_path):
            gif_path = ""

        entries.append(HelpEntry(
            id=entry_id, title=title, category=category,
            tags=tags, body_md=body, signature=signature,
            gif_path=gif_path,
        ))

    return entries


def _parse_ui_help(filepath: str) -> list[HelpEntry]:
    """Parse docs/ui_help.md en liste de HelpEntry (aide contextuelle UI)."""
    entries: list[HelpEntry] = []
    if not os.path.isfile(filepath):
        return entries

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r'\n(?=## )', content)
    for sec in sections:
        if not sec.strip() or not sec.startswith('## '):
            continue
        lines = sec.strip().splitlines()
        entry_id = lines[0][3:].strip()

        # Titre
        title_match = re.search(r'\*\*Titre\*\* : (.+)', sec)
        title = title_match.group(1).strip() if title_match else entry_id

        # Tags
        tags_match = re.search(r'\*\*Tags\*\* : (.+)', sec)
        tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []

        body = "\n".join(lines[1:]).strip()

        gif_path = os.path.join(GIFS_DIR, f"{entry_id}.gif")
        if not os.path.isfile(gif_path):
            gif_path = ""

        entries.append(HelpEntry(
            id=entry_id, title=title, category="Interface",
            tags=tags, body_md=body, gif_path=gif_path,
        ))

    return entries


def _parse_tutorials(filepath: str) -> list[Tutorial]:
    """Parse docs/tutorials/tutorials.md en liste de Tutorial."""
    tutorials: list[Tutorial] = []
    if not os.path.isfile(filepath):
        return tutorials

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Sépare par ## tutorial:xxx
    tuto_blocks = re.split(r'\n(?=## tutorial:)', content)
    for block in tuto_blocks:
        if not block.strip() or '## tutorial:' not in block:
            continue

        lines = block.strip().splitlines()
        header = lines[0]
        tuto_id = re.search(r'## tutorial:(\S+)', header)
        if not tuto_id:
            continue
        tid = tuto_id.group(1)

        title_m   = re.search(r'\*\*Titre\*\* : (.+)', block)
        dur_m     = re.search(r'\*\*Durée\*\* : (.+)', block)
        tags_m    = re.search(r'\*\*Tags\*\* : (.+)', block)

        title    = title_m.group(1).strip()   if title_m else tid
        duration = dur_m.group(1).strip()     if dur_m   else ""
        tags     = [t.strip() for t in tags_m.group(1).split(",")] if tags_m else []

        # Description (première phrase hors méta)
        desc_lines = [l for l in lines[1:5]
                      if l and not l.startswith('**') and not l.startswith('#')]
        desc = " ".join(desc_lines[:2])

        # Étapes : ### Étape N
        steps: list[TutorialStep] = []
        step_blocks = re.split(r'\n(?=### Étape )', block)
        for sb in step_blocks[1:]:
            step_lines = sb.strip().splitlines()
            step_header = step_lines[0]
            step_num_m  = re.search(r'### Étape (\d+)', step_header)
            step_num    = int(step_num_m.group(1)) if step_num_m else len(steps) + 1
            step_title  = re.sub(r'### Étape \d+ — ', '', step_header).strip()

            step_body = "\n".join(step_lines[1:]).strip()

            # Action
            action_m = re.search(r'\*\*Action\*\* : (.+)', step_body)
            action   = action_m.group(1).strip() if action_m else ""

            gif_path = os.path.join(GIFS_DIR, f"{tid}_step{step_num}.gif")
            if not os.path.isfile(gif_path):
                gif_path = ""

            steps.append(TutorialStep(
                number=step_num, title=step_title,
                body_md=step_body, action=action, gif_path=gif_path,
            ))

        tutorials.append(Tutorial(
            id=tid, title=title, desc=desc, tags=tags,
            steps=steps, duration=duration,
        ))

    return tutorials


# ═════════════════════════════════════════════════════════════════════════════
#  HelpDatabase — index complet
# ═════════════════════════════════════════════════════════════════════════════

class HelpDatabase:
    """Charge et indexe toutes les entrées d'aide depuis docs/."""

    def __init__(self):
        self._entries:   list[HelpEntry] = []
        self._tutorials: list[Tutorial]  = []
        self._index:     dict[str, HelpEntry] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        # GLSL docs
        if os.path.isdir(GLSL_DIR):
            for fname in sorted(os.listdir(GLSL_DIR)):
                if fname.endswith(".md"):
                    fpath = os.path.join(GLSL_DIR, fname)
                    try:
                        entries = _parse_glsl_md(fpath)
                        self._entries.extend(entries)
                        log.debug("GLSL help chargé : %s (%d entrées)", fname, len(entries))
                    except Exception as e:
                        log.warning("Erreur parse %s : %s", fname, e)

        # UI help
        if os.path.isfile(UI_HELP_FILE):
            try:
                ui_entries = _parse_ui_help(UI_HELP_FILE)
                self._entries.extend(ui_entries)
            except Exception as e:
                log.warning("Erreur parse ui_help.md : %s", e)

        # Tutorials
        tuto_file = os.path.join(TUTO_DIR, "tutorials.md")
        if os.path.isfile(tuto_file):
            try:
                self._tutorials = _parse_tutorials(tuto_file)
                log.debug("Tutoriels chargés : %d", len(self._tutorials))
            except Exception as e:
                log.warning("Erreur parse tutorials.md : %s", e)

        # Index par ID
        self._index = {e.id: e for e in self._entries}
        self._loaded = True
        log.info("HelpDatabase chargée : %d entrées, %d tutoriels",
                 len(self._entries), len(self._tutorials))

    def get(self, entry_id: str) -> Optional[HelpEntry]:
        self.load()
        return self._index.get(entry_id)

    def search(self, query: str, max_results: int = 20) -> list[HelpEntry]:
        self.load()
        if not query.strip():
            return self._entries[:max_results]
        return [e for e in self._entries if e.matches(query)][:max_results]

    def by_category(self) -> dict[str, list[HelpEntry]]:
        self.load()
        cats: dict[str, list[HelpEntry]] = {}
        for e in self._entries:
            cats.setdefault(e.category, []).append(e)
        return cats

    def tutorials(self) -> list[Tutorial]:
        self.load()
        return self._tutorials

    def get_tutorial(self, tid: str) -> Optional[Tutorial]:
        self.load()
        for t in self._tutorials:
            if t.id == tid:
                return t
        return None

    @property
    def count(self) -> int:
        self.load()
        return len(self._entries)


# Singleton
_DB = HelpDatabase()


# ═════════════════════════════════════════════════════════════════════════════
#  ContextResolver — résout le sujet d'aide depuis le contexte Qt
# ═════════════════════════════════════════════════════════════════════════════

# Mapping objectName Qt → id d'aide
_WIDGET_HELP_MAP: dict[str, str] = {
    "DockLeft":     "left_panel",
    "DockEditor":   "editor",
    "DockTimeline": "timeline",
    "DockNodeGraph":"node_graph",
    "DockScript":   "editor",
    "GLWidget":     "viewport",
    "gl_widget":    "viewport",
}

# Mapping classes Qt → id d'aide
_CLASS_HELP_MAP: dict[str, str] = {
    "GLWidget":        "viewport",
    "CodeEditor":      "editor",
    "TimelineWidget":  "timeline",
    "LeftPanel":       "left_panel",
    "NodeGraphWidget": "node_graph",
}


def resolve_context(widget: QWidget) -> Optional[str]:
    """
    Résout l'identifiant d'aide le plus pertinent pour le widget donné.
    Remonte la hiérarchie jusqu'à trouver un mapping.
    """
    w: Optional[QWidget] = widget
    while w is not None:
        # Par objectName
        name = w.objectName()
        if name in _WIDGET_HELP_MAP:
            return _WIDGET_HELP_MAP[name]

        # Par nom de classe
        cls = type(w).__name__
        if cls in _CLASS_HELP_MAP:
            return _CLASS_HELP_MAP[cls]

        w = w.parentWidget()

    return None


def resolve_glsl_word(word: str) -> Optional[str]:
    """Résout un mot GLSL en identifiant d'aide (ex: 'smoothstep' → entrée doc)."""
    _DB.load()
    return _DB._index.get(word.lower(), None) and word.lower()


# ═════════════════════════════════════════════════════════════════════════════
#  HelpPanel — fenêtre de documentation principale
# ═════════════════════════════════════════════════════════════════════════════

_STYLE = """
QDialog, QWidget { background: #0d0f16; color: #cdd6f4; }
QTreeWidget { background: #0a0c12; border: 1px solid #1e2235;
              color: #a0a8c0; font: 9px 'Segoe UI'; }
QTreeWidget::item:selected { background: #2a3060; color: #cdd6f4; }
QTreeWidget::item:hover    { background: #1a1d2e; }
QTextBrowser { background: #0a0c12; color: #cdd6f4; border: 1px solid #1e2235;
               font: 10px 'Segoe UI'; }
QLineEdit { background: #12141e; color: #cdd6f4; border: 1px solid #2a2d3a;
            border-radius: 4px; padding: 4px 8px; font: 9px 'Segoe UI'; }
QLineEdit:focus { border-color: #4e7fff88; }
QTabWidget::pane { border: 1px solid #1e2235; background: #0d0f16; }
QTabBar::tab { background: #12141e; color: #5a6080; padding: 5px 14px;
               border: 1px solid #1e2235; margin-right: 2px; font: 9px 'Segoe UI'; }
QTabBar::tab:selected { color: #cdd6f4; border-bottom: 2px solid #4e7fff; }
QScrollBar:vertical { background: #0a0c12; width: 8px; }
QScrollBar::handle:vertical { background: #2a2d3a; border-radius: 4px; }
QPushButton { background: #12141e; color: #7090c0; border: 1px solid #2a2d3a;
              border-radius: 3px; padding: 3px 10px; font: 9px 'Segoe UI'; }
QPushButton:hover { background: #1e2235; color: #89b4fa; }
"""


class HelpPanel(QDialog):
    """
    Fenêtre principale de documentation.
    Onglets : Référence GLSL | Interface | Tutoriels | Recherche
    """

    def __init__(self, parent=None, initial_id: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Documentation — OpenShader")
        self.resize(1000, 660)
        self.setStyleSheet(_STYLE)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        _DB.load()
        self._build_ui()

        if initial_id:
            self._show_entry_id(initial_id)
        else:
            self._show_welcome()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre de recherche ────────────────────────────────────────────────
        search_bar = QWidget()
        search_bar.setFixedHeight(44)
        search_bar.setStyleSheet("background:#090b11;border-bottom:1px solid #1e2235;")
        sb_lay = QHBoxLayout(search_bar)
        sb_lay.setContentsMargins(12, 6, 12, 6)

        lbl = QLabel("📖  OpenShader Docs")
        lbl.setStyleSheet("color:#cba6f7;font:bold 12px 'Segoe UI';letter-spacing:1px;")
        sb_lay.addWidget(lbl)
        sb_lay.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Rechercher dans la documentation…")
        self._search.setFixedWidth(320)
        self._search.textChanged.connect(self._on_search)
        sb_lay.addWidget(self._search)

        root.addWidget(search_bar)

        # ── Corps principal ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ── Panneau gauche : arbre de navigation ──────────────────────────────
        left = QWidget()
        left.setFixedWidth(240)
        left.setStyleSheet("background:#090b11;border-right:1px solid #1e2235;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._tabs_left = QTabWidget()
        self._tabs_left.setStyleSheet(
            "QTabBar::tab{padding:4px 10px;font:8px 'Segoe UI';}"
        )

        # Tab Référence
        self._tree_ref = QTreeWidget()
        self._tree_ref.setHeaderHidden(True)
        self._tree_ref.setRootIsDecorated(True)
        self._tree_ref.itemClicked.connect(self._on_tree_click)
        self._populate_ref_tree()
        self._tabs_left.addTab(self._tree_ref, "📚 Référence")

        # Tab Tutoriels
        self._tree_tuto = QTreeWidget()
        self._tree_tuto.setHeaderHidden(True)
        self._tree_tuto.itemClicked.connect(self._on_tuto_click)
        self._populate_tuto_tree()
        self._tabs_left.addTab(self._tree_tuto, "🎓 Tutoriels")

        ll.addWidget(self._tabs_left)
        splitter.addWidget(left)

        # ── Panneau droit : contenu ───────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        self._content = QTextBrowser()
        self._content.setOpenExternalLinks(True)
        self._content.setStyleSheet(
            "QTextBrowser{background:#090b11;padding:16px;line-height:1.6;}"
        )
        rl.addWidget(self._content, 1)

        # Barre GIF (masquée par défaut)
        self._gif_bar = QLabel()
        self._gif_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_bar.setFixedHeight(0)
        self._gif_bar.setStyleSheet("background:#080a10;border-top:1px solid #1e2235;")
        rl.addWidget(self._gif_bar)

        splitter.addWidget(right)
        splitter.setSizes([240, 760])

        # ── Résultats de recherche (overlay) ──────────────────────────────────
        self._search_results = QTreeWidget()
        self._search_results.setHeaderHidden(True)
        self._search_results.setStyleSheet(
            "QTreeWidget{background:#090b11;border:none;}"
        )
        self._search_results.itemClicked.connect(self._on_search_result_click)
        self._search_results.hide()
        ll.addWidget(self._search_results)

    def _populate_ref_tree(self):
        cats = _DB.by_category()
        for cat, entries in sorted(cats.items()):
            parent_item = QTreeWidgetItem([cat])
            parent_item.setForeground(0, QColor("#89b4fa"))
            parent_item.setFont(0, QFont("Segoe UI", 9, QFont.Weight.Bold))
            for entry in sorted(entries, key=lambda e: e.title):
                child = QTreeWidgetItem([entry.title])
                child.setData(0, Qt.ItemDataRole.UserRole, entry.id)
                child.setForeground(0, QColor("#a0a8c0"))
                parent_item.addChild(child)
            self._tree_ref.addTopLevelItem(parent_item)

    def _populate_tuto_tree(self):
        for tuto in _DB.tutorials():
            item = QTreeWidgetItem([f"{tuto.title}"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("tutorial", tuto.id))
            item.setForeground(0, QColor("#a6e3a1"))
            dur_item = QTreeWidgetItem([f"  ⏱ {tuto.duration}"])
            dur_item.setForeground(0, QColor("#5a6080"))
            dur_item.setFlags(dur_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.addChild(dur_item)
            self._tree_tuto.addTopLevelItem(item)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_tree_click(self, item: QTreeWidgetItem, col: int):
        entry_id = item.data(0, Qt.ItemDataRole.UserRole)
        if entry_id:
            self._show_entry_id(entry_id)

    def _on_tuto_click(self, item: QTreeWidgetItem, col: int):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "tutorial":
            self._show_tutorial(data[1])

    def _show_entry_id(self, entry_id: str):
        entry = _DB.get(entry_id)
        if entry:
            self._show_entry(entry)

    def _show_entry(self, entry: HelpEntry):
        # GIF
        if entry.gif_path and os.path.isfile(entry.gif_path):
            movie = QMovie(entry.gif_path)
            self._gif_bar.setMovie(movie)
            self._gif_bar.setFixedHeight(180)
            movie.start()
        else:
            self._gif_bar.setFixedHeight(0)
            self._gif_bar.clear()

        # Contenu HTML
        header_html = (
            f'<div style="background:#12141e;padding:12px 16px;'
            f'border-bottom:2px solid #4e7fff22;">'
            f'<span style="color:#cba6f7;font-size:14px;font-weight:bold;">'
            f'{entry.title}</span>'
            f'<span style="color:#5a6080;font-size:9px;margin-left:12px;">'
            f'{entry.category}</span>'
        )
        if entry.signature:
            header_html += (
                f'<br><code style="color:#89dceb;font-size:10px;background:none;">'
                f'{entry.signature}</code>'
            )
        header_html += '</div>'

        tags_html = ""
        if entry.tags:
            tags_html = '<div style="padding:6px 16px;background:#0a0c12;">'
            for tag in entry.tags:
                tags_html += (
                    f'<span style="background:#4e7fff18;color:#4e7fff;'
                    f'border:1px solid #4e7fff44;border-radius:3px;'
                    f'padding:1px 6px;margin-right:4px;font-size:9px;">#{tag}</span>'
                )
            tags_html += '</div>'

        body_html = _md_to_html(entry.body_md)

        full_html = (
            f'<html><body style="background:#090b11;margin:0;padding:0;font-family:Segoe UI;">'
            f'{header_html}{tags_html}'
            f'<div style="padding:16px;">{body_html}</div>'
            f'</body></html>'
        )
        self._content.setHtml(full_html)

    def _show_tutorial(self, tuto_id: str):
        tuto = _DB.get_tutorial(tuto_id)
        if not tuto:
            return

        html = (
            f'<html><body style="background:#090b11;color:#cdd6f4;'
            f'font-family:Segoe UI;margin:0;padding:0;">'
            f'<div style="background:#12141e;padding:12px 16px;'
            f'border-bottom:2px solid #a6e3a144;">'
            f'<span style="color:#a6e3a1;font-size:14px;font-weight:bold;">'
            f'🎓 {tuto.title}</span>'
            f'<span style="color:#5a6080;font-size:9px;margin-left:12px;">'
            f'⏱ {tuto.duration}</span></div>'
            f'<div style="padding:16px;">'
        )

        for step in tuto.steps:
            html += (
                f'<div style="margin-bottom:20px;border-left:3px solid #a6e3a1;'
                f'padding-left:12px;">'
                f'<div style="color:#a6e3a1;font-weight:bold;margin-bottom:6px;">'
                f'Étape {step.number} — {step.title}</div>'
                f'{_md_to_html(step.body_md)}'
            )
            if step.action:
                html += (
                    f'<div style="background:#0d1a0d;border:1px solid #a6e3a144;'
                    f'border-radius:4px;padding:8px 12px;margin-top:8px;color:#a6e3a1;">'
                    f'▶ <b>Action</b> : {step.action}</div>'
                )
            html += '</div>'

        html += '</div></body></html>'
        self._content.setHtml(html)

    def _show_welcome(self):
        html = (
            '<html><body style="background:#090b11;color:#cdd6f4;'
            'font-family:Segoe UI;padding:24px;">'
            '<h1 style="color:#cba6f7;">📖 Documentation OpenShader</h1>'
            f'<p style="color:#a0a8c0;">Base de connaissance : '
            f'<b style="color:#89b4fa;">{_DB.count}</b> entrées</p>'
            '<hr style="border-color:#1e2235;">'
            '<h3 style="color:#89b4fa;">Accès rapide</h3>'
            '<ul style="color:#a0a8c0;line-height:2;">'
            '<li><b style="color:#f5c2e7;">F1</b> — Aide contextuelle sur le widget actif</li>'
            '<li><b style="color:#f5c2e7;">Hover GLSL</b> — Tooltip sur les fonctions GLSL</li>'
            '<li><b style="color:#f5c2e7;">Recherche</b> — Barre en haut de cette fenêtre</li>'
            '</ul>'
            '<h3 style="color:#89b4fa;">Sections disponibles</h3>'
            '<ul style="color:#a0a8c0;line-height:2;">'
            '<li>📐 Mathématiques GLSL (sin, cos, smoothstep, length…)</li>'
            '<li>🎨 Couleur, palettes, post-process</li>'
            '<li>📦 SDF — Signed Distance Functions</li>'
            '<li>🌊 Textures, bruit, hash, FBM, Voronoï</li>'
            '<li>🖥 Interface OpenShader</li>'
            '<li>🎓 Tutoriels interactifs</li>'
            '</ul>'
            '</body></html>'
        )
        self._content.setHtml(html)

    # ── Recherche ─────────────────────────────────────────────────────────────

    def _on_search(self, query: str):
        if not query.strip():
            self._search_results.hide()
            self._tabs_left.show()
            return

        results = _DB.search(query)
        self._search_results.clear()
        for entry in results:
            item = QTreeWidgetItem([f"[{entry.category}]  {entry.title}"])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.id)
            item.setForeground(0, QColor("#cdd6f4" if entry.category != "Interface"
                                         else "#a6e3a1"))
            self._search_results.addTopLevelItem(item)

        self._tabs_left.hide()
        self._search_results.show()

        if results:
            self._show_entry(results[0])

    def _on_search_result_click(self, item: QTreeWidgetItem, col: int):
        entry_id = item.data(0, Qt.ItemDataRole.UserRole)
        if entry_id:
            self._show_entry_id(entry_id)

    def show_for(self, entry_id: str):
        """Ouvre et navigue vers une entrée spécifique."""
        self._show_entry_id(entry_id)
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()


# ═════════════════════════════════════════════════════════════════════════════
#  TutorialOverlay — overlay step-by-step sur la MainWindow
# ═════════════════════════════════════════════════════════════════════════════

class TutorialOverlay(QWidget):
    """
    Overlay semi-transparent affiché par-dessus la MainWindow pendant un tutoriel.
    Affiche une étape à la fois avec navigation Précédent/Suivant/Fermer.
    """

    finished = pyqtSignal()

    def __init__(self, parent: QWidget, tutorial: Tutorial):
        super().__init__(parent)
        self._tuto   = tutorial
        self._step   = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.Tool)
        self._build_card()
        self._refresh()

    def _build_card(self):
        self._card = QFrame(self)
        self._card.setFixedWidth(380)
        self._card.setStyleSheet(
            "QFrame{background:#0d0f16;border:1px solid #4e7fff88;"
            "border-radius:8px;}"
        )
        cl = QVBoxLayout(self._card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)

        # Titre tutoriel
        self._lbl_tuto = QLabel()
        self._lbl_tuto.setStyleSheet(
            "color:#a6e3a1;font:bold 10px 'Segoe UI';letter-spacing:.05em;"
        )
        cl.addWidget(self._lbl_tuto)

        # Titre étape
        self._lbl_step = QLabel()
        self._lbl_step.setStyleSheet("color:#cdd6f4;font:bold 12px 'Segoe UI';")
        self._lbl_step.setWordWrap(True)
        cl.addWidget(self._lbl_step)

        # Corps
        self._body = QTextBrowser()
        self._body.setFixedHeight(160)
        self._body.setStyleSheet(
            "QTextBrowser{background:#090b11;border:1px solid #1e2235;"
            "border-radius:4px;padding:8px;font:9px 'Segoe UI';color:#cdd6f4;}"
            "QScrollBar:vertical{width:6px;background:#090b11;}"
            "QScrollBar::handle:vertical{background:#2a2d3a;border-radius:3px;}"
        )
        cl.addWidget(self._body)

        # Action box
        self._action_box = QLabel()
        self._action_box.setWordWrap(True)
        self._action_box.setStyleSheet(
            "QLabel{background:#0d1a0d;color:#a6e3a1;border:1px solid #a6e3a144;"
            "border-radius:4px;padding:8px;font:9px 'Segoe UI';}"
        )
        self._action_box.hide()
        cl.addWidget(self._action_box)

        # GIF
        self._gif_lbl = QLabel()
        self._gif_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_lbl.hide()
        cl.addWidget(self._gif_lbl)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(3)
        self._progress.setStyleSheet(
            "QProgressBar{background:#1e2235;border:none;border-radius:1px;}"
            "QProgressBar::chunk{background:#4e7fff;border-radius:1px;}"
        )
        cl.addWidget(self._progress)

        # Boutons
        btn_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Précédent")
        self._btn_next = QPushButton("Suivant ▶")
        btn_close      = QPushButton("✕ Fermer")
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next.clicked.connect(self._next)
        btn_close.clicked.connect(self._close)
        btn_row.addWidget(self._btn_prev)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        btn_row.addWidget(self._btn_next)
        cl.addLayout(btn_row)

    def _refresh(self):
        steps = self._tuto.steps
        if not steps:
            return
        step = steps[self._step]

        self._lbl_tuto.setText(
            f"🎓 {self._tuto.title}  —  Étape {step.number}/{len(steps)}"
        )
        self._lbl_step.setText(step.title)
        self._body.setHtml(_md_to_html(step.body_md))

        if step.action:
            self._action_box.setText(f"▶ {step.action}")
            self._action_box.show()
        else:
            self._action_box.hide()

        # GIF
        if step.gif_path and os.path.isfile(step.gif_path):
            movie = QMovie(step.gif_path)
            movie.setScaledSize(QSize(340, 120))
            self._gif_lbl.setMovie(movie)
            self._gif_lbl.setFixedHeight(130)
            self._gif_lbl.show()
            movie.start()
        else:
            self._gif_lbl.hide()

        self._progress.setMaximum(len(steps))
        self._progress.setValue(self._step + 1)

        self._btn_prev.setEnabled(self._step > 0)
        self._btn_next.setText("Terminer ✓" if self._step == len(steps) - 1
                                else "Suivant ▶")

        self._card.adjustSize()
        self._position_card()

    def _position_card(self):
        if self.parent():
            parent_rect = self.parent().rect()
            self.setGeometry(parent_rect)
            # Coin inférieur droit
            x = parent_rect.width()  - self._card.width()  - 20
            y = parent_rect.height() - self._card.height() - 20
            self._card.move(x, y)

    def _prev(self):
        if self._step > 0:
            self._step -= 1
            self._refresh()

    def _next(self):
        if self._step < len(self._tuto.steps) - 1:
            self._step += 1
            self._refresh()
        else:
            self._close()

    def _close(self):
        self.hide()
        self.finished.emit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_card()

    def paintEvent(self, e):
        # Fond semi-transparent minimaliste (juste un léger assombrissement)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 30))


# ═════════════════════════════════════════════════════════════════════════════
#  HelpSystem — façade principale + gestion F1
# ═════════════════════════════════════════════════════════════════════════════

class HelpSystem(QObject):
    """
    Façade singleton du système d'aide.
    Installe un event filter sur la MainWindow pour intercepter F1.

    Usage dans MainWindow.__init__ :
        self.help_system = HelpSystem(self)
        self.help_system.install()
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self._win    = main_window
        self._panel: Optional[HelpPanel] = None
        self._overlay: Optional[TutorialOverlay] = None

        # Pré-charge la DB en arrière-plan
        import threading
        threading.Thread(target=_DB.load, daemon=True, name="HelpDBLoad").start()

    def install(self):
        """Installe F1 comme raccourci global d'aide contextuelle."""
        self._win.installEventFilter(self)

        # Raccourci F1 global
        sc = QShortcut(QKeySequence(Qt.Key.Key_F1), self._win)
        sc.activated.connect(self._on_f1)

        log.debug("HelpSystem installé — F1 actif")

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # On laisse l'event se propager normalement
        return False

    def _on_f1(self):
        """Déclenche l'aide contextuelle selon le widget actif."""
        focused = QApplication.focusWidget()
        entry_id = None

        if focused:
            # Cas spécial : CodeEditor → essaie le mot GLSL sous le curseur
            cls = type(focused).__name__
            if cls == "CodeEditor":
                try:
                    word = focused.textUnderCursor()
                    if word:
                        candidate = resolve_glsl_word(word)
                        if candidate:
                            entry_id = candidate
                except Exception:
                    pass

            # Fallback : résolution par widget
            if not entry_id:
                entry_id = resolve_context(focused)

        self.show_help(entry_id or "")

    def show_help(self, entry_id: str = ""):
        """Ouvre le panneau d'aide, optionnellement sur une entrée spécifique."""
        if self._panel is None:
            self._panel = HelpPanel(self._win)

        if entry_id:
            self._panel.show_for(entry_id)
        else:
            if not self._panel.isVisible():
                self._panel.show()
            self._panel.raise_()
            self._panel.activateWindow()

    def start_tutorial(self, tutorial_id: str):
        """Démarre un tutoriel interactif overlay."""
        tuto = _DB.get_tutorial(tutorial_id)
        if not tuto:
            log.warning("Tutoriel introuvable : %s", tutorial_id)
            return

        if self._overlay:
            self._overlay.close()

        self._overlay = TutorialOverlay(self._win, tuto)
        self._overlay.finished.connect(lambda: setattr(self, '_overlay', None))
        self._overlay.show()
        log.info("Tutoriel démarré : %s", tutorial_id)

    def show_glsl_reference(self):
        """Ouvre directement la référence GLSL."""
        self.show_help("sin")  # première entrée de la référence math

    def show_tutorials(self):
        """Ouvre le panneau sur l'onglet Tutoriels."""
        if self._panel is None:
            self._panel = HelpPanel(self._win)
        self._panel.show()
        self._panel._tabs_left.setCurrentIndex(1)
        self._panel.raise_()
