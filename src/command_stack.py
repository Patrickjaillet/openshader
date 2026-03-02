"""
command_stack.py
----------------
CommandStack global — Undo/Redo multi-niveaux pour DemoMaker v2.5.

Architecture :
  • Un seul QUndoStack partagé par toute l'application (500 niveaux).
  • Des QUndoCommand concrètes pour chaque type d'action :
      - SetUniformCommand     : changement d'un uniform GLSL
      - SetFXStateCommand     : toggle / paramètre d'effet post-processing
      - LoadShaderCommand     : chargement d'un fichier shader dans un onglet
      - ConnectEdgeCommand    : ajout d'une connexion dans le Node Graph
      - DisconnectEdgeCommand : suppression d'une connexion dans le Node Graph
  • Un CommandStackPanel (QWidget) : visualiseur de l'historique des actions,
    conçu pour être intégré dans un QDockWidget.

Usage depuis main_window.py :
    self.cmd_stack = CommandStack(parent=self)
    self.cmd_stack.push(SetUniformCommand(engine, "uSpeed", old, new))
    self.cmd_stack.undo()    # ou Ctrl+Z via QAction
    self.cmd_stack.redo()    # ou Ctrl+Y via QAction
"""

from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                              QListWidgetItem, QPushButton, QLabel, QFrame,
                              QAbstractItemView)
from PyQt6.QtCore    import Qt, pyqtSignal, QObject
from PyQt6.QtGui     import (QUndoStack, QUndoCommand, QColor, QFont,
                              QKeySequence, QIcon)

from .logger import get_logger

log = get_logger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────

MAX_UNDO_LEVELS = 500


# ══════════════════════════════════════════════════════════════════════════════
#  CommandStack — wrapper autour de QUndoStack
# ══════════════════════════════════════════════════════════════════════════════

class CommandStack(QObject):
    """Stack Undo/Redo central partagé par toute l'application.

    Expose directement les méthodes push / undo / redo / clear du QUndoStack
    sous-jacent, ainsi que des fabriques de QAction prêtes-à-l'emploi.
    """

    # Émis à chaque push/undo/redo — le visualiseur s'y abonne
    history_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = QUndoStack(self)
        self._stack.setUndoLimit(MAX_UNDO_LEVELS)
        self._stack.indexChanged.connect(self._on_index_changed)

    def _on_index_changed(self, _index: int):
        """Slot dédié pour éviter les crashs liés aux lambdas sur objet détruit."""
        self.history_changed.emit()

    # ── Accès au QUndoStack brut (pour TimelineWidget, etc.) ──────────────
    @property
    def qt_stack(self) -> QUndoStack:
        return self._stack

    # ── Délégation ────────────────────────────────────────────────────────
    def push(self, cmd: QUndoCommand):
        self._stack.push(cmd)

    def undo(self):
        self._stack.undo()

    def redo(self):
        self._stack.redo()

    def clear(self):
        self._stack.clear()
        self.history_changed.emit()

    def can_undo(self) -> bool:
        return self._stack.canUndo()

    def can_redo(self) -> bool:
        return self._stack.canRedo()

    def undo_text(self) -> str:
        return self._stack.undoText()

    def redo_text(self) -> str:
        return self._stack.redoText()

    def index(self) -> int:
        return self._stack.index()

    def count(self) -> int:
        return self._stack.count()

    def command(self, idx: int) -> QUndoCommand:
        return self._stack.command(idx)

    # ── Fabriques QAction pour le menu Édition ────────────────────────────
    def create_undo_action(self, parent, prefix: str = "&Annuler"):
        act = self._stack.createUndoAction(parent, prefix)
        act.setShortcuts(QKeySequence.StandardKey.Undo)
        return act

    def create_redo_action(self, parent, prefix: str = "&Rétablir"):
        act = self._stack.createRedoAction(parent, prefix)
        act.setShortcuts(QKeySequence.StandardKey.Redo)
        return act


# ══════════════════════════════════════════════════════════════════════════════
#  Commandes concrètes
# ══════════════════════════════════════════════════════════════════════════════

class SetUniformCommand(QUndoCommand):
    """Change la valeur d'un uniform GLSL et permet de l'annuler."""

    def __init__(self, shader_engine, name: str, old_value, new_value,
                 parent: QUndoCommand | None = None):
        super().__init__(f"Uniform {name} = {new_value}", parent)
        self._engine    = shader_engine
        self._name      = name
        self._old_value = old_value
        self._new_value = new_value

    def redo(self):
        self._engine.set_uniform(self._name, self._new_value)

    def undo(self):
        self._engine.set_uniform(self._name, self._old_value)

    def mergeWith(self, other: QUndoCommand) -> bool:
        """Fusionne les changements consécutifs sur le même uniform (sliders)."""
        if other.id() != self.id():
            return False
        o = other  # type: ignore[assignment]
        if not hasattr(o, '_name') or o._name != self._name:
            return False
        self._new_value = o._new_value
        self.setText(f"Uniform {self._name} = {self._new_value}")
        return True

    def id(self) -> int:
        # ID unique par nom d'uniform (pour mergeWith)
        return hash(("uniform", self._name)) & 0x7FFFFFFF


class SetFXStateCommand(QUndoCommand):
    """Sauvegarde/restaure un état FX complet (toggles + valeurs de sliders)."""

    def __init__(self, left_panel, shader_path: str,
                 old_state: dict, new_state: dict,
                 parent: QUndoCommand | None = None):
        super().__init__("Effets post-processing", parent)
        self._panel       = left_panel
        self._shader_path = shader_path
        self._old_state   = old_state
        self._new_state   = new_state

    def redo(self):
        # emit=False : évite la récursion infinie.
        # restore_fx_state(emit=True) → _emit_composed_shader → effect_changed
        # → _on_fx_state_changed → cmd_stack.push(SetFXStateCommand) → redo()
        # → boucle infinie → RecursionError → app freeze/hang
        self._panel.restore_fx_state(self._new_state, emit=False)

    def undo(self):
        self._panel.restore_fx_state(self._old_state, emit=False)


class LoadShaderCommand(QUndoCommand):
    """Charge un shader dans un onglet éditeur et permet d'annuler le chargement."""

    def __init__(self, main_window, pass_name: str,
                 old_source: str, new_source: str,
                 new_path: str | None = None,
                 parent: QUndoCommand | None = None):
        label = (f"Charger {pass_name} : "
                 f"{(new_path or '').split('/')[-1].split(chr(92))[-1] or '(source)'}")
        super().__init__(label, parent)
        self._win        = main_window
        self._pass_name  = pass_name
        self._old_source = old_source
        self._new_source = new_source
        self._new_path   = new_path

    def redo(self):
        self._apply(self._new_source, self._new_path)

    def undo(self):
        self._apply(self._old_source, None)

    def _apply(self, source: str, path: str | None):
        editors = getattr(self._win, 'editors', {})
        ed = editors.get(self._pass_name)
        if ed:
            ed.set_code(source)
        # Recompiler
        try:
            self._win.shader_engine.load_shader_source(
                source, self._pass_name,
                source_path=path or ""
            )
        except Exception as exc:
            log.warning("LoadShaderCommand._apply error: %s", exc)


class ConnectEdgeCommand(QUndoCommand):
    """Ajoute une connexion dans le Node Graph."""

    def __init__(self, scene, src_pass: str, dst_pass: str,
                 src_port: int = 0, dst_port: int = 0,
                 parent: QUndoCommand | None = None):
        super().__init__(f"Connecter {src_pass} → {dst_pass}", parent)
        self._scene    = scene
        self._src_pass = src_pass
        self._dst_pass = dst_pass
        self._src_port = src_port
        self._dst_port = dst_port

    def redo(self):
        self._scene.connect_passes(
            self._src_pass, self._dst_pass,
            self._src_port, self._dst_port
        )

    def undo(self):
        self._scene.disconnect_passes(
            self._src_pass, self._dst_pass,
            self._src_port, self._dst_port
        )


class DisconnectEdgeCommand(QUndoCommand):
    """Supprime une connexion dans le Node Graph."""

    def __init__(self, scene, src_pass: str, dst_pass: str,
                 src_port: int = 0, dst_port: int = 0,
                 parent: QUndoCommand | None = None):
        super().__init__(f"Déconnecter {src_pass} → {dst_pass}", parent)
        self._scene    = scene
        self._src_pass = src_pass
        self._dst_pass = dst_pass
        self._src_port = src_port
        self._dst_port = dst_port

    def redo(self):
        self._scene.disconnect_passes(
            self._src_pass, self._dst_pass,
            self._src_port, self._dst_port
        )

    def undo(self):
        self._scene.connect_passes(
            self._src_pass, self._dst_pass,
            self._src_port, self._dst_port
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CommandStackPanel — visualiseur de l'historique (Dock Widget)
# ══════════════════════════════════════════════════════════════════════════════

_ITEM_STYLE_DONE   = "#c0c4d0"   # actions déjà effectuées
_ITEM_STYLE_UNDONE = "#404060"   # actions annulées (grises)
_ITEM_STYLE_CURSOR = "#5090e0"   # action courante (surlignée)

_PANEL_STYLE = """
QWidget {
    background: #0e1016;
    color: #c0c4d0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10px;
}
QListWidget {
    background: #0e1016;
    border: none;
    outline: none;
}
QListWidget::item {
    padding: 3px 8px;
    border-radius: 2px;
}
QListWidget::item:selected {
    background: #1e2030;
}
QPushButton {
    background: #1e2030; color: #8090b0;
    border: 1px solid #2a2d3a; border-radius: 3px;
    padding: 2px 10px; font: 9px 'Segoe UI';
}
QPushButton:hover  { background: #2a2d3a; color: #c0c8e0; }
QPushButton:pressed{ background: #3a4060; }
QPushButton:disabled { color: #3a3d4d; border-color: #1e2030; }
"""


class CommandStackPanel(QWidget):
    """Panneau flottant affichant l'historique Undo/Redo en temps réel.

    - Les actions effectuées s'affichent en clair (blanc/bleu)
    - Les actions annulées apparaissent en grisé
    - L'action courante est surlignée
    - Double-clic sur une entrée → saut direct à cet état (jump_to_index)
    """

    def __init__(self, cmd_stack: CommandStack, parent=None):
        super().__init__(parent)
        self._cmd_stack = cmd_stack
        self._building  = False   # garde anti-récursion pendant refresh
        self._setup_ui()
        cmd_stack.history_changed.connect(self._refresh)
        self._refresh()

    # ── Construction UI ───────────────────────────────────────────────────

    def _setup_ui(self):
        self.setStyleSheet(_PANEL_STYLE)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # En-tête
        hdr = QLabel("  📋  Historique des actions")
        hdr.setStyleSheet(
            "background:#12141a; color:#6a7090; font:bold 9px 'Segoe UI';"
            "padding:5px 0; border-bottom:1px solid #1e2030;"
        )
        vl.addWidget(hdr)

        # Liste des actions
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        vl.addWidget(self._list, 1)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#1e2030;")
        vl.addWidget(sep)

        # Barre d'outils bas
        bottom = QHBoxLayout()
        bottom.setContentsMargins(8, 4, 8, 4)
        bottom.setSpacing(6)

        self._btn_undo = QPushButton("↩ Annuler")
        self._btn_undo.setToolTip("Ctrl+Z")
        self._btn_redo = QPushButton("↪ Rétablir")
        self._btn_redo.setToolTip("Ctrl+Y")
        self._btn_clear = QPushButton("🗑 Effacer")
        self._btn_clear.setToolTip("Effacer tout l'historique")

        self._lbl_count = QLabel("0 actions")
        self._lbl_count.setStyleSheet("color:#404060; font:9px 'Segoe UI';")

        self._btn_undo.clicked.connect(self._cmd_stack.undo)
        self._btn_redo.clicked.connect(self._cmd_stack.redo)
        self._btn_clear.clicked.connect(self._on_clear)

        bottom.addWidget(self._btn_undo)
        bottom.addWidget(self._btn_redo)
        bottom.addWidget(self._btn_clear)
        bottom.addStretch()
        bottom.addWidget(self._lbl_count)
        vl.addLayout(bottom)

    # ── Rafraîchissement ──────────────────────────────────────────────────

    def _refresh(self):
        if self._building:
            return
        self._building = True
        try:
            self._list.clear()
            stack  = self._cmd_stack
            count  = stack.count()
            cursor = stack.index()   # index de la prochaine action à faire

            # Entrée spéciale "État initial"
            init_item = QListWidgetItem("  ◆ État initial")
            init_item.setData(Qt.ItemDataRole.UserRole, -1)
            if cursor == 0:
                init_item.setForeground(QColor(_ITEM_STYLE_CURSOR))
                f = init_item.font(); f.setBold(True); init_item.setFont(f)
            else:
                init_item.setForeground(QColor(_ITEM_STYLE_DONE))
            self._list.addItem(init_item)

            for i in range(count):
                cmd   = stack.command(i)
                text  = cmd.text() if cmd else f"Action {i + 1}"
                label = f"  {'▶' if i == cursor - 1 else '  '} {text}"
                item  = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, i + 1)   # index = i+1

                if i < cursor:
                    # Action effectuée
                    color = _ITEM_STYLE_CURSOR if i == cursor - 1 else _ITEM_STYLE_DONE
                    item.setForeground(QColor(color))
                    if i == cursor - 1:
                        f = item.font(); f.setBold(True); item.setFont(f)
                else:
                    # Action annulée
                    item.setForeground(QColor(_ITEM_STYLE_UNDONE))

                self._list.addItem(item)

            # Scroll vers l'action courante
            if cursor >= 0:
                self._list.scrollToItem(
                    self._list.item(cursor),
                    QAbstractItemView.ScrollHint.PositionAtCenter
                )

            # Boutons
            self._btn_undo.setEnabled(stack.can_undo())
            self._btn_redo.setEnabled(stack.can_redo())
            undo_lbl = f"↩ {stack.undo_text()[:20]}" if stack.can_undo() else "↩ Annuler"
            redo_lbl = f"↪ {stack.redo_text()[:20]}" if stack.can_redo() else "↪ Rétablir"
            self._btn_undo.setText(undo_lbl)
            self._btn_redo.setText(redo_lbl)
            self._lbl_count.setText(f"{count} action{'s' if count != 1 else ''}")

        finally:
            self._building = False

    # ── Interactions ──────────────────────────────────────────────────────

    def _on_double_click(self, item: QListWidgetItem):
        """Saute directement à l'état correspondant (jump to index)."""
        target = item.data(Qt.ItemDataRole.UserRole)
        if target is None:
            return
        target = int(target)
        current = self._cmd_stack.index()
        if target == current:
            return
        # Appliquer undo/redo jusqu'à atteindre target
        if target < current:
            for _ in range(current - target):
                self._cmd_stack.undo()
        else:
            for _ in range(target - current):
                self._cmd_stack.redo()

    def _on_clear(self):
        from PyQt6.QtWidgets import QMessageBox
        rep = QMessageBox.question(
            self, "Effacer l'historique",
            "Effacer tout l'historique des actions ?\nCette opération est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if rep == QMessageBox.StandardButton.Yes:
            self._cmd_stack.clear()
