"""
node_graph.py
-------------
v2.0 — Éditeur visuel de connexions entre passes (DAG).

Architecture :
  - NodeGraphScene   : QGraphicsScene contenant les nœuds et les arêtes
  - NodeItem         : nœud représentant une passe de rendu (Image, Post, BufferA…)
  - PortItem         : port d'entrée/sortie d'un nœud
  - EdgeItem         : connexion entre un port sortant et un port entrant
  - NodeGraphWidget  : QGraphicsView + toolbar (wrapper intégrable dans MainWindow)

Chaque modification de connexion émet `graph_changed(dag)` où `dag` est un dict
{ output_node: [input_node, …] } utilisable par ShaderEngine pour ordonnancer
le rendu multi-passe.

Passes supportées :
  Buffer A/B/C/D  ← peuvent lire leur propre sortie (ping-pong)
  Image           ← passe principale (lit les Buffers)
  Post            ← post-processing (lit Image)
  Output          ← sortie finale (nœud terminal fixe)
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsPathItem,
    QGraphicsRectItem, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PyQt6.QtCore    import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui     import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QFont, QLinearGradient, QCursor
)

from .logger import get_logger

log = get_logger(__name__)

# ── Palette de couleurs par type de passe ────────────────────────────────────

_NODE_COLORS: dict[str, tuple[str, str]] = {
    'buffer':  ('#1a2840', '#2a5888'),   # (bg_top, accent)
    'image':   ('#1a2820', '#2a8840'),
    'post':    ('#281a28', '#882888'),
    'trans':   ('#282018', '#886028'),
    'output':  ('#141414', '#606060'),
    'audio':   ('#1a1828', '#4050a0'),
}

_PORT_RADIUS   = 6
_NODE_WIDTH    = 140
_NODE_HEIGHT   = 80
_HEADER_HEIGHT = 26


# ── PortItem ─────────────────────────────────────────────────────────────────

class PortItem(QGraphicsEllipseItem):
    """Port d'entrée ou de sortie d'un nœud."""

    def __init__(self, parent: 'NodeItem', port_index: int,
                 is_output: bool, label: str = ""):
        r = _PORT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent)
        self.node        = parent
        self.port_index  = port_index
        self.is_output   = is_output
        self.label       = label
        self._edges: list[EdgeItem] = []

        self.setBrush(QBrush(QColor('#2a5888' if is_output else '#206040')))
        self.setPen(QPen(QColor('#c0c8e0'), 1.5))
        self.setZValue(3)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor('#60a0ff' if self.is_output else '#40c080')))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor('#2a5888' if self.is_output else '#206040')))
        super().hoverLeaveEvent(event)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def add_edge(self, edge: 'EdgeItem'):
        self._edges.append(edge)

    def remove_edge(self, edge: 'EdgeItem'):
        if edge in self._edges:
            self._edges.remove(edge)

    def edges(self) -> list['EdgeItem']:
        return list(self._edges)


# ── NodeItem ─────────────────────────────────────────────────────────────────

class NodeItem(QGraphicsItem):
    """Nœud visuel représentant une passe de rendu."""

    def __init__(self, pass_name: str, node_type: str = 'image',
                 num_inputs: int = 1, num_outputs: int = 1):
        super().__init__()
        self.pass_name   = pass_name
        self.node_type   = node_type
        self.num_inputs  = num_inputs
        self.num_outputs = num_outputs

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)

        bg_top, accent = _NODE_COLORS.get(node_type, ('#1a1a24', '#3a5888'))
        self._bg_top = bg_top
        self._accent = accent

        # Création des ports
        self._input_ports:  list[PortItem] = []
        self._output_ports: list[PortItem] = []
        self._build_ports()

    def _build_ports(self):
        for i in range(self.num_inputs):
            p = PortItem(self, i, is_output=False, label=f"in{i}")
            y = _HEADER_HEIGHT + (_NODE_HEIGHT - _HEADER_HEIGHT) * (i + 1) / (self.num_inputs + 1)
            p.setPos(-_PORT_RADIUS, y)
            self._input_ports.append(p)

        for i in range(self.num_outputs):
            p = PortItem(self, i, is_output=True, label=f"out{i}")
            y = _HEADER_HEIGHT + (_NODE_HEIGHT - _HEADER_HEIGHT) * (i + 1) / (self.num_outputs + 1)
            p.setPos(_NODE_WIDTH + _PORT_RADIUS, y)
            self._output_ports.append(p)

    def boundingRect(self) -> QRectF:
        return QRectF(-_PORT_RADIUS, 0, _NODE_WIDTH + 2 * _PORT_RADIUS, _NODE_HEIGHT)

    def paint(self, painter: QPainter, option, widget=None):
        r = QRectF(0, 0, _NODE_WIDTH, _NODE_HEIGHT)

        # Corps
        grad = QLinearGradient(0, 0, 0, _NODE_HEIGHT)
        grad.setColorAt(0.0, QColor(self._bg_top))
        grad.setColorAt(1.0, QColor('#0e1018'))
        painter.setBrush(QBrush(grad))
        if self.isSelected():
            painter.setPen(QPen(QColor('#80b0ff'), 2))
        else:
            painter.setPen(QPen(QColor('#2a2d3a'), 1))
        painter.drawRoundedRect(r, 6, 6)

        # Header
        hr = QRectF(0, 0, _NODE_WIDTH, _HEADER_HEIGHT)
        painter.setBrush(QBrush(QColor(self._accent)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(hr, 6, 6)
        painter.drawRect(QRectF(0, 6, _NODE_WIDTH, _HEADER_HEIGHT - 6))

        # Titre
        painter.setPen(QPen(QColor('#e0e8ff')))
        font = QFont('Segoe UI', 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(6, 0, _NODE_WIDTH - 12, _HEADER_HEIGHT),
                         Qt.AlignmentFlag.AlignVCenter, self.pass_name)

        # Type
        painter.setPen(QPen(QColor('#606880')))
        font2 = QFont('Segoe UI', 7)
        painter.setFont(font2)
        painter.drawText(QRectF(6, _HEADER_HEIGHT + 4, _NODE_WIDTH - 12, 20),
                         Qt.AlignmentFlag.AlignLeft, self.node_type.upper())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for p in self._input_ports + self._output_ports:
                for edge in p.edges():
                    edge.update_path()
        return super().itemChange(change, value)

    def input_port(self, i: int = 0) -> PortItem | None:
        return self._input_ports[i] if i < len(self._input_ports) else None

    def output_port(self, i: int = 0) -> PortItem | None:
        return self._output_ports[i] if i < len(self._output_ports) else None

    def all_ports(self) -> list[PortItem]:
        return self._input_ports + self._output_ports


# ── EdgeItem ─────────────────────────────────────────────────────────────────

class EdgeItem(QGraphicsPathItem):
    """Connexion courbe entre deux ports."""

    def __init__(self, src: PortItem, dst: PortItem):
        super().__init__()
        self.src = src
        self.dst = dst
        self.setPen(QPen(QColor('#4080c0'), 2, Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setZValue(0)
        src.add_edge(self)
        dst.add_edge(self)
        self.update_path()

    def update_path(self):
        p0 = self.src.scene_pos()
        p1 = self.dst.scene_pos()
        dx = abs(p1.x() - p0.x()) * 0.5
        path = QPainterPath(p0)
        path.cubicTo(p0 + QPointF(dx, 0),
                     p1 - QPointF(dx, 0),
                     p1)
        self.setPath(path)

    def remove_from_scene(self):
        self.src.remove_edge(self)
        self.dst.remove_edge(self)
        if self.scene():
            self.scene().removeItem(self)


# ── NodeGraphScene ────────────────────────────────────────────────────────────

class NodeGraphScene(QGraphicsScene):
    """Scène gérant les nœuds, les arêtes, et le drag de connexion."""

    graph_changed = pyqtSignal(dict)   # { source_pass: [dest_pass, …] }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor('#0e1018')))
        self._nodes: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem]      = []
        self._drag_port: PortItem | None  = None
        self._drag_path: QGraphicsPathItem | None = None

    # ── Construction du graphe par défaut ────────────────────────────────────

    def build_default(self):
        """Crée le graphe de passes par défaut du DemoMaker."""
        self._nodes.clear()
        self._edges.clear()
        for item in list(self.items()):
            self.removeItem(item)

        nodes_spec = [
            ('Buffer A', 'buffer',  1, 1, QPointF(60,  40)),
            ('Buffer B', 'buffer',  1, 1, QPointF(60, 160)),
            ('Buffer C', 'buffer',  1, 1, QPointF(60, 280)),
            ('Buffer D', 'buffer',  1, 1, QPointF(60, 400)),
            ('Image',    'image',   4, 1, QPointF(280, 200)),
            ('Post',     'post',    1, 1, QPointF(500, 200)),
            ('Output',   'output',  1, 0, QPointF(700, 200)),
        ]

        for name, ntype, ni, no, pos in nodes_spec:
            node = NodeItem(name, ntype, ni, no)
            node.setPos(pos)
            self.addItem(node)
            self._nodes[name] = node

        # Connexions par défaut : Buffer* → Image, Image → Post → Output
        self._connect('Buffer A', 'Image', 0, 0)
        self._connect('Buffer B', 'Image', 0, 1)
        self._connect('Buffer C', 'Image', 0, 2)
        self._connect('Buffer D', 'Image', 0, 3)
        self._connect('Image', 'Post',   0, 0)
        self._connect('Post',  'Output', 0, 0)

    def _connect(self, src_name: str, dst_name: str,
                 src_port: int = 0, dst_port: int = 0):
        src_node = self._nodes.get(src_name)
        dst_node = self._nodes.get(dst_name)
        if not src_node or not dst_node:
            return
        sp = src_node.output_port(src_port)
        dp = dst_node.input_port(dst_port)
        if sp and dp:
            edge = EdgeItem(sp, dp)
            self.addItem(edge)
            self._edges.append(edge)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dag(self) -> dict[str, list[str]]:
        """Convertit les connexions en DAG : { source_pass: [dest_pass, …] }."""
        dag: dict[str, list[str]] = {}
        for edge in self._edges:
            src = edge.src.node.pass_name
            dst = edge.dst.node.pass_name
            dag.setdefault(src, [])
            if dst not in dag[src]:
                dag[src].append(dst)
        return dag

    def to_dict(self) -> dict:
        nodes = {
            name: {'x': int(n.pos().x()), 'y': int(n.pos().y()),
                   'type': n.node_type, 'num_inputs': n.num_inputs,
                   'num_outputs': n.num_outputs}
            for name, n in self._nodes.items()
        }
        edges = [
            {'src': e.src.node.pass_name, 'src_port': e.src.port_index,
             'dst': e.dst.node.pass_name, 'dst_port': e.dst.port_index}
            for e in self._edges
        ]
        return {'nodes': nodes, 'edges': edges}

    def from_dict(self, data: dict):
        for item in list(self.items()):
            self.removeItem(item)
        self._nodes.clear()
        self._edges.clear()

        for name, nd in data.get('nodes', {}).items():
            node = NodeItem(name, nd.get('type', 'image'),
                            nd.get('num_inputs', 1), nd.get('num_outputs', 1))
            node.setPos(QPointF(nd.get('x', 0), nd.get('y', 0)))
            self.addItem(node)
            self._nodes[name] = node

        for ed in data.get('edges', []):
            self._connect(ed['src'], ed['dst'],
                          ed.get('src_port', 0), ed.get('dst_port', 0))

    # ── Mouse events pour le drag de connexion ────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform()
                               if self.views() else __import__('PyQt6.QtGui', fromlist=['QTransform']).QTransform())
            if isinstance(item, PortItem) and item.is_output:
                self._drag_port = item
                self._drag_path = QGraphicsPathItem()
                self._drag_path.setPen(QPen(QColor('#80c0ff'), 2,
                                            Qt.PenStyle.DashLine))
                self._drag_path.setZValue(10)
                self.addItem(self._drag_path)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_port and self._drag_path:
            p0 = self._drag_port.scene_pos()
            p1 = event.scenePos()
            dx = abs(p1.x() - p0.x()) * 0.5
            path = QPainterPath(p0)
            path.cubicTo(p0 + QPointF(dx, 0), p1 - QPointF(dx, 0), p1)
            self._drag_path.setPath(path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_port and self._drag_path:
            self.removeItem(self._drag_path)
            self._drag_path = None

            item = self.itemAt(event.scenePos(), self.views()[0].transform()
                               if self.views() else __import__('PyQt6.QtGui', fromlist=['QTransform']).QTransform())
            if isinstance(item, PortItem) and not item.is_output:
                src_pass = self._drag_port.node.pass_name
                dst_pass = item.node.pass_name
                src_port = self._drag_port.port_index
                dst_port = item.port_index

                cmd_stack = getattr(self, '_cmd_stack', None)
                if cmd_stack is not None:
                    # Passer par le CommandStack pour un undo propre
                    from .command_stack import DisconnectEdgeCommand, ConnectEdgeCommand
                    for old in list(item.edges()):
                        old_src  = old.src.node.pass_name
                        old_pi   = old.src.port_index
                        cmd_stack.push(DisconnectEdgeCommand(
                            self, old_src, dst_pass, old_pi, dst_port))
                    cmd_stack.push(ConnectEdgeCommand(
                        self, src_pass, dst_pass, src_port, dst_port))
                else:
                    # Mutation directe (fallback sans CommandStack)
                    for old in list(item.edges()):
                        old.remove_from_scene()
                        if old in self._edges:
                            self._edges.remove(old)
                    edge = EdgeItem(self._drag_port, item)
                    self.addItem(edge)
                    self._edges.append(edge)
                    self.graph_changed.emit(self.to_dag())
                    log.debug("Connexion : %s → %s", src_pass, dst_pass)

            self._drag_port = None
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            for item in self.selectedItems():
                if isinstance(item, NodeItem):
                    for p in item.all_ports():
                        for e in list(p.edges()):
                            e.remove_from_scene()
                            if e in self._edges:
                                self._edges.remove(e)
                    self.removeItem(item)
                    name = item.pass_name
                    if name in self._nodes:
                        del self._nodes[name]
        super().keyPressEvent(event)

    # ── API publique pour CommandStack ────────────────────────────────────────

    def connect_passes(self, src_pass: str, dst_pass: str,
                       src_port: int = 0, dst_port: int = 0):
        """Ajoute une connexion entre deux passes (appelé par ConnectEdgeCommand)."""
        self._connect(src_pass, dst_pass, src_port, dst_port)
        self.graph_changed.emit(self.to_dag())

    def disconnect_passes(self, src_pass: str, dst_pass: str,
                          src_port: int = 0, dst_port: int = 0):
        """Supprime une connexion entre deux passes (appelé par DisconnectEdgeCommand)."""
        src_node = self._nodes.get(src_pass)
        dst_node = self._nodes.get(dst_pass)
        if not src_node or not dst_node:
            return
        for edge in list(self._edges):
            if (edge.src.node is src_node and edge.dst.node is dst_node
                    and edge.src.port_index == src_port
                    and edge.dst.port_index == dst_port):
                edge.remove_from_scene()
                self._edges.remove(edge)
                self.graph_changed.emit(self.to_dag())
                return


    # ── v2.8 : Intégration nœuds audio ──────────────────────────────────────────

    def sync_audio_nodes(self, synth_dag: dict,
                         audio_node_labels: dict[str, str] | None = None):
        """
        Synchronise les nœuds audio dans le Node Graph visuel existant.
        Ajoute un nœud 'audio' pour chaque nœud du SynthGraphScene,
        câblé sur le nœud 'Output' via la chaîne audio.

        synth_dag      : DAG retourné par SynthGraphScene._make_dag()
        audio_node_labels : {node_id: label} pour nommer les nœuds visuels
        """
        labels = audio_node_labels or {}

        # Supprime les nœuds audio existants
        for name in [n for n in list(self._nodes.keys()) if n.startswith('🎵')]:
            node = self._nodes.pop(name)
            # Supprime les arêtes liées
            for edge in list(self._edges):
                if edge.src.parentItem() is node or edge.dst.parentItem() is node:
                    edge.remove_from_scene()
                    if edge in self._edges:
                        self._edges.remove(edge)
            self.removeItem(node)

        if not synth_dag:
            return

        # Crée un nœud visuel « 🎵 Synth » agrégé (représente tout le graphe audio)
        # positionné en bas du canvas
        synth_node = NodeItem('🎵 Synth', 'audio', num_inputs=0, num_outputs=1)
        synth_node.setPos(QPointF(700, 400))
        self.addItem(synth_node)
        self._nodes['🎵 Synth'] = synth_node
        self.graph_changed.emit(self.to_dag())


# ── NodeGraphWidget ───────────────────────────────────────────────────────────

class NodeGraphWidget(QWidget):
    """Widget complet : vue + toolbar."""

    graph_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = NodeGraphScene()
        self._scene.graph_changed.connect(self.graph_changed)

        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setStyleSheet("background: #0e1018; border: none;")

        # Toolbar
        tb = QWidget()
        tb.setFixedHeight(28)
        tb.setStyleSheet("background: #12141a; border-bottom: 1px solid #1e2030;")
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(8, 2, 8, 2)
        tbl.setSpacing(4)

        def _btn(text, tip, cb):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedHeight(22)
            b.setStyleSheet("""
                QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                              border-radius:3px; padding:0 8px; font:9px 'Segoe UI'; }
                QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            """)
            b.clicked.connect(cb)
            return b

        tbl.addWidget(_btn("⟳ Reset", "Réinitialiser le graphe par défaut", self._reset))
        tbl.addWidget(_btn("⊕ Fit",   "Ajuster la vue",                     self._fit))
        tbl.addStretch()
        lbl = QLabel("DAG des passes")
        lbl.setStyleSheet("color:#404860; font:9px 'Segoe UI';")
        tbl.addWidget(lbl)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(tb)
        layout.addWidget(self._view)

        self._scene.build_default()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._view.scale(factor, factor)

    def _reset(self):
        self._scene.build_default()
        self._fit()

    def _fit(self):
        self._view.fitInView(self._scene.itemsBoundingRect(),
                             Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def scene(self) -> NodeGraphScene:
        return self._scene

    def to_dict(self) -> dict:
        return self._scene.to_dict()

    def from_dict(self, data: dict):
        self._scene.from_dict(data)
