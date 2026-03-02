"""
main_window.py
--------------
Fenêtre principale du OpenShader.
Orchestre tous les modules : ShaderEngine, AudioEngine, Timeline, UI.
"""

import os
import json
import tempfile
import time
import shutil
import subprocess
import webbrowser
import zipfile
import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QSplitter, QLabel, QPushButton, QToolBar, QTabWidget,
                              QStatusBar, QMenuBar, QMenu, QFileDialog, QInputDialog,
                              QMessageBox, QFrame, QSizePolicy, QProgressDialog,
                              QApplication, QDockWidget, QDialog, QComboBox)
from PyQt6.QtCore  import Qt, QTimer, pyqtSlot, QSettings, QMimeData
from PyQt6.QtGui   import QAction, QKeySequence, QIcon, QFont, QScreen

from .logger        import get_logger
from .shader_engine import ShaderEngine, get_header_line_count
from .audio_engine     import AudioEngine
from .timeline         import Timeline
from .gl_widget        import GLWidget, VIEWPORT_W, VIEWPORT_H, PRESET_RESOLUTIONS

# ModernGL est optionnel — utilisé uniquement pour le rendu offline
try:
    import moderngl
except ImportError:
    moderngl = None  # type: ignore
from .code_editor      import CodeEditor, SplitEditorView, GLSL_SNIPPETS, THEMES
from .left_panel       import LeftPanel
from .timeline_widget  import TimelineWidget, AddKeyframeCommand
# v2.0 — Nouvelles fonctionnalités majeures
from .midi_engine      import MidiEngine, MidiMapping
from .osc_engine       import OscEngine, OscMapping
from .dmx_engine       import DmxEngine, DmxFixture, DmxMapping as DmxMappingDef, DmxPanel  # v4.0 — DMX/Artnet
from .script_engine    import ScriptEngine, _SCRIPT_TEMPLATE
from .plugin_manager   import PluginManager, PostProcessPlugin
from .node_graph       import NodeGraphWidget
# v2.1 ─ imports
from .hot_reload       import HotReloadManager
from .gpu_profiler     import GPUProfiler
from .audio_analyzer   import AudioAnalyzer
from .audio_sync       import AudioSyncEngine, AudioSyncPanel  # v2.3 — Sync auto
from .ai_upscaler      import UpscalerPanel, UPSCALE_MODES      # v2.3 — Upscaling IA
# v2.3 ─ Export & Packaging
from .export_dialog              import ExportDialog
# v6.1 ─ Offline renderer (TAA + Motion Blur + DCP)
from .offline_renderer_dialog    import OfflineRendererDialog
from .shadertoy_multipass_export import show_multipass_export_dialog
from .standalone_player          import StandaloneExporter
from .gallery_exporter           import GalleryPublishDialog
from .wasm_exporter              import WasmExporter          # v3.0 — WASM player
from .ai_shader_generator        import AIShaderGenerator      # v3.5 — IA génération
from .intro_toolkit              import IntroBuilderDialog, IntroSizeEstimator
# v2.5 ─ CommandStack global
from .command_stack import (CommandStack, CommandStackPanel,
                             SetUniformCommand, SetFXStateCommand,
                             LoadShaderCommand)
# v2.6 ─ Raccourcis configurables
from .shortcut_manager import ShortcutManager, ShortcutEditor, ACTION_REGISTRY
# v2.7 ─ Backend Vulkan alternatif
from .vulkan_shader_engine import (
    VulkanShaderEngine, vulkan_available, has_ray_tracing,
    BACKEND_OPENGL, BACKEND_VULKAN, load_backend_pref, save_backend_pref
)
# v2.8 ─ Éditeur de synthétiseur procédural visuel
from .synth_editor import SynthEditorWidget
# v2.9 - Integration VR OpenXR
from .vr_window import VRWindow as VRWindow, openxr_available, VR_GLSL_HEADER, VR_GLSL_HELPERS
# v3.6 — Aide interactive intégrée
from .help_system import HelpSystem
from .scene_graph import (SceneGraph, SceneItem, SceneGraphWidget,    # v6.0 — Scene Graph
                          create_scene_graph_dock)
from .arrangement_view import (Arrangement, ArrangementView,                # v6.0 — Arrangement
                               create_arrangement_dock)

log = get_logger(__name__)


_SHADER_TEMPLATES = {
    "GLSL Minimal": """#version 330 core
uniform vec2  uResolution;
uniform float uTime;
out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    fragColor = vec4(uv, 0.5 + 0.5 * sin(uTime), 1.0);
}
""",
    "Shadertoy Minimal": """// https://www.shadertoy.com/new
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}
""",
    "Shadertoy Plasma": """// https://www.shadertoy.com/view/Xds3zN
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
	float t = iTime * 0.2;
	float c = sin(sin(uv.x*2. + t) + sin(uv.y*3. - t) + sin(uv.x*5. - t*2.) + sin((uv.x+uv.y)*4. + t) * 2. + t);
	vec3 col = vec3(c*0.5+0.5, c*0.2+0.2, 0.0);
	fragColor = vec4(col, 1.0);
}
""",
    "Post: Passthrough": """// Shader de post-processing minimal.
// iChannel0 contient le rendu de la passe 'Image'.
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    fragColor = texture(iChannel0, uv);
}
""",
    "Post: Chromatic Aberration": """// Effet d'aberration chromatique simple.
// iChannel0 contient le rendu de la passe 'Image'.
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    float amount = 0.005;
    
    float r = texture(iChannel0, vec2(uv.x + amount, uv.y)).r;
    float g = texture(iChannel0, uv).g;
    float b = texture(iChannel0, vec2(uv.x - amount, uv.y)).b;

    fragColor = vec4(r, g, b, 1.0);
}
"""
}

class MainWindow(QMainWindow):
    """Fenêtre principale du OpenShader."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenShader — Shader Edition")
        self.setMinimumSize(1400, 800)

        # ── Moteurs ──────────────────────────────────────────────────────────
        # v2.7 — Factory backend : OpenGL ou Vulkan selon la préférence
        from .vulkan_shader_engine import create_engine as _create_engine
        self.shader_engine = _create_engine(VIEWPORT_W, VIEWPORT_H)
        self.audio_engine  = AudioEngine()
        self.timeline      = Timeline(duration=60.0)

        # ── Pistes par défaut ─────────────────────────────────────────────────
        self._init_default_tracks()

        # ── État de lecture ───────────────────────────────────────────────────
        self._pending_audio_path: str | None = None  # chemin en attente de playback_ready
        self._is_playing   = False
        self._play_start_wall = 0.0
        self._play_offset     = 0.0  # temps en secondes au moment du play
        self._current_time    = 0.0
        self._render_is_dirty = True
        self._last_rendered_time = -1.0
        self._texture_paths   = [None] * 4  # Chemins des textures chargées
        self._is_recording    = False
        self._recording_track = None
        self._active_image_shader_path = None
        # États FX mémorisés par chemin de shader : { path → dict retourné par get_fx_state() }
        self._shader_fx_states: dict[str, dict] = {}

        # ── État de transition ────────────────────────────────────────────────
        # Chemins actifs pour les scènes A/B et le shader de transition
        self._active_scene_a_path: str | None = None
        self._active_scene_b_path: str | None = None
        self._active_trans_path:   str | None = None
        # Temps de début/fin de la transition en cours
        self._trans_start_time: float = 0.0
        self._trans_end_time:   float = 0.0

        # ── v2.0 — Nouvelles fonctionnalités majeures ────────────────────────
        self.midi_engine   = MidiEngine(self)
        self.osc_engine    = OscEngine(self)
        self.dmx_engine    = DmxEngine(self)      # v4.0 — DMX/Artnet/sACN
        self.script_engine = ScriptEngine(self)
        self.plugin_manager = PluginManager(self)
        self._vj_window: 'VJWindow | None' = None   # plein-écran VJing
        self._vr_window: 'VRWindow | None' = None   # rendu VR OpenXR (v2.9)
        self._xr_saved_mappings: list = []          # mappings XR persistés
        self._benchmark_active = False
        # v3.5 — IA génération de shaders
        self.ai_generator = AIShaderGenerator(self)
        self.ai_generator.generation_error.connect(
            lambda msg: self._status.showMessage(f"❌ IA : {msg}", 6000))

        # v3.6 — Serveur REST local
        self._rest_server = None  # instancié lazily au démarrage via menu

        # v5.0 — Co-édition temps réel
        self._collab_session  = None   # CollabSession
        self._collab_panel    = None   # CollabPanel (dock)
        self._collab_overlay  = None   # CollabCursorOverlay

        # v5.0 — Cloud Sync
        self._cloud_manager = None    # CloudSyncManager (lazy)
        self._cloud_panel   = None    # CloudSyncPanel (lazy)

        # v6.0 — Scene Graph multi-shaders
        self._scene_graph:     SceneGraph          = SceneGraph()
        self._scene_graph_wgt: SceneGraphWidget | None = None
        self._dock_scene_graph = None   # QDockWidget (lazy — créé dans _setup_docks)

        # v6.0 — Arrangement View
        self._arrangement:      Arrangement        = Arrangement()
        self._arrangement_view: ArrangementView | None = None
        self._dock_arrangement  = None   # QDockWidget

        # v3.6 — Système d'aide interactive
        self.help_system = HelpSystem(self)
        self.help_system.install()

        # v2.1 ─ Performance & Hot-Reload
        self.hot_reload      = HotReloadManager(self)
        self.gpu_profiler    = GPUProfiler()
        self.audio_analyzer  = AudioAnalyzer()
        self.audio_sync      = AudioSyncEngine(self.audio_analyzer)  # v2.3
        self._audio_sync_panel = None  # v2.3 — instancié lazily
        self._upscaler_panel: "UpscalerPanel | None" = None  # v2.3 — instancié lazily
        self._upscale_btns:   dict = {}   # rempli dans _build_viewport_controls
        self._pending_upscale_mode: str = "off"  # appliqué après init GL
        self._hot_reload_enabled   = False
        self._hot_reload_action    = None
        self.hot_reload.file_changed.connect(self._on_hot_reload_file_changed)
        self._benchmark_report_timer = QTimer(self)
        self._benchmark_report_timer.setInterval(1000)
        self._benchmark_report_timer.timeout.connect(self._update_benchmark_status)

        # v2.0 — état éditeur de script et node graph
        self._last_script:      str | None = None
        self._node_graph_data:  dict | None = None
        self.script_engine.set_timeline(self.timeline)

        # Connecter les signaux du moteur de script
        self.script_engine.uniform_set.connect(
            lambda name, val: self.shader_engine.set_uniform(name, val)
        )
        self.script_engine.transport_command.connect(self._on_script_transport)
        self.script_engine.seek_requested.connect(self._on_seek)

        # Connecter le MIDI → uniforms
        self.midi_engine.uniform_changed.connect(
            lambda name, val: self.shader_engine.set_uniform(name, float(val))
        )

        # Connecter l'OSC → uniforms (même pipeline que MIDI)
        self.osc_engine.uniform_changed.connect(
            lambda name, val: self.shader_engine.set_uniform(name, float(val))
        )

        # Connecter le shader → DMX (v4.0 — uniforms GLSL pilotent l'éclairage)
        # ShaderEngine n'est pas un QObject → pas de signal Qt.
        # Le routage se fait via _on_uniform_changed (voir méthode dédiée).
        # self.shader_engine.uniform_changed.connect(...)  ← supprimé (v5.0 fix)

        # Charger les plugins intégrés
        self.plugin_manager.scan_and_load()
        self._current_project_path: str | None = None   # chemin .demomaker actif
        self._project_is_modified:  bool = False         # flag "non sauvegardé"
        self._folder_mode: bool = False                  # v5.0 — sauvegarde en dossier (Git-friendly)
        self._autosave_dir = os.path.join(
            os.path.expanduser("~"), ".demomaker", "autosave"
        )
        os.makedirs(self._autosave_dir, exist_ok=True)
        self._autosave_interval = 120  # secondes entre deux auto-sauvegardes
        self._last_autosave_time = time.time()

        # Timer principal (mis à jour ~60 fps)
        self._main_timer = QTimer(self)
        self._main_timer.setInterval(16)
        self._main_timer.timeout.connect(self._tick)
        self._main_timer.start()

        # v2.4 ─ Thème UI adaptatif (dark / light / auto)
        self._ui_theme_name: str = "auto"

        # v2.5 ─ CommandStack global (Undo/Redo multi-niveaux)
        self.cmd_stack = CommandStack(self)

        # v2.6 ─ Gestionnaire de raccourcis configurables
        self.shortcut_mgr = ShortcutManager(self)

        self._setup_ui()
        self._setup_menu()

        # Connect signals after UI is set up
        self.audio_engine.waveform_ready.connect(self.gl_widget.on_waveform_data)
        self.audio_engine.fft_ready.connect(self.gl_widget.on_fft_data)
        self.audio_engine.amplitude_ready.connect(self._on_audio_amplitude)
        self.audio_engine.amplitude_ready.connect(self.gl_widget.on_amplitude_data)
        self.audio_engine.amplitude_ready.connect(self._on_amplitude_for_analysis)  # v2.1
        # Durée disponible de façon asynchrone une fois le média chargé
        self.audio_engine.playback_ready.connect(self._on_audio_playback_ready)

        self._load_default_shader()
        self._load_settings()
        self.setAcceptDrops(True)

    def _init_default_tracks(self):
        """Crée les pistes par défaut au démarrage."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Piste Audio (en premier, en haut)
        audio_track = self.timeline.add_audio_track("🔊 Audio", "")
        audio_track.color = "#111a11"

        # Piste Shader 1 — Scène A
        t_shader1 = self.timeline.add_track("🎬 Scène A", "_scene_a", "shader")
        t_shader1.color = "#141828"
        plasma = os.path.join(base, 'shaders', 'stoy', 'plasma.st')
        if os.path.isfile(plasma):
            t_shader1.add_keyframe(0.0, plasma, 'step')

        # Piste Transition (type 'trans' — reconnue par _tick)
        t_trans = self.timeline.add_track("🔀 Transition", "_trans", "trans")
        t_trans.color = "#1a1420"

        # Piste Shader 2 — Scène B
        t_shader2 = self.timeline.add_track("🎬 Scène B", "_scene_b", "shader")
        t_shader2.color = "#141828"
        tunnel = os.path.join(base, 'shaders', 'stoy', 'tunnel.st')
        if os.path.isfile(tunnel):
            t_shader2.add_keyframe(0.0, tunnel, 'step')

    # ── Construction UI ───────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setStyleSheet(_build_app_style(resolve_theme("auto")))

        # ── Central Widget (Viewport + Header) ───────────────────────────────
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Viewport OpenGL (centré) — en haut
        viewport_container = self._build_viewport_container()
        central_layout.addWidget(viewport_container)

        # Header (transport controls) — en bas
        central_layout.addWidget(self._build_header())

        # ── Configuration Docking ────────────────────────────────────────────
        # v2.4 — Interface entièrement redockable
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.GroupedDragging
        )

        # 1. Panneau Gauche (Explorateur) — toutes zones autorisées
        self.dock_left = QDockWidget("Explorateur", self)
        self.dock_left.setObjectName("DockLeft")
        self.dock_left.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.left_panel = LeftPanel()
        self.left_panel.shader_file_requested.connect(self._load_shader_file)
        self.left_panel.audio_file_requested.connect(self._load_audio_file)
        self.left_panel.uniform_value_changed.connect(self._on_uniform_changed)
        self.left_panel.effect_changed.connect(self._on_effect_changed)
        self.left_panel.effect_changed.connect(self._on_fx_state_changed)
        self.left_panel.shader_save_requested.connect(self._save_shader_to_file)
        self.left_panel.export_requested.connect(self._on_export_video)
        # v2.2 — Thumbnail
        self.left_panel.thumbnail_requested.connect(self._generate_thumbnail)
        # v2.3 — Auto-paramétrage IA
        self.left_panel.params_scan_requested.connect(self._on_params_scan_requested)
        self.left_panel.apply_param_to_shader.connect(self._on_apply_param_to_shader)

        self.dock_left.setWidget(self.left_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_left)

        # 2. Éditeur de code (Droite)
        self.dock_editor = QDockWidget("Éditeur GLSL", self)
        self.dock_editor.setObjectName("DockEditor")
        self.dock_editor.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.dock_editor.setWidget(self._build_editor_panel())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_editor)

        # 3. Timeline (Bas)
        self.dock_timeline = QDockWidget("Timeline", self)
        self.dock_timeline.setObjectName("DockTimeline")
        self.dock_timeline.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.timeline_widget = TimelineWidget(self.timeline)
        # v2.5 — Partager le CommandStack global avec la timeline
        self.timeline_widget.undo_stack = self.cmd_stack.qt_stack
        self.timeline_widget.time_changed.connect(self._on_timeline_seek)
        self.timeline_widget.timeline_data_changed.connect(self._on_timeline_data_changed)
        self.timeline_widget.audio_file_dropped.connect(self._load_audio_file)

        self.dock_timeline.setWidget(self.timeline_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_timeline)

        # 4. Node Graph (dock secondaire, masqué par défaut)
        self.dock_node_graph = QDockWidget("🕸 Node Graph", self)
        self.dock_node_graph.setObjectName("DockNodeGraph")
        self.dock_node_graph.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        _ng_placeholder = QWidget()
        self.dock_node_graph.setWidget(_ng_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_node_graph)
        self.tabifyDockWidget(self.dock_editor, self.dock_node_graph)
        self.dock_node_graph.hide()
        # Construction lazy du widget dès la première ouverture du dock
        self.dock_node_graph.visibilityChanged.connect(
            lambda vis: self._show_node_graph() if vis else None
        )

        # 5. Script Python (dock secondaire, masqué par défaut)
        self.dock_script = QDockWidget("🐍 Script Python", self)
        self.dock_script.setObjectName("DockScript")
        self.dock_script.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        _sc_placeholder = QWidget()
        self.dock_script.setWidget(_sc_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_script)
        self.tabifyDockWidget(self.dock_editor, self.dock_script)
        self.dock_script.hide()
        # Construction lazy du widget dès la première ouverture du dock
        self.dock_script.visibilityChanged.connect(
            lambda vis: self._show_script_editor() if vis else None
        )

        # v2.8 ─ Dock Synthétiseur Procédural
        self.dock_synth = QDockWidget("🎹 Synth", self)
        self.dock_synth.setObjectName("DockSynth")
        self.dock_synth.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._synth_editor = SynthEditorWidget()
        self._synth_editor.wav_exported.connect(
            lambda p: self._status.showMessage(f"WAV exporté : {p}", 4000)
        )
        self._synth_editor.graph_changed.connect(
            lambda dag: log.debug("Synth DAG updated: %d edges", sum(len(v) for v in dag.values()))
        )
        self.dock_synth.setWidget(self._synth_editor)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_synth)
        self.tabifyDockWidget(self.dock_editor, self.dock_synth)
        self.dock_synth.hide()

        # S'assurer que l'éditeur est l'onglet actif par défaut
        self.dock_editor.raise_()

        # 6. Historique Undo/Redo (v2.5)
        self.dock_history = QDockWidget("📋 Historique", self)
        self.dock_history.setObjectName("DockHistory")
        self.dock_history.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        # Le panel sera construit après cmd_stack (qui existe déjà ici)
        self._history_panel = CommandStackPanel(self.cmd_stack)
        self.dock_history.setWidget(self._history_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_history)
        self.tabifyDockWidget(self.dock_editor, self.dock_history)
        self.dock_history.hide()

        # 8. Scene Graph (v6.0)
        self._dock_scene_graph, self._scene_graph_wgt = create_scene_graph_dock(
            self._scene_graph, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_scene_graph)
        self._dock_scene_graph.hide()
        # Signaux
        self._scene_graph_wgt.scene_activated.connect(self._on_scene_graph_activate)
        self._scene_graph_wgt.scene_preview_requested.connect(self._on_scene_thumb_requested)

        # 9. Arrangement View (v6.0)
        self._dock_arrangement, self._arrangement_view = create_arrangement_dock(
            self._arrangement, self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_arrangement)
        self._dock_arrangement.hide()
        # Signaux
        self._arrangement_view.time_changed.connect(self._on_timeline_seek)
        self._arrangement_view.arrangement_data_changed.connect(self._on_arrangement_changed)
        self._arrangement_view.scene_block_activated.connect(self._on_arrangement_scene)
        self._arrangement_view.cue_activated.connect(
            lambda cue: self._status.showMessage(f"🔖 Cue : {cue.label}  ({cue.time:.2f}s)", 2000))

        # ── Status bar ───────────────────────────────────────────────────────
        self._status = QStatusBar()
        self._status.setStyleSheet("background: #0e1016; color: #6a7090;")
        self.setStatusBar(self._status)
        self._lbl_status_time = QLabel("00:00.000")
        self._lbl_status_time.setStyleSheet(
            "color: #8a90aa; font: 10px 'Cascadia Code', monospace;"
        )
        self._status.addPermanentWidget(self._lbl_status_time)

        # v2.8 — Estimation taille intro (4K/64K) en temps réel
        self._lbl_intro_size = QLabel("")
        self._lbl_intro_size.setStyleSheet(
            "color: #506080; font: 10px 'Cascadia Code', monospace;"
            "padding: 0 8px;"
        )
        self._lbl_intro_size.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lbl_intro_size.setToolTip(
            "Taille estimée compressée (LZMA) des shaders chargés.\n"
            "Budget : 4K = 4096 B · 64K = 65536 B")
        self._lbl_intro_size.setVisible(False)  # masqué tant qu'aucun shader chargé
        self._status.addPermanentWidget(self._lbl_intro_size)
        # Timer dédié : mise à jour toutes les 3s (pas à chaque frame)
        self._intro_size_timer = QTimer(self)
        self._intro_size_timer.setInterval(3000)
        self._intro_size_timer.timeout.connect(self._update_intro_size_label)
        self._intro_size_timer.start()

    def _build_editor_panel(self) -> QWidget:
        """Construit le panneau éditeur multi-pass complet avec toolbar."""
        container = QWidget()
        container.setStyleSheet("background: #1a1c24;")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # ── Toolbar de l'éditeur ─────────────────────────────────────────────
        tb = QWidget()
        tb.setFixedHeight(30)
        tb.setStyleSheet("background: #12141a; border-bottom: 1px solid #1e2030;")
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(6, 2, 6, 2)
        tb_lay.setSpacing(4)

        lbl_pass = QLabel("PASS :")
        lbl_pass.setStyleSheet("color:#3a4060; font:bold 8px 'Segoe UI'; min-width:32px;")
        tb_lay.addWidget(lbl_pass)

        # Bouton Compiler (F5)
        self._btn_compile = QPushButton("▶ Compiler")
        self._btn_compile.setFixedHeight(22)
        self._btn_compile.setToolTip("Recompiler la passe active (F5)")
        self._btn_compile.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_compile.clicked.connect(self._recompile_current)
        tb_lay.addWidget(self._btn_compile)

        # Bouton Effacer la passe
        self._btn_clear_pass = QPushButton("✕ Effacer")
        self._btn_clear_pass.setFixedHeight(22)
        self._btn_clear_pass.setToolTip("Vider la passe active (désactive le rendu de cette passe)")
        self._btn_clear_pass.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_clear_pass.clicked.connect(self._clear_current_pass)
        tb_lay.addWidget(self._btn_clear_pass)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#1e2030; max-width:1px;")
        tb_lay.addWidget(sep)

        # Menu des templates de la passe
        self._btn_template = QPushButton("⊞ Template…")
        self._btn_template.setFixedHeight(22)
        self._btn_template.setToolTip("Insérer un template GLSL pour cette passe")
        self._btn_template.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_template.clicked.connect(self._insert_pass_template)
        tb_lay.addWidget(self._btn_template)

        # Snippets (Ctrl+J)
        self._btn_snippet = QPushButton("✂ Snippet")
        self._btn_snippet.setFixedHeight(22)
        self._btn_snippet.setToolTip("Insérer un snippet GLSL (Ctrl+J)")
        self._btn_snippet.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_snippet.clicked.connect(self._insert_snippet)
        tb_lay.addWidget(self._btn_snippet)

        # Rechercher / Remplacer (Ctrl+H)
        self._btn_find = QPushButton("🔍 Chercher")
        self._btn_find.setFixedHeight(22)
        self._btn_find.setToolTip("Rechercher / Remplacer (Ctrl+H)")
        self._btn_find.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_find.clicked.connect(self._toggle_find_replace)
        tb_lay.addWidget(self._btn_find)

        # Split view
        self._btn_split = QPushButton("⧉ Split")
        self._btn_split.setFixedHeight(22)
        self._btn_split.setToolTip("Activer/désactiver la vue Split (deux éditeurs côte à côte)")
        self._btn_split.setStyleSheet(_build_editor_btn_style(resolve_theme("auto")))
        self._btn_split.setCheckable(True)
        self._btn_split.clicked.connect(self._toggle_split_view)
        tb_lay.addWidget(self._btn_split)

        # Thème éditeur
        self._cmb_theme = QComboBox()
        self._cmb_theme.setFixedHeight(22)
        self._cmb_theme.setToolTip("Thème de l'éditeur")
        self._cmb_theme.addItems(list(THEMES.keys()))
        self._cmb_theme.setStyleSheet(
            "QComboBox { background:#1a1d2e; color:#89b4fa; border:1px solid #2a3060;"
            "            border-radius:3px; padding:1px 4px; font-size:9px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background:#12141a; color:#cdd6f4; }"
        )
        self._cmb_theme.currentTextChanged.connect(self._apply_editor_theme)
        tb_lay.addWidget(self._cmb_theme)

        tb_lay.addStretch()

        # Indicateur de type de shader courant
        self._lbl_pass_type = QLabel("")
        self._lbl_pass_type.setStyleSheet("color:#3a4060; font:8px 'Segoe UI';")
        tb_lay.addWidget(self._lbl_pass_type)

        # Indicateur d'état (OK / erreur / vide)
        self._lbl_pass_status = QLabel("○ vide")
        self._lbl_pass_status.setStyleSheet("color:#3a4060; font:8px 'Segoe UI'; min-width:50px;")
        tb_lay.addWidget(self._lbl_pass_status)

        vlay.addWidget(tb)

        # ── Onglets ──────────────────────────────────────────────────────────
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setStyleSheet(_build_tab_style(resolve_theme("auto")))
        self.editors = {}

        # Descriptions et rôles de chaque passe
        _pass_tooltips = {
            'Image':    "Image  —  Passe principale (rendu final).\n"
                        "iChannel0–3 = Buffer A–D ou textures.",
            'Buffer A': "Buffer A  —  Passe auxiliaire A (ping-pong).\n"
                        "iChannel0 = résultat précédent de Buffer A.",
            'Buffer B': "Buffer B  —  Passe auxiliaire B (ping-pong).\n"
                        "iChannel0 = résultat précédent de Buffer B.",
            'Buffer C': "Buffer C  —  Passe auxiliaire C (ping-pong).\n"
                        "iChannel0 = résultat précédent de Buffer C.",
            'Buffer D': "Buffer D  —  Passe auxiliaire D (ping-pong).\n"
                        "iChannel0 = résultat précédent de Buffer D.",
            'Post':     "Post  —  Post-processing appliqué à la sortie Image.\n"
                        "iChannel0 = texture finale de la passe Image.\n"
                        "Géré aussi par le panneau Effets (FX).",
            'Trans':    "Trans  —  Shader de transition entre deux scènes.\n"
                        "iChannel0 = Scène A (sortante)\n"
                        "iChannel1 = Scène B (entrante)\n"
                        "iProgress = avancement [0.0 → 1.0]\n"
                        "Chargé automatiquement depuis la piste Transition de la timeline.",
        }

        # Passes moteur + onglet Trans dédié
        all_tabs = list(self.shader_engine.pass_names) + ['Trans']
        for pass_name in all_tabs:
            editor = CodeEditor()
            editor.code_changed.connect(
                lambda source, p=pass_name: self._on_code_changed(source, p))
            # v2.8 — déclenche la mise à jour du label taille intro au prochain tick
            editor.code_changed.connect(
                lambda _src: self._intro_size_timer.start())
            self.editors[pass_name] = editor
            idx = self.editor_tabs.addTab(editor, f"○  {pass_name}")
            self.editor_tabs.setTabToolTip(idx, _pass_tooltips.get(pass_name, pass_name))
            # v2.6 — enregistrer les QShortcuts de l'éditeur dans le ShortcutManager
            if hasattr(self, 'shortcut_mgr'):
                editor.register_shortcuts_in_manager(self.shortcut_mgr)
            # v3.6 — synchronise config complétion IA
            editor.sync_ai_completion_config(self.ai_generator)

        self.editor_tabs.currentChanged.connect(self._on_editor_tab_changed)
        vlay.addWidget(self.editor_tabs)

        # ── Raccourcis Ctrl+1…6 ──────────────────────────────────────────────
        from PyQt6.QtGui import QShortcut
        for i, pname in enumerate(self.shader_engine.pass_names, start=1):
            sc = QShortcut(QKeySequence(f"Ctrl+{i}"), self)
            sc.activated.connect(lambda idx=i - 1: self.editor_tabs.setCurrentIndex(idx))
            # v2.6 — enregistrer dans le ShortcutManager
            if hasattr(self, 'shortcut_mgr'):
                self.shortcut_mgr.register_qshortcut(f"tab_{i}", sc)

        return container

    # ── Helpers statut des onglets ────────────────────────────────────────────

    # Préfixes d'icône selon l'état
    _TAB_ICON_EMPTY = "○ "    # gris — passe vide
    _TAB_ICON_OK    = "● "    # vert — compilée OK
    _TAB_ICON_ERR   = "✕ "    # rouge — erreur

    def _update_tab_status(self, pass_name: str):
        """Met à jour l'icône et la couleur de l'onglet pour une passe donnée."""
        # Cherche l'onglet par nom (avec ou sans préfixe icône)
        target_idx = -1
        for i in range(self.editor_tabs.count()):
            raw = self.editor_tabs.tabText(i)
            if raw.lstrip("○●✕ ").strip() == pass_name:
                target_idx = i
                break
        if target_idx < 0:
            return

        if pass_name == 'Trans':
            src   = self.shader_engine.trans_source
            error = self.shader_engine.trans_error
        else:
            src   = self.shader_engine.sources.get(pass_name, '')
            error = self.shader_engine.errors.get(pass_name)

        if not src.strip():
            label = f"{self._TAB_ICON_EMPTY}{pass_name}"
            color = "#3a4060"
        elif error:
            label = f"{self._TAB_ICON_ERR}{pass_name}"
            color = "#c04040"
        else:
            label = f"{self._TAB_ICON_OK}{pass_name}"
            color = "#40a060"

        self.editor_tabs.setTabText(target_idx, label)
        from PyQt6.QtGui import QColor
        self.editor_tabs.tabBar().setTabTextColor(target_idx, QColor(color))

    def _update_pass_toolbar(self, pass_name: str):
        """Met à jour la toolbar de l'éditeur (statut, type) pour la passe active."""
        if pass_name == 'Trans':
            src   = self.shader_engine.trans_source
            error = self.shader_engine.trans_error
            stype_label = "Transition"
        else:
            src   = self.shader_engine.sources.get(pass_name, '')
            error = self.shader_engine.errors.get(pass_name)
            stype = self.shader_engine.get_shader_type(pass_name)
            stype_label = "Shadertoy" if stype == 'shadertoy' else "GLSL pur"

        if not src.strip():
            self._lbl_pass_status.setText("○ vide")
            self._lbl_pass_status.setStyleSheet("color:#3a4060; font:8px 'Segoe UI';")
        elif error:
            self._lbl_pass_status.setText("✕ erreur")
            self._lbl_pass_status.setStyleSheet("color:#c04040; font:8px 'Segoe UI';")
        else:
            self._lbl_pass_status.setText("● OK")
            self._lbl_pass_status.setStyleSheet("color:#40a060; font:8px 'Segoe UI';")

        self._lbl_pass_type.setText(f"{stype_label}  |")
        self._btn_clear_pass.setEnabled(bool(src.strip()))

    def _clear_current_pass(self):
        """Vide le code de la passe active (désactive son rendu)."""
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        pass_name = raw.lstrip("○●✕ ").strip()
        editor = self.editors.get(pass_name)
        if not editor:
            return
        editor.set_code("")
        if pass_name == 'Trans':
            self.shader_engine.load_trans_source("")
            self._active_trans_path = None
            self.shader_engine.set_transition(0.0, active=False)
        else:
            self.shader_engine.load_shader_source("", pass_name)
        self._update_tab_status(pass_name)
        self._update_pass_toolbar(pass_name)
        self._render_is_dirty = True
        self._status.showMessage(f"Passe {pass_name} effacée", 2000)

    def _insert_pass_template(self):
        """Insère un template adapté à la passe active (si l'éditeur est vide)."""
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        pass_name = raw.lstrip("○●✕ ").strip()
        for pname in self.shader_engine.pass_names:
            if pass_name == pname:
                pass_name = pname
                break
        editor = self.editors.get(pass_name)
        if not editor:
            return

        templates = {
            'Buffer A': _TEMPLATE_BUFFER,
            'Buffer B': _TEMPLATE_BUFFER,
            'Buffer C': _TEMPLATE_BUFFER,
            'Buffer D': _TEMPLATE_BUFFER,
            'Image':    _TEMPLATE_IMAGE,
            'Post':     _SHADER_TEMPLATES.get('Post: Passthrough', _TEMPLATE_POST),
            'Trans':    _TEMPLATE_TRANS,
        }
        tpl = templates.get(pass_name, _TEMPLATE_IMAGE)

        # Demande confirmation si l'éditeur n'est pas vide
        if editor.get_code().strip():
            from PyQt6.QtWidgets import QMessageBox
            r = QMessageBox.question(
                self, f"Remplacer la passe {pass_name}",
                "L'éditeur contient déjà du code.\nRemplacer par le template ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        editor.set_code(tpl)
        self._compile_source(tpl, pass_name)

    # ── v1.4 — Éditeur enrichi ───────────────────────────────────────────────

    def _current_editor(self) -> "CodeEditor | None":
        """Retourne l'éditeur de la passe active."""
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        pass_name = raw.lstrip("○●✕ ").strip()
        return self.editors.get(pass_name)

    def _apply_editor_theme(self, theme_name: str):
        """Applique le thème choisi à tous les éditeurs."""
        for editor in self.editors.values():
            editor.apply_theme(theme_name)

    # ── v2.4 — Thème UI global ────────────────────────────────────────────────

    def _apply_ui_theme(self, theme_name: str | None = None):
        """Applique le thème global 'dark', 'light' ou 'auto' à toute l'UI."""
        global _CURRENT_THEME_NAME, _CURRENT_THEME
        if theme_name is not None:
            self._ui_theme_name = theme_name
        _CURRENT_THEME_NAME = self._ui_theme_name
        _CURRENT_THEME = resolve_theme(self._ui_theme_name)
        t = _CURRENT_THEME
        app = QApplication.instance()

        # ── QPalette : assure la cohérence des widgets natifs Qt ────────────
        from PyQt6.QtGui import QPalette, QColor
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window,          QColor(t['bg2']))
        palette.setColor(QPalette.ColorRole.WindowText,      QColor(t['text']))
        palette.setColor(QPalette.ColorRole.Base,            QColor(t['bg2']))
        palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(t['bg1']))
        palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(t['bg1']))
        palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(t['text']))
        palette.setColor(QPalette.ColorRole.Text,            QColor(t['text']))
        palette.setColor(QPalette.ColorRole.Button,          QColor(t['bg3']))
        palette.setColor(QPalette.ColorRole.ButtonText,      QColor(t['text']))
        palette.setColor(QPalette.ColorRole.BrightText,      QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight,       QColor(t['accent']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Link,            QColor(t['accent']))
        palette.setColor(QPalette.ColorRole.Mid,             QColor(t['border']))
        palette.setColor(QPalette.ColorRole.Midlight,        QColor(t['bg1']))
        palette.setColor(QPalette.ColorRole.Dark,            QColor(t['bg0']))
        palette.setColor(QPalette.ColorRole.Shadow,          QColor(t['border']))
        # Disabled
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(t['text_dim']))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(t['text_dim']))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(t['text_dim']))
        if app:
            app.setPalette(palette)

        # ── Stylesheet global ────────────────────────────────────────────────
        self.setStyleSheet(_build_app_style(t))
        # Retab style
        if hasattr(self, 'editor_tabs'):
            self.editor_tabs.setStyleSheet(_build_tab_style(t))
        # Boutons de l'éditeur
        for btn_name in ("_btn_compile", "_btn_clear_pass", "_btn_template",
                         "_btn_snippet", "_btn_find", "_btn_split"):
            btn = getattr(self, btn_name, None)
            if btn:
                btn.setStyleSheet(_build_editor_btn_style(t))
        QSettings("OpenShader", "OpenShader").setValue("ui_theme", self._ui_theme_name)
        log.info(f"Thème UI appliqué : {self._ui_theme_name}")

    def _change_ui_theme(self, name: str):
        self._apply_ui_theme(name)
        # Synchroniser les coches du menu
        for attr, key in (("_action_theme_dark", "dark"),
                          ("_action_theme_light", "light"),
                          ("_action_theme_auto", "auto")):
            action = getattr(self, attr, None)
            if action:
                action.setChecked(key == name)

    # ── v2.6 — Éditeur de raccourcis ─────────────────────────────────────────

    def _show_shortcut_editor(self):
        """Ouvre l'éditeur de raccourcis clavier (style VS Code)."""
        dlg = ShortcutEditor(self.shortcut_mgr, self)
        dlg.exec()

    # ── v2.4 — Profils de disposition des docks ───────────────────────────────

    _LAYOUT_PROFILES = {
        "studio": "studio",
        "vj":     "vj",
        "code":   "code",
        "timeline": "timeline",
    }

    def _apply_layout_profile(self, profile: str):
        """Applique un profil de disposition prédéfini."""
        all_docks = [self.dock_left, self.dock_editor, self.dock_timeline,
                     self.dock_node_graph, self.dock_script]

        if profile == "studio":
            # Layout par défaut : tout visible
            for d in all_docks:
                d.setFloating(False)
            self.dock_node_graph.hide()
            self.dock_script.hide()
            self.dock_left.show()
            self.dock_editor.show()
            self.dock_timeline.show()
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self.dock_left)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,  self.dock_editor)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_timeline)

        elif profile == "vj":
            # VJ Compact : explorateur + viewport seulement, timeline masquée
            self.dock_left.show()
            self.dock_left.setFloating(False)
            self.dock_editor.hide()
            self.dock_timeline.hide()
            self.dock_node_graph.hide()
            self.dock_script.hide()

        elif profile == "code":
            # Code Only : éditeur plein écran, tout le reste masqué
            self.dock_left.hide()
            self.dock_timeline.hide()
            self.dock_node_graph.hide()
            self.dock_script.hide()
            self.dock_editor.show()
            self.dock_editor.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_editor)

        elif profile == "timeline":
            # Timeline Only : explorateur + timeline, éditeur masqué
            self.dock_left.show()
            self.dock_left.setFloating(False)
            self.dock_editor.hide()
            self.dock_node_graph.hide()
            self.dock_script.hide()
            self.dock_timeline.show()
            self.dock_timeline.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,   self.dock_left)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_timeline)

        log.info("Profil de disposition appliqué : %s", profile)

    def _save_layout_profile_dialog(self):
        """Sauvegarde la disposition courante sous un nom utilisateur."""
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Sauvegarder la disposition",
            "Nom du profil :", text="Mon layout"
        )
        if ok and name.strip():
            key = f"layout_profile_{name.strip()}"
            QSettings("OpenShader", "OpenShader").setValue(key, self.saveState())
            self._status.showMessage(f"Disposition « {name.strip()} » sauvegardée.", 3000)
            log.info("Layout sauvegardé : %s", key)

    def _load_layout_profile_dialog(self):
        """Charge une disposition sauvegardée."""
        from PyQt6.QtWidgets import QInputDialog
        settings = QSettings("OpenShader", "OpenShader")
        # Trouver les profils existants
        profiles = [k.replace("layout_profile_", "")
                    for k in settings.allKeys()
                    if k.startswith("layout_profile_")]
        if not profiles:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Aucun profil",
                                    "Aucune disposition sauvegardée.\n"
                                    "Utilisez « Sauvegarder la disposition… » d'abord.")
            return
        name, ok = QInputDialog.getItem(
            self, "Charger une disposition",
            "Profil :", profiles, 0, False
        )
        if ok and name:
            state = settings.value(f"layout_profile_{name}")
            if state:
                self.restoreState(state)
                self._status.showMessage(f"Disposition « {name} » chargée.", 3000)


    def _insert_snippet(self):
        """Ouvre le menu de snippets GLSL dans l'éditeur actif (Ctrl+J)."""
        editor = self._current_editor()
        if editor:
            editor.show_snippet_menu()

    def _toggle_find_replace(self):
        """Affiche / masque la barre Rechercher-Remplacer (Ctrl+H)."""
        editor = self._current_editor()
        if editor:
            editor.toggle_find_replace()

    def _toggle_split_view(self, checked: bool):
        """Active ou désactive la vue Split pour la passe active."""
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        pass_name = raw.lstrip("○●✕ ").strip()
        editor = self.editors.get(pass_name)
        if editor is None:
            return
        idx = self.editor_tabs.indexOf(editor)
        if checked:
            # Crée un SplitEditorView qui partage le document de l'éditeur
            split = SplitEditorView()
            split.editor_left.setDocument(editor.document())
            split.editor_right.setDocument(editor.document())
            # Garde une référence pour restaurer
            editor._split_view = split
            self.editor_tabs.removeTab(idx)
            self.editor_tabs.insertTab(idx, split, self.editor_tabs.tabText(idx) if idx >= 0 else pass_name)
            self.editor_tabs.setCurrentIndex(idx)
        else:
            split = getattr(editor, "_split_view", None)
            if split is not None:
                self.editor_tabs.removeTab(self.editor_tabs.indexOf(split))
                self.editor_tabs.insertTab(idx, editor, self.editor_tabs.tabText(idx) if idx >= 0 else pass_name)
                self.editor_tabs.setCurrentIndex(idx)
                editor._split_view = None

    def _build_header(self) -> QWidget:
        """Construit la barre de contrôle en-tête."""
        hdr = QWidget()
        hdr.setFixedHeight(48)
        hdr.setStyleSheet("background: #0e1016; border-bottom: 1px solid #1e2030;")
        layout = QHBoxLayout(hdr)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        # Logo / titre
        lbl_logo = QLabel()
        _logo_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "logo.png")
        if os.path.isfile(_logo_path):
            from PyQt6.QtGui import QPixmap
            _pix = QPixmap(_logo_path)
            if not _pix.isNull():
                lbl_logo.setPixmap(
                    _pix.scaled(32, 32,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation))
        else:
            # Fallback texte si logo introuvable
            lbl_logo.setText("DM")
            lbl_logo.setStyleSheet("""
                color: #ffffff;
                font: bold 18px 'Segoe UI';
                background: #2a3a5a;
                border-radius: 4px;
                padding: 2px 8px;
            """)
        lbl_logo.setFixedSize(36, 36)
        lbl_title = QLabel("OpenShader")
        lbl_title.setStyleSheet(
            "color: #c8ccd8; font: bold 13px 'Segoe UI'; letter-spacing: 1px;"
        )

        layout.addWidget(lbl_logo)
        layout.addWidget(lbl_title)
        layout.addSpacing(20)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #2a2d3a;")
        layout.addWidget(sep)
        layout.addSpacing(8)

        # Contrôles de lecture
        self._btn_rewind = self._make_btn("⏮", "Retour au début (Home)")
        self._btn_play   = self._make_btn("▶", "Lecture / Pause (Espace)")
        self._btn_stop   = self._make_btn("⏹", "Stop")

        self._btn_rewind.clicked.connect(self._on_rewind)
        self._btn_play.clicked.connect(self._on_play_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        for btn in (self._btn_rewind, self._btn_play, self._btn_stop):
            layout.addWidget(btn)

        layout.addSpacing(12)

        # Temps courant
        self._lbl_time = QLabel("00:00.000")
        self._lbl_time.setStyleSheet(
            "color: #a0a8c0; font: 12px 'Cascadia Code', monospace;"
        )
        layout.addWidget(self._lbl_time)

        layout.addStretch()

        # Audio info
        self._lbl_audio = QLabel("Pas d'audio")
        self._lbl_audio.setStyleSheet("color: #5a6080; font: 9px 'Segoe UI';")
        layout.addWidget(self._lbl_audio)

        layout.addSpacing(12)

        # Export
        btn_export = QPushButton("⬇ Export…")
        btn_export.setFixedHeight(28)
        btn_export.setStyleSheet("""
            QPushButton {
                background: #1f2d4a; color: #6090c0;
                border: 1px solid #2a4060; border-radius: 4px;
                font: 10px 'Segoe UI'; padding: 0 12px;
            }
            QPushButton:hover { background: #263856; }
        """)
        btn_export.clicked.connect(self._on_export)
        layout.addWidget(btn_export)

        return hdr

    def _build_viewport_container(self) -> QWidget:
        """Centre : viewport OpenGL professionnel avec header info + frame stylisée."""
        container = QWidget()
        container.setStyleSheet("background: #080910;")
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(12, 10, 12, 10)
        outer_layout.setSpacing(0)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ── Bloc central (header + frame + footer) ────────────────────────────
        center_block = QWidget()
        center_block.setStyleSheet("background: transparent;")
        center_block.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        block_layout = QVBoxLayout(center_block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(0)

        # ── Barre supérieure info viewport ───────────────────────────────────
        top_bar = QWidget()
        top_bar.setFixedHeight(26)
        top_bar.setStyleSheet("""
            QWidget {
                background: #0e1018;
                border-top: 1px solid #1e2235;
                border-left: 1px solid #1e2235;
                border-right: 1px solid #1e2235;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
        """)
        top_bar_l = QHBoxLayout(top_bar)
        top_bar_l.setContentsMargins(10, 0, 10, 0)
        top_bar_l.setSpacing(14)

        # Indicateur LIVE animé
        self._vp_live_dot = QLabel("●")
        self._vp_live_dot.setStyleSheet("color: #3a9e5a; font: bold 9px 'Segoe UI'; background: transparent; border: none;")
        lbl_live = QLabel("LIVE")
        lbl_live.setStyleSheet("color: #3a9e5a; font: bold 9px 'Segoe UI'; letter-spacing: 1px; background: transparent; border: none;")

        # Separateur vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #1e2235; background: #1e2235; border: none; max-width: 1px;")
        sep1.setFixedWidth(1)

        # Label FPS dans la top bar
        self._vp_fps_label = QLabel("-- fps")
        self._vp_fps_label.setStyleSheet("color: #505878; font: 9px 'Consolas'; background: transparent; border: none;")
        self._vp_fps_label.setMinimumWidth(52)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #1e2235; background: #1e2235; border: none; max-width: 1px;")
        sep2.setFixedWidth(1)

        # Label résolution dans top bar
        self._vp_res_label = QLabel(f"{VIEWPORT_W} × {VIEWPORT_H}")
        self._vp_res_label.setStyleSheet("color: #505878; font: 9px 'Consolas'; background: transparent; border: none;")

        top_bar_l.addWidget(self._vp_live_dot)
        top_bar_l.addWidget(lbl_live)
        top_bar_l.addWidget(sep1)
        top_bar_l.addWidget(self._vp_fps_label)
        top_bar_l.addWidget(sep2)
        top_bar_l.addWidget(self._vp_res_label)
        top_bar_l.addStretch()

        # Label "OPENGL 3.3" à droite
        lbl_api = QLabel("OpenGL 3.3")
        lbl_api.setStyleSheet("color: #303450; font: 9px 'Segoe UI'; background: transparent; border: none;")
        top_bar_l.addWidget(lbl_api)

        block_layout.addWidget(top_bar)

        # ── Frame viewport avec bordure et coin-markers ───────────────────────
        # Wrapper avec bordure lumineuse fine
        frame_wrapper = QWidget()
        frame_wrapper.setStyleSheet("""
            QWidget {
                background: #000000;
                border-left: 1px solid #22253a;
                border-right: 1px solid #22253a;
                border-top: none;
                border-bottom: none;
            }
        """)
        frame_wrapper_l = QVBoxLayout(frame_wrapper)
        frame_wrapper_l.setContentsMargins(2, 2, 2, 2)
        frame_wrapper_l.setSpacing(0)
        frame_wrapper_l.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ── Viewport GL ───────────────────────────────────────────────────────
        self.gl_widget = GLWidget(self.shader_engine)
        self.gl_widget.fps_updated.connect(self._on_fps_updated)
        self.gl_widget.fps_updated.connect(self._on_viewport_fps_display)
        self.gl_widget.render_error.connect(self._on_render_error)
        frame_wrapper_l.addWidget(self.gl_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        block_layout.addWidget(frame_wrapper)

        # ── Barre inférieure contrôles compacts ───────────────────────────────
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(30)
        bottom_bar.setStyleSheet("""
            QWidget {
                background: #0e1018;
                border-bottom: 1px solid #1e2235;
                border-left: 1px solid #1e2235;
                border-right: 1px solid #1e2235;
                border-top: 1px solid #141720;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)
        bottom_bar_l = QHBoxLayout(bottom_bar)
        bottom_bar_l.setContentsMargins(10, 0, 10, 0)
        bottom_bar_l.setSpacing(8)

        # Sélecteur résolution compact
        lbl_res_title = QLabel("RES")
        lbl_res_title.setStyleSheet("color: #383c55; font: bold 8px 'Segoe UI'; letter-spacing: 1px; background: transparent; border: none;")
        self._res_combo = QComboBox()
        self._res_combo.setFixedHeight(20)
        self._res_combo.setStyleSheet("""
            QComboBox {
                background: #131520; color: #8890b0;
                border: 1px solid #1e2235; border-radius: 3px;
                font: 9px 'Consolas'; padding: 0px 6px;
                min-width: 90px;
            }
            QComboBox:hover { border-color: #2e3460; color: #c0c8e0; }
            QComboBox::drop-down { border: none; width: 14px; }
            QComboBox QAbstractItemView {
                background: #131520; color: #9098b8;
                selection-background-color: #1e2235;
                border: 1px solid #2a2d45;
            }
        """)
        for (rw, rh) in PRESET_RESOLUTIONS:
            self._res_combo.addItem(f"{rw} × {rh}", (rw, rh))
        default_idx = next(
            (i for i, (rw, rh) in enumerate(PRESET_RESOLUTIONS)
             if rw == VIEWPORT_W and rh == VIEWPORT_H), 0)
        self._res_combo.setCurrentIndex(default_idx)
        self._res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self._res_combo.currentIndexChanged.connect(self._on_viewport_res_combo_changed)

        # Séparateur
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet("color: #1e2235; background: #1e2235; border: none;")
        sep3.setFixedWidth(1)
        sep3.setFixedHeight(16)

        # Upscaling IA compact
        lbl_up = QLabel("AI")
        lbl_up.setStyleSheet("color: #383c55; font: bold 8px 'Segoe UI'; letter-spacing: 1px; background: transparent; border: none;")

        _upscale_btn_style = """
            QPushButton {{
                background: {bg}; color: {fg};
                border: 1px solid {bd}; border-radius: 3px;
                font: bold 8px 'Segoe UI'; padding: 1px 5px;
                min-width: 32px;
            }}
            QPushButton:hover {{ background: {hov}; border-color: {fg}; color: {fgh}; }}
            QPushButton:checked {{ background: {chk}; color: {fg}; border-color: {fg}; }}
        """
        self._upscale_btns: dict[str, QPushButton] = {}
        _modes = [
            ("off",         "OFF",    "#131520", "#30344a", "#1e2235", "#505878", "#131520"),
            ("quality",     "×2",     "#111a14", "#3a7a45", "#162218", "#50a060", "#111a14"),
            ("performance", "×3",     "#111820", "#365a78", "#162030", "#4a80a8", "#111820"),
            ("ultra",       "×4",     "#18101e", "#6a3890", "#201530", "#9060c0", "#18101e"),
        ]
        for mode_key, label, bg, fg, hov, fgh, chk in _modes:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(20)
            btn.setStyleSheet(_upscale_btn_style.format(bg=bg, fg=fg, bd=fg, hov=hov, fgh=fgh, chk=chk))
            btn.setToolTip(UPSCALE_MODES[mode_key].description)
            btn.clicked.connect(lambda checked, m=mode_key: self._on_upscale_btn_clicked(m))
            self._upscale_btns[mode_key] = btn
        self._upscale_btns["off"].setChecked(True)

        bottom_bar_l.addWidget(lbl_res_title)
        bottom_bar_l.addWidget(self._res_combo)
        bottom_bar_l.addWidget(sep3)
        bottom_bar_l.addWidget(lbl_up)
        for mode_key in ("off", "quality", "performance", "ultra"):
            bottom_bar_l.addWidget(self._upscale_btns[mode_key])
        bottom_bar_l.addStretch()

        block_layout.addWidget(bottom_bar)

        # ── Timer animation du point LIVE (clignotement) ─────────────────────
        self._live_dot_state = True
        self._live_dot_timer = QTimer(self)
        self._live_dot_timer.setInterval(900)
        self._live_dot_timer.timeout.connect(self._blink_live_dot)
        self._live_dot_timer.start()

        outer_layout.addWidget(center_block)
        return container

    @pyqtSlot()
    def _blink_live_dot(self):
        """Fait clignoter le point LIVE dans la top bar du viewport."""
        self._live_dot_state = not self._live_dot_state
        color = "#3a9e5a" if self._live_dot_state else "#1a3a28"
        self._vp_live_dot.setStyleSheet(
            f"color: {color}; font: bold 9px 'Segoe UI'; background: transparent; border: none;"
        )

    @pyqtSlot(float)
    def _on_viewport_fps_display(self, fps: float):
        """Met à jour le label FPS dans la top bar du viewport."""
        color = "#3a9e5a" if fps >= 55 else ("#c8a030" if fps >= 30 else "#b83030")
        self._vp_fps_label.setStyleSheet(
            f"color: {color}; font: 9px 'Consolas'; background: transparent; border: none;"
        )
        self._vp_fps_label.setText(f"{fps:5.1f} fps")

    @pyqtSlot(int)
    def _on_viewport_res_combo_changed(self, index: int):
        """Met à jour le label résolution dans la top bar du viewport."""
        data = self._res_combo.itemData(index)
        if data:
            rw, rh = data
            self._vp_res_label.setText(f"{rw} × {rh}")
            self._vp_res_label.setStyleSheet(
                "color: #505878; font: 9px 'Consolas'; background: transparent; border: none;"
            )

    def _make_btn(self, text: str, tooltip: str = "") -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(34, 34)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                background: #1a1c24; color: #c8ccd8;
                border: 1px solid #2a2d3a; border-radius: 4px;
                font: 14px;
            }
            QPushButton:hover { background: #242730; }
            QPushButton:pressed { background: #2e3240; }
            QPushButton:checked { background: #1f3a2a; color: #5dd88a; border-color: #2a5a3a; }
        """)
        return btn

    # ── Menu ─────────────────────────────────────────────────────────────────

    def _setup_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet("""
            QMenuBar {
                background: #0d0f15;
                color: #b8bdd0;
                font: 11px 'Segoe UI';
                spacing: 2px;
                padding: 1px 4px;
                border-bottom: 1px solid #1e2130;
            }
            QMenuBar::item {
                padding: 4px 10px;
                border-radius: 3px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #1e2335;
                color: #dde1f0;
            }
            QMenuBar::item:pressed {
                background: #2a3a6a;
                color: #ffffff;
            }
            QMenu {
                background: #13151e;
                color: #c0c4d8;
                border: 1px solid #252838;
                border-radius: 4px;
                padding: 4px 0px;
                font: 11px 'Segoe UI';
            }
            QMenu::item {
                padding: 5px 28px 5px 20px;
                border-radius: 2px;
                margin: 1px 4px;
            }
            QMenu::item:selected {
                background: #253060;
                color: #e8ecff;
            }
            QMenu::item:disabled {
                color: #454860;
            }
            QMenu::separator {
                height: 1px;
                background: #1e2130;
                margin: 4px 8px;
            }
            QMenu::icon {
                padding-left: 6px;
            }
            QMenu[title="section"] {
                color: #5a6080;
                font: 9px 'Segoe UI';
            }
        """)

        # ── Fichier ──────────────────────────────────────────────────────────
        file_m = mb.addMenu("  Fichier  ")

        # Projet
        self._add_action(file_m, "Nouveau projet…",             "Ctrl+N",       self._new_project_from_template, action_id="new_project")
        self._add_action(file_m, "Ouvrir projet…",              "Ctrl+O",       self._open_project,              action_id="open_project")
        self._recent_menu = file_m.addMenu("Projets récents")
        self._update_recent_menu()
        file_m.addSeparator()

        # Sauvegarder
        self._add_action(file_m, "Enregistrer",                 "Ctrl+S",       self._save_project_quick,        action_id="save_project")
        self._add_action(file_m, "Enregistrer sous…",           "Ctrl+Shift+S", self._save_project,              action_id="save_project_as")
        self._add_action(file_m, "Restaurer depuis auto-save…", "",             self._restore_autosave_dialog)
        file_m.addSeparator()

        # Versioning & Cloud
        versioning_m = file_m.addMenu("Versioning")
        self._add_action(versioning_m, "Enregistrer en dossier versionné…", "Ctrl+Shift+D", self._save_project_folder, action_id="save_project_folder")
        self._add_action(versioning_m, "Ouvrir dossier versionné…",         "",             self._open_project_folder, action_id="open_project_folder")
        self._folder_mode_action = self._add_action(
            versioning_m, "Mode dossier versionné (Git)",       "",
            self._toggle_folder_mode, is_checkable=True, action_id="toggle_folder_mode")
        cloud_m = file_m.addMenu("Cloud Sync")
        self._add_action(cloud_m, "Sauvegarder dans le cloud",  "Ctrl+Shift+U", self._cloud_save_quick,  action_id="cloud_save")
        self._add_action(cloud_m, "Panneau Cloud Sync…",        "",             self._show_cloud_panel,  action_id="cloud_panel")
        file_m.addSeparator()

        # Import
        import_m = file_m.addMenu("Importer")
        self._add_action(import_m, "Shader GLSL…",              "",             self._open_shader_dialog)
        self._add_action(import_m, "Fichier audio…",            "",             self._open_audio_dialog)
        self._add_action(import_m, "Timeline…",                 "",             self._load_timeline)
        for i in range(4):
            self._add_action(import_m, f"Texture iChannel{i}…","",             lambda checked=False, ch=i: self._load_texture_dialog(ch))

        # Export
        export_m = file_m.addMenu("Exporter")
        self._add_action(export_m, "Vidéo haute qualité…",      "Ctrl+Shift+E", self._export_video_hq,              action_id="export_video")
        self._add_action(export_m, "Rendu offline (TAA / DCP)…","Ctrl+Shift+O", self._export_offline,               action_id="export_offline")
        export_m.addSeparator()
        self._add_action(export_m, "Timeline…",                 "",             self._save_timeline)
        self._add_action(export_m, "Vers Shadertoy…",           "",             self._export_shadertoy)
        self._add_action(export_m, "Shadertoy multipass…",      "",             self._export_shadertoy_multipass)
        self._add_action(export_m, "Standalone démo (64K)…",    "",             self._export_standalone)
        self._add_action(export_m, "Bundle WASM…",              "",             self._export_wasm,                  action_id="export_wasm")
        self._add_action(export_m, "Packaging exécutable…",     "",             self._export_packaging)
        export_m.addSeparator()
        self._add_action(export_m, "Publier vers la galerie…",  "",             self._export_gallery,               action_id="export_gallery")
        file_m.addSeparator()

        # Capture & Scene
        self._add_action(file_m, "Capture d'écran…",            "F12",          self._on_screenshot,                action_id="screenshot")
        self._add_action(file_m, "Sauvegarder dans Scene Graph…","",            self._save_scene_to_graph,          action_id="save_scene_graph")
        file_m.addSeparator()
        self._add_action(file_m, "Quitter",                      "Ctrl+Q",      self.close,                         action_id="quit")

        # ── Édition ──────────────────────────────────────────────────────────
        edit_m = mb.addMenu("  Édition  ")
        act_undo = self.cmd_stack.create_undo_action(self, "Annuler")
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        edit_m.addAction(act_undo)
        act_redo = self.cmd_stack.create_redo_action(self, "Rétablir")
        act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        edit_m.addAction(act_redo)
        edit_m.addSeparator()
        edit_m.addAction(self.dock_history.toggleViewAction())
        edit_m.addSeparator()
        self._add_action(edit_m, "Presets…",                     "",             self._update_presets_menu)
        self.presets_menu = edit_m.addMenu("Charger un preset")
        self._update_presets_menu()
        edit_m.addSeparator()
        self._add_action(edit_m, "Préférences…",                 "Ctrl+,",       self._show_preferences,             action_id="preferences")
        self._add_action(edit_m, "Raccourcis clavier…",          "Ctrl+K, Ctrl+S", self._show_shortcut_editor,       action_id="open_shortcut_editor")

        # ── Shader ───────────────────────────────────────────────────────────
        shader_m = mb.addMenu("  Shader  ")
        self._add_action(shader_m, "Recompiler",                 "F5",           self._recompile_current,            action_id="recompile")
        self._hot_reload_action = self._add_action(
            shader_m, "Hot-Reload (watchdog)",                   "",
            self._toggle_hot_reload, is_checkable=True, action_id="hotreload")
        shader_m.addSeparator()

        # IA
        ai_m = shader_m.addMenu("Intelligence Artificielle")
        self._add_action(ai_m, "Générateur IA Shader…",          "Ctrl+Shift+A", self._show_ai_panel,                action_id="show_ai_panel")
        self._ai_completion_action = self._add_action(
            ai_m, "Complétion GLSL automatique (Tab)",            "Ctrl+Shift+Space",
            self._toggle_ai_completion, is_checkable=True, action_id="ai_completion")
        self._ai_completion_action.setChecked(True)
        self._add_action(ai_m, "Upscaling IA…",                  "",             self._show_upscaler_panel,          action_id="show_upscaler")
        shader_m.addSeparator()

        # Node / Script
        self._add_action(shader_m, "Node Graph…",                "Ctrl+G",       self._show_node_graph,              action_id="show_node_graph")
        self._add_action(shader_m, "Script Python…",             "Ctrl+P",       self._show_script_editor,           action_id="show_script")
        self._add_action(shader_m, "Scene Graph…",               "Ctrl+Shift+G", self._show_scene_graph,             action_id="show_scene_graph")
        shader_m.addSeparator()

        # Références
        self._add_action(shader_m, "Référence GLSL…",            "Ctrl+F1",      lambda: self.help_system.show_glsl_reference())
        self._add_action(shader_m, "Aide expressions keyframes…","",             self._show_expression_help)

        # ── Timeline ─────────────────────────────────────────────────────────
        tl_m = mb.addMenu("  Timeline  ")
        self._add_action(tl_m, "Play / Pause",                   "Space",        self._on_play_pause,                action_id="play_pause")
        self._add_action(tl_m, "Stop",                           "Escape",       self._on_stop,                      action_id="stop")
        self._add_action(tl_m, "Retour au début",                "Home",         self._on_rewind,                    action_id="rewind")
        tl_m.addSeparator()
        self._add_action(tl_m, "Arrangement…",                   "Ctrl+Shift+R", self._show_arrangement,             action_id="show_arrangement")
        self._add_action(tl_m, "Ajouter une piste Caméra 3D…",   "",             self._add_camera_track_action)
        tl_m.addSeparator()

        # Audio
        audio_m = tl_m.addMenu("Audio")
        self._add_action(audio_m, "Importer fichier audio…",     "",             self._open_audio_dialog)
        self._add_action(audio_m, "Enregistrer depuis le micro…","",             self._start_audio_record_dialog)
        self._add_action(audio_m, "Analyse Audio…",              "",             self._show_audio_analysis_panel)
        self._add_action(audio_m, "Sync Audio automatique…",     "",             self._show_audio_sync_panel)
        self._add_action(audio_m, "Synth procédural…",           "Ctrl+Y",       self._show_synth_editor,            action_id="show_synth")
        tl_m.addSeparator()

        # MIDI / OSC / DMX
        proto_m = tl_m.addMenu("Protocoles temps réel")
        self._add_action(proto_m, "MIDI…",                       "",             self._show_midi_panel)
        self._add_action(proto_m, "OSC…",                        "",             self._show_osc_panel)
        self._add_action(proto_m, "DMX / Artnet…",               "",             self._show_dmx_panel)
        self._rest_server_action = self._add_action(
            proto_m, "API REST locale…",                          "Ctrl+Shift+R",
            self._toggle_rest_server, is_checkable=True, action_id="rest_server")

        # ── Performance ──────────────────────────────────────────────────────
        perf_m = mb.addMenu("  Performance  ")
        self._add_action(perf_m, "Mode VJ — Plein écran",        "F11",          self._start_vj_mode,                action_id="vj_start")
        self._add_action(perf_m, "Quitter le mode VJ",           "Escape",       self._stop_vj_mode,                 action_id="vj_stop")
        perf_m.addSeparator()
        self._add_action(perf_m, "Mode VR — OpenXR",             "Ctrl+Shift+V", self._start_vr_mode,                action_id="vr_start")
        self._add_action(perf_m, "Quitter le mode VR",           "",             self._stop_vr_mode,                 action_id="vr_stop")
        self._add_action(perf_m, "Mappings contrôleurs XR…",     "",             self._show_xr_mapping_panel,        action_id="vr_mappings")
        perf_m.addSeparator()
        self._collab_action = self._add_action(
            perf_m, "Co-édition temps réel…",                     "Ctrl+Shift+C",
            self._show_collab_panel, action_id="collab_session")
        perf_m.addSeparator()

        # Visualisation live
        self._add_action(perf_m, "Spectre audio (FFT)",          "",             self._toggle_fft_display,           is_checkable=True)
        self._add_action(perf_m, "Oscilloscope",                 "",             self._toggle_oscilloscope_display,  is_checkable=True)

        # ── Outils ───────────────────────────────────────────────────────────
        outils_m = mb.addMenu("  Outils  ")
        self._add_action(outils_m, "Asset Store…",               "Ctrl+Shift+Z", self._show_asset_store,             action_id="asset_store")
        self._add_action(outils_m, "Plugins…",                   "",             self._show_plugin_panel)
        outils_m.addSeparator()
        self._add_action(outils_m, "4K / 64K Intro Toolkit…",    "Ctrl+Shift+K", self._show_intro_toolkit,           action_id="intro_toolkit")
        self._add_action(outils_m, "Estimation taille intro",     "",             self._toggle_intro_size_label,      is_checkable=True)
        outils_m.addSeparator()
        self._add_action(outils_m, "Profiler GPU…",              "",             self._show_gpu_profiler_panel)
        self._add_action(outils_m, "Benchmark…",                 "",             self._show_benchmark_panel)

        # ── Fenêtre ───────────────────────────────────────────────────────────
        win_m = mb.addMenu("  Fenêtre  ")

        # Panneaux
        panels_m = win_m.addMenu("Panneaux")
        panels_m.addAction(self.dock_left.toggleViewAction())
        panels_m.addAction(self.dock_editor.toggleViewAction())
        panels_m.addAction(self.dock_timeline.toggleViewAction())
        panels_m.addAction(self.dock_node_graph.toggleViewAction())
        panels_m.addAction(self.dock_script.toggleViewAction())
        panels_m.addAction(self.dock_synth.toggleViewAction())
        panels_m.addAction(self.dock_history.toggleViewAction())
        win_m.addSeparator()

        # Profils de disposition
        layout_m = win_m.addMenu("Profil de disposition")
        self._add_action(layout_m, "Studio Full",                "",             lambda: self._apply_layout_profile("studio"))
        self._add_action(layout_m, "VJ Compact",                 "",             lambda: self._apply_layout_profile("vj"))
        self._add_action(layout_m, "Code Only",                  "",             lambda: self._apply_layout_profile("code"))
        self._add_action(layout_m, "Timeline Only",              "",             lambda: self._apply_layout_profile("timeline"))
        layout_m.addSeparator()
        self._add_action(layout_m, "Sauvegarder la disposition…","",             self._save_layout_profile_dialog)
        self._add_action(layout_m, "Charger une disposition…",   "",             self._load_layout_profile_dialog)

        # Thème
        theme_m = win_m.addMenu("Thème de l'interface")
        self._action_theme_dark  = self._add_action(theme_m, "Sombre",           "", lambda: self._change_ui_theme("dark"),  is_checkable=True)
        self._action_theme_light = self._add_action(theme_m, "Clair",            "", lambda: self._change_ui_theme("light"), is_checkable=True)
        self._action_theme_auto  = self._add_action(theme_m, "Auto (système)",   "", lambda: self._change_ui_theme("auto"),  is_checkable=True)
        self._action_theme_auto.setChecked(True)
        win_m.addSeparator()
        self._add_action(win_m, "Galerie publique…",             "",             self._export_gallery,               action_id="export_gallery_outils")

        # ── Aide ─────────────────────────────────────────────────────────────
        help_m = mb.addMenu("  Aide  ")
        self._add_action(help_m, "Documentation interactive…",   "Shift+F1",    lambda: self.help_system.show_help())
        self._add_action(help_m, "Référence GLSL…",              "Ctrl+F1",     lambda: self.help_system.show_glsl_reference())
        help_m.addSeparator()
        tutos_m = help_m.addMenu("Tutoriels")
        self._add_action(tutos_m, "Tous les tutoriels…",          "",            lambda: self.help_system.show_tutorials())
        tutos_m.addSeparator()
        self._add_action(tutos_m, "Premier shader",               "",            lambda: self.help_system.start_tutorial("first_shader"))
        self._add_action(tutos_m, "Audio-réactif",                "",            lambda: self.help_system.start_tutorial("audio_reactive"))
        self._add_action(tutos_m, "Raymarching",                  "",            lambda: self.help_system.start_tutorial("raymarching_intro"))
        help_m.addSeparator()
        ressources_m = help_m.addMenu("Ressources en ligne")
        self._add_action(ressources_m, "Documentation GLSL (Khronos)…", "", lambda: self._open_url("https://www.khronos.org/opengl/wiki/Fragment_Shader"))
        self._add_action(ressources_m, "Shadertoy…",              "",            lambda: self._open_url("https://www.shadertoy.com"))
        self._add_action(ressources_m, "The Book of Shaders…",    "",            lambda: self._open_url("https://thebookofshaders.com"))
        help_m.addSeparator()
        self._add_action(help_m, "À propos de OpenShader…",       "",            self._show_about)

        # v2.6 — Appliquer les bindings sauvegardés à toutes les actions enregistrées
        self.shortcut_mgr.apply_all()

    def _add_action(self, menu: QMenu, label: str, shortcut: str, slot,
                    is_checkable=False, action_id: str = ""):
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if is_checkable:
            act.setCheckable(True)
        act.triggered.connect(slot)
        menu.addAction(act)
        # v2.6 — enregistrement dans le ShortcutManager
        if action_id and hasattr(self, 'shortcut_mgr'):
            self.shortcut_mgr.register_action(action_id, act)
        return act

    # ── Chargement ────────────────────────────────────────────────────────────

    def _load_default_shader(self):
        """Charge le shader par défaut au démarrage.

        Priorité :
          1. shaders/stoy/plasma.st  (si présent dans le répertoire du projet)
          2. Shader Shadertoy minimal inline  (fallback garanti — plus jamais de grille rouge)
        """
        base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default = os.path.join(base, 'shaders', 'stoy', 'plasma.st')

        if os.path.isfile(default):
            QTimer.singleShot(500, lambda: self._load_shader_file(default))
        else:
            # Fallback inline : shader coloré simple, garanti de compiler
            QTimer.singleShot(500, self._load_fallback_shader)

    def _load_fallback_shader(self):
        """Injecte un shader par défaut inline quand aucun fichier n'est disponible."""
        fallback = _SHADER_TEMPLATES.get("Shadertoy Minimal", "")
        if not fallback:
            fallback = (
                "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n"
                "{\n"
                "    vec2 uv = fragCoord / iResolution.xy;\n"
                "    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));\n"
                "    fragColor = vec4(col, 1.0);\n"
                "}\n"
            )
        if 'Image' in self.editors:
            self.editors['Image'].set_code(fallback)
        self._compile_source(fallback, 'Image')

    @pyqtSlot(str)
    def _load_shader_file(self, path: str):
        """Charge un fichier shader selon son extension.
        .trans  -> onglet Trans + load_trans_source()
        .st/.glsl -> onglet actif courant + _compile_source()
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            self._status.showMessage(f"Erreur lecture : {e}", 4000)
            return

        ext = os.path.splitext(path)[1].lower()

        if ext == '.trans':
            # Fichier de transition : toujours dans l onglet Trans
            if 'Trans' not in self.editors:
                self._status.showMessage("Onglet Trans introuvable", 4000)
                return
            # Bascule sur l onglet Trans
            for i in range(self.editor_tabs.count()):
                if self.editor_tabs.tabText(i).lstrip("○●✕ ").strip() == 'Trans':
                    self.editor_tabs.setCurrentIndex(i)
                    break
            self.editors['Trans'].set_code(source)
            ok, err = self.shader_engine.load_trans_source(source, source_path=path)
            self._active_trans_path = path
            self.hot_reload.watch(path)
            self._update_tab_status_trans(ok, err if not ok else None)
            if ok:
                self._status.showMessage(
                    f"Transition chargee : {os.path.basename(path)}", 3000)
                self.left_panel.update_shader_info("Trans", "transition", True, "")
            else:
                self._status.showMessage(
                    f"Erreur transition : {err[:80]}", 5000)
                self.left_panel.update_shader_info("Trans", "transition", False, err or "")
        else:
            # Shader normal : onglet actif
            try:
                raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
                current_tab_name = raw.lstrip("○●✕ ").strip()
                editor = self.editors[current_tab_name]
                editor.set_code(source)
                self._compile_source(source, current_tab_name, path)
            except KeyError as e:
                self._status.showMessage(f"Onglet invalide : {e}", 4000)

    @pyqtSlot(str)
    def _load_audio_file(self, path: str):
        ok, msg = self.audio_engine.load(path)
        # Ne pas lire audio_engine.duration ici : le chargement est asynchrone.
        # La durée sera mise à jour dans _on_audio_playback_ready via le signal
        # playback_ready, émis par QMediaPlayer une fois le fichier prêt.
        self._pending_audio_path = path   # mémorisé pour _on_audio_playback_ready
        if ok:
            self._lbl_audio.setText(f"🔊 {self.audio_engine.file_name}")
            self._status.showMessage(msg, 3000)
            self._on_timeline_seek(0.0)
        else:
            self._status.showMessage(f"Audio : {msg}", 4000)

    @pyqtSlot(float)
    def _on_audio_playback_ready(self, duration: float):
        """Appelé quand QMediaPlayer a fini de charger et que la durée est connue."""
        if duration > 0:
            self.timeline.duration = duration
            self.timeline_widget.set_duration(duration)
        path = getattr(self, '_pending_audio_path', None)
        if path:
            self.timeline_widget.refresh_audio_waveform(path, duration)
            self._pending_audio_path = None


    def _load_texture_dialog(self, channel: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Charger texture iChannel{channel}", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tif)"
        )
        if path:
            self._load_texture(channel, path)

    def _load_texture(self, channel: int, path: str):
        ok, msg = self.shader_engine.load_texture(channel, path)
        if ok:
            self._texture_paths[channel] = path
            self.left_panel.update_texture_label(channel, os.path.basename(path))
            self._status.showMessage(f"iChannel{channel} : {os.path.basename(path)}", 3000)
            self._render_is_dirty = True
        else:
            self._status.showMessage(f"Erreur iChannel{channel} : {msg}", 5000)

    def _load_image_shader(self, path: str):
        """Charge un shader dans la passe Image et gère les FX mémorisés."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            self._compile_source(source, 'Image', path)
            self._active_image_shader_path = path
            self.hot_reload.watch(path)  # v2.1

            if path in self._shader_fx_states:
                self.left_panel.restore_fx_state(
                    self._shader_fx_states[path], emit=True)
                log.debug("FX restaurés pour '%s'", os.path.basename(path))
            else:
                self.left_panel.restore_fx_state(
                    {"active": [False] * len(self.left_panel._fx_active), "params": {}},
                    emit=True)
        except (OSError, UnicodeDecodeError) as e:
            self._status.showMessage(f"Erreur chargement shader clip : {e}", 4000)
            self._active_image_shader_path = None

    def _update_tab_status_trans(self, ok: bool, error: str | None = None):
        """Met à jour l'onglet Trans (icône, couleur, surlignage erreur)."""
        if 'Trans' not in self.editors:
            return
        editor = self.editors['Trans']
        from PyQt6.QtGui import QColor
        # Mise à jour de l'onglet
        for i in range(self.editor_tabs.count()):
            raw = self.editor_tabs.tabText(i)
            if raw.lstrip("○●✕ ").strip() == 'Trans':
                if not self.shader_engine.trans_source.strip():
                    self.editor_tabs.setTabText(i, "○  Trans")
                    self.editor_tabs.tabBar().setTabTextColor(i, QColor("#3a4060"))
                elif ok:
                    self.editor_tabs.setTabText(i, "●  Trans")
                    self.editor_tabs.tabBar().setTabTextColor(i, QColor("#40a060"))
                else:
                    self.editor_tabs.setTabText(i, "✕  Trans")
                    self.editor_tabs.tabBar().setTabTextColor(i, QColor("#c04040"))
                break
        # Surlignage d'erreur dans l'éditeur
        if ok:
            editor.clear_error()
            from .shader_engine import TRANS_HEADER
            editor.set_header_lines(len(TRANS_HEADER.splitlines()))
        elif error:
            editor.show_error(error)

    def _compile_source(self, source: str, pass_name: str, path: str = ""):
        ok, error = self.shader_engine.load_shader_source(source, pass_name,
                                                              source_path=path or None)
        name  = os.path.basename(path) if path else "source directe"

        # Marque le projet comme modifié (v1.5)
        self._project_is_modified = True
        self._update_title()

        # Toujours mettre à jour l'onglet (icône + couleur)
        self._update_tab_status(pass_name)

        # Met à jour la toolbar et le panneau info seulement si l'onglet est visible
        current_raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        current_pass = current_raw.lstrip("○●✕ ").strip()
        if current_pass == pass_name:
            self._update_pass_toolbar(pass_name)
            stype = self.shader_engine.get_shader_type(pass_name)
            self.left_panel.update_shader_info(f"{name} ({pass_name})", stype, ok, error)
            editor = self.editors[pass_name]
            if ok:
                editor.clear_error()
                editor.set_header_lines(get_header_line_count(source))
                self._status.showMessage(f"✓ {pass_name} compilé", 3000)
            else:
                editor.show_error(error)
                self._status.showMessage(f"✗ Erreur compilation {pass_name}", 5000)

        self._render_is_dirty = True

        # v5.0 — Propage aux pairs si session co-édition active
        if ok:
            self._collab_on_shader_compiled(pass_name, source)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def _on_code_changed(self, source: str, pass_name: str):
        if pass_name == 'Trans':
            ok, err = self.shader_engine.load_trans_source(source)
            self._update_tab_status_trans(ok, err if not ok else None)
            if ok:
                self._active_trans_path = None  # force rechargement depuis timeline si besoin
                self._status.showMessage("✓ Transition compilée", 3000)
                self.left_panel.update_shader_info("Trans", "transition", True, "")
            else:
                self._status.showMessage(f"✗ Erreur transition : {err[:80]}", 5000)
                self.left_panel.update_shader_info("Trans", "transition", False, err or "")
            self._render_is_dirty = True
        else:
            self._compile_source(source, pass_name)

    @pyqtSlot(int)
    def _on_editor_tab_changed(self, index: int):
        raw = self.editor_tabs.tabText(index)
        pass_name = raw.lstrip("○●✕ ").strip()
        if pass_name == 'Trans':
            error = self.shader_engine.trans_error
            ok    = error is None
            self.left_panel.update_shader_info("Trans", "transition", ok, error or "")
        else:
            error = self.shader_engine.errors.get(pass_name)
            stype = self.shader_engine.get_shader_type(pass_name)
            self.left_panel.update_shader_info(f"({pass_name})", stype,
                                               error is None, error or "")
        self._update_pass_toolbar(pass_name)

    @pyqtSlot(str, object)
    def _on_uniform_changed(self, name: str, value):
        # v2.5 — passer par le CommandStack
        old_value = self.shader_engine.get_uniform(name)
        self.cmd_stack.push(SetUniformCommand(self.shader_engine, name, old_value, value))
        self._render_is_dirty = True
        # v4.0 — propager au DMX engine (ShaderEngine n'est pas un QObject)
        try:
            self.dmx_engine.uniform_changed_slot(name, float(value))
        except Exception:
            pass

    @pyqtSlot(float)
    def _on_audio_amplitude(self, amplitude: float):
        if self._is_recording and self._recording_track:
            cmd = AddKeyframeCommand(self._recording_track, self._current_time, amplitude, self.timeline_widget.canvas, 'linear')
            self.cmd_stack.push(cmd)

    @pyqtSlot(object)
    def _on_effect_changed(self, glsl_or_none):
        """Active/désactive les effets post-processing (shader composé ou None)."""
        if glsl_or_none is None:
            self.shader_engine.load_shader_source("", "Post")
            self._status.showMessage("Effets désactivés", 2000)
        else:
            ok, err = self.shader_engine.load_shader_source(glsl_or_none, "Post")
            if ok:
                self._status.showMessage("✓ Effets appliqués", 2000)
            else:
                self._status.showMessage(f"✗ Erreur FX : {err[:80]}", 5000)
                log.error("Erreur compilation FX composé : %s", err)
        self._render_is_dirty = True

    @pyqtSlot(object)
    def _on_fx_state_changed(self, _glsl_or_none):
        """Sauvegarde l'état FX courant sur le shader actif dès que l'utilisateur
        change un toggle ou un paramètre — v2.5 : passe par le CommandStack."""
        if self._active_image_shader_path:
            old_state = self._shader_fx_states.get(self._active_image_shader_path, {})
            new_state = self.left_panel.get_fx_state()
            if old_state != new_state:
                # ── Fix récursion infinie ──────────────────────────────────────
                # _shader_fx_states DOIT être mis à jour AVANT le push().
                # push() appelle redo() → restore_fx_state(emit=False maintenant)
                # mais si emit=True, il rappellerait _on_fx_state_changed() alors
                # que old_state serait encore {} → boucle infinie → RecursionError.
                # En mettant à jour le dict en premier, tout appel récursif éventuel
                # trouverait old_state == new_state et sortirait immédiatement.
                self._shader_fx_states[self._active_image_shader_path] = new_state
                self.cmd_stack.push(SetFXStateCommand(
                    self.left_panel,
                    self._active_image_shader_path,
                    old_state, new_state
                ))
            else:
                self._shader_fx_states[self._active_image_shader_path] = new_state
            log.debug("FX sauvegardé pour '%s'",
                      os.path.basename(self._active_image_shader_path))

    @pyqtSlot(float)
    def _on_fps_updated(self, fps: float):
        self.left_panel.update_fps(fps)
        # v2.3 — Init paresseuse du panneau upscaler (GL contexte prêt dès le 1er FPS)
        if self._upscaler_panel is None:
            self._init_upscaler_panel()
        # Rafraîchit les stats d'upscaling dans le panneau (si ouvert)
        if self._upscaler_panel is not None:
            self._upscaler_panel.refresh_stats()

    # ── Upscaling IA (v2.3) ──────────────────────────────────────────────────

    def _init_upscaler_panel(self):
        """Crée le UpscalerPanel après que le contexte GL est initialisé."""
        ctrl = getattr(self.gl_widget, "upscaler_ctrl", None)
        if ctrl is None:
            return
        self._upscaler_panel = UpscalerPanel(ctrl, parent=self)
        ctrl.mode_changed.connect(self._on_upscale_mode_changed)
        ctrl.render_size_changed.connect(self._on_upscale_render_size_changed)
        # Applique un mode en attente (ex: sauvegardé dans la session)
        if self._pending_upscale_mode != "off":
            ctrl.set_mode(self._pending_upscale_mode)
        log.info("UpscalerPanel créé")

    def _on_upscale_btn_clicked(self, mode: str):
        """Clic sur un bouton inline de mode upscaling."""
        # Sync visuel des boutons
        for k, btn in self._upscale_btns.items():
            btn.setChecked(k == mode)
        # Applique le mode
        ctrl = getattr(self.gl_widget, "upscaler_ctrl", None)
        if ctrl is not None:
            ctrl.set_mode(mode)
        else:
            # GL pas encore initialisé — on mémorise pour appliquer au démarrage
            self._pending_upscale_mode = mode

    def _on_upscale_mode_changed(self, mode: str):
        """Upscaler mode changed — synchronise les boutons inline."""
        # Sync boutons toolbar
        for k, btn in self._upscale_btns.items():
            btn.setChecked(k == mode)
        # Sync panneau UpscalerPanel si ouvert
        if self._upscaler_panel is not None:
            self._upscaler_panel._on_mode_changed_external(mode)
        # Forcer refresh du FBO Qt (la résolution vient de changer)
        self.gl_widget._qt_fbo = None

    def _on_upscale_render_size_changed(self, rw: int, rh: int):
        """Résolution de rendu changée — refresh FBO Qt."""
        self.gl_widget._qt_fbo = None
        log.info("Upscaler render size → %d×%d", rw, rh)

    def _show_upscaler_panel(self):
        """Ouvre le panneau de configuration détaillé de l'upscaler IA."""
        if self._upscaler_panel is None:
            self._init_upscaler_panel()
        if self._upscaler_panel is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Upscaling IA",
                "L'upscaler n'est pas encore disponible.\n"
                "Attendez que le contexte OpenGL soit initialisé (première frame)."
            )
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("⚡ Upscaling IA — ESRGAN-lite GPU")
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(
            "QDialog{background:#0e1016;color:#c8ccd8;}"
            "QWidget{background:#0e1016;}"
        )
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._upscaler_panel)
        footer = QHBoxLayout()
        footer.setContentsMargins(8, 6, 8, 8)
        from PyQt6.QtWidgets import QPushButton
        btn_close = QPushButton("Fermer")
        btn_close.setStyleSheet(
            "QPushButton{background:#161820;color:#7880a0;border:1px solid #1a1d28;"
            "border-radius:4px;padding:4px 14px;font:9px 'Segoe UI';}"
            "QPushButton:hover{background:#1e2232;color:#c0c8e0;}"
        )
        btn_close.clicked.connect(dlg.close)
        footer.addStretch()
        footer.addWidget(btn_close)
        lay.addLayout(footer)
        dlg.exec()
        # Le panneau est détaché du dialog à la fermeture — on efface la ref
        # pour qu'il soit reconstruit la prochaine fois (évite double-parent)
        self._upscaler_panel.setParent(None)
        self._upscaler_panel = None

    def _on_resolution_changed(self, idx: int):
        """Applique la résolution choisie dans le combobox."""
        w, h = self._res_combo.itemData(idx)
        # v2.3 — Si l'upscaler est actif, notifie la nouvelle résolution native
        # (il recalculera render_w/h et resizera le ShaderEngine lui-même)
        ctrl = getattr(self.gl_widget, "upscaler_ctrl", None)
        if ctrl is not None and ctrl.is_active:
            ctrl.set_native_size(w, h)
        else:
            self.gl_widget.set_resolution(w, h)

    @pyqtSlot(str)
    def _on_render_error(self, error: str):
        # Erreur déjà gérée lors de la compilation
        pass

    @pyqtSlot(float)
    def _on_timeline_seek(self, t: float):
        """L'utilisateur a cliqué sur la timeline → seek."""
        self._current_time = t
        self._play_offset = t
        self._play_start_wall = time.perf_counter()
        if self.audio_engine.has_file:
            self.audio_engine.seek(t)

    @pyqtSlot()
    def _on_timeline_data_changed(self):
        self._render_is_dirty = True
        # Synchronise la durée dans l'onglet Export
        self.left_panel.set_export_duration(self.timeline.duration)

    # ── Transport ─────────────────────────────────────────────────────────────

    def _on_play_pause(self):
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self._play_start_wall = time.perf_counter()
        self._is_playing = True
        self._btn_play.setText("⏸")
        if self.audio_engine.has_file:
            self.audio_engine.play()

    def _pause(self):
        self._play_offset = self._current_time
        self._is_playing = False
        self._btn_play.setText("▶")
        if self.audio_engine.has_file:
            self.audio_engine.pause()

    def _on_stop(self):
        if self._is_recording:
            self._is_recording = False
            self.audio_engine.stop_recording()
            self._recording_track = None

        self._is_playing = False
        self._play_offset = 0.0
        self._current_time = 0.0
        self._btn_play.setText("▶")
        if self.audio_engine.has_file:
            self.audio_engine.stop()

    def _on_rewind(self):
        self._on_timeline_seek(0.0)

    def _recompile_current(self):
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        current_tab_name = raw.lstrip("○●✕ ").strip()
        editor = self.editors.get(current_tab_name)
        if editor:
            source = editor.get_code()
            if source.strip():
                self._compile_source(source, current_tab_name)

    # ── Tick principal ────────────────────────────────────────────────────────

    def _tick(self):
        """Appelé ~60 fois/sec. Met à jour le temps et le viewport."""
        if self._is_playing:
            if self.audio_engine.has_file and self.audio_engine.is_playing:
                # Synchronisation audio → shader
                self._current_time = self.audio_engine.get_position()
            else:
                elapsed = time.perf_counter() - self._play_start_wall
                self._current_time = self._play_offset + elapsed

            # ── Bouclage loop region ──────────────────────────────────────
            tl = self.timeline
            if getattr(tl, 'loop_enabled', False):
                loop_in  = getattr(tl, 'loop_in',  0.0)
                loop_out = getattr(tl, 'loop_out', tl.duration)
                if loop_out > loop_in and self._current_time >= loop_out:
                    # Repart depuis loop_in
                    self._play_offset = loop_in
                    self._play_start_wall = time.perf_counter()
                    self._current_time = loop_in
                    if self.audio_engine.has_file:
                        self.audio_engine.seek(loop_in)
            elif self._current_time >= tl.duration:
                # Fin de timeline sans loop : arrêt
                self._on_stop()

        time_changed = abs(self._current_time - self._last_rendered_time) > 1e-6

        if self._is_playing or self._render_is_dirty or time_changed:
            # Évalue la timeline et met à jour les uniforms
            uniforms = self.timeline.evaluate(self._current_time)

            # ── Collecte des pistes scène/transition ──────────────────────────
            # Toutes les pistes shader actives → rendu multi-layer (overlay)
            active_layer_paths: list[str] = []   # ordonnées par piste (bas → haut)
            trans_path    = None   # piste trans uniforme _trans
            trans_start   = None
            trans_end     = None

            for track in self.timeline.tracks:
                if track.value_type == 'shader':
                    # Trouve le clip actif sur cette piste au temps courant
                    kfs = track.keyframes
                    active_path = None
                    for i, kf in enumerate(kfs):
                        kf_end = kfs[i+1].time if i+1 < len(kfs) else float('inf')
                        if kf.time <= self._current_time < kf_end and kf.value:
                            active_path = str(kf.value)
                            break
                    # Compat legacy : piste _scene_a charge aussi la passe Image principale
                    if track.uniform_name == '_scene_a' and active_path:
                        # ── FIX FREEZE : ne pas recompiler le shader pendant un drag ──
                        # Pendant le drag d'un clip, les kf.time changent à chaque
                        # mouseMoveEvent → active_path change en boucle → _load_image_shader
                        # appelle _compile_source (compilation GLSL bloquante ~50-200ms)
                        # → thread principal bloqué → UI freeze.
                        # On diffère le rechargement jusqu'à la fin du drag.
                        _canvas = getattr(self.timeline_widget, 'canvas', None)
                        _dragging = _canvas is not None and _canvas.is_dragging
                        if active_path != self._active_scene_a_path and not _dragging:
                            self._load_image_shader(active_path)
                            self._active_scene_a_path = active_path
                    elif track.uniform_name not in ('_scene_a', '_scene_b'):
                        uniforms.pop(track.uniform_name, None)
                    # Toutes les pistes shader (y compris _scene_a) contribuent aux layers
                    active_layer_paths.append(active_path or '')

                elif track.value_type == 'trans' and track.uniform_name == '_trans':
                    uniforms.pop(track.uniform_name, None)
                    kfs = track.keyframes
                    for i, kf in enumerate(kfs):
                        kf_end = kfs[i+1].time if i+1 < len(kfs) else self.timeline.duration
                        if kf.time <= self._current_time < kf_end and kf.value:
                            trans_path  = str(kf.value)
                            trans_start = kf.time
                            trans_end   = kf_end
                            break

            # ── Mise à jour des layers multi-pistes ───────────────────────────
            self.shader_engine.set_active_layers(active_layer_paths)

            # ── Chargement scène B (compatibilité transition) ─────────────────
            scene_b_path = None
            if trans_path:
                # Pour la transition, scene_b = couche suivante si disponible
                for track in self.timeline.tracks:
                    if track.value_type == 'shader' and track.uniform_name == '_scene_b':
                        kfs = track.keyframes
                        for i, kf in enumerate(kfs):
                            kf_end = kfs[i+1].time if i+1 < len(kfs) else float('inf')
                            if kf.time <= self._current_time < kf_end and kf.value:
                                scene_b_path = str(kf.value)
                                break
            if scene_b_path and scene_b_path != self._active_scene_b_path:
                try:
                    with open(scene_b_path, 'r', encoding='utf-8') as f:
                        src_b = f.read()
                    ok, err = self.shader_engine.load_scene_b_source(src_b, source_path=scene_b_path)
                    if not ok:
                        log.error("Erreur scène B '%s': %s", scene_b_path, err)
                    self._active_scene_b_path = scene_b_path
                except (OSError, UnicodeDecodeError) as e:
                    log.error("Impossible de charger scène B: %s", e)

            # ── Chargement du shader de transition ────────────────────────────
            if trans_path and trans_path != self._active_trans_path:
                try:
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        src_t = f.read()
                    ok, err = self.shader_engine.load_trans_source(src_t, source_path=trans_path)
                    if ok:
                        self._active_trans_path = trans_path
                        if 'Trans' in self.editors:
                            self.editors['Trans'].set_code(src_t)
                        self._update_tab_status_trans(True, None)
                        log.debug("Transition chargée : %s", os.path.basename(trans_path))
                    else:
                        log.error("Erreur transition '%s': %s", trans_path, err)
                        self._active_trans_path = None
                        if 'Trans' in self.editors:
                            self.editors['Trans'].set_code(src_t)
                        self._update_tab_status_trans(False, err)
                except (OSError, UnicodeDecodeError) as e:
                    log.error("Impossible de charger la transition: %s", e)

            # ── Calcul de iProgress + activation/désactivation ────────────────
            if trans_path and trans_start is not None and trans_end is not None:
                span = trans_end - trans_start
                progress = (self._current_time - trans_start) / span if span > 0 else 0.0
                progress = max(0.0, min(1.0, progress))
                self.shader_engine.set_transition(progress, active=True)
            else:
                self.shader_engine.set_transition(0.0, active=False)

            for name, val in uniforms.items():
                self.shader_engine.set_uniform(name, val)

            # Synchronise le viewport
            self.gl_widget.set_time(self._current_time)

            # Synchronise la tête de lecture de la timeline
            self.timeline_widget.set_current_time(self._current_time)

            self._last_rendered_time = self._current_time
            self._render_is_dirty = False

        # Affichage du temps
        m   = int(self._current_time) // 60
        s   = int(self._current_time) % 60
        ms  = int((self._current_time % 1) * 1000)
        ts  = f"{m:02d}:{s:02d}.{ms:03d}"
        self._lbl_time.setText(ts)
        self._lbl_status_time.setText(ts)

        # ── Auto-save périodique (v1.5) ───────────────────────────────────────
        now = time.time()
        if now - self._last_autosave_time >= self._autosave_interval:
            self._autosave_session()
            self._last_autosave_time = now

        # ── v5.0 — Co-édition : envoi curseur local ───────────────────────────
        self._collab_send_cursor()

        # ── v2.0 — Script engine tick ─────────────────────────────────────────
        self.script_engine.tick(self._current_time)

        # ── v2.8 — Synth procédural : synchronisation iTime ─────────────────
        if hasattr(self, '_synth_editor'):
            self._synth_editor.sync_itime(self._current_time)

        # ── v3.0 — Plugins natifs C++ : tick agrégé → uniforms GLSL ─────────
        if self.plugin_manager.native.get_all():
            dt    = getattr(self, '_last_dt', 0.016)
            rms   = getattr(self.audio_engine, 'rms', 0.0)
            bpm   = getattr(self.timeline, 'bpm', 120.0)
            beat  = getattr(self, '_current_beat', 0.0)
            native_uniforms = self.plugin_manager.native.tick_all(
                dt=dt, itime=self._current_time,
                rms=rms, bpm=bpm, beat=beat
            )
            for uname, uval in native_uniforms.items():
                self.shader_engine.set_uniform(uname, uval)

    # ── v1.5 — Gestionnaire de Projets ────────────────────────────────────────

    def _collect_project_data(self) -> dict:
        """Sérialise l'état courant en dict JSON."""
        timeline_data = {}
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
                tmp.close()
                self.timeline.save(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    timeline_data = json.load(f)
                os.unlink(tmp.name)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            log.warning("Erreur sérialisation timeline: %s", e)
        # v2.8 — Synth procédural
        synth_data = {}
        if hasattr(self, '_synth_editor'):
            try:
                synth_data = self._synth_editor.to_dict()
            except Exception as e:
                log.warning("Erreur sérialisation synth : %s", e)

        return {
            "version": "2.8",
            "shaders": {name: editor.get_code() for name, editor in self.editors.items()},
            "audio":   {"filename": os.path.basename(self.audio_engine.file_path) if self.audio_engine.file_path else None},
            "timeline": timeline_data,
            "textures": [os.path.basename(p) if p else None for p in self._texture_paths],
            "shader_fx_states": self._shader_fx_states,
            "synth": synth_data,
            "xr_mappings": (self._vr_window._ctrl.to_dict()  # v2.9
                            if self._vr_window else self._xr_saved_mappings),
            "osc": self.osc_engine.to_dict(),   # v3.0 — config OSC par scène
            "dmx": self.dmx_engine.to_dict(),   # v4.0 — config DMX/Artnet par scène
            "ai":  self.ai_generator.to_dict(), # v3.5 — config IA (hôtes, modèle)
            "scene_graph": self._scene_graph.to_dict(),  # v6.0 — Scene Graph
            "arrangement": self._arrangement.to_dict(), # v6.0 — Arrangement
        }

    def _save_demomaker_bundle(self, dest_path: str) -> bool:
        """Crée un bundle .demomaker (ZIP)."""
        import zipfile as _zf
        try:
            data = self._collect_project_data()
            with _zf.ZipFile(dest_path, 'w', compression=_zf.ZIP_DEFLATED) as zf:
                zf.writestr("project.json", json.dumps(data, indent=2, ensure_ascii=False))
                if self.audio_engine.file_path and os.path.exists(self.audio_engine.file_path):
                    zf.write(self.audio_engine.file_path,
                             os.path.join("audio", os.path.basename(self.audio_engine.file_path)))
                for p in self._texture_paths:
                    if p and os.path.exists(p):
                        zf.write(p, os.path.join("textures", os.path.basename(p)))
            return True
        except (OSError, zipfile.BadZipFile, ValueError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder le bundle :\n{e}")
            return False

    def _load_demomaker_bundle(self, src_path: str) -> bool:
        """Charge un bundle .demomaker (ZIP) ou .dmk (JSON)."""
        import zipfile as _zf
        try:
            if _zf.is_zipfile(src_path):
                extract_dir = tempfile.mkdtemp(prefix="demomaker_load_")
                with _zf.ZipFile(src_path, 'r') as zf:
                    zf.extractall(extract_dir)
                with open(os.path.join(extract_dir, "project.json"), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                audio_dir   = os.path.join(extract_dir, "audio")
                texture_dir = os.path.join(extract_dir, "textures")
            else:
                with open(src_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                extract_dir = None
                audio_dir   = os.path.dirname(src_path)
                texture_dir = os.path.dirname(src_path)

            if "shaders" in data:
                for pass_name, code in data["shaders"].items():
                    if pass_name in self.editors:
                        self.editors[pass_name].set_code(code)
                        self._compile_source(code, pass_name)

            audio_info = data.get("audio", {})
            audio_fn   = audio_info.get("filename") or audio_info.get("path")
            if audio_fn:
                for candidate in [os.path.join(audio_dir, audio_fn), audio_fn]:
                    if os.path.exists(candidate):
                        self._load_audio_file(candidate)
                        break
                else:
                    self._status.showMessage(f"Audio introuvable : {audio_fn}", 5000)
            else:
                self.audio_engine.stop()

            if "timeline" in data:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
                    json.dump(data["timeline"], tmp)
                    tmp.close()
                    try:
                        self.timeline.load(tmp.name)
                    except ValueError as _tl_err:
                        log.warning("Chargement timeline échoué : %s", _tl_err)
                    finally:
                        os.unlink(tmp.name)
                self.timeline_widget.set_duration(self.timeline.duration)
                self.timeline_widget.sync_bpm_controls()
                self.timeline_widget.sync_loop_controls()
                self.timeline_widget.canvas.update()
                self.timeline_widget.timeline_data_changed.emit()
                self._render_is_dirty = True

            if "textures" in data:
                for i, tex in enumerate(data["textures"]):
                    if not tex:
                        self._texture_paths[i] = None
                        continue
                    for candidate in [os.path.join(texture_dir, tex) if texture_dir else None, tex]:
                        if candidate and os.path.exists(candidate):
                            self._load_texture(i, candidate)
                            break
                    else:
                        self._texture_paths[i] = None

            if "shader_fx_states" in data:
                self._shader_fx_states = data["shader_fx_states"]
                if self._active_image_shader_path and \
                        self._active_image_shader_path in self._shader_fx_states:
                    self.left_panel.restore_fx_state(
                        self._shader_fx_states[self._active_image_shader_path], emit=True)

            # v6.0 — Restaure l'Arrangement View
            if "arrangement" in data and data["arrangement"]:
                try:
                    self._arrangement.from_dict(data["arrangement"])
                    if self._arrangement_view:
                        self._arrangement_view.refresh_cue_panel()
                        self._arrangement_view._canvas.update()
                    log.info("Arrangement restauré (%d pistes)", len(self._arrangement.tracks))
                except Exception as e:
                    log.warning("Restauration Arrangement : %s", e)

            # v6.0 — Restaure le Scene Graph
            if "scene_graph" in data and data["scene_graph"]:
                try:
                    self._scene_graph.from_dict(data["scene_graph"])
                    if self._scene_graph_wgt:
                        self._scene_graph_wgt.refresh()
                    log.info("Scene Graph restauré (%d scènes)", len(self._scene_graph.scenes))
                except Exception as e:
                    log.warning("Restauration Scene Graph : %s", e)

            # v2.8 — Restaure l'état du synthétiseur procédural
            if "synth" in data and data["synth"] and hasattr(self, '_synth_editor'):
                try:
                    self._synth_editor.from_dict(data["synth"])
                except Exception as e:
                    log.warning("Restauration synth : %s", e)

            # v2.9 — Restaure les mappings contrôleurs XR
            if "xr_mappings" in data and data["xr_mappings"]:
                self._xr_saved_mappings = data["xr_mappings"]
                log.info("XR mappings restaurés (%d)", len(self._xr_saved_mappings))

            # v3.0 — Restaure la configuration OSC
            if "osc" in data and data["osc"]:
                try:
                    self.osc_engine.from_dict(data["osc"])
                    log.info("Configuration OSC restaurée (%d mappings)",
                             len(self.osc_engine.get_mappings()))
                except Exception as e:
                    log.warning("Erreur restauration OSC : %s", e)

            # v4.0 — Restaure la configuration DMX/Artnet
            if "dmx" in data and data["dmx"]:
                try:
                    self.dmx_engine.from_dict(data["dmx"])
                    log.info("Configuration DMX restaurée (%d fixtures, %d mappings)",
                             len(self.dmx_engine.get_fixtures()),
                             len(self.dmx_engine.get_mappings()))
                except Exception as e:
                    log.warning("Erreur restauration DMX : %s", e)

            # v3.5 — Restaure la configuration IA (hôtes, modèle préféré)
            if "ai" in data and data["ai"]:
                try:
                    self.ai_generator.from_dict(data["ai"])
                except Exception as e:
                    log.warning("Erreur restauration config IA : %s", e)

            if extract_dir:
                shutil.rmtree(extract_dir, ignore_errors=True)
            return True
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError, ValueError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir le projet :\n{e}")
            return False

    def _save_project_quick(self):
        """Ctrl+S — sauvegarde rapide sur le fichier courant."""
        if self._folder_mode and self._current_project_path:
            # En mode dossier, _current_project_path est le dossier projet
            folder = self._current_project_path
            if os.path.isdir(folder):
                ok = self._save_project_to_folder(folder)
                if ok:
                    self._project_is_modified = False
                    self._update_title()
                    self._status.showMessage(
                        f"Projet sauvegardé : {os.path.basename(folder)}/", 4000)
                    self._cloud_notify_save()   # v5.0
                return
        if self._current_project_path and not self._folder_mode:
            ok = self._save_demomaker_bundle(self._current_project_path)
            if ok:
                self._project_is_modified = False
                self._update_title()
                self._status.showMessage(
                    f"Projet sauvegardé : {os.path.basename(self._current_project_path)}", 4000)
                self._cloud_notify_save()       # v5.0
        else:
            self._save_project()

    def _save_project(self):
        """Enregistrer sous… — bundle .demomaker ou .dmk."""
        default = ""
        if self._current_project_path:
            default = os.path.splitext(self._current_project_path)[0] + ".demomaker"
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le projet", default,
            "OpenShader Bundle (*.demomaker);;OpenShader JSON (*.dmk)"
        )
        if not path:
            return
        if not (path.endswith('.demomaker') or path.endswith('.dmk')):
            path += '.demomaker'

        if path.endswith('.demomaker'):
            ok = self._save_demomaker_bundle(path)
        else:
            data = self._collect_project_data()
            data["audio"] = {"path": self.audio_engine.file_path}
            data["textures"] = self._texture_paths
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                ok = True
            except (OSError, TypeError) as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer :\n{e}")
                ok = False

        if ok:
            self._current_project_path = path
            self._project_is_modified  = False
            self._update_title()
            self._add_to_recent(path)
            self._status.showMessage(f"Projet enregistré : {os.path.basename(path)}", 4000)

    def _open_project(self):
        """Charge un projet .demomaker ou .dmk."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un projet", "",
            "Projets OpenShader (*.demomaker *.dmk)"
        )
        if not path:
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Fichier introuvable", f"Introuvable :\n{path}")
            self._remove_from_recent(path)
            self._update_recent_menu()
            return
        # v5.0 — Détection automatique : dossier = format versionné
        if os.path.isdir(path):
            ok = self._load_project_from_folder(path)
            if ok:
                self._current_project_path = path
                self._folder_mode = True
                if hasattr(self, '_folder_mode_action'):
                    self._folder_mode_action.setChecked(True)
                self._project_is_modified = False
                self._update_title()
                self._add_to_recent(path)
                self._status.showMessage(f"Projet chargé : {os.path.basename(path)}/", 4000)
            return
        ok = self._load_demomaker_bundle(path)
        if ok:
            self._current_project_path = path
            self._folder_mode = False
            if hasattr(self, '_folder_mode_action'):
                self._folder_mode_action.setChecked(False)
            self._project_is_modified  = False
            self._update_title()
            self._add_to_recent(path)
            self._status.showMessage(f"Projet chargé : {os.path.basename(path)}", 4000)

    def _autosave_session(self):
        try:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(self._autosave_dir, f"autosave_{ts}.demomaker")
            self._save_demomaker_bundle(dest)
            saved = sorted(f for f in os.listdir(self._autosave_dir)
                           if f.startswith("autosave_") and f.endswith(".demomaker"))
            for old in saved[:-10]:
                try:
                    os.remove(os.path.join(self._autosave_dir, old))
                except OSError:
                    pass
        except (OSError, zipfile.BadZipFile, ValueError) as e:
            log.warning("Échec auto-save : %s", e)

    def _restore_autosave_dialog(self):
        saves = sorted(
            (f for f in os.listdir(self._autosave_dir)
             if f.startswith("autosave_") and f.endswith(".demomaker")),
            reverse=True
        )
        if not saves:
            QMessageBox.information(self, "Auto-save", "Aucune sauvegarde automatique disponible.")
            return
        item, ok = QInputDialog.getItem(
            self, "Restaurer depuis l'auto-save",
            "Choisir une session :", saves, 0, False
        )
        if ok and item:
            self._load_demomaker_bundle(os.path.join(self._autosave_dir, item))
            self._status.showMessage(f"Session restaurée : {item}", 5000)

    # ──────────────────────────────────────────────────────────────────────────
    # v5.0 — Sauvegarde en dossier versionné (Git-friendly)
    # ──────────────────────────────────────────────────────────────────────────

    _DEMOMAKERIGNORE_TEMPLATE = """\
# .demomakerignore — fichiers exclus du dossier projet (ex: gros binaires audio)
# Syntaxe : un pattern glob par ligne, # = commentaire, ! = négation
audio/*.mp3
audio/*.wav
audio/*.ogg
audio/*.flac
textures/*.png
textures/*.jpg
textures/*.jpeg
textures/*.exr
textures/*.hdr
"""

    def _toggle_folder_mode(self):
        """Active/désactive le mode sauvegarde en dossier versionné."""
        self._folder_mode = self._folder_mode_action.isChecked()
        state = "activé" if self._folder_mode else "désactivé"
        self._status.showMessage(f"Mode dossier versionné {state}", 3000)

    def _read_demomakerignore(self, folder: str) -> list:
        """Lit le fichier .demomakerignore et retourne la liste de patterns."""
        import fnmatch as _fnmatch
        ignore_path = os.path.join(folder, ".demomakerignore")
        if not os.path.exists(ignore_path):
            return []
        patterns = []
        with open(ignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
        return patterns

    def _is_ignored(self, rel_path: str, patterns: list) -> bool:
        """Retourne True si rel_path correspond à un pattern .demomakerignore."""
        import fnmatch as _fnmatch
        for pattern in patterns:
            negate = pattern.startswith('!')
            p = pattern[1:] if negate else pattern
            if _fnmatch.fnmatch(rel_path, p) or _fnmatch.fnmatch(os.path.basename(rel_path), p):
                if negate:
                    return False
        for pattern in patterns:
            negate = pattern.startswith('!')
            if negate:
                continue
            p = pattern
            if _fnmatch.fnmatch(rel_path, p) or _fnmatch.fnmatch(os.path.basename(rel_path), p):
                return True
        return False

    def _save_project_to_folder(self, folder: str) -> bool:
        """Sauvegarde le projet en dossier décompressé (format Git-friendly).

        Structure générée :
            <folder>/
                project.json          ← métadonnées & uniforms (sans shaders)
                shaders/
                    Image.glsl
                    BufA.glsl          ← un fichier par passe
                    Trans.glsl
                timeline.json          ← keyframes & BPM
                mappings/
                    midi.json
                    osc.json
                    dmx.json
                    xr.json
                audio/                 ← copie audio sauf si ignoré
                textures/              ← copie textures sauf si ignoré
                .demomakerignore       ← créé si absent
        """
        try:
            os.makedirs(folder, exist_ok=True)
            ignore_file = os.path.join(folder, ".demomakerignore")
            if not os.path.exists(ignore_file):
                with open(ignore_file, 'w', encoding='utf-8') as f:
                    f.write(self._DEMOMAKERIGNORE_TEMPLATE)
            ignore_patterns = self._read_demomakerignore(folder)

            data = self._collect_project_data()

            # ── Shaders : un fichier .glsl par passe ──────────────────────────
            shaders_dir = os.path.join(folder, "shaders")
            os.makedirs(shaders_dir, exist_ok=True)
            shader_files = {}
            for pass_name, code in data.get("shaders", {}).items():
                safe_name = pass_name.replace(" ", "_").replace("/", "_")
                glsl_file = f"{safe_name}.glsl"
                with open(os.path.join(shaders_dir, glsl_file), 'w', encoding='utf-8') as f:
                    f.write(code)
                shader_files[pass_name] = glsl_file
            # On n'embarque plus le code inline dans project.json
            meta = {k: v for k, v in data.items()
                    if k not in ("shaders", "timeline", "midi", "osc", "dmx", "xr_mappings")}
            meta["shader_files"] = shader_files
            meta["_format"] = "folder_v1"

            # ── project.json (métadonnées légères) ───────────────────────────
            with open(os.path.join(folder, "project.json"), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # ── timeline.json (keyframes diff-friendly) ───────────────────────
            tl_data = data.get("timeline", {})
            with open(os.path.join(folder, "timeline.json"), 'w', encoding='utf-8') as f:
                json.dump(tl_data, f, indent=2, ensure_ascii=False)

            # ── mappings/ (MIDI, OSC, DMX, XR) ───────────────────────────────
            mappings_dir = os.path.join(folder, "mappings")
            os.makedirs(mappings_dir, exist_ok=True)
            for key, filename in [("osc", "osc.json"), ("dmx", "dmx.json"),
                                   ("xr_mappings", "xr.json")]:
                payload = data.get(key)
                if payload:
                    with open(os.path.join(mappings_dir, filename), 'w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)

            # ── Audio ─────────────────────────────────────────────────────────
            if self.audio_engine.file_path and os.path.exists(self.audio_engine.file_path):
                audio_dir = os.path.join(folder, "audio")
                os.makedirs(audio_dir, exist_ok=True)
                rel = os.path.join("audio", os.path.basename(self.audio_engine.file_path))
                if not self._is_ignored(rel, ignore_patterns):
                    dest = os.path.join(folder, rel)
                    if not os.path.exists(dest):
                        shutil.copy2(self.audio_engine.file_path, dest)

            # ── Textures ──────────────────────────────────────────────────────
            for tex_path in self._texture_paths:
                if tex_path and os.path.exists(tex_path):
                    tex_dir = os.path.join(folder, "textures")
                    os.makedirs(tex_dir, exist_ok=True)
                    rel = os.path.join("textures", os.path.basename(tex_path))
                    if not self._is_ignored(rel, ignore_patterns):
                        dest = os.path.join(folder, rel)
                        if not os.path.exists(dest):
                            shutil.copy2(tex_path, dest)

            log.info("Projet sauvegardé en dossier : %s", folder)
            return True
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder le dossier projet :\n{e}")
            return False

    def _load_project_from_folder(self, folder: str) -> bool:
        """Charge un projet depuis un dossier versionné."""
        try:
            project_json = os.path.join(folder, "project.json")
            if not os.path.exists(project_json):
                QMessageBox.critical(self, "Erreur",
                                     f"Dossier projet invalide (project.json manquant) :\n{folder}")
                return False

            with open(project_json, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # ── Shaders ───────────────────────────────────────────────────────
            shaders_dir = os.path.join(folder, "shaders")
            shader_files = meta.get("shader_files", {})
            shaders = {}
            for pass_name, glsl_file in shader_files.items():
                glsl_path = os.path.join(shaders_dir, glsl_file)
                if os.path.exists(glsl_path):
                    with open(glsl_path, 'r', encoding='utf-8') as f:
                        shaders[pass_name] = f.read()
            for pass_name, code in shaders.items():
                if pass_name in self.editors:
                    self.editors[pass_name].set_code(code)
                    self._compile_source(code, pass_name)

            # ── Timeline ──────────────────────────────────────────────────────
            tl_path = os.path.join(folder, "timeline.json")
            if os.path.exists(tl_path):
                try:
                    self.timeline.load(tl_path)
                except ValueError as _tl_err:
                    log.warning("Chargement timeline '%s' échoué : %s", tl_path, _tl_err)
                self.timeline_widget.set_duration(self.timeline.duration)
                self.timeline_widget.sync_bpm_controls()
                self.timeline_widget.sync_loop_controls()
                self.timeline_widget.canvas.update()
                self.timeline_widget.timeline_data_changed.emit()
                self._render_is_dirty = True

            # ── Audio ─────────────────────────────────────────────────────────
            audio_info = meta.get("audio", {})
            audio_fn = audio_info.get("filename") or audio_info.get("path")
            if audio_fn:
                for candidate in [
                    os.path.join(folder, "audio", audio_fn),
                    os.path.join(folder, "audio", os.path.basename(audio_fn)),
                    audio_fn,
                ]:
                    if os.path.exists(candidate):
                        self._load_audio_file(candidate)
                        break
                else:
                    self._status.showMessage(f"Audio introuvable : {audio_fn}", 5000)

            # ── Textures ──────────────────────────────────────────────────────
            tex_dir = os.path.join(folder, "textures")
            for i, tex in enumerate(meta.get("textures", [])):
                if not tex:
                    self._texture_paths[i] = None
                    continue
                for candidate in [os.path.join(tex_dir, tex),
                                   os.path.join(tex_dir, os.path.basename(tex)), tex]:
                    if os.path.exists(candidate):
                        self._load_texture(i, candidate)
                        break
                else:
                    self._texture_paths[i] = None

            # ── Mappings ──────────────────────────────────────────────────────
            mappings_dir = os.path.join(folder, "mappings")
            for key, filename, loader in [
                ("osc",  "osc.json",  lambda d: self.osc_engine.from_dict(d)),
                ("dmx",  "dmx.json",  lambda d: self.dmx_engine.from_dict(d)),
                ("xr_mappings", "xr.json", lambda d: setattr(self, '_xr_saved_mappings', d)),
            ]:
                mp = os.path.join(mappings_dir, filename)
                if os.path.exists(mp):
                    try:
                        with open(mp, 'r', encoding='utf-8') as f:
                            loader(json.load(f))
                    except Exception as e:
                        log.warning("Erreur restauration %s : %s", key, e)

            # ── FX states & synth ─────────────────────────────────────────────
            if "shader_fx_states" in meta:
                self._shader_fx_states = meta["shader_fx_states"]
            if "synth" in meta and meta["synth"] and hasattr(self, '_synth_editor'):
                try:
                    self._synth_editor.from_dict(meta["synth"])
                except Exception as e:
                    log.warning("Restauration synth : %s", e)
            if "ai" in meta and meta["ai"]:
                try:
                    self.ai_generator.from_dict(meta["ai"])
                except Exception as e:
                    log.warning("Erreur restauration config IA : %s", e)

            log.info("Projet chargé depuis dossier : %s", folder)
            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir le dossier projet :\n{e}")
            return False

    def _save_project_folder(self):
        """Enregistrer le projet en dossier versionné (Git-friendly)."""
        default = ""
        if self._current_project_path:
            base = os.path.splitext(self._current_project_path)[0]
            default = base if os.path.isdir(base) else base
        folder = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier projet (sera créé si nécessaire)", default or os.path.expanduser("~"))
        if not folder:
            # Fallback: laisser l'utilisateur créer un nouveau dossier via getSaveFileName
            path, _ = QFileDialog.getSaveFileName(
                self, "Nom du dossier projet", default or os.path.expanduser("~/mon_projet"),
                "Dossier projet OpenShader (*)")
            if not path:
                return
            folder = path  # le dossier sera créé par _save_project_to_folder

        ok = self._save_project_to_folder(folder)
        if ok:
            self._current_project_path = folder
            self._folder_mode = True
            if hasattr(self, '_folder_mode_action'):
                self._folder_mode_action.setChecked(True)
            self._project_is_modified = False
            self._update_title()
            self._add_to_recent(folder)
            self._status.showMessage(f"Projet sauvegardé : {os.path.basename(folder)}/", 4000)

    def _open_project_folder(self):
        """Charge un projet depuis un dossier versionné."""
        folder = QFileDialog.getExistingDirectory(
            self, "Ouvrir un dossier projet OpenShader", os.path.expanduser("~"))
        if not folder:
            return
        ok = self._load_project_from_folder(folder)
        if ok:
            self._current_project_path = folder
            self._folder_mode = True
            if hasattr(self, '_folder_mode_action'):
                self._folder_mode_action.setChecked(True)
            self._project_is_modified = False
            self._update_title()
            self._add_to_recent(folder)
            self._status.showMessage(f"Projet chargé : {os.path.basename(folder)}/", 4000)

    def _get_recent_projects(self) -> list:
        settings = QSettings("OpenShader", "OpenShader")
        return settings.value("recentProjects", []) or []

    def _add_to_recent(self, path: str):
        settings = QSettings("OpenShader", "OpenShader")
        recent = self._get_recent_projects()
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        settings.setValue("recentProjects", recent[:12])
        self._update_recent_menu()

    def _remove_from_recent(self, path: str):
        settings = QSettings("OpenShader", "OpenShader")
        recent = [p for p in self._get_recent_projects() if p != path]
        settings.setValue("recentProjects", recent)

    def _update_recent_menu(self):
        if not hasattr(self, '_recent_menu'):
            return
        self._recent_menu.clear()
        recent = self._get_recent_projects()
        if not recent:
            act = self._recent_menu.addAction("(aucun projet récent)")
            act.setEnabled(False)
        else:
            for path in recent:
                act = self._recent_menu.addAction(f"📄 {os.path.basename(path)}")
                act.setToolTip(path)
                act.triggered.connect(lambda checked=False, p=path: self._load_project_from_path(p))
            self._recent_menu.addSeparator()
            self._recent_menu.addAction("🗑 Effacer la liste", self._clear_recent_projects)

    def _clear_recent_projects(self):
        QSettings("OpenShader", "OpenShader").setValue("recentProjects", [])
        self._update_recent_menu()

    # ── v2.3 — Export & Packaging ─────────────────────────────────────────────

    def _export_render_frame(self, t: float, width: int, height: int):
        """
        Callback utilisé par ExportWorker pour rendre une frame à t.
        Retourne une QImage RGBA.
        """
        from PyQt6.QtGui import QImage as _QImage

        # Met à jour les uniforms de la timeline
        uniforms = self.timeline.evaluate(t)
        for name, val in uniforms.items():
            self.shader_engine.set_uniform(name, val)

        # Redimensionne temporairement si nécessaire
        orig_w, orig_h = self.shader_engine.width, self.shader_engine.height
        if width != orig_w or height != orig_h:
            self.shader_engine.resize(width, height)

        self.gl_widget.makeCurrent()
        try:
            self.gl_widget.set_time(t)
            img = self.gl_widget.grabFramebuffer()
        finally:
            self.gl_widget.doneCurrent()

        if width != orig_w or height != orig_h:
            self.shader_engine.resize(orig_w, orig_h)

        # Retourne l'image à la bonne taille
        if img.width() != width or img.height() != height:
            from PyQt6.QtCore import Qt as _Qt
            img = img.scaled(width, height,
                             _Qt.AspectRatioMode.IgnoreAspectRatio,
                             _Qt.TransformationMode.SmoothTransformation)
        return img

    def _export_video_hq(self):
        """v2.3 — Ouvre le dialog d'export vidéo haute qualité."""
        audio_path = getattr(self.audio_engine, 'file_path', None)
        dlg = ExportDialog(
            self,
            viewport_w=self.shader_engine.width,
            viewport_h=self.shader_engine.height,
            timeline_duration=self.timeline.duration,
            audio_path=audio_path,
        )
        dlg.exec()

    def _export_offline(self):
        """v6.1 — Ouvre le dialog de rendu offline ultra-qualité (TAA, Motion Blur, DCP)."""
        audio_path = getattr(self.audio_engine, 'file_path', None)
        dlg = OfflineRendererDialog(
            self,
            viewport_w  = self.shader_engine.width,
            viewport_h  = self.shader_engine.height,
            timeline_duration = self.timeline.duration,
            audio_path  = audio_path,
            render_fn   = self._offline_render_fn,
        )
        dlg.exec()

    def _offline_render_fn(self, t: float, jitter_x: float, jitter_y: float):
        """
        v6.1 — Callback pour OfflineRenderEngine.
        Rend une frame à t avec un léger jitter sub-pixel et retourne np.ndarray RGBA uint8.

        Le jitter (jx, jy) est injecté dans l'uniform iJitter du shader.
        Si le shader n'utilise pas iJitter, la frame est simplement rendue à t.
        """
        import numpy as np
        from PyQt6.QtGui import QImage

        w = self.shader_engine.width
        h = self.shader_engine.height

        # Évalue les uniforms timeline
        uniforms = self.timeline.evaluate(t)
        for name, val in uniforms.items():
            self.shader_engine.set_uniform(name, val)

        # Injecte le jitter sub-pixel (en UV normalisé) pour le TAA
        if jitter_x != 0.0 or jitter_y != 0.0:
            self.shader_engine.set_uniform('iJitter', (jitter_x / w, jitter_y / h))
        else:
            self.shader_engine.set_uniform('iJitter', (0.0, 0.0))

        self.gl_widget.makeCurrent()
        try:
            self.gl_widget.set_time(t)
            # Rend dans image_fbo et lit les pixels directement (plus rapide que grabFramebuffer)
            self.shader_engine.render(t, screen_fbo=self.shader_engine.image_fbo)
            raw = self.shader_engine.image_fbo.read(components=4)
        finally:
            self.gl_widget.doneCurrent()

        # bytes → numpy RGBA uint8 (flip vertical : GL origine bas-gauche)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).copy()
        arr = arr[::-1, :, :]   # flip Y
        return arr

    def _export_shadertoy_multipass(self):
        """v2.3 — Export Shadertoy multipass (JSON clipboard)."""
        # Collecte les sources de toutes les passes
        sources = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                sources[pass_name] = src

        if not sources:
            QMessageBox.information(self, "Export Shadertoy multipass",
                                    "Aucun shader n'est chargé.")
            return

        dag = getattr(self, '_node_graph_data', None) or {}
        show_multipass_export_dialog(self, sources, dag)

    def _export_standalone(self):
        """v2.3 — Export standalone démo (lecteur minimal autonome)."""
        fmt_items = ["Dossier décompressé", "Archive ZIP"]
        fmt, ok = QInputDialog.getItem(
            self, "Export standalone", "Format de sortie :", fmt_items, 0, False)
        if not ok:
            return

        # Collecte shaders
        shaders = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                shaders[pass_name] = src

        audio_path = getattr(self.audio_engine, 'file_path', None)
        project_data = self._collect_project_data() if hasattr(self, '_collect_project_data') else {}

        exporter = StandaloneExporter(
            project_data=project_data,
            project_path=getattr(self, '_current_project_path', None),
            audio_path=audio_path,
            shaders=shaders,
        )

        if fmt == "Archive ZIP":
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Export standalone ZIP",
                os.path.expanduser("~/demo_standalone.zip"),
                "Archive ZIP (*.zip)")
            if not out_path:
                return
            if not out_path.endswith(".zip"):
                out_path += ".zip"
            try:
                n = exporter.export_to_zip(out_path)
                size_mb = os.path.getsize(out_path) / 1024 / 1024
                QMessageBox.information(self, "Export standalone",
                    f"✓ ZIP créé ({n} fichiers, {size_mb:.1f} Mo) :\n{out_path}\n\n"
                    "Lancez : python player.py")
            except OSError as e:
                QMessageBox.critical(self, "Erreur", str(e))
        else:
            out_dir = QFileDialog.getExistingDirectory(
                self, "Dossier de sortie standalone",
                os.path.expanduser("~"))
            if not out_dir:
                return
            out_dir = os.path.join(out_dir, "demo_standalone")
            try:
                files = exporter.export_to_dir(out_dir)
                QMessageBox.information(self, "Export standalone",
                    f"✓ {len(files)} fichiers exportés dans :\n{out_dir}\n\n"
                    "Lancez : python player.py")
                # Ouvrir dans l'explorateur
                import sys as _sys
                if _sys.platform == "win32":
                    subprocess.Popen(["explorer", out_dir])
                elif _sys.platform == "darwin":
                    subprocess.Popen(["open", out_dir])
                else:
                    subprocess.Popen(["xdg-open", out_dir])
            except OSError as e:
                QMessageBox.critical(self, "Erreur", str(e))

    def _export_packaging(self):
        """v2.3 — Lance build.py pour packaging PyInstaller."""
        import sys as _sys
        build_py = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "build.py")

        if not os.path.isfile(build_py):
            QMessageBox.critical(self, "Packaging",
                "build.py introuvable.\nAssurez-vous d'utiliser OpenShader v2.3+.")
            return

        # Vérifie PyInstaller
        if not shutil.which("pyinstaller"):
            reply = QMessageBox.question(
                self, "PyInstaller manquant",
                "PyInstaller n'est pas installé.\n\n"
                "Voulez-vous l'installer maintenant ?\n"
                "  pip install pyinstaller>=6.0",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                subprocess.Popen([_sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])
                QMessageBox.information(self, "Installation",
                    "Installation lancée dans un terminal.\n"
                    "Relancez l'export packaging une fois terminé.")
            return

        # Dialog options
        opts_items = [
            "One-dir (démarrage rapide, recommandé)",
            "One-file (exécutable unique, démarrage lent)",
            "One-file + mode debug",
        ]
        opt, ok = QInputDialog.getItem(
            self, "Packaging PyInstaller",
            "Mode de packaging :", opts_items, 0, False)
        if not ok:
            return

        cmd = [_sys.executable, build_py]
        if "One-file" in opt:
            cmd.append("--onefile")
        if "debug" in opt:
            cmd.append("--debug")

        # Lance dans un terminal visible
        plat = _sys.platform
        try:
            if plat == "win32":
                subprocess.Popen(["cmd", "/k"] + cmd,
                                  creationflags=subprocess.CREATE_NEW_CONSOLE)
            elif plat == "darwin":
                script = " ".join(f'"{c}"' for c in cmd)
                subprocess.Popen(["osascript", "-e",
                    f'tell application "Terminal" to do script "{script}"'])
            else:
                term = shutil.which("x-terminal-emulator") or shutil.which("xterm") or shutil.which("gnome-terminal")
                if term:
                    subprocess.Popen([term, "--"] + cmd)
                else:
                    subprocess.Popen(cmd)

            QMessageBox.information(self, "Packaging lancé",
                "Build PyInstaller lancé dans un terminal.\n\n"
                "Le résultat sera dans le dossier dist/\n"
                "une fois la compilation terminée.")
        except OSError as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lancer build.py :\n{e}")

    # ── v2.8 — 4K / 64K Intro Toolkit ───────────────────────────────────────

    def _show_intro_toolkit(self):
        """Ouvre le dialog 4K/64K intro toolkit."""
        shaders = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                shaders[pass_name] = src

        if not shaders:
            QMessageBox.information(
                self, "4K / 64K Intro Toolkit",
                "Aucun shader chargé.\nChargez un shader avant d'ouvrir le toolkit.")
            return

        dlg = IntroBuilderDialog(shaders=shaders, parent=self)
        dlg.exec()
        # Forcer la mise à jour du label status bar après fermeture
        self._update_intro_size_label()

    def _update_intro_size_label(self):
        """Met à jour le label de taille estimée dans la status bar (toutes les 3s)."""
        shaders = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                shaders[pass_name] = src

        if not shaders:
            self._lbl_intro_size.setVisible(False)
            return

        try:
            est    = IntroSizeEstimator(shaders)
            report = est.estimate()
            total  = report.total_estimated

            # Couleur selon le budget
            if total <= 4096:
                color = "#40c070"   # vert — sous 4K
            elif total <= 65536:
                color = "#c0a030"   # jaune — sous 64K
            else:
                color = "#c05050"   # rouge — dépasse 64K

            label = f"⬡ {est.format_bytes(total)}"
            if total <= 4096:
                label += " ≤4K"
            elif total <= 65536:
                label += " ≤64K"
            else:
                label += " >64K"

            self._lbl_intro_size.setText(label)
            self._lbl_intro_size.setStyleSheet(
                f"color: {color}; font: 10px 'Cascadia Code', monospace;"
                "padding: 0 8px;"
            )
            self._lbl_intro_size.setVisible(True)
        except Exception as e:
            log.debug("Erreur estimation taille intro : %s", e)
            self._lbl_intro_size.setVisible(False)

    def _toggle_intro_size_label(self):
        """Bascule la visibilité du label d'estimation dans la status bar."""
        self._lbl_intro_size.setVisible(not self._lbl_intro_size.isVisible())
        if self._lbl_intro_size.isVisible():
            self._update_intro_size_label()

    def _export_gallery(self):
        """v2.7 — Ouvre le dialog de publication vers la galerie en ligne."""
        # Collecte les shaders chargés
        shaders = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                shaders[pass_name] = src

        if not shaders:
            QMessageBox.information(
                self, "Galerie en ligne",
                "Aucun shader chargé.\nChargez un shader avant de publier.")
            return

        # Capture du viewport pour la preview
        preview_pixmap = None
        try:
            from PyQt6.QtGui import QPixmap
            if hasattr(self, 'gl_widget') and self.gl_widget.isVisible():
                preview_pixmap = self.gl_widget.grab()
        except Exception as e:
            log.warning("Impossible de capturer le viewport pour la preview : %s", e)

        res = (self.shader_engine.width, self.shader_engine.height)

        dlg = GalleryPublishDialog(
            shaders=shaders,
            preview_pixmap=preview_pixmap,
            resolution=res,
            parent=self,
        )
        dlg.exec()

    def _export_wasm(self):
        """v3.0 — Exporte un bundle WebAssembly jouable dans le navigateur."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                      QPushButton, QLineEdit, QTextEdit,
                                      QFormLayout, QFileDialog, QMessageBox,
                                      QProgressBar, QGroupBox)
        from PyQt6.QtCore import Qt

        # Collecte les shaders
        shaders = {}
        for pass_name, editor in self.editors.items():
            src = editor.get_code().strip() if hasattr(editor, 'get_code') else ''
            if src:
                shaders[pass_name] = src

        if not shaders:
            QMessageBox.information(
                self, "Export WASM",
                "Aucun shader chargé.\nChargez un shader avant d'exporter.")
            return

        # ── Dialog de configuration ────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Export bundle WebAssembly")
        dlg.resize(520, 420)
        dlg.setStyleSheet("""
            QDialog,QWidget { background:#0e1018; color:#c0c4d0; font:10px 'Segoe UI'; }
            QLineEdit,QTextEdit {
                background:#1e2030; color:#c0c4d0; border:1px solid #2a2d3a;
                border-radius:3px; padding:4px 8px; }
            QFormLayout QLabel { color:#7a8099; }
            QGroupBox { border:1px solid #2a2d3a; border-radius:4px;
                        margin-top:8px; color:#7a8099; padding:6px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; }
        """)
        _btn = """
            QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                          border-radius:3px; padding:4px 12px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            QPushButton#primary { background:#1e3a6a; color:#80b0ff;
                                  border-color:#3a5888; }
            QPushButton#primary:hover { background:#2a4a7a; }
        """

        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(16, 16, 16, 12)
        vl.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        edit_title = QLineEdit(getattr(self, '_project_title', 'Mon Shader'))
        edit_author = QLineEdit()
        edit_desc   = QTextEdit()
        edit_desc.setFixedHeight(60)
        edit_tags   = QLineEdit(); edit_tags.setPlaceholderText("glsl, demoscene, audio…")

        form.addRow("Titre :", edit_title)
        form.addRow("Auteur :", edit_author)
        form.addRow("Description :", edit_desc)
        form.addRow("Tags (virgule) :", edit_tags)
        vl.addLayout(form)

        # Dossier de sortie
        grp_out = QGroupBox("Dossier de sortie")
        hl_out  = QHBoxLayout(grp_out)
        edit_out = QLineEdit(
            str(Path.home() / 'Desktop' / 'openshader_wasm')
        )
        btn_browse = QPushButton("…"); btn_browse.setStyleSheet(_btn)
        btn_browse.setFixedWidth(30)
        hl_out.addWidget(edit_out, 1)
        hl_out.addWidget(btn_browse)
        vl.addWidget(grp_out)

        # Info
        info_lbl = QLabel(
            "📦  Génère : index.html · player.html · embed.js · sw.js "
            "(PWA offline) · manifest · shaders/ · runtime/ (Emscripten stub)"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color:#4a5570; font:9px 'Segoe UI'; padding:4px 0;")
        vl.addWidget(info_lbl)

        # Progress
        prog = QProgressBar()
        prog.setRange(0, 0); prog.setFixedHeight(4)
        prog.setStyleSheet("""
            QProgressBar { background:#1e2030; border:none; border-radius:2px; }
            QProgressBar::chunk { background:#3a88ff; border-radius:2px; }
        """)
        prog.hide()
        vl.addWidget(prog)

        # Boutons
        hl_btns = QHBoxLayout()
        btn_cancel = QPushButton("Annuler"); btn_cancel.setStyleSheet(_btn)
        btn_export = QPushButton("🌍 Exporter"); btn_export.setStyleSheet(_btn)
        btn_export.setObjectName("primary")
        hl_btns.addStretch()
        hl_btns.addWidget(btn_cancel)
        hl_btns.addWidget(btn_export)
        vl.addLayout(hl_btns)

        def _browse():
            d = QFileDialog.getExistingDirectory(dlg, "Dossier de sortie WASM")
            if d:
                edit_out.setText(d)

        def _export():
            title   = edit_title.text().strip() or 'Shader'
            author  = edit_author.text().strip() or 'Anonyme'
            desc    = edit_desc.toPlainText().strip()
            tags    = [t.strip() for t in edit_tags.text().split(',') if t.strip()]
            out_dir = edit_out.text().strip()

            if not out_dir:
                QMessageBox.warning(dlg, "Export WASM", "Choisissez un dossier de sortie.")
                return

            res = (getattr(self.shader_engine, 'width', 1920),
                   getattr(self.shader_engine, 'height', 1080))

            meta = {
                'title':       title,
                'author':      author,
                'description': desc,
                'tags':        tags,
                'licence':     'CC0 1.0',
                'resolution':  res,
            }

            prog.show()
            btn_export.setEnabled(False)
            dlg.repaint()

            try:
                exporter = WasmExporter(meta=meta, shaders=shaders)
                files = exporter.export(out_dir)

                prog.hide()
                btn_export.setEnabled(True)

                self._status.showMessage(
                    f"Bundle WASM exporté : {len(files)} fichiers → {out_dir}", 6000)

                msg = QMessageBox(dlg)
                msg.setWindowTitle("Export WASM — Succès")
                msg.setText(
                    f"✅  {len(files)} fichiers générés dans :\n{out_dir}\n\n"
                    "Ouvrez index.html dans Chrome/Firefox pour tester.\n\n"
                    "Déployez sur GitHub Pages, Netlify ou itch.io\n"
                    "pour partager sans serveur."
                )
                msg.addButton("Ouvrir le dossier", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("Fermer", QMessageBox.ButtonRole.RejectRole)
                if msg.exec() == 0:  # Ouvrir le dossier
                    import subprocess, sys
                    if sys.platform == 'win32':
                        subprocess.Popen(['explorer', out_dir])
                    elif sys.platform == 'darwin':
                        subprocess.Popen(['open', out_dir])
                    else:
                        subprocess.Popen(['xdg-open', out_dir])
                dlg.accept()

            except (OSError, ValueError) as e:
                prog.hide()
                btn_export.setEnabled(True)
                QMessageBox.critical(dlg, "Erreur export WASM", str(e))

        btn_browse.clicked.connect(_browse)
        btn_export.clicked.connect(_export)
        btn_cancel.clicked.connect(dlg.reject)
        dlg.exec()

    def _export_shadertoy(self):
        """Exporte le shader Image comme fichier .st Shadertoy."""
        code = self.editors.get('Image')
        if code is None:
            QMessageBox.information(self, "Export Shadertoy", "Aucun éditeur 'Image' trouvé.")
            return
        source = code.get_code().strip()
        if not source:
            QMessageBox.information(self, "Export Shadertoy", "L'éditeur Image est vide.")
            return
        default = ""
        if self._current_project_path:
            default = os.path.splitext(self._current_project_path)[0] + ".st"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter vers Shadertoy", default, "Shadertoy Shader (*.st)")
        if not path:
            return
        if not path.endswith('.st'):
            path += '.st'
        try:
            output = source if "void mainImage" in source else _glsl_to_shadertoy(source)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(output)
            self._status.showMessage(f"Exporté : {os.path.basename(path)}", 4000)
            QMessageBox.information(self, "Export Shadertoy",
                f"Shader exporté :\n{path}\n\nCopiez le contenu sur https://www.shadertoy.com/new")
        except (OSError, UnicodeEncodeError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'exporter :\n{e}")

    def _update_title(self):
        base = "OpenShader — Shader Edition"
        if self._current_project_path:
            name = os.path.splitext(os.path.basename(self._current_project_path))[0]
            mod  = " •" if self._project_is_modified else ""
            self.setWindowTitle(f"{name}{mod} — {base}")
        else:
            mod = " •" if self._project_is_modified else ""
            self.setWindowTitle(f"Sans titre{mod} — {base}")

    # ── Fichier / Export ──────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _save_shader_to_file(self, path: str):
        """Sauvegarde le code de l'éditeur actif dans le fichier donné."""
        raw = self.editor_tabs.tabText(self.editor_tabs.currentIndex())
        current_tab = raw.lstrip("○●✕ ").strip()
        editor = self.editors.get(current_tab)
        if not editor:
            self._status.showMessage("Aucun éditeur actif.", 3000)
            return
        code = editor.get_code()
        if not code.strip():
            self._status.showMessage("L'éditeur est vide — rien à sauvegarder.", 3000)
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)
            self._status.showMessage(
                f"✓ Shader sauvegardé : {os.path.basename(path)}", 4000)
            # Rafraîchit l'arbre du panneau gauche
            self.left_panel.refresh_tree()
        except (OSError, UnicodeEncodeError) as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder :\n{e}")

    def _open_shader_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un shader", "",
            "Shaders (*.st *.glsl);;Shadertoy (*.st);;GLSL (*.glsl)"
        )
        if path:
            self._load_shader_file(path)

    def _open_audio_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier audio", "",
            "Audio (*.wav *.mp3 *.ogg)"
        )
        if path:
            self._load_audio_file(path)

    def _save_timeline(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la timeline", "", "JSON (*.json)"
        )
        if path:
            self.timeline.save(path)
            self._status.showMessage(f"Timeline enregistrée : {path}", 3000)

    def _load_timeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une timeline", "", "JSON (*.json)"
        )
        if path:
            self.timeline.load(path)
            self.timeline_widget.set_duration(self.timeline.duration)
            self.timeline_widget.sync_bpm_controls()
            self.timeline_widget.sync_loop_controls()
            self.timeline_widget.canvas.update()
            self._render_is_dirty = True
            self._status.showMessage(f"Timeline chargée : {path}", 3000)

    @pyqtSlot(dict)
    def _on_export_video(self, params: dict):
        """Export vidéo MP4, WebM ou GIF via FFmpeg."""
        fmt      = params["format"]
        width    = params["width"]
        height   = params["height"]
        fps      = params["fps"]
        duration = params["duration"]
        crf      = params["crf"]
        gif_fps  = params["gif_fps"]
        gif_loop = params["gif_loop"]
        with_audio = params.get("include_audio", True)

        # ── Vérification FFmpeg ───────────────────────────────────────────────
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            QMessageBox.critical(self, "FFmpeg manquant",
                "FFmpeg est requis pour exporter une vidéo.\n"
                "Téléchargez-le sur https://ffmpeg.org/download.html\n"
                "et assurez-vous qu'il est dans votre PATH.")
            self.left_panel.set_export_progress(-1)
            return

        # ── Chemin de sortie ──────────────────────────────────────────────────
        filters = {"mp4": "Vidéo MP4 (*.mp4)", "webm": "Vidéo WebM (*.webm)",
                   "gif": "GIF animé (*.gif)"}
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Exporter en {fmt.upper()}",
            os.path.expanduser("~/export." + fmt),
            filters.get(fmt, f"*.{fmt}"))
        if not out_path:
            self.left_panel.set_export_progress(-1)
            return
        if not out_path.lower().endswith(f".{fmt}"):
            out_path += f".{fmt}"

        # ── Dossier temporaire pour les frames ────────────────────────────────
        tmp_dir = tempfile.mkdtemp(prefix="openshader_export_")
        total_frames = max(1, int(duration * fps))
        was_playing  = self._is_playing
        if was_playing:
            self._pause()

        self.left_panel.set_export_progress(0.0, f"Rendu de {total_frames} frames…")
        QApplication.processEvents()

        # ── Rendu des frames ─────────────────────────────────────────────────
        self.gl_widget.makeCurrent()
        try:
            for frame in range(total_frames):
                t = frame / fps
                uniforms = self.timeline.evaluate(t)
                for name, val in uniforms.items():
                    self.shader_engine.set_uniform(name, val)
                self.gl_widget.set_time(t)

                img = self.gl_widget.grabFramebuffer()

                # Redimensionner si nécessaire
                if img.width() != width or img.height() != height:
                    img = img.scaled(width, height,
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

                img.save(os.path.join(tmp_dir, f"frame_{frame:05d}.png"))

                pct = (frame + 1) / total_frames * 0.7  # 70% pour le rendu
                self.left_panel.set_export_progress(
                    pct, f"Frame {frame+1}/{total_frames}")
                QApplication.processEvents()

        except (OSError, RuntimeError, Exception) as e:
            QMessageBox.critical(self, "Erreur de rendu", str(e))
            self.left_panel.set_export_progress(-1)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self.gl_widget.doneCurrent()
            return
        finally:
            self.gl_widget.doneCurrent()

        # ── Encodage FFmpeg ───────────────────────────────────────────────────
        self.left_panel.set_export_progress(0.72, "Encodage FFmpeg…")
        QApplication.processEvents()

        audio_path = getattr(self.audio_engine, 'file_path', None)
        input_pattern = os.path.join(tmp_dir, "frame_%05d.png")

        try:
            if fmt == "mp4":
                cmd = [ffmpeg, "-y",
                       "-framerate", str(fps),
                       "-i", input_pattern]
                if with_audio and audio_path and os.path.isfile(audio_path):
                    cmd += ["-i", audio_path, "-t", str(duration)]
                cmd += ["-c:v", "libx264",
                        "-crf", str(crf),
                        "-preset", "slow",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart"]
                if with_audio and audio_path and os.path.isfile(audio_path):
                    cmd += ["-c:a", "aac", "-b:a", "192k"]
                cmd += [out_path]

            elif fmt == "webm":
                cmd = [ffmpeg, "-y",
                       "-framerate", str(fps),
                       "-i", input_pattern]
                if with_audio and audio_path and os.path.isfile(audio_path):
                    cmd += ["-i", audio_path, "-t", str(duration)]
                cmd += ["-c:v", "libvpx-vp9",
                        "-crf", str(crf),
                        "-b:v", "0",
                        "-pix_fmt", "yuv420p"]
                if with_audio and audio_path and os.path.isfile(audio_path):
                    cmd += ["-c:a", "libopus", "-b:a", "128k"]
                cmd += [out_path]

            elif fmt == "gif":
                # GIF deux passes : palette → dithering
                palette_path = os.path.join(tmp_dir, "palette.png")

                # Passe 1 : générer la palette
                cmd1 = [ffmpeg, "-y",
                        "-framerate", str(fps),
                        "-i", input_pattern,
                        "-vf", f"fps={gif_fps},scale={width}:{height}:flags=lanczos,palettegen",
                        palette_path]
                r1 = subprocess.run(cmd1, capture_output=True, timeout=120)
                if r1.returncode != 0:
                    raise RuntimeError(f"Erreur palette GIF:\n{r1.stderr.decode(errors='replace')}")

                self.left_panel.set_export_progress(0.85, "Assemblage GIF…")
                QApplication.processEvents()

                # Passe 2 : encoder le GIF avec dithering
                loop_val = str(gif_loop)
                cmd = [ffmpeg, "-y",
                       "-framerate", str(fps),
                       "-i", input_pattern,
                       "-i", palette_path,
                       "-lavfi", f"fps={gif_fps},scale={width}:{height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                       "-loop", loop_val,
                       out_path]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            if result.returncode != 0:
                err = result.stderr.decode(errors='replace')
                raise RuntimeError(f"FFmpeg a retourné une erreur :\n{err[-800:]}")

        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, "Export timeout", "L'encodage a dépassé le délai imparti.")
            self.left_panel.set_export_progress(-1)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        except (OSError, RuntimeError) as e:
            QMessageBox.critical(self, "Erreur FFmpeg", str(e))
            self.left_panel.set_export_progress(-1)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        # ── Nettoyage & succès ────────────────────────────────────────────────
        shutil.rmtree(tmp_dir, ignore_errors=True)
        self.left_panel.set_export_progress(1.0,
            f"✓ Export terminé : {os.path.basename(out_path)}")
        QApplication.processEvents()

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        self._status.showMessage(
            f"✓ {fmt.upper()} exporté : {os.path.basename(out_path)}  ({size_mb:.1f} Mo)", 6000)

        reply = QMessageBox.question(
            self, "Export terminé",
            f"Fichier généré :\n{out_path}\n({size_mb:.1f} Mo)\n\nOuvrir dans l'explorateur ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            target = os.path.dirname(out_path)
            if os.sys.platform == 'win32':
                subprocess.Popen(['explorer', '/select,', out_path])
            elif os.sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', out_path])
            else:
                subprocess.Popen(['xdg-open', target])

        # Réinitialiser la progression après 3 secondes
        QTimer.singleShot(3000, lambda: self.left_panel.set_export_progress(-1))

        if was_playing:
            self._play()

    def _on_export(self):
        # 1. Choisir le dossier de sortie
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier d'export (Séquence PNG)")
        if not folder:
            return

        # 2. Choisir les FPS
        fps, ok = QInputDialog.getInt(self, "Export PNG", "Images par seconde (FPS) :", 60, 1, 120)
        if not ok:
            return

        # 3. Configuration
        duration = self.timeline.duration
        total_frames = int(duration * fps)
        
        # Pause si lecture en cours
        was_playing = self._is_playing
        if was_playing:
            self._pause()

        # Barre de progression
        progress = QProgressDialog("Export des frames...", "Annuler", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Boucle de rendu
        self.gl_widget.makeCurrent()
        try:
            for frame in range(total_frames):
                if progress.wasCanceled():
                    break
                
                t = frame / fps
                
                # Mise à jour simulation
                uniforms = self.timeline.evaluate(t)
                for name, val in uniforms.items():
                    self.shader_engine.set_uniform(name, val)
                
                self.gl_widget.set_time(t)
                
                # Capture (déclenche un paintGL interne dans un FBO)
                image = self.gl_widget.grabFramebuffer()
                
                # Sauvegarde
                filename = f"frame_{frame:05d}.png"
                image.save(os.path.join(folder, filename))
                
                progress.setValue(frame + 1)
                QApplication.processEvents()

        except (OSError, RuntimeError, Exception) as e:
            QMessageBox.critical(self, "Erreur Export", f"Une erreur est survenue :\n{e}")

        finally:
            self.gl_widget.doneCurrent()
            progress.close()
            
            # Restauration état
            self._on_timeline_seek(self._current_time)
            if was_playing:
                self._play()

        if not progress.wasCanceled():
            # ── Multiplexage Audio / Vidéo avec FFmpeg ───────────────────────
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                reply = QMessageBox.question(
                    self, "Créer la vidéo ?",
                    "FFmpeg a été détecté.\nVoulez-vous générer un fichier vidéo MP4 maintenant ?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    output_mp4 = os.path.join(folder, "output.mp4")
                    audio_path = self.audio_engine.file_path
                    
                    # Commande de base
                    cmd = [
                        ffmpeg_path, "-y",
                        "-framerate", str(fps),
                        "-i", os.path.join(folder, "frame_%05d.png"),
                    ]
                    
                    # Ajout audio si présent
                    if audio_path:
                        cmd.extend(["-i", audio_path, "-t", str(duration)])
                    
                    # Encodage
                    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_mp4])
                    
                    subprocess.run(cmd, capture_output=True)
                    self._status.showMessage(f"Vidéo générée : {output_mp4}", 5000)
                    return

            self._status.showMessage(f"Séquence d'images exportée dans {folder}", 5000)

    def _on_screenshot(self):
        """Capture la frame courante du viewport et la sauvegarde en PNG."""
        # grabFramebuffer() est la méthode la plus simple pour capturer le contenu d'un QOpenGLWidget
        image = self.gl_widget.grabFramebuffer()

        if image.isNull():
            self._status.showMessage("Erreur de capture du framebuffer.", 4000)
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la capture", "", "Image PNG (*.png)"
        )

        if path:
            if not path.lower().endswith('.png'):
                path += '.png'

            if image.save(path):
                self._status.showMessage(f"Capture enregistrée : {os.path.basename(path)}", 4000)
            else:
                self._status.showMessage("Erreur lors de l'enregistrement de l'image.", 5000)

    def _new_project_from_template(self):
        """Crée un nouveau projet à partir d'un modèle (v1.5 — templates enrichis)."""
        templates = {
            **_SHADER_TEMPLATES,
            "🎵 AudioVisualizer — Bars":  _TEMPLATE_AUDIOBARS,
            "🌀 Demo Intro — Tunnel":     _TEMPLATE_INTRO_TUNNEL,
            "🔁 Loopable BG — Voronoi":   _TEMPLATE_LOOP_VORONOI,
        }
        items = list(templates.keys())
        item, ok = QInputDialog.getItem(self, "Nouveau Projet",
                                        "Choisir un modèle de shader :", items, 0, False)
        if not (ok and item):
            return

        self._on_stop()
        self.timeline.clear()
        self._init_default_tracks()
        self.timeline_widget.set_duration(self.timeline.duration)
        self.timeline_widget.canvas.update()
        self.cmd_stack.clear()   # v2.5 — vider le CommandStack global
        self.audio_engine.stop()
        self._texture_paths = [None] * 4
        self.shader_engine.textures = [None] * 4
        for editor in self.editors.values():
            editor.set_code("")

        template_code = templates[item]
        target_pass   = 'Post' if item.startswith('Post:') else 'Image'

        self.editors[target_pass].set_code(template_code)
        self._compile_source(template_code, target_pass, f"Modèle : {item}")

        if target_pass == 'Post' and not self.editors['Image'].get_code().strip():
            img_code = _SHADER_TEMPLATES['Shadertoy Minimal']
            self.editors['Image'].set_code(img_code)
            self._compile_source(img_code, 'Image', "Modèle : Shadertoy Minimal")

        self._current_project_path = None
        self._project_is_modified  = True
        self._update_title()
        self._status.showMessage(f"Nouveau projet : {item}", 3000)

    def _start_audio_record_dialog(self):
        """Démarre une session d'enregistrement audio sur une piste."""
        if self._is_recording:
            QMessageBox.information(self, "Info", "Un enregistrement est déjà en cours.")
            return

        float_tracks = [t.name for t in self.timeline.tracks if t.value_type == 'float']
        if not float_tracks:
            QMessageBox.warning(self, "Enregistrement Audio", "Aucune piste de type 'float' n'a été trouvée pour l'enregistrement.")
            return
        
        track_name, ok = QInputDialog.getItem(self, "Enregistrement Audio",
                                              "Choisir la piste pour l'enregistrement :", float_tracks, 0, False)
        if not (ok and track_name):
            return

        self._recording_track = next(t for t in self.timeline.tracks if t.name == track_name)
        
        reply = QMessageBox.question(self, "Confirmation", 
                                     f"Effacer les keyframes existants sur la piste '{track_name}' ?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            self._recording_track.keyframes.clear()

        if self.audio_engine.start_recording():
            self._is_recording = True
            self._status.showMessage(f"🔴 Enregistrement sur '{track_name}'...", 10000)
            self._play()
        else:
            QMessageBox.critical(self, "Erreur Audio", "Impossible de démarrer la capture du microphone.")

    def _toggle_fft_display(self, checked: bool):
        self.gl_widget.toggle_fft(checked)
        self._render_is_dirty = True

    def _toggle_oscilloscope_display(self, checked: bool):
        self.gl_widget.toggle_oscilloscope(checked)
        self._render_is_dirty = True

    # ── À propos ──────────────────────────────────────────────────────────────

    def _open_url(self, url: str):
        webbrowser.open(url)

    # ── v2.2 — Camera track & Expression help actions ────────────────────────

    def _add_camera_track_action(self):
        """v2.2 — Ajoute une piste de caméra 3D à la timeline via le menu."""
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Nouvelle piste Caméra", "Nom :", text="Caméra")
        if ok and name:
            track = self.timeline.add_camera_track(name.strip())
            self.timeline_widget.canvas.update()
            log.info(f"Piste caméra '{name}' ajoutée — uniforms : uCamPos, uCamTarget, uCamFOV")
            self.statusBar().showMessage(
                "🎥 Piste caméra créée. Uniforms disponibles : uCamPos (vec3), uCamTarget (vec3), uCamFOV (float)",
                6000
            )

    def _show_expression_help(self):
        """v2.2 — Affiche l'aide sur les expressions de keyframe."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("📐 Expressions dans les keyframes")
        dlg.setMinimumWidth(500)
        dlg.setStyleSheet("background: #1c1e24; color: #d0d3de;")
        lay = QVBoxLayout(dlg)

        text = """<style>
            body { font-family: 'Segoe UI'; font-size: 10px; }
            code { background: #0d0f14; color: #cdd6f4; padding: 2px 4px;
                   border-radius: 2px; font-family: Consolas; }
            h3   { color: #89b4fa; }
            td   { padding: 4px 8px; }
            th   { color: #6272a4; text-align: left; }
        </style>
        <h3>Expressions Python sandboxées dans les keyframes</h3>
        <p>Clic droit sur un keyframe → <b>Ajouter une expression</b></p>
        <p>L'expression remplace la valeur fixe du keyframe et est évaluée à chaque frame.</p>
        <table>
        <tr><th>Variable</th><th>Type</th><th>Description</th></tr>
        <tr><td><code>t</code></td><td>float</td><td>Temps en secondes</td></tr>
        <tr><td><code>beat</code></td><td>float</td><td>Numéro de beat courant</td></tr>
        <tr><td><code>bpm</code></td><td>float</td><td>BPM du projet</td></tr>
        <tr><td><code>rms</code></td><td>float [0,1]</td><td>Amplitude audio</td></tr>
        <tr><td><code>fft[n]</code></td><td>float [0,1]</td><td>Bande FFT n</td></tr>
        </table>
        <h3>Exemples</h3>
        <ul>
        <li><code>sin(t * 2.0) * 0.5 + 0.5</code> — oscillation entre 0 et 1</li>
        <li><code>clamp(rms * 2.0, 0.0, 1.0)</code> — amplitude audio, clampée</li>
        <li><code>fract(beat)</code> — avance par beat (effet strobe)</li>
        <li><code>smoothstep(0.0, 1.0, t / 10.0)</code> — fade-in sur 10s</li>
        <li><code>0.5 + 0.5 * sin(beat * 3.14159)</code> — pulse par beat</li>
        </ul>
        """

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setOpenExternalLinks(False)
        lay.addWidget(lbl)

        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(dlg.accept)
        btn_close.setStyleSheet(
            "background:#2a2d3a; color:#c8ccd8; border:1px solid #3a3d4d;"
            "border-radius:3px; padding:4px 16px;"
        )
        lay.addWidget(btn_close)
        dlg.exec()

    # ── v2.2 — Thumbnail generation ───────────────────────────────────────────

    def _generate_thumbnail(self, shader_path: str):
        """
        v2.2 — Génère un thumbnail 120×68 pour le shader donné.
        Charge le shader, rend à t=5s, capture l'image, restaure le shader courant.
        """
        from PyQt6.QtGui import QPixmap
        import os as _os

        if not _os.path.isfile(shader_path):
            return

        # Sauvegarde du temps courant (attribut float, pas méthode)
        saved_time = self._current_time
        try:
            with open(shader_path, 'r', encoding='utf-8', errors='replace') as f:
                src = f.read()

            # Positionne le temps à 5s pour le thumbnail
            self.shader_engine._last_time = 5.0

            # Compile via la vraie méthode de ShaderEngine
            ok, err = self.shader_engine.load_shader_source(
                src, pass_name='Image', source_path=shader_path)
            if not ok:
                return

            # Render + capture
            self.gl_widget.set_time(5.0)
            self.gl_widget.repaint()
            img = self.gl_widget.grabFramebuffer()
            if img.isNull():
                return

            pixmap = QPixmap.fromImage(img).scaled(
                self.left_panel.THUMB_W,
                self.left_panel.THUMB_H,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.left_panel.set_thumbnail(shader_path, pixmap)

        except Exception as exc:
            log.warning(f"Thumbnail generation failed for {shader_path}: {exc}", exc_info=True)
        finally:
            # Restaure l'état : _current_time est un float attribut, non callable
            self.shader_engine._last_time = saved_time
            self._recompile_current()

    def _current_pass_name(self) -> str:
        """Retourne le nom de la passe courante (Image, Buffer A, etc.)."""
        idx = self.editor_tabs.currentIndex()
        return self.editor_tabs.tabText(idx) if idx >= 0 else 'Image'

    # ── v2.7 — Préférences ────────────────────────────────────────────────────

    def _show_preferences(self):
        """Ouvre la boîte de dialogue des préférences (toggle backend OpenGL / Vulkan)."""
        _make_preferences_dialog(self)

    def _show_about(self):
        import os as _os
        from PyQt6.QtGui     import QPixmap as _QPixmap
        from PyQt6.QtWidgets import (QScrollArea as _SA, QTabWidget as _TW,
                                     QGridLayout as _GL, QDialogButtonBox as _DBB,
                                     QGraphicsDropShadowEffect as _Shadow)
        from PyQt6.QtCore    import QPropertyAnimation as _Anim, QEasingCurve as _Ease

        dlg = QDialog(self)
        dlg.setWindowTitle("OpenShader  —  À propos")
        dlg.setFixedSize(720, 620)
        dlg.setStyleSheet("""
            QDialog {
                background: #080a12;
            }
            QTabWidget::pane {
                border: none;
                background: #080a12;
            }
            QTabWidget::tab-bar { left: 0px; }
            QTabBar {
                background: transparent;
            }
            QTabBar::tab {
                background: transparent;
                color: #363a55;
                font: 11px 'Segoe UI';
                padding: 10px 22px;
                border: none;
                border-bottom: 2px solid transparent;
                margin-right: 1px;
            }
            QTabBar::tab:selected {
                color: #c8d0f0;
                border-bottom: 2px solid #5c7cff;
            }
            QTabBar::tab:hover:!selected {
                color: #666a90;
                border-bottom: 2px solid #252840;
            }
            QLabel { background: transparent; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: transparent;
                width: 4px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #1e2240;
                border-radius: 2px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QPushButton {
                background: #0e1120;
                color: #4a5080;
                border: 1px solid #1a1e35;
                border-radius: 5px;
                font: 10px 'Segoe UI';
                padding: 6px 18px;
            }
            QPushButton:hover {
                background: #141830;
                color: #7a85b8;
                border-color: #282e50;
            }
            QPushButton#btn_primary {
                background: #1a2e70;
                color: #a0b8ff;
                border: 1px solid #2a44a8;
            }
            QPushButton#btn_primary:hover {
                background: #203890;
                color: #c8d8ff;
                border-color: #3a5acc;
            }
        """)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(148)
        header.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #060810, stop:0.5 #0a0e1e, stop:1 #060810);"
            "border-bottom: 1px solid #10142a;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(32, 24, 32, 24)
        hl.setSpacing(24)

        # Logo
        _logo_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "logo.png")
        lbl_logo = QLabel()
        if _os.path.isfile(_logo_path):
            _px = _QPixmap(_logo_path).scaled(
                88, 88,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_logo.setPixmap(_px)
        else:
            lbl_logo.setText("OS")
            lbl_logo.setStyleSheet(
                "color:#a0b0ff; font: bold 24px 'Segoe UI';"
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 #0c1840, stop:1 #060e28);"
                "border: 1px solid #1e2e60;"
                "border-radius: 16px;"
            )
        lbl_logo.setFixedSize(88, 88)
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(lbl_logo)

        info_col = QVBoxLayout()
        info_col.setSpacing(5)
        info_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Nom + version inline
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lbl_name = QLabel("OpenShader")
        lbl_name.setStyleSheet(
            "color: #e0e6ff;"
            "font: bold 26px 'Segoe UI';"
            "letter-spacing: -0.5px;"
        )
        title_row.addWidget(lbl_name)
        ver_badge = QLabel("v 6.1")
        ver_badge.setStyleSheet(
            "color: #7ec880;"
            "background: #091a0e;"
            "border: 1px solid #1a4022;"
            "border-radius: 4px;"
            "font: bold 10px 'Segoe UI';"
            "padding: 3px 10px;"
        )
        title_row.addWidget(ver_badge)
        title_row.addStretch()
        info_col.addLayout(title_row)

        lbl_tagline = QLabel("Shader Edition  ·  Demoscene & VJing Tools")
        lbl_tagline.setStyleSheet("color: #3a4060; font: 11px 'Segoe UI';")
        info_col.addWidget(lbl_tagline)
        info_col.addSpacing(10)

        # Badges
        badges_row = QHBoxLayout()
        badges_row.setSpacing(7)
        for text, fg, bg, bd in [
            ("OpenGL 3.3 · Vulkan 1.1",  "#6a9cff", "#070f24", "#102048"),
            ("PyQt6 · Python 3.12",       "#b07aff", "#0f0720", "#281848"),
            ("MIT License",               "#50a878", "#061410", "#0e3020"),
            ("Février 2026",              "#404660", "#080a14", "#141828"),
        ]:
            b = QLabel(text)
            b.setStyleSheet(
                f"color:{fg}; background:{bg}; border:1px solid {bd};"
                "border-radius: 3px; font: 9px 'Segoe UI'; padding: 2px 8px;"
            )
            badges_row.addWidget(b)
        badges_row.addStretch()
        info_col.addLayout(badges_row)
        hl.addLayout(info_col, 1)
        root.addWidget(header)

        # ── helpers ────────────────────────────────────────────────────────
        def make_scroll_tab():
            outer = QWidget()
            outer.setStyleSheet("background:#080a12;")
            ov = QVBoxLayout(outer)
            ov.setContentsMargins(0, 0, 0, 0)
            sa = _SA()
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            inner = QWidget()
            inner.setStyleSheet("background:#080a12;")
            il = QVBoxLayout(inner)
            il.setContentsMargins(32, 20, 32, 24)
            il.setSpacing(0)
            sa.setWidget(inner)
            ov.addWidget(sa)
            return outer, il

        def section(layout, title, color="#5c7cff"):
            layout.addSpacing(18)
            row = QHBoxLayout()
            row.setSpacing(10)
            # Petite pastille colorée
            dot = QLabel()
            dot.setFixedSize(4, 14)
            dot.setStyleSheet(
                f"background: {color}; border-radius: 2px;"
            )
            lbl = QLabel(title.upper())
            lbl.setStyleSheet(
                f"color: {color};"
                "font: bold 9px 'Segoe UI';"
                "letter-spacing: 0.15em;"
            )
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            w = QWidget()
            w.setLayout(row)
            layout.addWidget(w)
            layout.addSpacing(3)
            # Ligne séparatrice
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"color: {color}18; max-height: 1px; margin-bottom: 10px;")
            layout.addWidget(sep)

        def info_row(layout, label, value, val_color="#7880aa"):
            row_w = QWidget()
            row_w.setStyleSheet("background:transparent;")
            rh = QHBoxLayout(row_w)
            rh.setContentsMargins(0, 3, 0, 3)
            rh.setSpacing(0)
            lbl_l = QLabel(label)
            lbl_l.setStyleSheet(
                "color: #353850; font: 10px 'Segoe UI';"
                "min-width: 170px; max-width: 170px;"
            )
            lbl_v = QLabel(value)
            lbl_v.setStyleSheet(f"color: {val_color}; font: 10px 'Segoe UI';")
            lbl_v.setWordWrap(True)
            rh.addWidget(lbl_l)
            rh.addWidget(lbl_v, 1)
            layout.addWidget(row_w)

        def feature_card(layout, icon, title, desc, accent="#5c7cff"):
            card = QWidget()
            card.setStyleSheet(
                "background: #0c0e1a;"
                "border: 1px solid #14182e;"
                "border-radius: 8px;"
                "margin-bottom: 5px;"
            )
            ch = QHBoxLayout(card)
            ch.setContentsMargins(14, 12, 16, 12)
            ch.setSpacing(14)
            # Icône
            ic = QLabel(icon)
            ic.setFixedSize(36, 36)
            ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ic.setStyleSheet(
                f"font: 17px;"
                f"background: {accent}12;"
                f"border: 1px solid {accent}28;"
                "border-radius: 8px;"
            )
            ch.addWidget(ic)
            # Texte
            tv = QVBoxLayout()
            tv.setSpacing(3)
            t1 = QLabel(title)
            t1.setStyleSheet("color: #9098c8; font: bold 10px 'Segoe UI';")
            t2 = QLabel(desc)
            t2.setStyleSheet("color: #404568; font: 9px 'Segoe UI';")
            t2.setWordWrap(True)
            tv.addWidget(t1)
            tv.addWidget(t2)
            ch.addLayout(tv, 1)
            layout.addWidget(card)

        # ── Onglet 1 — Vue d'ensemble ─────────────────────────────────────
        tab1, l1 = make_scroll_tab()

        section(l1, "Présentation", "#5c7cff")
        desc_lbl = QLabel(
            "OpenShader est un environnement de création de shaders GLSL temps réel, "
            "conçu pour la demoscène, le VJing et l'art génératif. Il combine un éditeur "
            "GLSL complet, une timeline d'animation, un moteur audio-réactif, et des outils "
            "de rendu professionnel dans une interface unifiée."
        )
        desc_lbl.setStyleSheet("color: #5a6080; font: 10px 'Segoe UI'; line-height: 160%;")
        desc_lbl.setWordWrap(True)
        l1.addWidget(desc_lbl)

        section(l1, "Stack technique", "#5c7cff")
        for label, value in [
            ("Langage",         "Python 3.12+"),
            ("Interface",       "PyQt6 ≥ 6.4"),
            ("Rendu",           "OpenGL 3.3 core  ·  Vulkan 1.1  ·  ModernGL ≥ 5.8"),
            ("Audio",           "pygame  ·  FFmpeg  ·  librosa  ·  scipy"),
            ("Intelligence IA", "OpenAI GPT-4o  ·  Ollama  ·  llama.cpp"),
            ("VR / XR",         "OpenXR 1.0  ·  pyopenxr (optionnel)"),
            ("Formats shader",  ".glsl  ·  .st  ·  .trans  ·  .demomaker"),
            ("Export vidéo",    "H.264  ·  H.265  ·  ProRes  ·  VP9  ·  AV1  ·  GIF  ·  WebP"),
        ]:
            info_row(l1, label, value)

        section(l1, "Compatibilité", "#5c7cff")
        for label, value in [
            ("Systèmes",        "Windows 10/11  ·  macOS 12+  ·  Linux (X11 / Wayland)"),
            ("GPU minimum",     "OpenGL 3.3 — Intel HD 4000 / NVIDIA GT 630 / AMD R7"),
            ("GPU recommandé",  "Vulkan 1.1 — NVIDIA RTX / AMD RX 5000+ / Apple M1"),
            ("RAM",             "4 Go minimum  ·  16 Go recommandé pour le rendu offline"),
        ]:
            info_row(l1, label, value)
        l1.addStretch()

        # ── Onglet 2 — Fonctionnalités ────────────────────────────────────
        tab2, l2 = make_scroll_tab()

        section(l2, "Cœur du moteur", "#5c7cff")
        for icon, title, desc in [
            ("⚡", "Éditeur GLSL temps réel",
             "Coloration syntaxique  ·  complétion IA  ·  hot-reload  ·  multi-pass"),
            ("🎬", "Timeline d'animation",
             "Courbes Bézier  ·  keyframes  ·  clips shader  ·  audio-sync  ·  arrangement"),
            ("✦",  "Génération IA",
             "Shaders from prompt  ·  complétion token par token  ·  détection auto-paramètres"),
            ("🕸", "Node Graph",
             "Graphe de passes composables  ·  preview live  ·  export GLSL fusionné"),
        ]:
            feature_card(l2, icon, title, desc, "#5c7cff")

        section(l2, "Audio & Protocoles", "#a855f7")
        for icon, title, desc in [
            ("🎵", "Audio-réactivité",
             "FFT  ·  waveform  ·  MFCC  ·  détection beats ML  ·  sync automatique"),
            ("🎹", "MIDI & Synthétiseur",
             "MIDI Learn  ·  synthétiseur procédural  ·  oscillateurs · LFO · ADSR"),
            ("📡", "OSC · DMX · REST",
             "OSC 1.1  ·  DMX512 / Artnet / sACN  ·  API REST FastAPI locale"),
        ]:
            feature_card(l2, icon, title, desc, "#a855f7")

        section(l2, "Rendu & Export", "#10b981")
        for icon, title, desc in [
            ("🎞", "Rendu offline",
             "TAA  ·  Motion Blur N-échantillons  ·  DCP cinéma  ·  HDR10  ·  SSAA 2×"),
            ("📦", "Export multi-cible",
             "Vidéo  ·  Shadertoy  ·  WASM  ·  64K intro  ·  Standalone  ·  Packaging"),
            ("⚡", "Upscaling IA",
             "ESRGAN  ·  DLSS-like  ·  AMD FSR — rendu basse résolution × 4"),
        ]:
            feature_card(l2, icon, title, desc, "#10b981")

        section(l2, "Collaboration & Cloud", "#f59e0b")
        for icon, title, desc in [
            ("☁",  "Cloud Sync OAuth",
             "GitHub / Google PKCE  ·  30 révisions  ·  partage par lien JWT"),
            ("👥", "Co-édition temps réel",
             "WebSocket LAN  ·  curseurs partagés  ·  chat  ·  verrouillage de pistes"),
            ("🏪", "Asset Store",
             "Index GitHub  ·  import direct  ·  publication avec preview"),
        ]:
            feature_card(l2, icon, title, desc, "#f59e0b")
        l2.addStretch()

        # ── Onglet 3 — Changelog ─────────────────────────────────────────
        tab3, l3 = make_scroll_tab()

        changelog = [
            ("v6.1", "Février 2026",  "#7ec880", "#5c7cff", [
                ("🗂", "Scene Graph",          "Arbre de scènes en dock — drag & drop, transitions nommées"),
                ("🎼", "Arrangement View",     "Vue séquentielle — blocs colorés réordonnables"),
                ("🎞", "Offline Renderer",     "TAA · Motion Blur · DCP · HDR10 · anti-aliasing temporel"),
                ("🎛", "+20 FX post-process",  "ASCII · Aquarelle · Hologramme · Vitrail · Datamosh…"),
                ("📋", "Barre de menu",        "Restructuration professionnelle en 7 catégories logiques"),
            ]),
            ("v5.0", "2025",           "#6a9cff", "#4a6cee", [
                ("☁",  "Cloud Sync OAuth",    "GitHub / Google PKCE · 30 révisions · partage JWT"),
                ("👥", "Co-édition LAN",      "WebSocket · curseurs partagés · chat · verrouillage pistes"),
                ("🥽", "VR OpenXR",           "Rendu stéréo · contrôleurs XR → uniforms · overlay VR"),
            ]),
            ("v4.0", "2024",           "#e8956a", "#d06050", [
                ("💡", "DMX512 / Artnet",     "Artnet 4 · sACN E1.31 · USB-DMX ENTTEC — 44 Hz"),
                ("🏁", "Intro Toolkit",       "4K/64K · Crinkler/Kkrunchy · estimateur de taille live"),
                ("🕸", "Export WASM",         "Bundle zero-CDN · WebGL2 + WebGPU · PWA installable"),
            ]),
            ("v3.x", "2023",           "#80d0e8", "#a070e0", [
                ("🤖", "IA Shader",           "GPT-4o · Ollama · streaming token par token"),
                ("⌨",  "Raccourcis config.",  "ACTION_REGISTRY · profil Blender-like · JSON import/export"),
                ("🌐", "API REST locale",     "FastAPI · /uniforms · /shader · Swagger inclus"),
            ]),
            ("v2.x", "2022–2023",      "#88c888", "#88c8d8", [
                ("⚡", "Backend Vulkan",      "Rendu hautes performances · Ray-Tracing si disponible"),
                ("🕸", "Node Graph",          "Graphe de passes GLSL composables"),
                ("🎹", "Synthétiseur",        "Oscillateurs · LFO · ADSR · bruit procédural → uniform"),
                ("🎵", "Audio ML",            "Détection beats/drops · keyframes auto · palette mood"),
            ]),
            ("v1.x", "2021–2022",      "#303450", "#303450", [
                ("—",  "Fondations",          "Timeline Bézier · Multi-pass · Préprocesseur GLSL · CLI headless"),
            ]),
        ]

        for ver, date, ver_color, accent, entries in changelog:
            # Header de version
            block = QWidget()
            block.setStyleSheet(
                f"background: {accent}08;"
                f"border-left: 3px solid {accent}40;"
                "border-radius: 4px;"
                "margin-bottom: 6px;"
                "padding: 2px 0px;"
            )
            bl = QVBoxLayout(block)
            bl.setContentsMargins(14, 8, 14, 8)
            bl.setSpacing(4)

            # Titre de version
            vh = QHBoxLayout()
            vh.setSpacing(10)
            ver_lbl = QLabel(ver)
            ver_lbl.setStyleSheet(
                f"color: {ver_color};"
                "font: bold 12px 'Segoe UI';"
            )
            date_lbl = QLabel(date)
            date_lbl.setStyleSheet(
                "color: #303450; font: 9px 'Segoe UI';"
            )
            vh.addWidget(ver_lbl)
            vh.addWidget(date_lbl)
            vh.addStretch()
            bl.addLayout(vh)

            # Entrées
            for icon, title, desc in entries:
                row_w = QWidget()
                row_w.setStyleSheet("background:transparent;")
                rhl = QHBoxLayout(row_w)
                rhl.setContentsMargins(0, 1, 0, 1)
                rhl.setSpacing(10)
                ic_lbl = QLabel(icon)
                ic_lbl.setFixedWidth(18)
                ic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                ic_lbl.setStyleSheet("font: 11px;")
                t_lbl = QLabel(
                    f"<span style='color:#606890;font-weight:600;'>{title}</span>"
                    f"&nbsp;&nbsp;<span style='color:#383c58;'>{desc}</span>"
                )
                t_lbl.setStyleSheet("font: 9px 'Segoe UI';")
                t_lbl.setWordWrap(True)
                rhl.addWidget(ic_lbl)
                rhl.addWidget(t_lbl, 1)
                bl.addWidget(row_w)

            l3.addWidget(block)

        l3.addStretch()

        # ── Onglet 4 — Licence & Crédits ────────────────────────────────
        tab4, l4 = make_scroll_tab()

        section(l4, "Licence", "#50a878")
        lic_lbl = QLabel(
            "OpenShader est distribué sous la <b style='color:#50a878;'>licence MIT</b>.<br><br>"
            "Permission est accordée, sans frais, à toute personne obtenant une copie "
            "de ce logiciel d'utiliser, copier, modifier, fusionner, publier, distribuer, "
            "sous-licencier et/ou vendre des copies du Logiciel, sous réserve des conditions "
            "de la licence MIT complète disponible sur <span style='color:#4a70d0;'>opensource.org/licenses/MIT</span>."
        )
        lic_lbl.setStyleSheet("color: #484c70; font: 10px 'Segoe UI'; line-height: 165%;")
        lic_lbl.setWordWrap(True)
        l4.addWidget(lic_lbl)

        section(l4, "Bibliothèques tierces", "#5c7cff")
        # Header tableau
        hdr = QWidget()
        hdr.setStyleSheet("background:#0c0f1e; border-radius: 4px;")
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(10, 5, 10, 5)
        for txt, w in [("Bibliothèque", 150), ("Licence", 170), ("Site", -1)]:
            lh = QLabel(txt)
            lh.setStyleSheet("color:#2e3460; font:bold 9px 'Segoe UI'; letter-spacing:0.1em;")
            if w > 0:
                lh.setFixedWidth(w)
            hdr_l.addWidget(lh, 0 if w > 0 else 1)
        l4.addWidget(hdr)

        for lib, lic, url in [
            ("PyQt6",          "GPL v3 / Commercial",  "riverbankcomputing.com"),
            ("ModernGL",       "MIT",                   "github.com/moderngl/moderngl"),
            ("pygame",         "LGPL v2.1",             "pygame.org"),
            ("FFmpeg",         "LGPL v2.1 / GPL v2",   "ffmpeg.org"),
            ("librosa",        "ISC",                   "librosa.org"),
            ("numpy / scipy",  "BSD-3-Clause",          "scipy.org"),
            ("OpenXR SDK",     "Apache 2.0",            "khronos.org/openxr"),
            ("FastAPI",        "MIT",                   "fastapi.tiangolo.com"),
        ]:
            rw = QWidget()
            rw.setStyleSheet(
                "background: transparent;"
                "border-bottom: 1px solid #0e1020;"
            )
            rhl = QHBoxLayout(rw)
            rhl.setContentsMargins(10, 5, 10, 5)
            rhl.setSpacing(0)
            l_lib = QLabel(lib)
            l_lib.setStyleSheet("color: #5868a0; font: bold 9px 'Segoe UI'; min-width:150px; max-width:150px;")
            l_lic = QLabel(lic)
            l_lic.setStyleSheet("color: #383c60; font: 9px 'Segoe UI'; min-width:170px; max-width:170px;")
            l_url = QLabel(f"<span style='color:#304080;'>{url}</span>")
            l_url.setStyleSheet("font: 9px 'Segoe UI';")
            rhl.addWidget(l_lib)
            rhl.addWidget(l_lic)
            rhl.addWidget(l_url, 1)
            l4.addWidget(rw)

        section(l4, "Standards & Références", "#5c7cff")
        for label, value in [
            ("Spécification GLSL",  "Khronos Group — OpenGL Shading Language 4.60"),
            ("Standard OpenXR",     "Khronos Group — OpenXR 1.0"),
            ("Standard DMX",        "ESTA / ANSI E1.11 — DMX512-A"),
            ("Standard OSC",        "CNMAT — Open Sound Control 1.0"),
            ("Inspiration UI",      "Blender  ·  DaVinci Resolve  ·  TouchDesigner"),
        ]:
            info_row(l4, label, value, "#404870")
        l4.addStretch()

        # ── Assemblage onglets ────────────────────────────────────────────
        tabs = _TW()
        tabs.setDocumentMode(True)
        tabs.addTab(tab1, "  Vue d'ensemble  ")
        tabs.addTab(tab2, "  Fonctionnalités  ")
        tabs.addTab(tab3, "  Nouveautés  ")
        tabs.addTab(tab4, "  Licence & Crédits  ")
        root.addWidget(tabs, 1)

        # ── Footer ────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(54)
        footer.setStyleSheet(
            "background: #060810;"
            "border-top: 1px solid #0e1228;"
        )
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(32, 0, 16, 0)
        fl.setSpacing(8)

        copy_lbl = QLabel("© 2021–2026 OpenShader Project  ·  MIT License")
        copy_lbl.setStyleSheet("color: #1e2240; font: 9px 'Segoe UI';")
        fl.addWidget(copy_lbl)
        fl.addStretch()

        for label, url, primary in [
            ("GitHub",        "https://github.com",          False),
            ("Documentation", "https://openshader.io/docs",  False),
            ("Shadertoy",     "https://www.shadertoy.com",    False),
            ("Fermer",        None,                           True),
        ]:
            btn = QPushButton(label)
            if primary:
                btn.setObjectName("btn_primary")
                btn.clicked.connect(dlg.accept)
            else:
                btn.clicked.connect(lambda checked=False, u=url: self._open_url(u))
            fl.addWidget(btn)

        root.addWidget(footer)
        dlg.exec()
    # ── Presets ───────────────────────────────────────────────────────────────

    def _update_presets_menu(self):
        self.presets_menu.clear()
        self._add_action(self.presets_menu, "Sauvegarder le preset actuel...", "", self._save_preset)
        self.presets_menu.addSeparator()

        presets = self._get_presets()
        if presets:
            for name in sorted(presets.keys()):
                self._add_action(self.presets_menu, name, "", lambda checked=False, n=name: self._load_preset(n))
            self.presets_menu.addSeparator()
        
        self._add_action(self.presets_menu, "Supprimer un preset...", "", self._delete_preset)

    def _get_presets(self) -> dict:
        try:
            with open("presets.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Sauvegarder le preset", "Nom du preset :")
        if not (ok and name):
            return

        presets = self._get_presets()
        presets[name] = self.timeline.evaluate(0.0) # Sauvegarde les valeurs à t=0
        with open("presets.json", "w") as f:
            json.dump(presets, f, indent=2)
        self._update_presets_menu()

    def _load_preset(self, name: str):
        presets = self._get_presets()
        if name in presets:
            for uniform, value in presets[name].items():
                track = self.timeline.get_track_by_uniform(uniform)
                if track:
                    track.add_keyframe(0.0, value) # Ajoute/remplace le KF à t=0
            self.timeline_widget.timeline_data_changed.emit()
            self._status.showMessage(f"Preset '{name}' chargé.", 3000)

    def _delete_preset(self):
        presets = self._get_presets()
        if not presets: return
        name, ok = QInputDialog.getItem(self, "Supprimer un preset", "Choisir le preset à supprimer :", list(presets.keys()), 0, False)
        if ok and name and name in presets:
            del presets[name]
            with open("presets.json", "w") as f:
                json.dump(presets, f, indent=2)
            self._update_presets_menu()

    # ── Settings (Persistance) ────────────────────────────────────────────────

    # ── v2.0 — Méthodes helpers transport (pour ScriptEngine) ────────────────

    def _on_script_transport(self, cmd: str):
        if cmd == 'play':
            self._play()
        elif cmd == 'stop':
            self._on_stop()

    def _on_seek(self, t: float):
        self._current_time = t
        self._play_offset  = t
        if self._is_playing:
            self._play_start_wall = __import__('time').perf_counter()
        self.timeline_widget.set_current_time(t)

    # ── v2.0 — Node Graph ─────────────────────────────────────────────────────

    def _show_node_graph(self):
        """Affiche le Node Graph dans son dock dédié (v2.4 : non-modal)."""
        dock = self.dock_node_graph
        # Construire le widget une seule fois
        if not isinstance(dock.widget(), NodeGraphWidget):
            ng = NodeGraphWidget()
            if hasattr(self, '_node_graph_data') and self._node_graph_data:
                ng.from_dict(self._node_graph_data)
            ng.graph_changed.connect(self._on_graph_changed)
            # v2.5 — attacher le CommandStack à la scène du graph
            ng._scene._cmd_stack = self.cmd_stack
            # Sauvegarder l'état à chaque fermeture du dock
            dock.visibilityChanged.connect(
                lambda vis, _ng=ng: self._on_node_graph_dock_closed(vis, _ng)
            )
            dock.setWidget(ng)
        # N'appeler show/raise que si le dock n'est pas déjà visible
        # (évite boucle infinie avec le signal visibilityChanged)
        if not dock.isVisible():
            dock.show()
        dock.raise_()

    def _on_node_graph_dock_closed(self, visible: bool, ng: 'NodeGraphWidget'):
        if not visible:
            self._node_graph_data = ng.to_dict()


    def _on_graph_changed(self, dag: dict):
        """Appelé quand le DAG change — met à jour l'ordre de rendu."""
        log.info("Node graph mis à jour : %s", dag)
        # Persistance dans le projet
        self._project_is_modified = True
        self._update_title()

    # ── v2.8 — Synthétiseur Procédural Visuel ───────────────────────────────────

    def _show_synth_editor(self):
        """Affiche l'éditeur de synthétiseur procédural dans son dock."""
        dock = self.dock_synth
        if not dock.isVisible():
            dock.show()
        dock.raise_()

    # ── v2.0 — Script Python ─────────────────────────────────────────────────

    def _show_script_editor(self):
        """Affiche l'éditeur de script Python dans son dock dédié (v2.4 : non-modal)."""
        dock = self.dock_script
        # Construire le widget une seule fois
        if not hasattr(self, '_script_dock_built') or not self._script_dock_built:
            self._script_dock_built = True
            from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout,
                                          QPushButton, QTextEdit, QSplitter, QLabel)
            from PyQt6.QtGui import QFont, QKeySequence, QShortcut

            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(8, 8, 8, 8)
            vl.setSpacing(6)

            splitter = QSplitter(container)
            splitter.setOrientation(Qt.Orientation.Vertical)

            # Éditeur
            self._script_editor_widget = QTextEdit()
            self._script_editor_widget.setFont(QFont('Consolas', 10))
            self._script_editor_widget.setPlaceholderText("# Entrez votre script Python ici…")
            if hasattr(self, '_last_script') and self._last_script:
                self._script_editor_widget.setPlainText(self._last_script)
            else:
                self._script_editor_widget.setPlainText(_SCRIPT_TEMPLATE)
            splitter.addWidget(self._script_editor_widget)

            # Console
            self._script_console_widget = QTextEdit()
            self._script_console_widget.setReadOnly(True)
            self._script_console_widget.setFont(QFont('Consolas', 9))
            self._script_console_widget.setMaximumHeight(120)
            self._script_console_widget.setStyleSheet("background: #080a10; color: #40a060;")
            splitter.addWidget(self._script_console_widget)
            splitter.setSizes([400, 120])
            vl.addWidget(splitter)

            # Connecter sortie du moteur de script
            self.script_engine.output_line.connect(self._script_console_widget.append)

            # Toolbar
            tb = QWidget()
            tbl = QHBoxLayout(tb)
            tbl.setContentsMargins(0, 0, 0, 0)
            tbl.setSpacing(4)

            _btn_style = """
                QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                              border-radius:3px; padding:3px 12px; font:9px 'Segoe UI'; }
                QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            """
            btn_run      = QPushButton("▶ Exécuter")
            btn_run.setStyleSheet(_btn_style)
            btn_run.setToolTip("Ctrl+Return")
            btn_clear    = QPushButton("🗑 Console")
            btn_clear.setStyleSheet(_btn_style)
            btn_template = QPushButton("📋 Template")
            btn_template.setStyleSheet(_btn_style)
            lbl_status   = QLabel("Prêt")
            lbl_status.setStyleSheet("color:#505470; font:9px 'Segoe UI';")

            tbl.addWidget(btn_run)
            tbl.addWidget(btn_clear)
            tbl.addWidget(btn_template)
            tbl.addStretch()
            tbl.addWidget(lbl_status)
            vl.addWidget(tb)

            def _run():
                src = self._script_editor_widget.toPlainText()
                self._last_script = src
                self._script_console_widget.clear()
                ok = self.script_engine.execute(src)
                lbl_status.setText("✓ OK" if ok else "✗ Erreur")
                lbl_status.setStyleSheet(
                    "color:#40c060; font:9px 'Segoe UI';" if ok
                    else "color:#c04040; font:9px 'Segoe UI';"
                )

            btn_run.clicked.connect(_run)
            btn_clear.clicked.connect(self._script_console_widget.clear)
            btn_template.clicked.connect(
                lambda: self._script_editor_widget.setPlainText(_SCRIPT_TEMPLATE)
            )

            sc = QShortcut(QKeySequence("Ctrl+Return"), self._script_editor_widget)
            sc.activated.connect(_run)

            dock.setWidget(container)

        # N'appeler show/raise que si le dock n'est pas déjà visible
        # (évite boucle infinie avec le signal visibilityChanged)
        if not dock.isVisible():
            dock.show()
        dock.raise_()

    # ── v2.0 — MIDI ──────────────────────────────────────────────────────────

    def _show_midi_panel(self):
        """Ouvre le panneau de configuration MIDI."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                      QPushButton, QComboBox, QTableWidget,
                                      QTableWidgetItem, QHeaderView, QDoubleSpinBox,
                                      QSpinBox, QLineEdit)

        dlg = QDialog(self)
        dlg.setWindowTitle("MIDI — Mapping en temps réel")
        dlg.resize(720, 460)
        dlg.setStyleSheet("""
            QDialog,QWidget { background:#0e1018; color:#c0c4d0; font:10px 'Segoe UI'; }
            QTableWidget { background:#12141a; gridline-color:#1e2030; }
            QHeaderView::section { background:#14161c; color:#7a8099; font:9px; border:none;
                                   padding:4px; }
            QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox {
                background:#1e2030; color:#c0c4d0; border:1px solid #2a2d3a; border-radius:3px;
                padding:2px 6px; }
        """)

        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(12, 12, 12, 8)
        vl.setSpacing(8)

        _btn_style = """
            QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                          border-radius:3px; padding:3px 10px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            QPushButton:checked { background:#2a3a5a; color:#80b0ff; border-color:#3a5888; }
        """

        # Disponibilité mido
        if not self.midi_engine.is_available:
            lbl = QLabel("⚠ mido non installé. Exécutez : pip install mido python-rtmidi")
            lbl.setStyleSheet("color:#c08040; padding:20px;")
            vl.addWidget(lbl)
            close = QPushButton("Fermer"); close.setStyleSheet(_btn_style)
            close.clicked.connect(dlg.accept)
            vl.addWidget(close)
            dlg.exec()
            return

        # Port selector
        hl_port = QHBoxLayout()
        hl_port.addWidget(QLabel("Port MIDI :"))
        combo_port = QComboBox()
        combo_port.addItems(self.midi_engine.list_ports() or ["(aucun port détecté)"])
        hl_port.addWidget(combo_port, 1)

        btn_connect = QPushButton("Connecter")
        btn_connect.setCheckable(True)
        btn_connect.setStyleSheet(_btn_style)
        btn_connect.setChecked(self.midi_engine.is_running)
        hl_port.addWidget(btn_connect)

        lbl_status = QLabel("Déconnecté" if not self.midi_engine.is_running else "Connecté")
        lbl_status.setStyleSheet("color:#c08040; font:9px 'Segoe UI';")
        hl_port.addWidget(lbl_status)
        vl.addLayout(hl_port)

        # Tableau des mappings
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["CC / Note", "Canal", "Uniform", "Min", "Max"])
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        def _refresh_table():
            table.setRowCount(0)
            for m in self.midi_engine.get_mappings():
                row = table.rowCount()
                table.insertRow(row)
                cc_str = f"CC {m.cc}" if m.cc is not None else f"Note {m.note}"
                table.setItem(row, 0, QTableWidgetItem(cc_str))
                table.setItem(row, 1, QTableWidgetItem(str(m.channel) if m.channel >= 0 else "All"))
                table.setItem(row, 2, QTableWidgetItem(m.uniform))
                table.setItem(row, 3, QTableWidgetItem(f"{m.lo:.2f}"))
                table.setItem(row, 4, QTableWidgetItem(f"{m.hi:.2f}"))

        _refresh_table()
        vl.addWidget(table)

        # Boutons d'ajout / suppression
        hl_btns = QHBoxLayout()

        cc_spin = QSpinBox(); cc_spin.setRange(0, 127); cc_spin.setPrefix("CC ")
        cc_spin.setFixedWidth(80)
        uniform_edit = QLineEdit(); uniform_edit.setPlaceholderText("uniform (ex: uBrightness)")
        lo_spin = QDoubleSpinBox(); lo_spin.setRange(-10.0, 10.0); lo_spin.setValue(0.0); lo_spin.setSingleStep(0.1)
        hi_spin = QDoubleSpinBox(); hi_spin.setRange(-10.0, 10.0); hi_spin.setValue(1.0); hi_spin.setSingleStep(0.1)

        btn_add    = QPushButton("+ Ajouter"); btn_add.setStyleSheet(_btn_style)
        btn_delete = QPushButton("✕ Supprimer"); btn_delete.setStyleSheet(_btn_style)
        btn_learn  = QPushButton("🎹 MIDI Learn"); btn_learn.setStyleSheet(_btn_style)
        btn_clear  = QPushButton("🗑 Tout effacer"); btn_clear.setStyleSheet(_btn_style)

        for w in [cc_spin, uniform_edit, lo_spin, hi_spin,
                  btn_add, btn_delete, btn_learn, btn_clear]:
            hl_btns.addWidget(w)
        hl_btns.addStretch()
        vl.addLayout(hl_btns)

        def _add():
            u = uniform_edit.text().strip()
            if not u:
                return
            self.midi_engine.add_mapping(u, cc=cc_spin.value(),
                                          lo=lo_spin.value(), hi=hi_spin.value())
            _refresh_table()
            self._project_is_modified = True

        def _delete():
            rows = {i.row() for i in table.selectedIndexes()}
            mappings = self.midi_engine.get_mappings()
            for r in sorted(rows, reverse=True):
                if r < len(mappings):
                    self.midi_engine.remove_mapping(mappings[r])
            _refresh_table()

        def _learn():
            btn_learn.setText("… Touchez un contrôleur")
            self.midi_engine.start_learn()

            def _on_learn(ch, cc_or_note, val):
                if cc_or_note < 128:
                    cc_spin.setValue(cc_or_note)
                btn_learn.setText("🎹 MIDI Learn")
                self.midi_engine.learn_triggered.disconnect(_on_learn)

            self.midi_engine.learn_triggered.connect(_on_learn)

        def _toggle_connect(checked):
            if checked:
                self.midi_engine.start(combo_port.currentText())
                lbl_status.setText("Connecté")
                lbl_status.setStyleSheet("color:#40c060; font:9px 'Segoe UI';")
            else:
                self.midi_engine.stop()
                lbl_status.setText("Déconnecté")
                lbl_status.setStyleSheet("color:#c08040; font:9px 'Segoe UI';")

        btn_add.clicked.connect(_add)
        btn_delete.clicked.connect(_delete)
        btn_learn.clicked.connect(_learn)
        btn_clear.clicked.connect(lambda: (self.midi_engine.clear_mappings(), _refresh_table()))
        btn_connect.toggled.connect(_toggle_connect)

        # Fermer
        btn_close = QPushButton("Fermer"); btn_close.setStyleSheet(_btn_style)
        btn_close.clicked.connect(dlg.accept)
        vl.addWidget(btn_close)

        dlg.exec()

    def _show_osc_panel(self):
        """Ouvre le panneau de configuration OSC (Open Sound Control)."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                      QPushButton, QComboBox, QTableWidget,
                                      QTableWidgetItem, QHeaderView, QDoubleSpinBox,
                                      QSpinBox, QLineEdit, QGroupBox, QTabWidget,
                                      QWidget)

        dlg = QDialog(self)
        dlg.setWindowTitle("OSC — Open Sound Control")
        dlg.resize(820, 540)
        dlg.setStyleSheet("""
            QDialog,QWidget { background:#0e1018; color:#c0c4d0; font:10px 'Segoe UI'; }
            QTableWidget { background:#12141a; gridline-color:#1e2030; }
            QHeaderView::section { background:#14161c; color:#7a8099; font:9px; border:none;
                                   padding:4px; }
            QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox {
                background:#1e2030; color:#c0c4d0; border:1px solid #2a2d3a; border-radius:3px;
                padding:2px 6px; }
            QGroupBox { border:1px solid #2a2d3a; border-radius:4px; margin-top:8px;
                        font:9px 'Segoe UI'; color:#7a8099; padding:4px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; }
            QTabWidget::pane { border:1px solid #2a2d3a; }
            QTabBar::tab { background:#14161c; color:#7a8099; padding:4px 12px;
                           border:1px solid #2a2d3a; }
            QTabBar::tab:selected { background:#1e2030; color:#c0c4d0; }
        """)

        _btn_style = """
            QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                          border-radius:3px; padding:3px 10px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            QPushButton:checked { background:#2a3a5a; color:#80b0ff; border-color:#3a5888; }
        """

        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(12, 12, 12, 8)
        vl.setSpacing(8)

        # Disponibilité python-osc
        if not self.osc_engine.is_available:
            lbl = QLabel("⚠ python-osc non installé. Exécutez : pip install python-osc")
            lbl.setStyleSheet("color:#c08040; padding:20px;")
            vl.addWidget(lbl)
            close = QPushButton("Fermer"); close.setStyleSheet(_btn_style)
            close.clicked.connect(dlg.accept)
            vl.addWidget(close)
            dlg.exec()
            return

        tabs = QTabWidget()
        vl.addWidget(tabs)

        # ── Onglet 1 : Réception (uniforms) ──────────────────────────────────
        tab_rx = QWidget()
        vl_rx  = QVBoxLayout(tab_rx)
        vl_rx.setSpacing(8)
        tabs.addTab(tab_rx, "📥 Réception (uniforms)")

        # Groupe : connexion serveur
        grp_srv = QGroupBox("Serveur UDP entrant")
        hl_srv  = QHBoxLayout(grp_srv)
        hl_srv.addWidget(QLabel("Port :"))
        spin_port = QSpinBox()
        spin_port.setRange(1024, 65535)
        spin_port.setValue(self.osc_engine.in_port)
        spin_port.setFixedWidth(90)
        hl_srv.addWidget(spin_port)

        hl_srv.addWidget(QLabel("Hôte :"))
        edit_host = QLineEdit(self.osc_engine.in_host)
        edit_host.setFixedWidth(130)
        hl_srv.addWidget(edit_host)

        btn_srv_toggle = QPushButton("Démarrer")
        btn_srv_toggle.setCheckable(True)
        btn_srv_toggle.setStyleSheet(_btn_style)
        btn_srv_toggle.setChecked(self.osc_engine.is_running)

        lbl_srv_status = QLabel(
            "En écoute" if self.osc_engine.is_running else "Arrêté"
        )
        lbl_srv_status.setStyleSheet(
            "color:#40c060; font:9px 'Segoe UI';" if self.osc_engine.is_running
            else "color:#c08040; font:9px 'Segoe UI';"
        )
        hl_srv.addWidget(btn_srv_toggle)
        hl_srv.addWidget(lbl_srv_status)
        hl_srv.addStretch()
        vl_rx.addWidget(grp_srv)

        # Tableau des mappings entrants
        table_rx = QTableWidget(0, 5)
        table_rx.setHorizontalHeaderLabels(
            ["Adresse OSC", "Uniform GLSL", "Min", "Max", "Courbe"]
        )
        table_rx.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table_rx.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_rx.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        def _refresh_rx():
            table_rx.setRowCount(0)
            for m in self.osc_engine.get_mappings():
                row = table_rx.rowCount()
                table_rx.insertRow(row)
                table_rx.setItem(row, 0, QTableWidgetItem(m.address))
                table_rx.setItem(row, 1, QTableWidgetItem(m.uniform))
                table_rx.setItem(row, 2, QTableWidgetItem(f"{m.lo:.2f}"))
                table_rx.setItem(row, 3, QTableWidgetItem(f"{m.hi:.2f}"))
                table_rx.setItem(row, 4, QTableWidgetItem(m.curve))

        _refresh_rx()
        vl_rx.addWidget(table_rx)

        # Barre d'ajout de mapping
        hl_add = QHBoxLayout()
        edit_addr    = QLineEdit(); edit_addr.setPlaceholderText("/1/fader1")
        edit_uniform = QLineEdit(); edit_uniform.setPlaceholderText("uniform (ex: uBrightness)")
        lo_spin = QDoubleSpinBox(); lo_spin.setRange(-10.0, 10.0); lo_spin.setValue(0.0); lo_spin.setSingleStep(0.1)
        hi_spin = QDoubleSpinBox(); hi_spin.setRange(-10.0, 10.0); hi_spin.setValue(1.0); hi_spin.setSingleStep(0.1)
        btn_add_rx  = QPushButton("+ Ajouter");         btn_add_rx.setStyleSheet(_btn_style)
        btn_del_rx  = QPushButton("✕ Supprimer");       btn_del_rx.setStyleSheet(_btn_style)
        btn_learn   = QPushButton("📡 OSC Learn");       btn_learn.setStyleSheet(_btn_style)
        btn_clr_rx  = QPushButton("🗑 Tout effacer");   btn_clr_rx.setStyleSheet(_btn_style)

        # Label d'écoute pour le Learn
        lbl_learn = QLabel("")
        lbl_learn.setStyleSheet("color:#80b0ff; font:9px 'Segoe UI';")

        for w in [edit_addr, edit_uniform, lo_spin, hi_spin,
                  btn_add_rx, btn_del_rx, btn_learn, btn_clr_rx]:
            hl_add.addWidget(w)
        hl_add.addStretch()
        vl_rx.addLayout(hl_add)
        vl_rx.addWidget(lbl_learn)

        def _add_rx():
            addr = edit_addr.text().strip()
            u    = edit_uniform.text().strip()
            if not addr or not u:
                return
            self.osc_engine.add_mapping(addr, u, lo=lo_spin.value(), hi=hi_spin.value())
            _refresh_rx()
            self._project_is_modified = True

        def _del_rx():
            rows = {i.row() for i in table_rx.selectedIndexes()}
            mappings = self.osc_engine.get_mappings()
            for r in sorted(rows, reverse=True):
                if r < len(mappings):
                    self.osc_engine.remove_mapping(mappings[r])
            _refresh_rx()

        def _learn():
            btn_learn.setText("… Envoyez un message OSC")
            lbl_learn.setText("En attente d'un message OSC…")
            self.osc_engine.start_learn()

            def _on_msg(address, args):
                edit_addr.setText(address)
                btn_learn.setText("📡 OSC Learn")
                lbl_learn.setText(f"Capturé : {address}  args={args}")
                try:
                    self.osc_engine.message_received.disconnect(_on_msg)
                except RuntimeError:
                    pass

            self.osc_engine.message_received.connect(_on_msg)

        def _toggle_server(checked):
            if checked:
                self.osc_engine.start(host=edit_host.text().strip() or '0.0.0.0',
                                      port=spin_port.value())
                lbl_srv_status.setText("En écoute")
                lbl_srv_status.setStyleSheet("color:#40c060; font:9px 'Segoe UI';")
            else:
                self.osc_engine.stop()
                lbl_srv_status.setText("Arrêté")
                lbl_srv_status.setStyleSheet("color:#c08040; font:9px 'Segoe UI';")

        btn_add_rx.clicked.connect(_add_rx)
        btn_del_rx.clicked.connect(_del_rx)
        btn_learn.clicked.connect(_learn)
        btn_clr_rx.clicked.connect(lambda: (self.osc_engine.clear_mappings(), _refresh_rx()))
        btn_srv_toggle.toggled.connect(_toggle_server)

        # ── Onglet 2 : Émission (sortie vers DMX/lighting) ───────────────────
        tab_tx = QWidget()
        vl_tx  = QVBoxLayout(tab_tx)
        vl_tx.setSpacing(8)
        tabs.addTab(tab_tx, "📤 Émission (DMX / éclairage)")

        grp_tx = QGroupBox("Envoyer un message OSC de test")
        fl_tx  = QHBoxLayout(grp_tx)
        fl_tx.addWidget(QLabel("Hôte :"))
        edit_tx_host = QLineEdit("192.168.1.10"); edit_tx_host.setFixedWidth(130)
        fl_tx.addWidget(edit_tx_host)
        fl_tx.addWidget(QLabel("Port :"))
        spin_tx_port = QSpinBox(); spin_tx_port.setRange(1, 65535); spin_tx_port.setValue(7000)
        spin_tx_port.setFixedWidth(80)
        fl_tx.addWidget(spin_tx_port)
        fl_tx.addWidget(QLabel("Adresse :"))
        edit_tx_addr = QLineEdit("/dmx/ch1"); edit_tx_addr.setFixedWidth(140)
        fl_tx.addWidget(edit_tx_addr)
        fl_tx.addWidget(QLabel("Valeur :"))
        spin_tx_val = QDoubleSpinBox(); spin_tx_val.setRange(0.0, 1.0); spin_tx_val.setValue(1.0); spin_tx_val.setSingleStep(0.01)
        spin_tx_val.setFixedWidth(80)
        fl_tx.addWidget(spin_tx_val)
        btn_send = QPushButton("▶ Envoyer"); btn_send.setStyleSheet(_btn_style)
        fl_tx.addWidget(btn_send)
        fl_tx.addStretch()
        vl_tx.addWidget(grp_tx)

        lbl_tx_log = QLabel("—")
        lbl_tx_log.setStyleSheet("color:#7a8099; font:9px 'Segoe UI'; padding:4px;")
        vl_tx.addWidget(lbl_tx_log)

        presets_info = QLabel(
            "💡 Cas d'usage typiques :\n"
            "  • Bridge DMX OLA      → /dmx/universe/1/slot/N  val 0..1\n"
            "  • Resolume Arena      → /composition/layers/1/clips/1/connect  1\n"
            "  • grandMA3 / ETC Eos  → /eos/chan/N/param/intensity  val 0..1\n"
            "  • Pure Data (mrpeach) → adresse libre, valeur float"
        )
        presets_info.setStyleSheet("color:#5a6080; font:9px 'Segoe UI'; padding:8px;")
        vl_tx.addWidget(presets_info)
        vl_tx.addStretch()

        def _send_test():
            host = edit_tx_host.text().strip()
            port = spin_tx_port.value()
            addr = edit_tx_addr.text().strip()
            val  = spin_tx_val.value()
            if not host or not addr:
                return
            self.osc_engine.send(addr, val, host=host, port=port)
            lbl_tx_log.setText(f"✓ Envoyé → {host}:{port}  {addr}  {val:.3f}")
            lbl_tx_log.setStyleSheet("color:#40c060; font:9px 'Segoe UI'; padding:4px;")

        btn_send.clicked.connect(_send_test)

        # ── Fermer ────────────────────────────────────────────────────────────
        btn_close = QPushButton("Fermer"); btn_close.setStyleSheet(_btn_style)
        btn_close.clicked.connect(dlg.accept)
        vl.addWidget(btn_close)

        dlg.exec()

    # ── v2.0 — VJing mode (v2.4 : sélection multi-écrans) ────────────────────

    def _start_vj_mode(self):
        """Ouvre la fenêtre VJing plein écran.
        
        Si plusieurs moniteurs sont détectés, propose un dialogue de choix.
        """
        if self._vj_window and self._vj_window.isVisible():
            self._vj_window.activateWindow()
            return

        from PyQt6.QtWidgets import QApplication
        screens = QApplication.screens()

        target_screen = None
        if len(screens) > 1:
            target_screen = self._pick_screen_dialog(screens)
            if target_screen is None:
                return  # annulé
        else:
            target_screen = screens[0]

        self._vj_window = VJWindow(self)
        # Positionner sur l'écran cible avant le plein écran
        geo = target_screen.geometry()
        self._vj_window.setGeometry(geo)
        self._vj_window.show()
        self._vj_window.windowHandle().setScreen(target_screen)
        self._vj_window.showFullScreen()

    def _pick_screen_dialog(self, screens) -> 'QScreen | None':
        """Dialogue de sélection du moniteur pour la fenêtre VJ."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                      QLabel, QPushButton, QButtonGroup,
                                      QRadioButton, QFrame)
        dlg = QDialog(self)
        dlg.setWindowTitle("Moniteur de prévisualisation")
        dlg.setMinimumWidth(360)
        vl = QVBoxLayout(dlg)
        vl.setSpacing(10)
        vl.addWidget(QLabel("<b>Choisir l'écran de sortie VJ :</b>"))

        btn_group  = QButtonGroup(dlg)
        screen_map = {}

        for idx, screen in enumerate(screens):
            geo  = screen.geometry()
            dpr  = screen.devicePixelRatio()
            name = screen.name() or f"Écran {idx + 1}"
            label = (f"{'★ ' if screen == self.screen() else ''}"
                     f"{name} — {geo.width()}×{geo.height()}"
                     f"  ({dpr:.2g}× DPR)")
            rb = QRadioButton(label)
            if screen != self.screen():
                rb.setChecked(True)   # pré-sélectionne l'écran secondaire
            screen_map[rb] = screen
            btn_group.addButton(rb)
            vl.addWidget(rb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        vl.addWidget(sep)

        hl = QHBoxLayout()
        btn_ok     = QPushButton("Lancer le mode VJ")
        btn_cancel = QPushButton("Annuler")
        btn_ok.setDefault(True)
        hl.addStretch()
        hl.addWidget(btn_cancel)
        hl.addWidget(btn_ok)
        vl.addLayout(hl)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        for rb, screen in screen_map.items():
            if rb.isChecked():
                return screen
        return screens[0]


    def _stop_vj_mode(self):
        if self._vj_window:
            self._vj_window.close()
            self._vj_window = None

    # ── v2.9 — Réalité Virtuelle OpenXR ──────────────────────────────────────

    def _start_vr_mode(self):
        """Ouvre VRWindow en plein écran. Simulation si pyopenxr absent."""
        if self._vr_window and self._vr_window.isVisible():
            self._vr_window.activateWindow()
            return
        from PyQt6.QtWidgets import QApplication
        screens = QApplication.screens()
        target  = screens[-1] if len(screens) > 1 else screens[0]
        self._vr_window = VRWindow(self)
        if self._xr_saved_mappings:
            try:
                self._vr_window._ctrl.from_dict(self._xr_saved_mappings)
            except Exception as e:
                log.warning("Restore XR mappings: %s", e)
        self._vr_window.setGeometry(target.geometry())
        self._vr_window.show()
        self._vr_window.windowHandle().setScreen(target)
        self._vr_window.showFullScreen()
        self._vr_window.start()
        sim = " (simulation)" if not openxr_available() else ""
        self._status.showMessage(
            f"Mode VR démarré{sim}  |  Tab=HUD  T=Timeline  L=Learn  Esc=Quitter", 8000)

    def _stop_vr_mode(self):
        if self._vr_window:
            self._vr_window.close()
            self._vr_window = None

    def _load_next_shader(self):
        """Charge le shader suivant (utilisé par VRWindow)."""
        paths = self.left_panel.get_shader_paths() if hasattr(self.left_panel, 'get_shader_paths') else []
        if not paths: return
        cur = self._active_image_shader_path or ''
        self._load_shader_file(paths[(paths.index(cur) + 1) % len(paths)] if cur in paths else paths[0])

    def _load_prev_shader(self):
        """Charge le shader précédent (utilisé par VRWindow)."""
        paths = self.left_panel.get_shader_paths() if hasattr(self.left_panel, 'get_shader_paths') else []
        if not paths: return
        cur = self._active_image_shader_path or ''
        self._load_shader_file(paths[(paths.index(cur) - 1) % len(paths)] if cur in paths else paths[0])

    def _show_xr_mapping_panel(self):
        """Panneau de configuration des mappings contrôleurs XR."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QTableWidget, QTableWidgetItem, QHeaderView,
            QComboBox, QDoubleSpinBox, QGroupBox, QFormLayout, QLineEdit
        )
        from .vr_window import XRControllerInput
        dlg = QDialog(self)
        dlg.setWindowTitle("Mappings contrôleurs XR — v2.9")
        dlg.resize(700, 460)
        vl = QVBoxLayout(dlg)
        xr_ok = openxr_available()
        info  = QLabel(
            ("OpenXR disponible" if xr_ok else "Mode simulation (pip install pyopenxr)") +
            "  |  Les entrées XR sont mappables sur n'importe quel uniform GLSL.\n"
            "Activez le Mode VR (Ctrl+Shift+V) puis appuyez sur L pour le XR Learn interactif."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: palette(mid); font-size: 11px; padding: 4px;")
        vl.addWidget(info)
        tbl = QTableWidget(0, 5)
        tbl.setHorizontalHeaderLabels(["Entrée XR", "Uniform", "Min", "Max", "Courbe"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setMinimumHeight(200)
        vl.addWidget(tbl)
        vr_ctrl  = getattr(self._vr_window, '_ctrl', None) if self._vr_window else None
        mappings = vr_ctrl.get_mappings() if vr_ctrl else []
        def _populate():
            tbl.setRowCount(0)
            for mp in mappings:
                r = tbl.rowCount(); tbl.insertRow(r)
                for c, v in enumerate([mp.xr_input, mp.uniform, f"{mp.lo:.3f}", f"{mp.hi:.3f}", mp.curve]):
                    tbl.setItem(r, c, QTableWidgetItem(v))
        _populate()
        grp = QGroupBox("Ajouter un mapping")
        fg  = QFormLayout(grp)
        cmb_inp   = QComboBox(); cmb_inp.addItems(XRControllerInput.ALL)
        edt_unif  = QLineEdit(); edt_unif.setPlaceholderText("ex: uSpeed, uBrightness…")
        spn_lo    = QDoubleSpinBox(); spn_lo.setRange(-1000, 1000); spn_lo.setValue(0.0)
        spn_hi    = QDoubleSpinBox(); spn_hi.setRange(-1000, 1000); spn_hi.setValue(1.0)
        cmb_curve = QComboBox(); cmb_curve.addItems(["linear", "exp", "log"])
        for lbl, w in [("Entrée XR", cmb_inp), ("Uniform", edt_unif),
                       ("Min", spn_lo), ("Max", spn_hi), ("Courbe", cmb_curve)]:
            fg.addRow(lbl, w)
        vl.addWidget(grp)
        hl = QHBoxLayout()
        btn_add, btn_del, btn_def, btn_ok = (
            QPushButton("＋ Ajouter"), QPushButton("✕ Supprimer"),
            QPushButton("↺ Défauts"), QPushButton("Fermer")
        )
        btn_ok.setDefault(True)
        for b in (btn_add, btn_del, btn_def): hl.addWidget(b)
        hl.addStretch(); hl.addWidget(btn_ok)
        vl.addLayout(hl)
        def _add():
            if not vr_ctrl or not edt_unif.text().strip(): return
            vr_ctrl.add_mapping(cmb_inp.currentText(), edt_unif.text().strip(),
                                spn_lo.value(), spn_hi.value(), cmb_curve.currentText())
            mappings[:] = vr_ctrl.get_mappings(); _populate()
        def _del():
            r = tbl.currentRow()
            if vr_ctrl and 0 <= r < len(mappings):
                vr_ctrl.remove_mapping(mappings[r])
                mappings[:] = vr_ctrl.get_mappings(); _populate()
        def _defaults():
            if vr_ctrl: vr_ctrl.default_mappings(); mappings[:] = vr_ctrl.get_mappings(); _populate()
        btn_add.clicked.connect(_add); btn_del.clicked.connect(_del)
        btn_def.clicked.connect(_defaults); btn_ok.clicked.connect(dlg.accept)
        dlg.exec()

    # ── v2.0 — Plugin panel ───────────────────────────────────────────────────

    # ── v5.0 — Asset Store ────────────────────────────────────────────────────

    def _show_asset_store(self):
        """Ouvre l'Asset Store intégré (shaders communautaires)."""
        from .asset_store import create_asset_store_browser

        dlg = QDialog(self)
        dlg.setWindowTitle("🗃 Asset Store — Shaders communautaires")
        dlg.setMinimumSize(820, 600)
        dlg.setStyleSheet("QDialog{background:#0d0f16;color:#c0c8e0;}")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)

        # glsl_getter : retourne le code de l'éditeur actif pour pré-remplir
        # le dialog de publication
        def _get_active_glsl() -> str:
            active = self._tab_widget.currentIndex() if hasattr(self, '_tab_widget') else -1
            for name, editor in self.editors.items():
                try:
                    return editor.get_code()
                except Exception:
                    pass
            return ""

        mgr, browser = create_asset_store_browser(
            glsl_getter=_get_active_glsl, parent=dlg)

        # Import one-click → charge dans l'éditeur actif
        browser.import_requested.connect(self._on_asset_imported)

        lay.addWidget(browser)

        # Fermer
        from PyQt6.QtWidgets import QDialogButtonBox
        btn_close = QPushButton("Fermer")
        btn_close.setStyleSheet(
            "QPushButton{background:#12141e;color:#5a6080;border:1px solid #1e2235;"
            "border-radius:3px;padding:4px 16px;margin:6px;}"
            "QPushButton:hover{background:#1e2235;color:#c0c8e0;}"
        )
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignRight)

        dlg.exec()

    def _on_asset_imported(self, glsl_source: str):
        """Charge un shader importé depuis l'Asset Store dans l'éditeur actif."""
        if not glsl_source.strip():
            return
        # Détermine le nom de la passe active (Image par défaut)
        pass_name = "Image"
        try:
            # Si un onglet d'éditeur est actif on prend son nom
            current_tab = self.code_tabs.currentIndex()
            pass_name   = self.code_tabs.tabText(current_tab).strip() or "Image"
        except Exception:
            pass

        if pass_name not in self.editors:
            pass_name = "Image"

        editor = self.editors.get(pass_name)
        if editor is None:
            return

        editor.set_code(glsl_source)
        self._compile_source(glsl_source, pass_name)
        self._project_is_modified = True
        self._update_title()
        self._status.showMessage(
            f"✓ Shader importé depuis l'Asset Store → passe « {pass_name} »", 5000)

    def _show_plugin_panel(self):
        """Ouvre le gestionnaire de plugins (Python + C++ natifs)."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                      QPushButton, QListWidget, QListWidgetItem,
                                      QSlider, QGroupBox, QCheckBox, QScrollArea,
                                      QTabWidget, QWidget, QTableWidget,
                                      QTableWidgetItem, QHeaderView, QLineEdit,
                                      QFileDialog)
        from PyQt6.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Plugins — OpenShader v3.0")
        dlg.resize(800, 540)
        dlg.setStyleSheet("""
            QDialog,QWidget { background:#0e1018; color:#c0c4d0; font:10px 'Segoe UI'; }
            QGroupBox { border:1px solid #2a2d3a; border-radius:4px; margin-top:8px;
                        padding:8px; color:#7a8099; }
            QGroupBox::title { subcontrol-origin:margin; padding:0 4px; }
            QSlider::groove:horizontal { background:#1e2030; height:4px; border-radius:2px; }
            QSlider::handle:horizontal { background:#4080c0; width:12px; height:12px;
                                         margin:-4px 0; border-radius:6px; }
        """)

        _btn_style = """
            QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                          border-radius:3px; padding:3px 10px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
        """

        hl = QHBoxLayout(dlg)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.setSpacing(8)

        # Liste des plugins
        plugin_list = QListWidget()
        plugin_list.setFixedWidth(200)
        plugin_list.setStyleSheet("background:#12141a; border:1px solid #1e2030;")

        plugins = self.plugin_manager.get_all()
        for p in plugins:
            item = QListWidgetItem(f"{'✓' if p.enabled else '○'} {p.name}")
            item.setData(32, p.name)  # Qt.UserRole = 32
            plugin_list.addItem(item)

        hl.addWidget(plugin_list)

        # Panneau de détail + paramètres
        detail_area = QScrollArea()
        detail_area.setWidgetResizable(True)
        detail_area.setStyleSheet("border:1px solid #1e2030;")
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.addStretch()
        detail_area.setWidget(detail_widget)
        hl.addWidget(detail_area, 1)

        def _show_plugin(name: str):
            # Vider le panneau de détail
            while detail_layout.count() > 0:
                item = detail_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            p = self.plugin_manager.get_by_name(name)
            if not p:
                return

            # Infos
            grp = QGroupBox(p.name)
            grp_layout = QVBoxLayout(grp)

            desc_lbl = QLabel(p.description)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet("color:#505470; font:italic 9px 'Segoe UI';")
            grp_layout.addWidget(desc_lbl)

            meta_lbl = QLabel(f"v{p.version}  ·  {p.author}  ·  Type: {p.plugin_type}")
            meta_lbl.setStyleSheet("color:#404060; font:9px 'Segoe UI';")
            grp_layout.addWidget(meta_lbl)

            # Toggle
            chk = QCheckBox("Activer ce plugin")
            chk.setChecked(p.enabled)

            def _toggle(state, plugin_name=name):
                self.plugin_manager.toggle(plugin_name, state == 2)
                _refresh_list()

            chk.stateChanged.connect(_toggle)
            grp_layout.addWidget(chk)
            detail_layout.addWidget(grp)

            # Paramètres (pour PostProcessPlugin)
            if isinstance(p, PostProcessPlugin):
                params_grp = QGroupBox("Paramètres")
                pgl = QVBoxLayout(params_grp)
                for pd in p.get_param_descriptors():
                    row = QHBoxLayout()
                    lbl = QLabel(pd.label)
                    lbl.setFixedWidth(100)
                    slider = QSlider()
                    slider.setOrientation(__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.Orientation.Horizontal)
                    slider.setRange(0, 1000)
                    slider.setValue(int((p.get_param(pd.name) or pd.default - pd.min) / (pd.max - pd.min) * 1000))
                    val_lbl = QLabel(f"{p.get_param(pd.name) or pd.default:.2f}")
                    val_lbl.setFixedWidth(40)
                    val_lbl.setStyleSheet("color:#7090c0;")

                    def _on_slider(v, param=pd, vlbl=val_lbl, plugin_inst=p):
                        val = param.min + v / 1000.0 * (param.max - param.min)
                        plugin_inst.set_param(param.name, val)
                        self.shader_engine.set_uniform(param.uniform, val)
                        vlbl.setText(f"{val:.2f}")

                    slider.valueChanged.connect(_on_slider)
                    row.addWidget(lbl)
                    row.addWidget(slider, 1)
                    row.addWidget(val_lbl)
                    pgl.addLayout(row)
                detail_layout.addWidget(params_grp)

            detail_layout.addStretch()

        def _refresh_list():
            plugin_list.clear()
            for p in self.plugin_manager.get_all():
                item = QListWidgetItem(f"{'✓' if p.enabled else '○'} {p.name}")
                item.setData(32, p.name)
                plugin_list.addItem(item)

        plugin_list.currentItemChanged.connect(
            lambda cur, prev: _show_plugin(cur.data(32)) if cur else None
        )

        if plugin_list.count() > 0:
            plugin_list.setCurrentRow(0)
            _show_plugin(plugins[0].name if plugins else "")

        # Footer
        footer = QHBoxLayout()
        btn_reload = QPushButton("↻ Recharger"); btn_reload.setStyleSheet(_btn_style)
        btn_reload.clicked.connect(lambda: (self.plugin_manager.scan_and_load(), _refresh_list()))
        btn_close = QPushButton("Fermer"); btn_close.setStyleSheet(_btn_style)
        btn_close.clicked.connect(dlg.accept)

        # On doit ajouter le footer dans le layout principal dlg
        # Recréer avec QVBoxLayout
        footer.addWidget(btn_reload)
        footer.addStretch()
        footer.addWidget(btn_close)

        # Wrapper pour injecter le footer sous hl
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(6)

        # ── Onglets : Plugins Python / Plugins C++ natifs ─────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border:1px solid #2a2d3a; }
            QTabBar::tab { background:#14161c; color:#7a8099; padding:4px 14px;
                           border:1px solid #2a2d3a; }
            QTabBar::tab:selected { background:#1e2030; color:#c0c4d0; }
        """)

        # Tab 1 — Plugins Python existants
        tab_py = QWidget()
        tab_py.setLayout(hl)
        tabs.addTab(tab_py, "🐍 Plugins Python")

        # Tab 2 — Plugins natifs C++
        tab_native = QWidget()
        vl_native = QVBoxLayout(tab_native)
        vl_native.setSpacing(8)
        vl_native.setContentsMargins(8, 8, 8, 8)

        # Table des plugins natifs chargés
        tbl_native = QTableWidget(0, 4)
        tbl_native.setHorizontalHeaderLabels(["Nom", "Version", "SDK", "Fichier"])
        tbl_native.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tbl_native.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        tbl_native.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl_native.setStyleSheet("background:#12141a; gridline-color:#1e2030;")

        def _refresh_native():
            tbl_native.setRowCount(0)
            for np in self.plugin_manager.native.get_all():
                r = tbl_native.rowCount()
                tbl_native.insertRow(r)
                tbl_native.setItem(r, 0, QTableWidgetItem(np.name))
                tbl_native.setItem(r, 1, QTableWidgetItem(np.version))
                sdk_str = f"v{(np.sdk_version >> 16) & 0xFFFF}.{np.sdk_version & 0xFFFF}"
                compat_item = QTableWidgetItem(f"{'✓' if np.sdk_compatible else '✗'} {sdk_str}")
                compat_item.setForeground(
                    __import__('PyQt6.QtGui', fromlist=['QColor']).QColor(
                        '#40c060' if np.sdk_compatible else '#c04040'
                    )
                )
                tbl_native.setItem(r, 2, compat_item)
                tbl_native.setItem(r, 3, QTableWidgetItem(np.path))

        _refresh_native()
        vl_native.addWidget(tbl_native, 1)

        # Uniforms exportés par le plugin sélectionné
        grp_uniF = QGroupBox("Uniforms exportés (dernier frame)")
        grp_uniF.setStyleSheet("QGroupBox { border:1px solid #2a2d3a; border-radius:4px;"
                               " margin-top:6px; color:#7a8099; } "
                               "QGroupBox::title { subcontrol-origin:margin; left:8px; }")
        hl_uni = QHBoxLayout(grp_uniF)
        lbl_uniforms = QLabel("—")
        lbl_uniforms.setStyleSheet("color:#7090c0; font:9px 'Consolas', monospace; padding:4px;")
        lbl_uniforms.setWordWrap(True)
        hl_uni.addWidget(lbl_uniforms)
        vl_native.addWidget(grp_uniF)

        def _on_native_select():
            r = tbl_native.currentRow()
            plugins_n = self.plugin_manager.native.get_all()
            if 0 <= r < len(plugins_n):
                p = plugins_n[r]
                u = p.cached_uniforms
                if u:
                    txt = "  ".join(f"{k}: {v:.4f}" for k, v in sorted(u.items()))
                else:
                    txt = "(aucun uniform disponible — lancez la lecture)"
                lbl_uniforms.setText(txt)

        tbl_native.itemSelectionChanged.connect(_on_native_select)

        # Barre d'outils : charger / décharger / recharger / hot-reload
        hl_btns_n = QHBoxLayout()

        btn_load_native   = QPushButton("📂 Charger .dll/.so"); btn_load_native.setStyleSheet(_btn_style)
        btn_unload_native = QPushButton("✕ Décharger");         btn_unload_native.setStyleSheet(_btn_style)
        btn_reload_native = QPushButton("↻ Recharger");         btn_reload_native.setStyleSheet(_btn_style)
        btn_hr_toggle     = QPushButton("🔥 Hot-reload ON");    btn_hr_toggle.setStyleSheet(_btn_style)
        btn_hr_toggle.setCheckable(True)
        btn_hr_toggle.setChecked(self.plugin_manager.native._hr_running)

        lbl_sdk_hint = QLabel("SDK : plugins/sdk/include/openshader_sdk.h")
        lbl_sdk_hint.setStyleSheet("color:#404060; font:9px 'Segoe UI';")

        for w in [btn_load_native, btn_unload_native, btn_reload_native, btn_hr_toggle]:
            hl_btns_n.addWidget(w)
        hl_btns_n.addStretch()
        hl_btns_n.addWidget(lbl_sdk_hint)
        vl_native.addLayout(hl_btns_n)

        def _load_native():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "Charger un plugin natif",
                "plugins/native",
                "Plugins natifs (*.dll *.so *.dylib);;Tous les fichiers (*)"
            )
            if path:
                p = self.plugin_manager.native.load(path)
                if p:
                    p.init(width=self.gl_widget.width(), height=self.gl_widget.height())
                _refresh_native()

        def _unload_native():
            r = tbl_native.currentRow()
            ps = self.plugin_manager.native.get_all()
            if 0 <= r < len(ps):
                self.plugin_manager.native.unload(ps[r].path)
                _refresh_native()

        def _reload_native():
            r = tbl_native.currentRow()
            ps = self.plugin_manager.native.get_all()
            if 0 <= r < len(ps):
                self.plugin_manager.native.reload(ps[r].path)
                _refresh_native()

        def _toggle_hot_reload(checked):
            if checked:
                self.plugin_manager.native.hot_reload_enable()
                btn_hr_toggle.setText("🔥 Hot-reload ON")
            else:
                self.plugin_manager.native.hot_reload_disable()
                btn_hr_toggle.setText("🔥 Hot-reload OFF")

        btn_load_native.clicked.connect(_load_native)
        btn_unload_native.clicked.connect(_unload_native)
        btn_reload_native.clicked.connect(_reload_native)
        btn_hr_toggle.toggled.connect(_toggle_hot_reload)

        tabs.addTab(tab_native, "⚙️ Plugins C++ natifs")

        # Tab 3 — Marketplace
        from .marketplace import MarketplaceBrowser
        tab_market = MarketplaceBrowser(self.plugin_manager.marketplace)
        tabs.addTab(tab_market, "🛒 Marketplace")

        wl.addWidget(tabs, 1)
        wl.addLayout(footer)
        dlg.setLayout(wl)

        dlg.exec()

    # ── v3.5 — IA Shader Generator ────────────────────────────────────────────

    def _toggle_ai_completion(self):
        """Active / désactive la complétion GLSL IA dans tous les éditeurs."""
        enabled = self._ai_completion_action.isChecked()
        for editor in self.editors.values():
            if hasattr(editor, 'ai_completion_enabled'):
                editor.ai_completion_enabled = enabled
        status = "activée" if enabled else "désactivée"
        self.statusBar().showMessage(f"✦ Complétion GLSL IA {status}", 3000)

    # ── v5.0 — Cloud Sync ────────────────────────────────────────────────────

    def _cloud_ensure_init(self):
        """Initialise CloudSyncManager + Panel la première fois (lazy)."""
        if self._cloud_manager is not None:
            return
        from .cloud_sync import create_cloud_sync
        mgr, panel = create_cloud_sync(self)
        self._cloud_manager = mgr
        self._cloud_panel   = panel

        # Callbacks de collecte de données
        mgr.set_data_callbacks(
            get_data=self._cloud_collect_bytes,
            get_name=self._cloud_project_name,
        )

        # Signaux importants vers MainWindow
        mgr.restore_ready.connect(self._on_cloud_restore_ready)
        mgr.sync_done.connect(
            lambda ok, msg: self._status.showMessage(
                f"{'☁' if ok else '✗'} {msg}", 5000))
        mgr.auth_changed.connect(self._on_cloud_auth_changed)

    def _cloud_collect_bytes(self) -> bytes:
        """Sérialise le projet courant en bytes (.demomaker ZIP en mémoire)."""
        import io, zipfile as _zf
        data = self._collect_project_data()
        buf  = io.BytesIO()
        with _zf.ZipFile(buf, "w", compression=_zf.ZIP_DEFLATED) as zf:
            zf.writestr("project.json",
                        json.dumps(data, indent=2, ensure_ascii=False))
            if self.audio_engine.file_path and \
               os.path.exists(self.audio_engine.file_path):
                zf.write(self.audio_engine.file_path,
                         "audio/" + os.path.basename(self.audio_engine.file_path))
            for p in self._texture_paths:
                if p and os.path.exists(p):
                    zf.write(p, "textures/" + os.path.basename(p))
        return buf.getvalue()

    def _cloud_project_name(self) -> str:
        if self._current_project_path:
            return os.path.splitext(
                os.path.basename(self._current_project_path))[0]
        return "Sans titre"

    def _cloud_save_quick(self):
        """☁ Sauvegarder dans le cloud (Ctrl+Shift+U)."""
        self._cloud_ensure_init()
        if not self._cloud_manager.is_logged_in:
            # Ouvre le panel pour inviter à se connecter
            self._show_cloud_panel()
            self._status.showMessage(
                "☁ Connectez-vous pour sauvegarder dans le cloud", 4000)
            return
        self._cloud_manager.save_to_cloud()

    def _cloud_notify_save(self):
        """Appelé après chaque Ctrl+S local pour déclencher l'auto-sync cloud."""
        if self._cloud_manager:
            self._cloud_manager.notify_manual_save()

    def _show_cloud_panel(self):
        """Ouvre le panneau Cloud Sync dans un dialog flottant."""
        self._cloud_ensure_init()

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("☁  Cloud Sync — OpenShader")
        dlg.setMinimumSize(540, 620)
        dlg.setStyleSheet("QDialog{background:#0d0f16;color:#c0c8e0;}")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._cloud_panel)

        # Ré-parent le panel pour ce dialog (le récupère à la fermeture)
        self._cloud_panel.setParent(dlg)

        btn_close = QPushButton("Fermer")
        btn_close.setStyleSheet(
            "QPushButton{background:#12141e;color:#5a6080;"
            "border:1px solid #1e2235;border-radius:3px;"
            "padding:4px 16px;margin:6px;}"
            "QPushButton:hover{background:#1e2235;color:#c0c8e0;}"
        )
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignRight)

        dlg.finished.connect(
            lambda: self._cloud_panel.setParent(None))   # libère le panel
        dlg.exec()

    def _on_cloud_restore_ready(self, data: bytes):
        """Données cloud/révision reçues → charge le projet en mémoire."""
        import io, zipfile as _zf, tempfile
        try:
            with _zf.ZipFile(io.BytesIO(data)) as zf:
                if "project.json" not in zf.namelist():
                    QMessageBox.warning(self, "Cloud",
                                        "Bundle invalide (project.json manquant)")
                    return
                # Écrit dans un temp file pour réutiliser _load_demomaker_bundle
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".demomaker") as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
            ok = self._load_demomaker_bundle(tmp_path)
            os.unlink(tmp_path)
            if ok:
                self._status.showMessage(
                    "☁ Projet restauré depuis le cloud", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Cloud",
                                f"Erreur lors du chargement :\n{e}")

    def _on_cloud_auth_changed(self, logged_in: bool, user):
        if logged_in and user:
            self._status.showMessage(
                f"☁ Connecté en tant que @{user.login}", 4000)
        else:
            self._status.showMessage("☁ Déconnecté du cloud", 3000)

    # ── v6.0 — Arrangement View ──────────────────────────────────────────────────

    def _show_arrangement(self):
        """Affiche le dock Arrangement View."""
        if self._dock_arrangement is None:
            self._dock_arrangement, self._arrangement_view = create_arrangement_dock(
                self._arrangement, self)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                               self._dock_arrangement)
            self._arrangement_view.time_changed.connect(self._on_timeline_seek)
            self._arrangement_view.arrangement_data_changed.connect(
                self._on_arrangement_changed)
            self._arrangement_view.scene_block_activated.connect(
                self._on_arrangement_scene)
        # Injecte les noms des scènes disponibles
        scene_names = [s.name for s in self._scene_graph.scenes]
        self._arrangement_view.set_available_scenes(scene_names)
        self._dock_arrangement.show()
        self._dock_arrangement.raise_()

    def _on_arrangement_changed(self):
        """Déclenché par tout changement dans l'Arrangement View."""
        self._render_is_dirty = True

    def _on_arrangement_scene(self, scene_name: str):
        """Charge la scène correspondant au bloc activé."""
        if not scene_name:
            return
        for idx, scene in enumerate(self._scene_graph.scenes):
            if scene.name == scene_name:
                self._on_scene_graph_activate(idx)
                return

    # ── v6.0 — Scene Graph multi-shaders ────────────────────────────────────────

    def _show_scene_graph(self):
        """Affiche le dock Scene Graph (le crée s'il n'existe pas encore)."""
        if self._dock_scene_graph is None:
            self._dock_scene_graph, self._scene_graph_wgt = create_scene_graph_dock(
                self._scene_graph, self)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_scene_graph)
            self._scene_graph_wgt.scene_activated.connect(self._on_scene_graph_activate)
            self._scene_graph_wgt.scene_preview_requested.connect(self._on_scene_thumb_requested)
        self._dock_scene_graph.show()
        self._dock_scene_graph.raise_()

    def _snapshot_current_scene(self, name: str | None = None) -> SceneItem:
        """Sérialise l'état courant dans un SceneItem."""
        import tempfile, json as _json
        # Shaders
        shaders = {n: ed.get_code() for n, ed in self.editors.items()}
        # Timeline
        tl_data: dict = {}
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
                self.timeline.save(tmp.name)
                tmp.close()
                with open(tmp.name, encoding='utf-8') as f:
                    tl_data = _json.load(f)
                import os; os.unlink(tmp.name)
        except Exception as e:
            log.warning("_snapshot_current_scene timeline: %s", e)
        # FX state
        fx = {}
        if self._active_image_shader_path:
            fx = self._shader_fx_states.get(self._active_image_shader_path, {})
        scene = SceneItem(
            name        = name or f"Scène {len(self._scene_graph.scenes)+1}",
            shaders     = shaders,
            timeline    = tl_data,
            uniforms    = {},
            fx_state    = fx,
            audio_path  = self.audio_engine.file_path,
        )
        return scene

    def _save_scene_to_graph(self):
        """Sauvegarde la scène courante dans le graphe (ajoute ou met à jour)."""
        if not self._dock_scene_graph:
            self._show_scene_graph()

        idx = self._scene_graph.active_index
        if idx >= 0:
            reply = QMessageBox.question(
                self, "Mettre à jour la scène",
                f"Écraser la scène active « {self._scene_graph.active_scene.name} » avec l'état courant ?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No  |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                scene = self._snapshot_current_scene(
                    name=self._scene_graph.active_scene.name)
                self._scene_graph_wgt.replace_scene(idx, scene)
                self._status.showMessage(f"Scène « {scene.name} » mise à jour.", 3000)
                return
        # Nouvelle scène
        name, ok = QInputDialog.getText(
            self, "Nouvelle scène", "Nom de la scène :")
        if not ok or not name.strip():
            return
        scene = self._snapshot_current_scene(name=name.strip())
        self._scene_graph_wgt.add_scene_from_current(scene)
        self._status.showMessage(f"Scène « {scene.name} » ajoutée au Scene Graph.", 3000)
        # Demande miniature
        new_idx = len(self._scene_graph.scenes) - 1
        QTimer.singleShot(300, lambda: self._on_scene_thumb_requested(new_idx))

    def _on_scene_graph_activate(self, idx: int):
        """Charge une scène depuis le graphe dans le projet courant."""
        import tempfile, json as _json
        scene = self._scene_graph.scenes[idx]
        # Shaders
        for pass_name, code in scene.shaders.items():
            if pass_name in self.editors:
                self.editors[pass_name].set_code(code)
                self._compile_source(code, pass_name)
        # Timeline
        if scene.timeline:
            try:
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
                    _json.dump(scene.timeline, tmp)
                    tmp.close()
                    self.timeline.load(tmp.name)
                    import os; os.unlink(tmp.name)
                self.timeline_widget.set_duration(self.timeline.duration)
                self.timeline_widget.sync_bpm_controls()
                self.timeline_widget.canvas.update()
                self.timeline_widget.timeline_data_changed.emit()
            except Exception as e:
                log.warning("_on_scene_graph_activate timeline: %s", e)
        # FX
        if scene.fx_state:
            self.left_panel.restore_fx_state(scene.fx_state, emit=True)
        # Audio
        if scene.audio_path and os.path.exists(scene.audio_path):
            self._load_audio_file(scene.audio_path)
        self._status.showMessage(f"🎬 Scène « {scene.name} » chargée.", 3000)

    def _on_scene_thumb_requested(self, idx: int):
        """Génère la miniature pour la scène idx (capture du viewport courant)."""
        from PyQt6.QtGui import QPixmap
        try:
            img = self.gl_widget.grabFramebuffer()
            pixmap = QPixmap.fromImage(img).scaled(
                128, 72,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            if self._scene_graph_wgt:
                self._scene_graph_wgt.set_thumbnail(idx, pixmap)
        except Exception as e:
            log.warning("_on_scene_thumb_requested: %s", e)

    # ── v5.0 — Co-édition temps réel ─────────────────────────────────────────

    def _show_collab_panel(self):
        """Ouvre (ou réaffiche) le panneau de co-édition temps réel."""
        from .collab_session import create_collab_session
        from PyQt6.QtWidgets import QDockWidget

        # Crée la session une seule fois
        if self._collab_session is None:
            session, panel, overlay = create_collab_session(self)
            self._collab_session = session
            self._collab_panel   = panel
            self._collab_overlay = overlay
            self._collab_connect_signals()

        # Dock réutilisable
        if not hasattr(self, '_collab_dock') or self._collab_dock is None:
            dock = QDockWidget("🤝 Co-édition", self)
            dock.setObjectName("DockCollab")
            dock.setWidget(self._collab_panel)
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea |
                Qt.DockWidgetArea.RightDockWidgetArea)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetClosable |
                QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._collab_dock = dock
        else:
            self._collab_dock.show()
            self._collab_dock.raise_()

    def _collab_connect_signals(self):
        """Connecte les signaux de la CollabSession à MainWindow."""
        s = self._collab_session

        # Injecte l'overlay dans le canvas de la timeline pour le dessin
        try:
            self.timeline_widget.canvas._collab_overlay = self._collab_overlay
        except Exception:
            pass

        # Shader reçu → compile dans la passe correspondante
        s.shader_changed.connect(self._on_collab_shader)

        # Uniform reçu → injecte
        s.uniform_changed.connect(self._on_collab_uniform)

        # Timeline reçue → recharge
        s.timeline_changed.connect(self._on_collab_timeline)

        # Curseur reçu → met à jour l'overlay
        s.cursor_moved.connect(self._on_collab_cursor)

        # Pair parti → retire le curseur
        s.peer_left.connect(self._on_collab_peer_left)

        # Snapshot initial (late-join) → charge l'état complet
        s.snapshot_received.connect(self._on_collab_snapshot)

        # Verrous de piste
        s.track_locked.connect(self._on_collab_lock)

        # Statut → barre de statut principale
        s.status_changed.connect(
            lambda msg: self._status.showMessage(f"🤝 {msg}", 4000))

        # Envoi du curseur local à chaque tick de la timeline
        # (branché sur le timer existant _tick)
        # On injecte dans _on_timeline_data_changed aussi
        self.timeline_widget.timeline_data_changed.connect(
            self._collab_send_snapshot_if_host)

    def _on_collab_shader(self, peer_id: str, pass_name: str, code: str):
        """Reçoit un shader d'un pair — l'applique sans renvoyer (évite boucle)."""
        if pass_name not in self.editors:
            return
        editor = self.editors[pass_name]
        # Évite de mettre le curseur en fin de fichier si le code est identique
        if editor.get_code() == code:
            return
        editor.set_code(code)
        self._compile_source(code, pass_name)
        self._render_is_dirty = True

    def _on_collab_uniform(self, peer_id: str, name: str, value):
        self.shader_engine.set_uniform(name, value)
        self._render_is_dirty = True

    def _on_collab_timeline(self, peer_id: str, data: dict):
        """Recharge la timeline depuis les données reçues d'un pair."""
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', delete=False, suffix='.json') as tmp:
                json.dump(data, tmp)
                tmp_path = tmp.name
            self.timeline.load(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            log.warning("Collab timeline load: %s", e)
            return
        self.timeline_widget.set_duration(self.timeline.duration)
        self.timeline_widget.sync_bpm_controls()
        self.timeline_widget.sync_loop_controls()
        self.timeline_widget.canvas.update()
        self._render_is_dirty = True

    def _on_collab_cursor(self, peer_id: str, name: str, color: str,
                          time_: float, track_id):
        if self._collab_overlay:
            self._collab_overlay.update_cursor(peer_id, name, color, time_, track_id)
            # Redessine la timeline
            try:
                self.timeline_widget.canvas.update()
            except Exception:
                pass

    def _on_collab_peer_left(self, peer_id: str):
        if self._collab_overlay:
            self._collab_overlay.remove_cursor(peer_id)
            try:
                self.timeline_widget.canvas.update()
            except Exception:
                pass

    def _on_collab_snapshot(self, data: dict):
        """Snapshot reçu lors du late-join : charge l'état complet du serveur."""
        shaders = data.get("shaders", {})
        for pass_name, code in shaders.items():
            if pass_name in self.editors and code:
                self.editors[pass_name].set_code(code)
                self._compile_source(code, pass_name)

        tl = data.get("timeline", {})
        if tl:
            self._on_collab_timeline("__server__", tl)

        self._status.showMessage(
            f"🤝 Session rejointe — {len(data.get('peers',[]))} pair(s) connecté(s)", 5000)

    def _on_collab_lock(self, peer_id: str, name: str, track_id: int, locked: bool):
        state = "verrouillée" if locked else "déverrouillée"
        self._status.showMessage(
            f"🤝 Piste #{track_id} {state} par {name}", 3000)

    def _collab_send_cursor(self):
        """Envoie la position du curseur local si une session est active."""
        if self._collab_session and self._collab_session.active:
            self._collab_session.send_cursor(
                self._current_time,
                track_id=None,   # TODO: injecter la piste sélectionnée
            )

    def _collab_send_snapshot_if_host(self):
        """Quand la timeline change, met à jour le snapshot serveur (hôte seulement)."""
        if self._collab_session and self._collab_session.is_host:
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(
                        mode='w', delete=False, suffix='.json') as tmp:
                    self.timeline.save(tmp.name)
                    with open(tmp.name, 'r') as f:
                        tl_data = json.load(f)
                    os.unlink(tmp.name)
            except Exception:
                tl_data = {}
            shaders = {n: e.get_code() for n, e in self.editors.items()}
            self._collab_session.update_snapshot(shaders, tl_data)

    def _collab_on_shader_compiled(self, pass_name: str, code: str):
        """Appelé après chaque compilation locale → propage aux pairs."""
        if self._collab_session and self._collab_session.active:
            self._collab_session.send_shader(pass_name, code)

    def _toggle_rest_server(self):
        """Démarre ou arrête le serveur REST local."""
        from PyQt6.QtWidgets import QInputDialog
        from .rest_api import OpenShaderRESTServer

        if self._rest_server and self._rest_server.running:
            self._rest_server.stop()
            self._rest_server = None
            self._rest_server_action.setChecked(False)
            self.statusBar().showMessage("🌐 API REST arrêtée", 3000)
            return

        # Demande le port
        port, ok = QInputDialog.getInt(
            self, "API REST locale",
            "Port d'écoute (localhost seulement) :",
            8765, 1024, 65535, 1,
        )
        if not ok:
            self._rest_server_action.setChecked(False)
            return

        self._rest_server = OpenShaderRESTServer(self)
        self._rest_server.start(host="127.0.0.1", port=port)

        if self._rest_server.running:
            self._rest_server_action.setChecked(True)
            self.statusBar().showMessage(
                f"🌐 API REST démarrée → http://127.0.0.1:{port}  "
                f"(docs: http://127.0.0.1:{port}/docs)", 6000)
        else:
            self._rest_server_action.setChecked(False)
            self.statusBar().showMessage(
                "🌐 Erreur REST — pip install fastapi uvicorn", 5000)

    def _show_ai_panel(self):
        """
        Ouvre le panneau de génération de shaders par IA.
        Interface : terminal dark, streaming live, diff avant/après, historique.
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
            QLabel, QPushButton, QLineEdit, QTextEdit,
            QListWidget, QListWidgetItem, QGroupBox,
            QTabWidget, QWidget, QComboBox, QScrollArea,
            QSizePolicy, QFrame,
        )
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui  import QFont, QColor, QTextCursor, QTextCharFormat

        # ── Shader courant ────────────────────────────────────────────────────
        current_editor = self._current_editor()
        current_shader = current_editor.get_code() if current_editor else ""

        # ── Dialogue ──────────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("✦ IA Shader Generator — DemoMaker v3.5")
        dlg.resize(1060, 720)

        _MONO  = "Consolas, 'Courier New', monospace"
        _SANS  = "'Segoe UI', system-ui, sans-serif"
        _BG    = "#070a0f"
        _SURF  = "#0d1117"
        _SURF2 = "#111720"
        _BORD  = "#1c2535"
        _ACC   = "#3d8eff"
        _ACC2  = "#00d4aa"
        _MUTED = "#3a4a66"
        _TEXT  = "#c8d8f0"
        _DIM   = "#5a7090"

        dlg.setStyleSheet(f"""
            QDialog, QWidget {{ background:{_BG}; color:{_TEXT};
                                font:10px {_SANS}; }}
            QSplitter::handle {{ background:{_BORD}; width:1px; height:1px; }}
            QGroupBox {{
                border:1px solid {_BORD}; border-radius:4px;
                margin-top:10px; color:{_MUTED}; padding:8px 6px 6px 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin:margin; left:10px;
                font:9px {_SANS}; letter-spacing:.06em;
                text-transform:uppercase;
            }}
            QTabWidget::pane {{ border:1px solid {_BORD}; background:{_SURF}; }}
            QTabBar::tab {{
                background:{_BG}; color:{_MUTED}; padding:5px 16px;
                border:1px solid {_BORD}; border-bottom:none;
                font:9px {_SANS}; letter-spacing:.05em;
            }}
            QTabBar::tab:selected {{ background:{_SURF}; color:{_TEXT}; }}
            QScrollBar:vertical {{
                background:{_BG}; width:6px; margin:0;
            }}
            QScrollBar::handle:vertical {{
                background:{_BORD}; border-radius:3px; min-height:20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

        _input_style = f"""
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background:{_SURF}; color:{_TEXT};
                border:1px solid {_BORD}; border-radius:3px;
                padding:6px 8px; font:11px {_MONO};
                selection-background-color:{_ACC};
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border-color:{_ACC};
            }}
        """
        _btn = f"""
            QPushButton {{
                background:{_SURF2}; color:{_DIM};
                border:1px solid {_BORD}; border-radius:3px;
                padding:5px 14px; font:9px {_SANS};
                letter-spacing:.05em;
            }}
            QPushButton:hover {{ background:{_BORD}; color:{_TEXT}; }}
            QPushButton:disabled {{ color:{_MUTED}; opacity:.4; }}
        """
        _btn_primary = f"""
            QPushButton {{
                background:#132240; color:{_ACC};
                border:1px solid {_ACC}44; border-radius:3px;
                padding:6px 18px; font:700 9px {_SANS};
                letter-spacing:.08em;
            }}
            QPushButton:hover {{ background:#1a3050; color:#6aadff; }}
            QPushButton:disabled {{ background:{_SURF2}; color:{_MUTED};
                                    border-color:{_BORD}; }}
        """
        _btn_apply = f"""
            QPushButton {{
                background:#0d2a22; color:{_ACC2};
                border:1px solid {_ACC2}44; border-radius:3px;
                padding:6px 18px; font:700 9px {_SANS};
                letter-spacing:.08em;
            }}
            QPushButton:hover {{ background:#143830; color:#30ffcc; }}
            QPushButton:disabled {{ background:{_SURF2}; color:{_MUTED};
                                    border-color:{_BORD}; }}
        """

        # ── Layout principal — splitter horizontal ────────────────────────────
        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Barre du haut : titre + status backend
        topbar = QWidget()
        topbar.setFixedHeight(38)
        topbar.setStyleSheet(f"background:{_SURF}; border-bottom:1px solid {_BORD};")
        tbl = QHBoxLayout(topbar)
        tbl.setContentsMargins(14, 0, 14, 0)

        lbl_title = QLabel("✦  IA SHADER GENERATOR")
        lbl_title.setStyleSheet(f"color:{_ACC}; font:700 10px {_MONO}; letter-spacing:.12em;")
        lbl_backend = QLabel("● détection…")
        lbl_backend.setStyleSheet(f"color:{_MUTED}; font:9px {_MONO};")
        tbl.addWidget(lbl_title)
        tbl.addStretch()
        tbl.addWidget(lbl_backend)
        root.addWidget(topbar)

        # Corps — splitter gauche/droite
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        root.addWidget(splitter, 1)

        # ── PANNEAU GAUCHE — Prompt + streaming + actions ─────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{_BG};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 10, 14)
        ll.setSpacing(10)

        # Prompt
        grp_prompt = QGroupBox("PROMPT")
        grp_prompt.setStyleSheet(
            f"QGroupBox{{border:1px solid {_BORD};border-radius:4px;"
            f"margin-top:10px;color:{_MUTED};padding:8px 6px 6px 6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;"
            f"font:9px {_SANS};letter-spacing:.06em;}}")
        gpl = QVBoxLayout(grp_prompt)

        edit_prompt = QTextEdit()
        edit_prompt.setPlaceholderText(
            "Décris l'effet voulu…\n"
            "ex: « Tunnel raymarché psychédélique audio-réactif »\n"
            "ex: « Ajouter du bruit Perlin à ce shader »"
        )
        edit_prompt.setFixedHeight(80)
        edit_prompt.setStyleSheet(_input_style + f"QTextEdit{{font:11px {_SANS};}}")
        gpl.addWidget(edit_prompt)

        # Suggestions
        lbl_sugg = QLabel("SUGGESTIONS")
        lbl_sugg.setStyleSheet(f"color:{_MUTED}; font:8px {_SANS}; letter-spacing:.06em;")
        gpl.addWidget(lbl_sugg)

        sugg_area = QWidget()
        sugg_flow = QHBoxLayout(sugg_area)
        sugg_flow.setContentsMargins(0, 0, 0, 0)
        sugg_flow.setSpacing(5)
        _sugg_btns: list[QPushButton] = []

        def _populate_suggestions(suggestions: list[str]):
            # Vide et recrée
            for b in _sugg_btns:
                b.setParent(None)
            _sugg_btns.clear()

            for s in suggestions[:4]:
                b = QPushButton(s)
                b.setStyleSheet(f"""
                    QPushButton {{
                        background:{_SURF}; color:{_DIM};
                        border:1px solid {_BORD}; border-radius:3px;
                        padding:3px 8px; font:8px {_SANS};
                        text-align:left;
                    }}
                    QPushButton:hover {{ color:{_TEXT}; border-color:{_ACC}44; }}
                """)
                b.clicked.connect(lambda _, txt=s: edit_prompt.setPlainText(txt))
                sugg_flow.addWidget(b)
                _sugg_btns.append(b)
            sugg_flow.addStretch()

        gpl.addWidget(sugg_area)
        ll.addWidget(grp_prompt)

        # Options : modèle + température
        opts_row = QHBoxLayout()
        combo_model = QComboBox()
        combo_model.setEditable(True)
        combo_model.addItems(["auto", "gpt-4o", "gpt-4-turbo", "codestral",
                               "deepseek-coder:6.7b", "mistral", "llama3.1"])
        combo_model.setFixedWidth(180)
        combo_model.setStyleSheet(f"""
            QComboBox {{ background:{_SURF}; color:{_TEXT};
                         border:1px solid {_BORD}; border-radius:3px;
                         padding:4px 8px; font:9px {_SANS}; }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{ background:{_SURF2}; color:{_TEXT};
                                           border:1px solid {_BORD}; }}
        """)
        lbl_model = QLabel("Modèle :")
        lbl_model.setStyleSheet(f"color:{_DIM}; font:9px {_SANS};")

        lbl_temp = QLabel("Temp :")
        lbl_temp.setStyleSheet(f"color:{_DIM}; font:9px {_SANS};")
        combo_temp = QComboBox()
        combo_temp.addItems(["0.3 — précis", "0.7 — équilibré", "1.0 — créatif"])
        combo_temp.setCurrentIndex(1)
        combo_temp.setFixedWidth(130)
        combo_temp.setStyleSheet(combo_model.styleSheet())

        opts_row.addWidget(lbl_model)
        opts_row.addWidget(combo_model)
        opts_row.addSpacing(12)
        opts_row.addWidget(lbl_temp)
        opts_row.addWidget(combo_temp)
        opts_row.addStretch()
        ll.addLayout(opts_row)

        # Boutons d'action
        btn_row = QHBoxLayout()
        btn_generate = QPushButton("▶  GÉNÉRER")
        btn_generate.setStyleSheet(_btn_primary)
        btn_generate.setFixedHeight(30)
        btn_stop = QPushButton("■  Stop")
        btn_stop.setStyleSheet(_btn)
        btn_stop.setEnabled(False)
        btn_detect = QPushButton("⟳ Détecter backends")
        btn_detect.setStyleSheet(_btn)

        btn_row.addWidget(btn_generate)
        btn_row.addWidget(btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(btn_detect)
        ll.addLayout(btn_row)

        # Stream output
        grp_stream = QGroupBox("SORTIE LIVE")
        grp_stream.setStyleSheet(
            f"QGroupBox{{border:1px solid {_BORD};border-radius:4px;"
            f"margin-top:10px;color:{_MUTED};padding:8px 6px 6px 6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;"
            f"font:9px {_SANS};letter-spacing:.06em;}}")
        gsl = QVBoxLayout(grp_stream)

        stream_out = QTextEdit()
        stream_out.setReadOnly(True)
        stream_out.setFont(QFont("Consolas", 9))
        stream_out.setStyleSheet(
            f"background:{_SURF}; color:#7ecfaa; border:none;"
            f"font:9px {_MONO}; selection-background-color:{_ACC};")
        stream_out.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        gsl.addWidget(stream_out)

        # Barre de statut stream
        lbl_status = QLabel("En attente…")
        lbl_status.setStyleSheet(f"color:{_MUTED}; font:8px {_MONO}; padding:2px 0;")
        gsl.addWidget(lbl_status)

        # Boutons appliquer / annuler (cachés jusqu'à résultat)
        apply_row = QHBoxLayout()
        btn_apply   = QPushButton("✓  Appliquer au shader")
        btn_discard = QPushButton("✕  Ignorer")
        btn_apply.setStyleSheet(_btn_apply);  btn_apply.setEnabled(False)
        btn_discard.setStyleSheet(_btn);      btn_discard.setEnabled(False)
        apply_row.addWidget(btn_apply)
        apply_row.addWidget(btn_discard)
        apply_row.addStretch()
        gsl.addLayout(apply_row)

        ll.addWidget(grp_stream, 1)

        # ── Config backend (en bas à gauche)
        grp_cfg = QGroupBox("CONFIGURATION BACKENDS")
        grp_cfg.setStyleSheet(
            f"QGroupBox{{border:1px solid {_BORD}22;border-radius:4px;"
            f"margin-top:6px;color:{_MUTED}44;padding:6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;"
            f"font:8px {_SANS};letter-spacing:.06em;color:{_MUTED}88;}}")
        gcl = QVBoxLayout(grp_cfg)
        gcl.setSpacing(5)

        edit_openai_key = QLineEdit()
        edit_openai_key.setPlaceholderText("Clé API OpenAI (sk-…) — jamais stockée dans le projet")
        edit_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        edit_openai_key.setStyleSheet(_input_style + f"QLineEdit{{font:9px {_SANS};}}")

        edit_ollama_host = QLineEdit("http://localhost:11434")
        edit_ollama_host.setStyleSheet(_input_style + f"QLineEdit{{font:9px {_MONO};}}")
        edit_llamacpp_host = QLineEdit("http://localhost:8080")
        edit_llamacpp_host.setStyleSheet(_input_style + f"QLineEdit{{font:9px {_MONO};}}")

        for lbl_txt, widget in [
            ("OpenAI API key :", edit_openai_key),
            ("Ollama host :", edit_ollama_host),
            ("llama.cpp host :", edit_llamacpp_host),
        ]:
            row = QHBoxLayout()
            l = QLabel(lbl_txt); l.setFixedWidth(110)
            l.setStyleSheet(f"color:{_MUTED}; font:8px {_SANS};")
            row.addWidget(l); row.addWidget(widget)
            gcl.addLayout(row)

        ll.addWidget(grp_cfg)

        # ── PANNEAU DROIT — Tabs : Diff + Historique ──────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{_BG};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 14, 14, 14)
        rl.setSpacing(0)

        tabs_right = QTabWidget()
        rl.addWidget(tabs_right)

        # ── Tab 1 : Diff avant / après ────────────────────────────────────────
        tab_diff = QWidget()
        tdl = QVBoxLayout(tab_diff)
        tdl.setContentsMargins(8, 10, 8, 8)
        tdl.setSpacing(6)

        diff_splitter = QSplitter(Qt.Orientation.Vertical)

        def _make_code_view(title: str, color: str) -> tuple:
            w   = QWidget()
            wl  = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(2)
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{color}; font:8px {_SANS}; "
                               f"letter-spacing:.06em; padding:2px 0;")
            te  = QTextEdit()
            te.setReadOnly(True)
            te.setFont(QFont("Consolas", 9))
            te.setStyleSheet(f"background:{_SURF}; color:{_TEXT}; border:none; "
                              f"font:9px {_MONO};")
            te.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
            wl.addWidget(lbl)
            wl.addWidget(te)
            return w, te

        w_before, te_before = _make_code_view("▸ AVANT", _MUTED)
        w_after,  te_after  = _make_code_view("▸ APRÈS  (généré)", _ACC2)

        diff_splitter.addWidget(w_before)
        diff_splitter.addWidget(w_after)
        diff_splitter.setSizes([200, 300])
        tdl.addWidget(diff_splitter, 1)

        # Charge le shader courant dans "avant"
        te_before.setPlainText(current_shader)

        # Ligne diff stats
        lbl_diff_stats = QLabel("")
        lbl_diff_stats.setStyleSheet(f"color:{_MUTED}; font:8px {_MONO}; padding:2px 0;")
        tdl.addWidget(lbl_diff_stats)

        tabs_right.addTab(tab_diff, "⟷  Diff")

        # ── Tab 2 : Historique ────────────────────────────────────────────────
        tab_hist = QWidget()
        thl = QVBoxLayout(tab_hist)
        thl.setContentsMargins(8, 10, 8, 8)
        thl.setSpacing(6)

        hist_list = QListWidget()
        hist_list.setStyleSheet(f"""
            QListWidget {{
                background:{_SURF}; border:none;
                font:9px {_SANS}; color:{_TEXT};
            }}
            QListWidget::item {{
                padding:6px 8px; border-bottom:1px solid {_BORD};
            }}
            QListWidget::item:selected {{
                background:{_ACC}22; color:{_TEXT};
            }}
            QListWidget::item:hover {{
                background:{_SURF2};
            }}
        """)
        thl.addWidget(hist_list, 1)

        hist_detail = QTextEdit()
        hist_detail.setReadOnly(True)
        hist_detail.setFixedHeight(120)
        hist_detail.setStyleSheet(
            f"background:{_SURF}; color:{_DIM}; border:none; font:9px {_MONO};")
        thl.addWidget(hist_detail)

        # Boutons historique
        hl_hist_btns = QHBoxLayout()
        btn_hist_apply  = QPushButton("↩ Restaurer")
        btn_hist_clear  = QPushButton("⌫ Vider")
        btn_hist_apply.setStyleSheet(_btn)
        btn_hist_clear.setStyleSheet(_btn)
        hl_hist_btns.addWidget(btn_hist_apply)
        hl_hist_btns.addStretch()
        hl_hist_btns.addWidget(btn_hist_clear)
        thl.addLayout(hl_hist_btns)

        tabs_right.addTab(tab_hist, "◷  Historique")

        # ── Assemblage splitter ───────────────────────────────────────────────
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 520])

        # ── Barre du bas ──────────────────────────────────────────────────────
        botbar = QWidget()
        botbar.setFixedHeight(34)
        botbar.setStyleSheet(f"background:{_SURF}; border-top:1px solid {_BORD};")
        bbl = QHBoxLayout(botbar)
        bbl.setContentsMargins(14, 0, 14, 0)
        lbl_hint = QLabel(
            "Ctrl+Return = Générer  ·  "
            "Backends : OpenAI cloud · Ollama local · llama.cpp GGUF · Stub offline"
        )
        lbl_hint.setStyleSheet(f"color:{_MUTED}; font:8px {_SANS};")
        btn_close = QPushButton("Fermer")
        btn_close.setStyleSheet(_btn)
        btn_close.setFixedWidth(70)
        btn_close.clicked.connect(dlg.accept)
        bbl.addWidget(lbl_hint)
        bbl.addStretch()
        bbl.addWidget(btn_close)
        root.addWidget(botbar)

        # ══════════════════════════════════════════════════════════════════════
        #  Logique
        # ══════════════════════════════════════════════════════════════════════

        _current_result = {"glsl": "", "active": False}
        _history_snapshots: list = []  # copie locale synchronisée

        # ── Backend detection ─────────────────────────────────────────────────
        def _update_backend_label(name: str):
            colors = {
                "openai":  (_ACC,    "OpenAI cloud"),
                "ollama":  (_ACC2,   "Ollama local"),
                "llamacpp": ("#ffd080", "llama.cpp GGUF"),
                "stub":    (_MUTED,  "Stub offline"),
            }
            col, label = colors.get(name, (_MUTED, name))
            lbl_backend.setStyleSheet(f"color:{col}; font:9px {_MONO};")
            lbl_backend.setText(f"● {label}")

        self.ai_generator.backend_changed.connect(_update_backend_label)

        def _apply_config_and_detect():
            key = edit_openai_key.text().strip()
            if key:
                self.ai_generator.set_openai_key(key)
            self.ai_generator.set_ollama_host(edit_ollama_host.text().strip())
            self.ai_generator.set_llamacpp_host(edit_llamacpp_host.text().strip())
            # v3.6 — propage aux moteurs de complétion IA des éditeurs
            for editor in self.editors.values():
                if hasattr(editor, 'sync_ai_completion_config'):
                    editor.sync_ai_completion_config(self.ai_generator)
            lbl_backend.setText("● détection…")
            lbl_backend.setStyleSheet(f"color:{_MUTED}; font:9px {_MONO};")

            def _detect():
                available = self.ai_generator.detect_backends()
                if not available:
                    _update_backend_label("stub")

            import threading as _thr
            _thr.Thread(target=_detect, daemon=True).start()

        btn_detect.clicked.connect(_apply_config_and_detect)

        # Détection initiale silencieuse
        import threading as _thr
        _thr.Thread(target=self.ai_generator.detect_backends, daemon=True).start()

        # ── Suggestions initiales ─────────────────────────────────────────────
        self.ai_generator.suggestion_ready.connect(_populate_suggestions)
        self.ai_generator.suggest(current_shader)

        # Recharge les suggestions quand l'utilisateur efface le prompt
        def _on_prompt_changed():
            if not edit_prompt.toPlainText().strip():
                self.ai_generator.suggest(
                    current_editor.get_code() if current_editor else "")

        edit_prompt.textChanged.connect(_on_prompt_changed)

        # ── Streaming tokens → stream_out ─────────────────────────────────────
        def _on_token(tok: str):
            stream_out.moveCursor(QTextCursor.MoveOperation.End)
            stream_out.insertPlainText(tok)
            stream_out.moveCursor(QTextCursor.MoveOperation.End)

        self.ai_generator.token_received.connect(_on_token)

        # ── Génération terminée ───────────────────────────────────────────────
        def _on_done(result):
            btn_generate.setEnabled(True)
            btn_stop.setEnabled(False)
            btn_apply.setEnabled(True)
            btn_discard.setEnabled(True)

            _current_result["glsl"]   = result.glsl
            _current_result["active"] = True

            # Diff panel
            te_after.setPlainText(result.glsl)
            before = te_before.toPlainText()
            b_lines = before.splitlines()
            a_lines = result.glsl.splitlines()
            added   = sum(1 for l in a_lines if l not in b_lines)
            removed = sum(1 for l in b_lines if l not in a_lines)
            lbl_diff_stats.setText(
                f"+{added} lignes  −{removed} lignes  ·  "
                f"{result.backend} / {result.model}  ·  {result.duration_s:.1f}s"
            )
            tabs_right.setCurrentIndex(0)  # bascule sur Diff

            lbl_status.setText(
                f"✓  Généré en {result.duration_s:.1f}s "
                f"({result.backend} · {result.model})"
            )
            lbl_status.setStyleSheet(f"color:{_ACC2}; font:8px {_MONO}; padding:2px 0;")

            # Historique
            _refresh_history()

        self.ai_generator.generation_done.connect(_on_done)

        # ── Erreur ────────────────────────────────────────────────────────────
        def _on_error(msg: str):
            btn_generate.setEnabled(True)
            btn_stop.setEnabled(False)
            lbl_status.setText(f"❌  {msg}")
            lbl_status.setStyleSheet(f"color:#ff6060; font:8px {_MONO}; padding:2px 0;")

        self.ai_generator.generation_error.connect(_on_error)

        # ── Lancer la génération ──────────────────────────────────────────────
        def _generate():
            prompt = edit_prompt.toPlainText().strip()
            if not prompt:
                edit_prompt.setFocus()
                return
            if self.ai_generator.is_generating:
                return

            # Applique la config avant de lancer
            key = edit_openai_key.text().strip()
            if key:
                self.ai_generator.set_openai_key(key)

            stream_out.clear()
            btn_apply.setEnabled(False)
            btn_discard.setEnabled(False)
            btn_generate.setEnabled(False)
            btn_stop.setEnabled(True)
            lbl_status.setText("⟳  Génération en cours…")
            lbl_status.setStyleSheet(f"color:{_ACC}; font:8px {_MONO}; padding:2px 0;")
            _current_result["active"] = False

            # Snapshot before pour diff
            cur = current_editor.get_code() if current_editor else ""
            te_before.setPlainText(cur)
            te_after.clear()
            lbl_diff_stats.setText("")

            model = combo_model.currentText().strip()
            if model == "auto":
                model = ""
            temp_map = {"0.3 — précis": 0.3, "0.7 — équilibré": 0.7, "1.0 — créatif": 1.0}
            temp = temp_map.get(combo_temp.currentText(), 0.7)

            self.ai_generator.generate(
                prompt        = prompt,
                current_shader= cur,
                model         = model,
                temperature   = temp,
            )

        btn_generate.clicked.connect(_generate)

        # Ctrl+Return dans le prompt → générer
        from PyQt6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence("Ctrl+Return"), edit_prompt)
        sc.activated.connect(_generate)

        # ── Appliquer le résultat ─────────────────────────────────────────────
        def _apply():
            glsl = _current_result.get("glsl", "")
            if not glsl:
                return
            if current_editor:
                current_editor.set_code(glsl)
                self._on_code_changed()  # re-compile
                self._status.showMessage("✦ Shader IA appliqué", 4000)
            dlg.accept()

        def _discard():
            _current_result["active"] = False
            stream_out.clear()
            te_after.clear()
            lbl_diff_stats.setText("")
            btn_apply.setEnabled(False)
            btn_discard.setEnabled(False)
            lbl_status.setText("Résultat ignoré.")
            lbl_status.setStyleSheet(f"color:{_MUTED}; font:8px {_MONO}; padding:2px 0;")

        btn_apply.clicked.connect(_apply)
        btn_discard.clicked.connect(_discard)

        # ── Historique ────────────────────────────────────────────────────────
        def _refresh_history():
            hist_list.clear()
            _history_snapshots.clear()
            for e in reversed(self.ai_generator.history):
                icon  = "✓" if e.ok else "✗"
                color = _ACC2 if e.ok else "#ff6060"
                item  = QListWidgetItem(f"{icon}  {e.time_label}  ·  {e.prompt_short}")
                item.setForeground(QColor(color))
                hist_list.addItem(item)
                _history_snapshots.append(e)

        def _on_hist_select():
            r = hist_list.currentRow()
            if 0 <= r < len(_history_snapshots):
                e = _history_snapshots[r]
                hist_detail.setPlainText(
                    f"Backend : {e.backend}  ·  Modèle : {e.model}\n"
                    f"Durée   : {e.duration_s:.1f}s\n"
                    f"Statut  : {'OK' if e.ok else 'Erreur'}\n"
                    f"Prompt  : {e.prompt}\n"
                    f"---\n"
                    + (e.glsl_after[:400] + "…" if len(e.glsl_after) > 400 else e.glsl_after)
                )

        def _hist_apply():
            r = hist_list.currentRow()
            if 0 <= r < len(_history_snapshots):
                e = _history_snapshots[r]
                if e.ok and e.glsl_after and current_editor:
                    current_editor.set_code(e.glsl_after)
                    self._on_code_changed()
                    self._status.showMessage(f"✦ Shader restauré depuis l'historique ({e.time_label})", 4000)
                    dlg.accept()

        def _hist_clear():
            self.ai_generator._history.clear()
            _refresh_history()
            hist_detail.clear()

        hist_list.currentItemChanged.connect(lambda *_: _on_hist_select())
        btn_hist_apply.clicked.connect(_hist_apply)
        btn_hist_clear.clicked.connect(_hist_clear)

        _refresh_history()

        # ── Nettoyage des connexions à la fermeture du dialogue ───────────────
        def _cleanup():
            try:
                self.ai_generator.token_received.disconnect(_on_token)
                self.ai_generator.generation_done.disconnect(_on_done)
                self.ai_generator.generation_error.disconnect(_on_error)
                self.ai_generator.suggestion_ready.disconnect(_populate_suggestions)
                self.ai_generator.backend_changed.disconnect(_update_backend_label)
            except RuntimeError:
                pass

        dlg.finished.connect(_cleanup)
        dlg.exec()

    # ── v2.0 — Benchmark ──────────────────────────────────────────────────────

    def _show_benchmark_panel(self):
        """Ouvre le panneau Benchmark."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                      QPushButton, QTextEdit, QProgressBar)
        from PyQt6.QtCore import QTimer as _QTimer

        dlg = QDialog(self)
        dlg.setWindowTitle("Benchmark — Performance GPU/CPU")
        dlg.resize(440, 380)
        dlg.setStyleSheet("""
            QDialog,QWidget { background:#0e1018; color:#c0c4d0; font:10px 'Segoe UI'; }
            QTextEdit { background:#12141a; color:#40c060; font:10px 'Consolas';
                        border:1px solid #1e2030; }
        """)

        _btn_style = """
            QPushButton { background:#1e2030; color:#8090b0; border:1px solid #2a2d3a;
                          border-radius:3px; padding:4px 14px; font:9px 'Segoe UI'; }
            QPushButton:hover { background:#2a2d3a; color:#c0c8e0; }
            QPushButton:checked { background:#1a3020; color:#40c060; border-color:#306040; }
        """

        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(16, 16, 16, 12)
        vl.setSpacing(10)

        lbl_title = QLabel("📊 Métriques de rendu temps réel")
        lbl_title.setStyleSheet("color:#8090b0; font:bold 11px 'Segoe UI';")
        vl.addWidget(lbl_title)

        output = QTextEdit()
        output.setReadOnly(True)
        output.setMinimumHeight(200)
        vl.addWidget(output)

        fps_bar = QProgressBar()
        fps_bar.setRange(0, 120)
        fps_bar.setValue(0)
        fps_bar.setFormat("FPS moyen : %v")
        fps_bar.setStyleSheet("""
            QProgressBar { border:1px solid #1e2030; border-radius:3px;
                           background:#12141a; text-align:center; color:#c0c4d0; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1a4030, stop:0.5 #2a8050, stop:1 #40c080); border-radius:2px; }
        """)
        vl.addWidget(fps_bar)

        hl_btns = QHBoxLayout()
        btn_start = QPushButton("▶ Démarrer")
        btn_start.setCheckable(True)
        btn_start.setStyleSheet(_btn_style)
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setStyleSheet(_btn_style)
        btn_export = QPushButton("💾 Exporter rapport")
        btn_export.setStyleSheet(_btn_style)
        btn_close = QPushButton("Fermer")
        btn_close.setStyleSheet(_btn_style)

        hl_btns.addWidget(btn_start)
        hl_btns.addWidget(btn_reset)
        hl_btns.addWidget(btn_export)
        hl_btns.addStretch()
        hl_btns.addWidget(btn_close)
        vl.addLayout(hl_btns)

        refresh_timer = _QTimer(dlg)
        refresh_timer.setInterval(500)

        def _refresh():
            stats = self.gl_widget.get_benchmark_stats()
            if not stats:
                output.setPlainText("Pas encore de données…\nLancez le benchmark et attendez quelques secondes.")
                return
            txt = (
                f"FPS moyen       : {stats.get('fps_mean', 0):.1f}\n"
                f"FPS min         : {stats.get('fps_min', 0):.1f}\n"
                f"FPS max         : {stats.get('fps_max', 0):.1f}\n"
                f"───────────────────────────────\n"
                f"Frame time moy. : {stats.get('frame_time_mean_ms', 0):.2f} ms\n"
                f"Frame time p95  : {stats.get('frame_time_p95_ms', 0):.2f} ms\n"
                f"───────────────────────────────\n"
                f"Frames total    : {stats.get('total_frames', 0)}\n"
                f"Durée mesure    : {stats.get('elapsed_s', 0):.1f} s\n"
            )
            output.setPlainText(txt)
            fps_bar.setValue(int(stats.get('fps_mean', 0)))

        refresh_timer.timeout.connect(_refresh)

        def _toggle(checked):
            self._benchmark_active = checked
            self.gl_widget.enable_benchmark(checked)
            btn_start.setText("⏹ Arrêter" if checked else "▶ Démarrer")
            if checked:
                refresh_timer.start()
            else:
                refresh_timer.stop()
                _refresh()

        def _reset():
            self.gl_widget.reset_benchmark()
            output.clear()
            fps_bar.setValue(0)

        def _export():
            stats = self.gl_widget.get_benchmark_stats()
            if not stats:
                return
            path, _ = __import__('PyQt6.QtWidgets', fromlist=['QFileDialog']).QFileDialog.getSaveFileName(
                dlg, "Exporter rapport benchmark", "benchmark_report.txt", "Text (*.txt)"
            )
            if path:
                import datetime as _dt
                lines = [
                    f"OpenShader v2.0 — Rapport Benchmark",
                    f"Date : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Résolution : {self.gl_widget.shader_engine.width}×{self.gl_widget.shader_engine.height}",
                    "",
                ] + [f"{k}: {v}" for k, v in stats.items()]
                with open(path, 'w') as f:
                    f.write('\n'.join(lines))

        btn_start.toggled.connect(_toggle)
        btn_reset.clicked.connect(_reset)
        btn_export.clicked.connect(_export)
        btn_close.clicked.connect(dlg.accept)

        dlg.finished.connect(lambda: refresh_timer.stop())
        # Si benchmark déjà actif, montrer les données immédiatemment
        if self._benchmark_active:
            btn_start.setChecked(True)
            refresh_timer.start()

        dlg.exec()

    def _update_benchmark_status(self):
        """Mise à jour de la barre de statut avec les FPS benchmark."""
        if not self._benchmark_active:
            return
        stats = self.gl_widget.get_benchmark_stats()
        if stats:
            self._status.showMessage(
                f"Benchmark — FPS: {stats['fps_mean']} | "
                f"Frame: {stats['frame_time_mean_ms']}ms | "
                f"p95: {stats['frame_time_p95_ms']}ms",
                800
            )

    def _load_settings(self):
        settings = QSettings("OpenShader", "OpenShader")
        geom = settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

        # v2.4 — Thème UI
        saved_theme = settings.value("ui_theme", "auto")
        self._change_ui_theme(saved_theme)

        # v5.0 — Mode dossier versionné
        self._folder_mode = settings.value("folder_mode", False, type=bool)
        if hasattr(self, '_folder_mode_action'):
            self._folder_mode_action.setChecked(self._folder_mode)

    def _save_settings(self):
        settings = QSettings("OpenShader", "OpenShader")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("folder_mode", self._folder_mode)  # v5.0

    # ── Drag & Drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        SUPPORTED_SHADERS = ('.st', '.glsl')
        SUPPORTED_AUDIO = ('.wav', '.mp3', '.ogg')
        SUPPORTED_TEXTURES = ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tif')

        urls = event.mimeData().urls()
        
        # Traite l'audio en premier
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in SUPPORTED_AUDIO:
                    self._load_audio_file(path)
                    break # Un seul fichier audio à la fois

        # Traite les autres fichiers
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()

                if ext in SUPPORTED_SHADERS:
                    self._load_shader_file(path)
                elif ext in SUPPORTED_TEXTURES:
                    try:
                        # Charge dans le premier slot de texture libre
                        channel = self._texture_paths.index(None)
                        self._load_texture(channel, path)
                    except ValueError:
                        self._status.showMessage("Tous les slots de texture sont pleins.", 4000)
                        break

    # ── Nettoyage ─────────────────────────────────────────────────────────────


    # ═══════════════════════════════════════════════════════════════
    # v2.1 ─ Hot-Reload
    # ═══════════════════════════════════════════════════════════════

    def _toggle_hot_reload(self, checked: bool):
        """Active/désactive le hot-reload watchdog (Ctrl+Shift+R)."""
        if not self.hot_reload.available:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Hot-Reload",
                "watchdog n\'est pas installé.\n"
                "Installez avec :  pip install watchdog>=3.0"
            )
            if self._hot_reload_action:
                self._hot_reload_action.setChecked(False)
            return
        self.hot_reload.set_enabled(checked)
        self._hot_reload_enabled = checked
        self._status.showMessage(
            "🔥 Hot-Reload " + ("activé" if checked else "désactivé"), 3000)

    def _on_hot_reload_file_changed(self, path: str):
        """Slot Qt (thread-safe) — recharge le shader modifié."""
        import os as _os
        name = _os.path.basename(path)
        log.info("Hot-reload : %s", name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            log.warning("Hot-reload lecture : %s", e); return

        if self._active_image_shader_path == path:
            self._compile_source(source, "Image", path)
            if "Image" in self.editors: self.editors["Image"].set_code(source)
        elif self._active_trans_path == path:
            ok, err = self.shader_engine.load_trans_source(source, source_path=path)
            self._update_tab_status_trans(ok, err if not ok else None)
            if "Trans" in self.editors: self.editors["Trans"].set_code(source)
        else:
            for i, s in enumerate(self.shader_engine._layer_sources):
                if s == path: self.shader_engine._layer_sources[i] = ""
        self._status.showMessage(f"🔥 {name} rechargé", 2500)

    # v2.1 ─ GPU Profiler

    def _show_gpu_profiler_panel(self):
        """Panneau profiler GPU par passe."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QLabel, QTextEdit, QPushButton, QCheckBox)
        from PyQt6.QtGui import QFont
        from PyQt6.QtCore import QTimer
        dlg = QDialog(self); dlg.setWindowTitle("⏱ Profiler GPU — v2.1")
        dlg.setMinimumSize(520, 440)
        lay = QVBoxLayout(dlg)
        mode = "GPU (GL_TIME_ELAPSED)" if self.gl_widget.gpu_profiler.gpu_mode else "CPU fallback"
        lay.addWidget(QLabel(f"Mesure du temps OpenGL par passe. Mode : <b>{mode}</b>"))
        chk = QCheckBox("Activer le profiler")
        chk.setChecked(self.gl_widget.gpu_profiler.enabled)
        lay.addWidget(chk)
        out = QTextEdit(); out.setReadOnly(True)
        out.setFont(QFont("Courier New", 9))
        out.setStyleSheet("background:#111318; color:#c0c4d0;")
        lay.addWidget(out)
        def refresh(): out.setPlainText(self.gl_widget.gpu_profiler.format_overlay())
        brow = QHBoxLayout()
        br = QPushButton("🔄 Rafraîchir"); brs = QPushButton("🗑 Reset"); bc = QPushButton("Fermer")
        brow.addWidget(br); brow.addWidget(brs); brow.addStretch(); brow.addWidget(bc)
        lay.addLayout(brow)
        chk.toggled.connect(lambda v: (self.gl_widget.gpu_profiler.set_enabled(v),
                                       setattr(self, "_gpu_profiler_enabled", v)))
        br.clicked.connect(refresh)
        brs.clicked.connect(lambda: (self.gl_widget.gpu_profiler.reset(), refresh()))
        bc.clicked.connect(dlg.close)
        t = QTimer(dlg); t.setInterval(500); t.timeout.connect(refresh); t.start()
        refresh(); dlg.exec()

    # v2.1 ─ Audio Analyzer

    def _on_amplitude_for_analysis(self, _amplitude: float):
        """Injecte MFCC / onset dans les uniforms GLSL."""
        if not self.audio_analyzer.is_ready(): return
        for name, val in self.audio_analyzer.get_features_at(self._current_time).as_uniforms().items():
            self.shader_engine.set_uniform(name, val)

    def _show_audio_analysis_panel(self):
        """Panneau analyse audio avancée."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QLabel, QTextEdit, QPushButton, QProgressBar)
        from PyQt6.QtGui import QFont
        dlg = QDialog(self); dlg.setWindowTitle("🎵 Analyse Audio — v2.1")
        dlg.setMinimumSize(480, 400); lay = QVBoxLayout(dlg)
        scipy_ok = False
        try: import scipy; scipy_ok = True  # noqa
        except ImportError: pass
        if not scipy_ok:
            lay.addWidget(QLabel(
                "<b>scipy non installé</b><br><br>"
                "pip install scipy>=1.10<br><br>"
                "Active : uMFCC0…12, uAudioOnset, uAudioCentroid, uAudioZCR"))
            lay.addWidget(QPushButton("Fermer", clicked=dlg.close))
            dlg.exec(); return
        lay.addWidget(QLabel(
            "Uniforms actifs après analyse :<br>"
            "<code>uAudioRMS, uAudioZCR, uAudioCentroid, "
            "uAudioOnset, uAudioOnsetStrength, uMFCC0…uMFCC12</code>"))
        slbl = QLabel("Statut : en attente"); lay.addWidget(slbl)
        bar = QProgressBar(); bar.setRange(0,0); bar.hide(); lay.addWidget(bar)
        out = QTextEdit(); out.setReadOnly(True)
        out.setFont(QFont("Courier New", 9))
        out.setStyleSheet("background:#111318; color:#c0c4d0;")
        lay.addWidget(out)
        def done(r):
            bar.hide()
            if r:
                s = self.audio_analyzer.get_summary()
                out.setPlainText("\n".join([
                    f"Durée       : {s.get('duration',0):.2f} s",
                    f"Sample rate : {s.get('sample_rate',0)} Hz",
                    f"Frames      : {s.get('n_frames',0)}",
                    f"Onsets      : {s.get('n_onsets',0)}",
                    f"RMS moyen   : {s.get('avg_rms',0):.4f}",
                ])); slbl.setText("✅ Terminé")
            else: slbl.setText("❌ Erreur")
        def run():
            fp = self.audio_engine.file_path
            if not fp: slbl.setText("⚠️  Pas de fichier audio"); return
            slbl.setText("⏳ Analyse..."); bar.show()
            self.audio_analyzer.analyze_file(fp, callback=done)
        brow = QHBoxLayout()
        brun = QPushButton("🔍 Analyser"); bc = QPushButton("Fermer")
        brow.addWidget(brun); brow.addStretch(); brow.addWidget(bc)
        lay.addLayout(brow)
        brun.clicked.connect(run); bc.clicked.connect(dlg.close)
        if self.audio_analyzer.is_ready(): done(self.audio_analyzer._result)
        dlg.exec()

    # ── Sync Audio Automatique (v2.3) ────────────────────────────────────────

    def _show_audio_sync_panel(self):
        """
        Ouvre le panneau de synchronisation audio automatique.
        Connecte les signaux d'injection/palette au moment de l'ouverture.
        """
        if not hasattr(self, "_audio_sync_panel") or self._audio_sync_panel is None:
            self._audio_sync_panel = AudioSyncPanel(self.audio_sync, parent=self)
            self._audio_sync_panel.inject_requested.connect(self._on_audio_sync_inject)
            self._audio_sync_panel.apply_palette_requested.connect(
                self._on_audio_sync_apply_palette
            )

        # Met à jour le curseur de temps si le panneau est ouvert
        self._audio_sync_panel.set_current_time(self._current_time)
        self._audio_sync_panel.show()
        self._audio_sync_panel.raise_()

    def _on_audio_sync_inject(self, event_types: list, interp: str):
        """
        Injecte les keyframes du plan audio dans la timeline.
        Déclenche un rafraîchissement de l'affichage.
        """
        if not self.audio_sync.plan:
            return
        counts = self.audio_sync.inject_into_timeline(
            self.timeline, event_types=event_types, interp=interp
        )
        total = sum(counts.values())
        log.info("AudioSync — %d keyframes injectées : %s", total, counts)

        # Rafraîchit la timeline
        if hasattr(self, "timeline_widget"):
            self.timeline_widget.canvas.update()
            self.timeline_widget.set_duration(self.timeline.duration)

        # Notification
        from PyQt6.QtWidgets import QMessageBox
        n_types = sum(1 for c in counts.values() if c > 0)
        QMessageBox.information(
            self,
            "Sync Audio — Keyframes injectées",
            f"✅ {total} keyframes injectées sur {n_types} piste(s).\n\n"
            + "\n".join(f"  • {t}: {n}" for t, n in counts.items() if n > 0),
        )

    def _on_audio_sync_apply_palette(self, palette):
        """
        Injecte la palette cosinus dans le shader courant.
        Ajoute le snippet palette() avant void mainImage.
        """
        from .audio_sync import PalettePreset
        if not isinstance(palette, PalettePreset):
            return

        pass_name = self._current_pass_name()
        editor    = self.editors.get(pass_name)
        if editor is None:
            return

        current_code = editor.get_code() if hasattr(editor, "get_code") else ""
        glsl_snippet = palette.to_glsl_code()

        # Si une palette existe déjà dans le shader, la remplace
        import re
        if "vec3 palette(" in current_code:
            # Remplace le bloc palette() existant
            new_code = re.sub(
                r'// Palette[^\n]*\n(?://[^\n]*\n)*vec3 palette\([^}]*\}\n?',
                glsl_snippet + "\n",
                current_code,
                count=1,
            )
        else:
            # Insère avant void mainImage
            insert_pat = re.search(r'void\s+mainImage', current_code)
            if insert_pat:
                pos = insert_pat.start()
                new_code = current_code[:pos] + glsl_snippet + "\n\n" + current_code[pos:]
            else:
                new_code = glsl_snippet + "\n\n" + current_code

        if hasattr(editor, "set_code"):
            editor.set_code(new_code)

        # Injecte aussi les uniforms palette
        for name, val in palette.to_glsl_uniforms().items():
            if isinstance(val, list):
                try:
                    self.shader_engine.set_uniform(name, val)
                except Exception:
                    pass
            else:
                try:
                    self.shader_engine.set_uniform(name, float(val))
                except Exception:
                    pass

        log.info("AudioSync — Palette '%s' injectée dans %s", palette.name, pass_name)

    # ── Auto-paramétrage IA (v2.3) ────────────────────────────────────────────

    def _on_params_scan_requested(self):
        """
        Récupère le code du shader actif et le transmet au panneau Params pour analyse.
        """
        pass_name = self._current_pass_name()
        editor = self.editors.get(pass_name)
        if editor is None:
            self.left_panel.scan_shader("")
            return
        glsl = editor.get_code() if hasattr(editor, "get_code") else ""
        self.left_panel.scan_shader(glsl)

    def _on_apply_param_to_shader(self, param):
        """
        Injecte un magic number comme uniform dans le shader courant.
        Utilise AIParamDetector.apply_param_to_shader() pour la transformation,
        puis recharge le shader et enregistre dans l'historique undo.
        """
        from .ai_param_detector import AIParamDetector
        pass_name = self._current_pass_name()
        editor = self.editors.get(pass_name)
        if editor is None:
            return

        original_code = editor.get_code() if hasattr(editor, "get_code") else ""
        if not original_code.strip():
            return

        detector = AIParamDetector()
        new_code = detector.apply_param_to_shader(original_code, param)

        if new_code == original_code:
            return  # Rien à faire

        # Appliquer dans l'éditeur
        if hasattr(editor, "set_code"):
            editor.set_code(new_code)

        # Recompiler
        try:
            self.shader_engine.compile(new_code)
            # Initialiser la valeur du nouvel uniform
            self.shader_engine.set_uniform(param.name, float(param.default))
        except Exception as e:
            log.warning("Recompilation après injection de param : %s", e)

        log.info("Param injecté : %s ← %s (défaut: %s)", param.name, param.original, param.default)


    # ── v4.0 — DMX / Artnet ──────────────────────────────────────────────────

    def _show_dmx_panel(self):
        """Ouvre le panneau DMX/Artnet (plan de salle, mappings, config réseau)."""
        if not hasattr(self, "_dmx_panel") or self._dmx_panel is None:
            self._dmx_panel = DmxPanel(self.dmx_engine, parent=self)
        self._dmx_panel.show()
        self._dmx_panel.raise_()

    def closeEvent(self, event):
        self._save_settings()
        self._main_timer.stop()
        self.hot_reload.stop()  # v2.1
        self.gl_widget.stop_render()
        self.audio_engine.cleanup()
        self.shader_engine.cleanup()
        if hasattr(self, "_synth_editor"):
            self._synth_editor.cleanup()  # v2.8
        if getattr(self, "_vr_window", None):
            self._vr_window.close()        # v2.9
        if self.dmx_engine.is_running:
            self.dmx_engine.stop()         # v4.0
        event.accept()


# ── Style global de l'application ────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  Thèmes UI  (dark / light / auto)
# ══════════════════════════════════════════════════════════════════════════════

_THEME_DARK = {
    "bg0":       "#0e1016",   # fond le plus sombre (barre de statut, header)
    "bg1":       "#12141a",   # fond intermédiaire
    "bg2":       "#1a1c24",   # fond principal
    "bg3":       "#1e2030",   # fond boutons / zones
    "border":    "#2a2d3a",
    "accent":    "#3a5888",
    "accent_hover": "#4a6898",
    "text":      "#c0c4d0",
    "text_dim":  "#6a7090",
    "scrollbar": "#2a2d3a",
    "scrollbar_bg": "#12141a",
    "selection": "#2a4060",
}

_THEME_LIGHT = {
    # Palette fidèle à Windows 10 / Fluent Design Light
    "bg0":       "#C8C8C8",   # barre de titre, header (gris Windows titlebars)
    "bg1":       "#F0F0F0",   # fond panneaux secondaires (Control color Windows 10)
    "bg2":       "#FFFFFF",   # fond principal (Window color)
    "bg3":       "#E1E1E1",   # fond boutons au repos (ButtonFace)
    "border":    "#ADADAD",   # bordures contrôles (Windows 10 border gray)
    "accent":    "#0078D4",   # accent Windows 10 bleu (Fluent Blue)
    "accent_hover": "#106EBE",# accent hover (bleu légèrement plus sombre)
    "text":      "#000000",   # texte principal
    "text_dim":  "#6E6E6E",   # texte secondaire / désactivé (GrayText Windows)
    "scrollbar": "#CDCDCD",   # poignée scrollbar Windows 10
    "scrollbar_bg": "#F0F0F0",# piste scrollbar
    "selection": "#CCE4F7",   # sélection texte (Highlight clair)
}


def _build_app_style(t: dict) -> str:
    """Génère le stylesheet global à partir du dictionnaire de thème *t*."""
    return f"""
/* ── Base ── */
QMainWindow, QWidget {{
    background: {t['bg2']};
    color: {t['text']};
    font-family: 'Segoe UI', sans-serif;
    font-size: 10px;
}}

/* ── ScrollBars (style Windows 10 fin) ── */
QScrollBar:vertical {{
    background: {t['scrollbar_bg']}; width: 10px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {t['scrollbar']}; border-radius: 5px; min-height: 24px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover {{ background: {t['border']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{
    background: {t['scrollbar_bg']}; height: 10px; border: none;
}}
QScrollBar::handle:horizontal {{
    background: {t['scrollbar']}; border-radius: 5px; min-width: 24px;
    margin: 1px;
}}
QScrollBar::handle:horizontal:hover {{ background: {t['border']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}

/* ── ToolTip ── */
QToolTip {{
    background: {t['bg1']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    padding: 4px 8px; font: 9px 'Segoe UI';
}}

/* ── MenuBar ── */
QMenuBar {{
    background: {t['bg1']}; color: {t['text']};
    border-bottom: 1px solid {t['border']};
    padding: 1px 0px;
}}
QMenuBar::item {{ padding: 3px 8px; border-radius: 2px; }}
QMenuBar::item:selected {{ background: {t['bg3']}; color: {t['text']}; }}
QMenuBar::item:pressed  {{ background: {t['accent']}; color: #ffffff; }}

/* ── Menu déroulant ── */
QMenu {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']};
    padding: 2px 0px;
}}
QMenu::item {{ padding: 4px 24px 4px 24px; }}
QMenu::item:selected {{ background: {t['accent']}; color: #ffffff; border-radius: 2px; }}
QMenu::item:disabled  {{ color: {t['text_dim']}; }}
QMenu::separator {{
    height: 1px; background: {t['border']}; margin: 2px 6px;
}}

/* ── ComboBox ── */
QComboBox {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    padding: 2px 6px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {t['accent']}; }}
QComboBox:focus {{ border-color: {t['accent']}; }}
QComboBox::drop-down {{
    border: none; width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']};
    selection-background-color: {t['accent']};
    selection-color: #ffffff;
    outline: none;
}}

/* ── LineEdit / SpinBox ── */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    padding: 2px 5px; selection-background-color: {t['accent']};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['accent']};
}}
QLineEdit:disabled, QSpinBox:disabled {{ color: {t['text_dim']}; background: {t['bg3']}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {t['bg3']}; border: none; width: 14px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t['border']};
}}

/* ── Buttons ── */
QPushButton {{
    background: {t['bg3']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    padding: 3px 10px; min-height: 18px;
}}
QPushButton:hover  {{ background: {t['accent_hover']}; color: #ffffff; border-color: {t['accent_hover']}; }}
QPushButton:pressed{{ background: {t['accent']}; color: #ffffff; }}
QPushButton:disabled {{ color: {t['text_dim']}; background: {t['bg1']}; border-color: {t['border']}; }}
QPushButton:default {{
    border: 2px solid {t['accent']};
}}

/* ── CheckBox / RadioButton ── */
QCheckBox, QRadioButton {{
    color: {t['text']}; spacing: 5px;
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {t['text_dim']}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {t['border']}; background: {t['bg2']};
    border-radius: 2px;
}}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {t['accent']}; border-color: {t['accent']};
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {t['accent']};
}}

/* ── ToolBar ── */
QToolBar {{
    background: {t['bg1']}; border-bottom: 1px solid {t['border']};
    spacing: 2px; padding: 1px 4px;
}}
QToolBar::separator {{
    background: {t['border']}; width: 1px; margin: 3px 2px;
}}
QToolButton {{
    background: transparent; color: {t['text']};
    border: 1px solid transparent; border-radius: 2px; padding: 3px 5px;
}}
QToolButton:hover  {{ background: {t['bg3']}; border-color: {t['border']}; }}
QToolButton:pressed{{ background: {t['accent']}; color: #ffffff; }}
QToolButton:checked{{ background: {t['selection']}; border-color: {t['accent']}; }}

/* ── DockWidget ── */
QDockWidget {{
    color: {t['text']}; font-weight: bold; font-size: 10px;
    titlebar-close-icon: none; titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {t['bg0']}; color: {t['text']};
    padding: 3px 6px; border-bottom: 1px solid {t['border']};
    text-align: left;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent; border: none; padding: 1px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {t['bg3']};
}}

/* ── GroupBox ── */
QGroupBox {{
    color: {t['text']}; border: 1px solid {t['border']};
    border-radius: 3px; margin-top: 8px; padding-top: 6px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 4px; left: 8px;
}}

/* ── Splitter ── */
QSplitter::handle {{
    background: {t['border']};
}}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical   {{ height: 4px; }}
QSplitter::handle:hover {{ background: {t['accent']}; }}

/* ── ListWidget / TreeWidget / TableWidget ── */
QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView, QTableView {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    alternate-background-color: {t['bg1']};
    gridline-color: {t['border']};
    outline: none;
}}
QListWidget::item:selected, QTreeWidget::item:selected,
QTableWidget::item:selected, QListView::item:selected,
QTreeView::item:selected, QTableView::item:selected {{
    background: {t['accent']}; color: #ffffff;
}}
QListWidget::item:hover, QTreeWidget::item:hover,
QListView::item:hover, QTreeView::item:hover {{
    background: {t['selection']};
}}
QHeaderView::section {{
    background: {t['bg1']}; color: {t['text']};
    border: none; border-right: 1px solid {t['border']};
    border-bottom: 1px solid {t['border']};
    padding: 3px 6px; font-weight: bold;
}}
QHeaderView::section:hover {{ background: {t['bg3']}; }}

/* ── Slider ── */
QSlider::groove:horizontal {{
    background: {t['bg3']}; height: 4px; border-radius: 2px;
    border: 1px solid {t['border']};
}}
QSlider::handle:horizontal {{
    background: {t['accent']}; width: 12px; height: 12px;
    border-radius: 6px; margin: -4px 0; border: none;
}}
QSlider::handle:horizontal:hover {{ background: {t['accent_hover']}; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::groove:vertical {{
    background: {t['bg3']}; width: 4px; border-radius: 2px;
    border: 1px solid {t['border']};
}}
QSlider::handle:vertical {{
    background: {t['accent']}; width: 12px; height: 12px;
    border-radius: 6px; margin: 0 -4px; border: none;
}}

/* ── ProgressBar ── */
QProgressBar {{
    background: {t['bg3']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    text-align: center; height: 14px;
}}
QProgressBar::chunk {{
    background: {t['accent']}; border-radius: 2px;
}}

/* ── StatusBar ── */
QStatusBar {{
    background: {t['bg0']}; color: {t['text']};
    border-top: 1px solid {t['border']};
}}
QStatusBar::item {{ border: none; }}

/* ── Label ── */
QLabel {{
    background: transparent; color: {t['text']};
}}
QLabel:disabled {{ color: {t['text_dim']}; }}

/* ── TextEdit / PlainTextEdit ── */
QTextEdit, QPlainTextEdit {{
    background: {t['bg2']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    selection-background-color: {t['selection']};
}}
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
"""


def _build_editor_btn_style(t: dict) -> str:
    return f"""
QPushButton {{
    background: {t['bg3']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: 2px;
    padding: 2px 8px; font: 9px 'Segoe UI';
}}
QPushButton:hover  {{ background: {t['accent_hover']}; color: #ffffff; border-color: {t['accent_hover']}; }}
QPushButton:pressed{{ background: {t['accent']}; color: #ffffff; }}
QPushButton:disabled {{ color: {t['text_dim']}; border-color: {t['bg3']}; }}
"""


def _build_tab_style(t: dict) -> str:
    return f"""
QTabWidget::pane {{
    background: {t['bg2']}; border: none;
}}
QTabBar::tab {{
    background: {t['bg1']}; color: {t['text_dim']};
    padding: 5px 12px; border: none;
    font: 9px 'Segoe UI'; min-width: 60px;
}}
QTabBar::tab:selected {{
    background: {t['bg2']}; color: {t['text']};
    border-top: 2px solid {t['accent']};
}}
QTabBar::tab:hover:!selected {{ background: {t['bg3']}; }}
"""


def _detect_system_dark_mode() -> bool:
    """Détecte si le thème système est sombre (Windows / macOS / Linux KDE/GNOME)."""
    from PyQt6.QtGui import QPalette, QColor
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        palette = app.palette()
        window_color = palette.color(QPalette.ColorRole.Window)
        # Luminosité < 128 → sombre
        return window_color.lightness() < 128
    return True  # défaut : dark


_CURRENT_THEME_NAME: str = "auto"   # "dark" | "light" | "auto"
_CURRENT_THEME: dict = _THEME_DARK  # résolu au démarrage


def resolve_theme(name: str) -> dict:
    """Renvoie le dict de couleurs pour 'dark', 'light' ou 'auto'."""
    if name == "light":
        return _THEME_LIGHT
    if name == "dark":
        return _THEME_DARK
    # auto
    return _THEME_DARK if _detect_system_dark_mode() else _THEME_LIGHT


# ══════════════════════════════════════════════════════════════════════════════

_TEMPLATE_BUFFER = """// Buffer — passe auxiliaire (ping-pong)
// iChannel0 = résultat du frame précédent de cette passe
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 prev = texture(iChannel0, uv);

    // Exemple : effet d'accumulation avec décroissance
    vec3 col = prev.rgb * 0.98;
    fragColor = vec4(col, 1.0);
}
"""

_TEMPLATE_IMAGE = """// Image — passe principale
// iChannel0–3 = Buffer A–D ou textures
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = texture(iChannel0, uv).rgb;
    fragColor = vec4(col, 1.0);
}
"""

_TEMPLATE_POST = """// Post — post-processing
// iChannel0 = sortie de la passe Image
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 col = texture(iChannel0, uv);
    fragColor = col;
}
"""

_TEMPLATE_TRANS = """// Transition — fondu entre deux scènes
// iChannel0 = scène A (sortante)
// iChannel1 = scène B (entrante)
// iProgress  = avancement [0.0 → 1.0]

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 a  = texture(iChannel0, uv);
    vec4 b  = texture(iChannel1, uv);

    // Exemple : fondu enchaîné avec courbe douce
    float t = smoothstep(0.0, 1.0, iProgress);
    fragColor = mix(a, b, t);
}
"""

# ── v1.5 — Templates supplémentaires ──────────────────────────────────────────

_TEMPLATE_AUDIOBARS = """// AudioVisualizer — Barres de fréquences (iChannel0 = texture audio FFT)
// Compatible Shadertoy : connecter iChannel0 = Mic/Music (FFT)
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    int  bars = 64;
    float barW = 1.0 / float(bars);
    float idx  = floor(uv.x / barW) / float(bars);
    float amp  = texture(iChannel0, vec2(idx, 0.0)).r;
    float glow = smoothstep(amp - 0.05, amp, uv.y) * (1.0 - smoothstep(amp, amp + 0.05, uv.y));
    vec3  col  = mix(vec3(0.1, 0.8, 1.0), vec3(1.0, 0.2, 0.5), uv.x);
    col *= step(uv.y, amp) * 1.5;
    col += glow * vec3(1.0, 1.0, 1.0) * 0.4;
    fragColor = vec4(col, 1.0);
}
"""

_TEMPLATE_INTRO_TUNNEL = """// Demo Intro — Tunnel infini
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2  uv  = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a   = atan(uv.y, uv.x);
    float r   = length(uv);
    float t   = iTime * 0.4;
    float u   = a / 3.14159 + t;
    float v   = 1.0 / r + t;
    vec3  col = 0.5 + 0.5 * cos(vec3(u, v, u + v) * 6.28 + vec3(0.0, 2.0, 4.0));
    col *= 0.8 + 0.2 * sin(r * 30.0 - t * 8.0);
    fragColor = vec4(col, 1.0);
}
"""

_TEMPLATE_LOOP_VORONOI = """// Loopable BG — Voronoi animé (loop toutes les 4 secondes)
// Modifiez LOOP_DUR pour ajuster la durée de la boucle.
#define LOOP_DUR 4.0
vec2 hash2(vec2 p){ p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*43758.5); }
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float lt = mod(iTime, LOOP_DUR) / LOOP_DUR;
    vec2 uv = fragCoord / iResolution.y * 3.0;
    float md = 1e9; int mx=-1, my=-1;
    for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
        vec2 g=floor(uv)+vec2(i,j);
        vec2 o=hash2(g);
        float phase=hash2(g+vec2(7.3,3.1)).x;
        vec2 c=g + o + 0.45*sin(2.0*3.14159*(lt+phase)*vec2(1.0,1.3));
        float d=length(uv-c);
        if(d<md){md=d;mx=int(g.x)&0xf;my=int(g.y)&0xf;}
    }
    vec3 col=0.5+0.5*cos(vec3(mx,my,mx+my)*0.9+vec3(0,2,4));
    col *= 0.6 + 0.4*md;
    fragColor = vec4(col, 1.0);
}
"""

# ── v1.5 — Conversion GLSL pur → Shadertoy ───────────────────────────────────

def _glsl_to_shadertoy(source: str) -> str:
    """Enveloppe un shader GLSL pur (uTime/uResolution) dans un wrapper Shadertoy."""
    # Remplace les uniforms OpenShader par les uniforms Shadertoy
    converted = source
    converted = converted.replace("uResolution", "iResolution")
    converted = converted.replace("uTime",       "iTime")

    # Si le shader contient déjà une fonction main(), l'emballer
    if "void main()" in converted:
        header = (
            "// Converti depuis GLSL pur par OpenShader v1.5\n"
            "// uResolution → iResolution, uTime → iTime\n\n"
        )
        wrapper = (
            "\n\nvoid mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n"
            "    // Appel du shader original\n"
            "    main();\n"
            "}\n"
        )
        # Remplace 'out vec4 fragColor' par une variable locale si besoin
        converted = converted.replace("out vec4 fragColor", "vec4 fragColor")
        return header + converted + wrapper
    return converted



# ── v2.7 — Préférences (backend OpenGL ↔ Vulkan) ─────────────────────────────

class PreferencesDialog:
    """
    Boîte de dialogue des préférences générales.
    Permet de basculer le backend de rendu entre OpenGL (ModernGL) et Vulkan.
    """
    # NOTE: Defined via __init__ below for Qt imports
    pass


def _make_preferences_dialog(parent=None):
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox,
                                 QRadioButton, QLabel,
                                 QDialogButtonBox)

    class _PrefDlg(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Preferences — OpenShader")
            self.setMinimumWidth(480)
            self.setModal(True)
            layout = QVBoxLayout(self)
            layout.setSpacing(12)

            grp = QGroupBox("Render Backend")
            grp_layout = QVBoxLayout(grp)

            self._rb_opengl = QRadioButton("OpenGL (ModernGL — stable, all GPUs)")
            self._rb_vulkan = QRadioButton("Vulkan (experimental — compute + ray tracing RTX/RDNA)")

            current = load_backend_pref()
            self._rb_opengl.setChecked(current == BACKEND_OPENGL)
            self._rb_vulkan.setChecked(current == BACKEND_VULKAN)
            self._rb_vulkan.setEnabled(vulkan_available())
            if not vulkan_available():
                self._rb_vulkan.setText(
                    "Vulkan  WARNING: not available — install pyvulkan + Vulkan SDK"
                )

            grp_layout.addWidget(self._rb_opengl)
            grp_layout.addWidget(self._rb_vulkan)

            self._info = QLabel()
            self._update_info()
            self._rb_opengl.toggled.connect(self._update_info)
            grp_layout.addWidget(self._info)
            layout.addWidget(grp)

            grp2 = QGroupBox("GPU Capabilities")
            grp2_layout = QVBoxLayout(grp2)
            rt = has_ray_tracing()
            rt_txt = ("Ray tracing (RTX / RDNA): available" if rt
                      else "Ray tracing: not available (incompatible GPU or Vulkan absent)")
            compute_txt = ("Compute shaders (GPU physics, particles): available" if vulkan_available()
                           else "Compute shaders: Vulkan required")
            grp2_layout.addWidget(QLabel(rt_txt))
            grp2_layout.addWidget(QLabel(compute_txt))
            layout.addWidget(grp2)

            note = QLabel("NOTE: Backend change takes effect on next launch.")
            note.setStyleSheet("color: orange; font-style: italic;")
            note.setWordWrap(True)
            layout.addWidget(note)

            bbox = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            bbox.accepted.connect(self._apply)
            bbox.rejected.connect(self.reject)
            layout.addWidget(bbox)

        def _update_info(self):
            if self._rb_vulkan.isChecked():
                txt = "Vulkan: compiles GLSL->SPIR-V via glslangValidator, exposes ray tracing extensions."
            else:
                txt = "OpenGL: stable ModernGL pipeline, GLSL 330 core, all GPUs."
            self._info.setText(txt)
            self._info.setWordWrap(True)
            self._info.setStyleSheet("color: gray; font-size: 11px;")

        def _apply(self):
            backend = BACKEND_VULKAN if self._rb_vulkan.isChecked() else BACKEND_OPENGL
            save_backend_pref(backend)
            log.info("Backend preference saved: %s", backend)
            self.accept()

    dlg = _PrefDlg(parent)
    dlg.exec()


# ── v2.0 — VJing Window ───────────────────────────────────────────────────────

class VJWindow(QMainWindow):
    """
    Fenêtre plein-écran dédiée au VJing.

    Stratégie : miroir par grabFramebuffer() — aucun reparenting, aucun second
    contexte OpenGL. Le gl_widget principal continue de rendre normalement ;
    VJWindow capture son framebuffer à ~60 Hz et l'affiche mis à l'échelle dans
    un QLabel.  À la fermeture : rien à restituer, le viewport principal est intact.

    Hotkeys : Escape (fermer), Tab (overlay), ←→ (shader précédent/suivant),
              1–9 (preset direct), F11 (toggle fullscreen).
    """

    def __init__(self, parent_window: 'MainWindow'):
        super().__init__()
        self.setWindowTitle("OpenShader — Mode VJ")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self._parent_window = parent_window

        # ── Affichage miroir ──────────────────────────────────────────────────
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("background: #000;")
        self._display.setScaledContents(False)   # on gère le scaling nous-mêmes

        central = QWidget()
        central.setStyleSheet("background: #000;")
        self.setCentralWidget(central)
        vl = QVBoxLayout(central)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self._display)

        # ── Overlay HUD ───────────────────────────────────────────────────────
        self._overlay = QLabel(central)
        self._overlay.setStyleSheet("""
            QLabel { color: rgba(255,255,255,210);
                     font: bold 11px 'Segoe UI';
                     background: rgba(0,0,0,140);
                     padding: 4px 12px; border-radius: 4px; }
        """)
        self._overlay.move(12, 12)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._show_overlay = True

        # ── Timer de capture + overlay ────────────────────────────────────────
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)   # ~60 Hz
        self._frame_timer.timeout.connect(self._capture_frame)
        self._frame_timer.start()

        # Présets VJ
        self._vj_shader_paths: list[str] = []
        self._vj_current_idx = 0

        self._setup_hotkeys()
        log.info("VJWindow ouverte (mode miroir grabFramebuffer).")

    # ── Capture framebuffer ───────────────────────────────────────────────────

    def _capture_frame(self):
        """Capture le framebuffer du gl_widget principal et l'affiche."""
        pw = self._parent_window
        if not pw or not pw.gl_widget.isVisible():
            return
        try:
            pw.gl_widget.makeCurrent()
            img = pw.gl_widget.grabFramebuffer()
            pw.gl_widget.doneCurrent()
        except (RuntimeError, OSError):
            return

        if img.isNull():
            return

        # Mise à l'échelle pour remplir la fenêtre en gardant le ratio
        target = self._display.size()
        if target.width() > 0 and target.height() > 0:
            scaled = img.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            from PyQt6.QtGui import QPixmap
            self._display.setPixmap(QPixmap.fromImage(scaled))

        # Rafraîchir overlay (moins souvent)
        if self._show_overlay:
            self._refresh_overlay()
            self._overlay.show()
            self._overlay.raise_()
        else:
            self._overlay.hide()

    def _refresh_overlay(self):
        pw = self._parent_window
        t  = pw._current_time if pw else 0.0
        m, s, ms = int(t) // 60, int(t) % 60, int((t % 1) * 1000)
        bpm  = getattr(pw.timeline, 'bpm', 120) if pw else 120
        beat = int(t / (60.0 / bpm)) + 1 if bpm > 0 else 0
        name = os.path.basename(pw._active_image_shader_path) \
               if pw and pw._active_image_shader_path else "—"
        dpr_str = f"  |  ×{self.devicePixelRatio():.3g}" if self.devicePixelRatio() != 1.0 else ""
        self._overlay.setText(
            f"⏱ {m:02d}:{s:02d}.{ms:03d}  |  ♩ {bpm:.0f} BPM  |  Beat {beat}  |  {name}{dpr_str}"
        )
        self._overlay.adjustSize()

    # ── Hotkeys ───────────────────────────────────────────────────────────────

    def _setup_hotkeys(self):
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.close)
        QShortcut(QKeySequence("Tab"),    self).activated.connect(self._toggle_overlay)
        QShortcut(QKeySequence("Right"),  self).activated.connect(self._next_shader)
        QShortcut(QKeySequence("Left"),   self).activated.connect(self._prev_shader)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                lambda checked=False, idx=i - 1: self._jump_to_shader(idx)
            )
        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fullscreen)

    def _toggle_overlay(self):
        self._show_overlay = not self._show_overlay

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ── Présets ───────────────────────────────────────────────────────────────

    def _set_shader_list(self, paths: list[str]):
        self._vj_shader_paths = paths

    def _jump_to_shader(self, idx: int):
        if 0 <= idx < len(self._vj_shader_paths) and self._parent_window:
            self._vj_current_idx = idx
            self._parent_window._load_shader_file(self._vj_shader_paths[idx])

    def _next_shader(self):
        self._jump_to_shader((self._vj_current_idx + 1) % max(1, len(self._vj_shader_paths)))

    def _prev_shader(self):
        self._jump_to_shader((self._vj_current_idx - 1) % max(1, len(self._vj_shader_paths)))

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._frame_timer.stop()
        if self._parent_window:
            self._parent_window._vj_window = None
        log.info("VJWindow fermée — viewport principal intact.")
        super().closeEvent(event)
