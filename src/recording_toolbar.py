"""
recording_toolbar.py
---------------------
Barre d'outils d'enregistrement live — à intégrer dans main_window.py
(typiquement sous la timeline ou dans la barre de transport).

Fournit :
  - Bouton ⏺ Record (bascule)
  - Bouton ✗ Annuler
  - Sélecteur de mode (Normal / Punch / Overdub)
  - Champs punch-in / punch-out (en secondes)
  - Indicateur visuel clignotant pendant l'enregistrement
  - Signal `export_requested` pour déclencher la sauvegarde .demomaker
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QFrame, QToolButton, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui  import QColor, QPalette

from .session_recorder import (
    SessionRecorder,
    RECORD_MODE_NORMAL, RECORD_MODE_PUNCH, RECORD_MODE_OVERDUB,
)


# ─────────────────────────────────────────────────────────────────────────────

class RecordingToolbar(QWidget):
    """
    Barre de contrôle d'enregistrement live.

    Connectez `time_provider` avant de démarrer :
        toolbar.recorder.set_time_provider(lambda: self._current_time)

    Connectez les sources d'uniforms :
        left_panel.uniform_value_changed.connect(
            lambda n, v: toolbar.recorder.record_event(n, v, 'ui'))
        midi_engine.uniform_changed.connect(
            lambda n, v: toolbar.recorder.record_event(n, v, 'midi'))
        osc_engine.uniform_changed.connect(
            lambda n, v: toolbar.recorder.record_event(n, v, 'osc'))

    Écoutez le signal d'export :
        toolbar.export_requested.connect(self._save_demomaker)
    """

    # Émis quand l'utilisateur clique sur "Exporter" ou que l'enregistrement
    # s'arrête et que l'export auto est activé.
    export_requested = pyqtSignal()

    def __init__(self, recorder: SessionRecorder, parent: QWidget | None = None):
        super().__init__(parent)
        self._recorder   = recorder
        self._blink_on   = False

        self._build_ui()
        self._connect_signals()

        # Timer de clignotement (500 ms)
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._blink)

    # ── Construction UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 3, 6, 3)
        root.setSpacing(8)

        # ── Indicateur ●
        self._indicator = QLabel("●")
        self._indicator.setFixedWidth(18)
        self._indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._indicator.setStyleSheet("color: #444; font-size: 16px;")
        root.addWidget(self._indicator)

        # ── Bouton Record
        self._btn_record = QPushButton("⏺  Enreg.")
        self._btn_record.setCheckable(True)
        self._btn_record.setFixedHeight(26)
        self._btn_record.setStyleSheet("""
            QPushButton { background: #2a1a1a; color: #cc4444;
                          border: 1px solid #cc4444; border-radius: 4px; padding: 0 8px; }
            QPushButton:checked { background: #cc2222; color: white;
                                  border: 1px solid #ff5555; }
            QPushButton:hover   { background: #3a1a1a; }
        """)
        root.addWidget(self._btn_record)

        # ── Bouton Annuler
        self._btn_cancel = QPushButton("✗")
        self._btn_cancel.setFixedWidth(28)
        self._btn_cancel.setFixedHeight(26)
        self._btn_cancel.setToolTip("Annuler l'enregistrement sans appliquer")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setStyleSheet("""
            QPushButton { background: #222; color: #888;
                          border: 1px solid #444; border-radius: 4px; }
            QPushButton:enabled { color: #ffaa44; border-color: #ffaa44; }
            QPushButton:hover:enabled { background: #332200; }
        """)
        root.addWidget(self._btn_cancel)

        # ── Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #333;")
        root.addWidget(sep)

        # ── Mode
        root.addWidget(QLabel("Mode :"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Normal", "Punch", "Overdub"])
        self._combo_mode.setFixedHeight(26)
        self._combo_mode.setToolTip(
            "Normal  : enregistrement libre depuis la position courante\n"
            "Punch   : enregistrement uniquement dans la plage [In, Out]\n"
            "Overdub : superpose sur les keyframes existants"
        )
        self._combo_mode.setStyleSheet(
            "QComboBox { background: #1e2028; color: #ccc; "
            "border: 1px solid #444; border-radius: 4px; padding: 0 4px; }"
        )
        root.addWidget(self._combo_mode)

        # ── Punch In / Out
        self._lbl_punch = QLabel("Punch :")
        root.addWidget(self._lbl_punch)

        self._spin_in = QDoubleSpinBox()
        self._spin_in.setRange(0.0, 3600.0)
        self._spin_in.setDecimals(2)
        self._spin_in.setSuffix(" s")
        self._spin_in.setFixedWidth(80)
        self._spin_in.setFixedHeight(26)
        self._spin_in.setToolTip("Punch-in : début de la zone d'enregistrement")
        self._spin_in.setStyleSheet(
            "QDoubleSpinBox { background: #1e2028; color: #ccc; "
            "border: 1px solid #444; border-radius: 4px; }"
        )
        root.addWidget(self._spin_in)

        root.addWidget(QLabel("→"))

        self._spin_out = QDoubleSpinBox()
        self._spin_out.setRange(0.0, 3600.0)
        self._spin_out.setDecimals(2)
        self._spin_out.setSuffix(" s")
        self._spin_out.setValue(10.0)
        self._spin_out.setFixedWidth(80)
        self._spin_out.setFixedHeight(26)
        self._spin_out.setToolTip("Punch-out : fin de la zone d'enregistrement")
        self._spin_out.setStyleSheet(self._spin_in.styleSheet())
        root.addWidget(self._spin_out)

        # ── Export
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #333;")
        root.addWidget(sep2)

        self._btn_export = QPushButton("💾 Exporter")
        self._btn_export.setFixedHeight(26)
        self._btn_export.setToolTip("Exporter la session en fichier .demomaker")
        self._btn_export.setStyleSheet("""
            QPushButton { background: #1a2a1a; color: #44cc88;
                          border: 1px solid #44cc88; border-radius: 4px; padding: 0 8px; }
            QPushButton:hover { background: #1a3a1a; }
        """)
        root.addWidget(self._btn_export)

        root.addStretch()

        # Initialise l'état des widgets punch
        self._update_punch_visibility()

    def _connect_signals(self):
        self._btn_record.toggled.connect(self._on_record_toggled)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_export.clicked.connect(self.export_requested)
        self._combo_mode.currentIndexChanged.connect(self._update_punch_visibility)

        self._recorder.recording_state_changed.connect(self._on_recording_state)
        self._recorder.event_captured.connect(self._on_event_captured)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_record_toggled(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        mode_map = {0: RECORD_MODE_NORMAL, 1: RECORD_MODE_PUNCH, 2: RECORD_MODE_OVERDUB}
        mode = mode_map.get(self._combo_mode.currentIndex(), RECORD_MODE_NORMAL)

        if mode == RECORD_MODE_PUNCH:
            pin  = self._spin_in.value()
            pout = self._spin_out.value()
            if pin >= pout:
                # Remet le bouton à l'état initial sans déclencher
                self._btn_record.blockSignals(True)
                self._btn_record.setChecked(False)
                self._btn_record.blockSignals(False)
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Punch invalide",
                                    "Punch-in doit être inférieur à Punch-out.")
                return
            self._recorder.set_punch(pin, pout)
        else:
            self._recorder.clear_punch()

        # Récupère la position courante depuis le provider (ou 0)
        t0 = self._recorder._time_provider() if self._recorder._time_provider else 0.0
        self._recorder.start(t0, mode=mode)

    def _stop_recording(self):
        self._recorder.stop()

    def _on_cancel(self):
        self._recorder.cancel()
        self._btn_record.blockSignals(True)
        self._btn_record.setChecked(False)
        self._btn_record.blockSignals(False)

    def _on_recording_state(self, recording: bool):
        self._btn_cancel.setEnabled(recording)
        self._combo_mode.setEnabled(not recording)
        self._spin_in.setEnabled(not recording)
        self._spin_out.setEnabled(not recording)

        if recording:
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self._indicator.setStyleSheet("color: #444; font-size: 16px;")
            # Désynchronise le bouton si l'arrêt vient de l'extérieur
            self._btn_record.blockSignals(True)
            self._btn_record.setChecked(False)
            self._btn_record.blockSignals(False)

    def _on_event_captured(self, uniform_name: str, t: float):
        """Flash visuel bref à chaque événement capturé (optionnel)."""
        pass  # L'indicateur clignotant suffit

    def _blink(self):
        self._blink_on = not self._blink_on
        color = "#ff2222" if self._blink_on else "#661111"
        self._indicator.setStyleSheet(f"color: {color}; font-size: 16px;")

    def _update_punch_visibility(self):
        punch_mode = self._combo_mode.currentIndex() == 1  # 1 = Punch
        self._lbl_punch.setVisible(punch_mode)
        self._spin_in.setVisible(punch_mode)
        self._spin_out.setVisible(punch_mode)
        # Le label "→" entre les deux spins
        root_layout = self.layout()
        # On retrouve le label "→" (index fixe : 9)
        item = root_layout.itemAt(9)
        if item and item.widget():
            item.widget().setVisible(punch_mode)

    # ── API publique ─────────────────────────────────────────────────────────

    @property
    def recorder(self) -> SessionRecorder:
        return self._recorder

    def set_punch_from_loop_region(self, loop_in: float, loop_out: float):
        """Synchronise les champs punch depuis la loop region de la timeline."""
        self._spin_in.setValue(loop_in)
        self._spin_out.setValue(loop_out)
