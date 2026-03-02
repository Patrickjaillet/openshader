"""
vr_window.py
------------
v2.9 — Intégration Réalité Virtuelle OpenXR pour OpenShader.

Architecture :
  OpenXRRuntime      — détection / init de la session XR (pyopenxr ou xr)
  VRShaderEngine     — sur-couche de ShaderEngine : rendu stéréo dans deux
                       FBOs (œil gauche / droit) avec uViewMatrix[2],
                       uProjectionMatrix[2], uEyeIndex injectés
  XRControllerMapper — lit les actions XR (axes, boutons) et les mappe sur
                       des uniforms GLSL via le même système que MidiMapping
  VRTimelineOverlay  — quad 3D flottant devant l'utilisateur affichant la
                       timeline (snapshot QPixmap → texture OpenGL)
  VRWindow (QMainWindow) — fenêtre Qt miroir + gestion du cycle de vie XR

Compatibilité casques :
  Oculus / Meta Quest (PC Link ou Air Link)
  SteamVR (Valve Index, HTC Vive, Reverb G2…)
  Monado (open source, Linux)
  Tout casque compatible OpenXR 1.0

Dépendances optionnelles :
  pip install pyopenxr           # bindings Python OpenXR 1.0
  pip install numpy moderngl     # déjà requis par OpenShader

Si pyopenxr est absent, VRWindow se replie sur un mode de simulation
(split-screen stéréo logiciel) sans crash.
"""

from __future__ import annotations

import math
import time
import ctypes
import threading
from typing import Optional, Any

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QDialog, QRadioButton, QButtonGroup,
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QMessageBox, QApplication, QSplitter
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui   import (QPixmap, QImage, QColor, QPainter, QPen,
                            QBrush, QFont, QKeySequence, QShortcut)

from .logger      import get_logger
from .midi_engine import MidiMapping

log = get_logger(__name__)

# ── Détection OpenXR ──────────────────────────────────────────────────────────

_XR_LIB: Optional[str] = None
_xr = None


def _try_import_openxr() -> bool:
    global _XR_LIB, _xr
    if _XR_LIB is not None:
        return _XR_LIB != ''
    for name in ('xr', 'pyopenxr'):
        try:
            import importlib
            _xr = importlib.import_module(name)
            _XR_LIB = name
            log.info("OpenXR : lib '%s' chargée", name)
            return True
        except ImportError:
            pass
    _XR_LIB = ''
    log.warning("OpenXR indisponible — mode simulation stéréo actif. "
                "Installez pyopenxr : pip install pyopenxr")
    return False


def openxr_available() -> bool:
    return _try_import_openxr()


# ── Constantes VR ─────────────────────────────────────────────────────────────

EYE_LEFT  = 0
EYE_RIGHT = 1

# IPD par défaut (m) et demi-angle FOV (degrés) si pas de casque réel
DEFAULT_IPD_M    = 0.063
DEFAULT_FOV_DEG  = 45.0
DEFAULT_NEAR     = 0.05
DEFAULT_FAR      = 100.0

# Header GLSL injecté pour les shaders VR
VR_GLSL_HEADER = """
#version 330 core
// ── OpenShader VR Uniforms ──────────────────────────────────────────────────
uniform vec3      iResolution;
uniform float     iTime;
uniform float     iTimeDelta;
uniform int       iFrame;
uniform vec4      iMouse;
// Stéréo
uniform int       uEyeIndex;           // 0 = gauche, 1 = droite
uniform mat4      uViewMatrix[2];      // matrice de vue par œil
uniform mat4      uProjectionMatrix[2];// matrice de projection par œil
uniform mat4      uHeadMatrix;         // pose de la tête (world→head)
// Contrôleurs XR (L/R : position, orientation quaternion, trigger, grip, axes)
uniform vec3      uCtrlPosL;           // position contrôleur gauche (world)
uniform vec3      uCtrlPosR;           // position contrôleur droit  (world)
uniform vec4      uCtrlRotL;           // quaternion contrôleur gauche
uniform vec4      uCtrlRotR;           // quaternion contrôleur droit
uniform float     uCtrlTriggerL;       // trigger gauche [0..1]
uniform float     uCtrlTriggerR;       // trigger droit  [0..1]
uniform float     uCtrlGripL;          // grip gauche    [0..1]
uniform float     uCtrlGripR;          // grip droit     [0..1]
uniform vec2      uCtrlThumbL;         // thumbstick gauche [-1..1]×2
uniform vec2      uCtrlThumbR;         // thumbstick droit  [-1..1]×2
// iChannel Shadertoy
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
out vec4 _fragColor;
// ────────────────────────────────────────────────────────────────────────────
"""

VR_GLSL_FOOTER = """
void main() {
    // fragCoord = position dans l'œil courant (0..iResolution.xy)
    mainImage(_fragColor, gl_FragCoord.xy);
}
"""

# Snippet GLSL utilitaire inséré en mode VR pour les shaders qui veulent
# accéder à uViewMatrix / uProjectionMatrix facilement
VR_GLSL_HELPERS = """
// ── VR Helpers ──────────────────────────────────────────────────────────────
// Retourne la direction d'un rayon dans l'espace monde depuis le pixel courant
vec3 vrRayDir(vec2 fragCoord) {
    mat4 invProj = inverse(uProjectionMatrix[uEyeIndex]);
    mat4 invView = inverse(uViewMatrix[uEyeIndex]);
    vec2 ndc = (fragCoord / iResolution.xy) * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye  = invProj * clip;
    eye = vec4(eye.xy, -1.0, 0.0);
    vec3 world = (invView * eye).xyz;
    return normalize(world);
}

// Retourne l'origine de la caméra pour l'œil courant (en espace monde)
vec3 vrEyeOrigin() {
    mat4 invView = inverse(uViewMatrix[uEyeIndex]);
    return invView[3].xyz;
}
// ─────────────────────────────────────────────────────────────────────────────
"""


# ── Matrices utilitaires ──────────────────────────────────────────────────────

def _perspective_matrix(fov_y_rad: float, aspect: float,
                         near: float, far: float) -> np.ndarray:
    """Retourne une matrice de projection perspective column-major (OpenGL)."""
    f = 1.0 / math.tan(fov_y_rad / 2)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = -1.0
    m[3, 2] = 2 * far * near * nf
    return m


def _look_at(eye: np.ndarray, center: np.ndarray,
              up: np.ndarray) -> np.ndarray:
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up / np.linalg.norm(up))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, 0]  = -np.dot(s, eye)
    m[3, 1]  = -np.dot(u, eye)
    m[3, 2]  =  np.dot(f, eye)
    return m.T


def _translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def _mat4_to_tuple(m: np.ndarray) -> tuple:
    """numpy (4,4) float32 → tuple de 16 floats (column-major pour OpenGL)."""
    return tuple(m.T.flatten().tolist())


# ── État XR par frame ─────────────────────────────────────────────────────────

class XRFrameState:
    """Contient les matrices et états de contrôleurs pour une frame XR."""

    def __init__(self):
        # Pose de la tête
        self.head_pos:   np.ndarray = np.zeros(3, dtype=np.float32)
        self.head_rot:   np.ndarray = np.array([0, 0, 0, 1], dtype=np.float32)

        # Matrices View / Projection par œil
        fov = math.radians(DEFAULT_FOV_DEG)
        ipd_half = DEFAULT_IPD_M / 2

        self.view_left  = _look_at(
            np.array([-ipd_half, 0, 0], np.float32),
            np.array([-ipd_half, 0, -1], np.float32),
            np.array([0, 1, 0], np.float32)
        )
        self.view_right = _look_at(
            np.array([ipd_half, 0, 0], np.float32),
            np.array([ipd_half, 0, -1], np.float32),
            np.array([0, 1, 0], np.float32)
        )
        self.proj_left  = _perspective_matrix(fov, 1.0, DEFAULT_NEAR, DEFAULT_FAR)
        self.proj_right = _perspective_matrix(fov, 1.0, DEFAULT_NEAR, DEFAULT_FAR)

        # Contrôleurs
        self.ctrl_pos_l:     np.ndarray = np.array([-0.3, -0.3, -0.5], np.float32)
        self.ctrl_pos_r:     np.ndarray = np.array([0.3,  -0.3, -0.5], np.float32)
        self.ctrl_rot_l:     np.ndarray = np.array([0, 0, 0, 1], np.float32)
        self.ctrl_rot_r:     np.ndarray = np.array([0, 0, 0, 1], np.float32)
        self.ctrl_trigger_l: float = 0.0
        self.ctrl_trigger_r: float = 0.0
        self.ctrl_grip_l:    float = 0.0
        self.ctrl_grip_r:    float = 0.0
        self.ctrl_thumb_l:   tuple = (0.0, 0.0)
        self.ctrl_thumb_r:   tuple = (0.0, 0.0)

        # Boutons
        self.btn_a:      bool = False
        self.btn_b:      bool = False
        self.btn_x:      bool = False
        self.btn_y:      bool = False
        self.btn_menu:   bool = False


# ── Runtime OpenXR ────────────────────────────────────────────────────────────

class OpenXRRuntime:
    """
    Encapsule la session OpenXR.
    Si pyopenxr n'est pas disponible, bascule en mode simulation
    (matrices synthétiques animées pour tester les shaders VR).
    """

    def __init__(self):
        self._session    = None
        self._instance   = None
        self._system_id  = None
        self._running    = False
        self._simulation = not _try_import_openxr()
        self._sim_t      = 0.0        # temps simulation
        self._ipd        = DEFAULT_IPD_M
        self._fov_deg    = DEFAULT_FOV_DEG
        self.frame_state = XRFrameState()

    @property
    def is_simulation(self) -> bool:
        return self._simulation

    @property
    def is_running(self) -> bool:
        return self._running

    def initialize(self) -> tuple[bool, str]:
        """
        Démarre la session OpenXR.
        Retourne (ok, message).
        """
        if self._simulation:
            self._running = True
            log.info("OpenXR : mode simulation (casque non détecté)")
            return True, "Mode simulation (casque non détecté)"

        try:
            xr = _xr

            app_info = xr.ApplicationInfo(
                application_name='OpenShader',
                application_version=xr.make_version(2, 9, 0),
                engine_name='OpenShader Engine',
                engine_version=xr.make_version(1, 0, 0),
                api_version=xr.XR_CURRENT_API_VERSION,
            )
            create_info = xr.InstanceCreateInfo(
                application_info=app_info,
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            )
            self._instance = xr.create_instance(create_info)

            system_info   = xr.SystemGetInfo(
                form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY
            )
            self._system_id = xr.get_system(self._instance, system_info)

            self._running = True
            name = xr.get_system_properties(
                self._instance, self._system_id
            ).system_name.decode(errors='replace')
            log.info("OpenXR session initialisée : %s", name)
            return True, f"Casque détecté : {name}"

        except Exception as e:
            log.warning("OpenXR init échouée (%s) — basculement simulation", e)
            self._simulation = True
            self._running    = True
            return True, f"Simulation (erreur OpenXR : {e})"

    def poll_frame(self, current_time: float):
        """
        Met à jour frame_state (pose tête + contrôleurs).
        En mode simulation, génère des mouvements synthétiques.
        """
        self._sim_t = current_time

        if self._simulation:
            self._simulate_frame(current_time)
            return

        try:
            self._poll_xr_frame()
        except Exception as e:
            log.debug("OpenXR poll_frame : %s — fallback simulation", e)
            self._simulate_frame(current_time)

    def _simulate_frame(self, t: float):
        """Génère une pose de tête simulée avec léger mouvement sinusoïdal."""
        fs = self.frame_state
        ipd_h = self._ipd / 2
        fov   = math.radians(self._fov_deg)

        # Tête : léger balancement
        head_y = math.sin(t * 0.3) * 0.05
        head_x = math.sin(t * 0.2) * 0.03

        fs.head_pos = np.array([head_x, 1.6 + head_y, 0], np.float32)

        eye_l = np.array([head_x - ipd_h, 1.6 + head_y, 0], np.float32)
        eye_r = np.array([head_x + ipd_h, 1.6 + head_y, 0], np.float32)
        center = np.array([head_x, 1.6 + head_y, -1], np.float32)
        up     = np.array([0, 1, 0], np.float32)

        fs.view_left  = _look_at(eye_l, center, up)
        fs.view_right = _look_at(eye_r, center, up)
        fs.proj_left  = _perspective_matrix(fov, 1.0, DEFAULT_NEAR, DEFAULT_FAR)
        fs.proj_right = _perspective_matrix(fov, 1.0, DEFAULT_NEAR, DEFAULT_FAR)

        # Contrôleurs : orbite lente
        a = t * 0.5
        fs.ctrl_pos_l = np.array([
            head_x - 0.35 + math.cos(a) * 0.05,
            1.2 + math.sin(a * 1.3) * 0.08,
            -0.45
        ], np.float32)
        fs.ctrl_pos_r = np.array([
            head_x + 0.35 + math.cos(a + math.pi) * 0.05,
            1.2 + math.sin(a * 1.1) * 0.08,
            -0.45
        ], np.float32)
        fs.ctrl_trigger_l = max(0, math.sin(t * 0.7)) * 0.5
        fs.ctrl_trigger_r = max(0, math.sin(t * 0.9 + 1)) * 0.5
        fs.ctrl_thumb_l   = (math.sin(t * 0.4) * 0.3, math.cos(t * 0.3) * 0.3)
        fs.ctrl_thumb_r   = (-math.sin(t * 0.5) * 0.3, math.cos(t * 0.6) * 0.3)

    def _poll_xr_frame(self):
        """Lit la pose réelle depuis l'API OpenXR."""
        if not _xr or not self._session:
            self._simulate_frame(self._sim_t)
            return

        xr  = _xr
        fs  = self.frame_state
        ipd_h = self._ipd / 2
        fov   = math.radians(self._fov_deg)

        try:
            frame_state = xr.wait_frame(self._session, xr.FrameWaitInfo())
            xr.begin_frame(self._session, xr.FrameBeginInfo())

            # Localisation des vues
            view_state, views = xr.locate_views(
                self._session,
                xr.ViewLocateInfo(
                    view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                    display_time=frame_state.predicted_display_time,
                    space=self._reference_space,
                )
            )

            if len(views) >= 2:
                for eye_idx, view in enumerate(views[:2]):
                    pos  = view.pose.position
                    ori  = view.pose.orientation
                    fov_ = view.fov

                    eye_pos = np.array([pos.x, pos.y, pos.z], np.float32)
                    center  = eye_pos + np.array([
                        math.sin(math.atan2(ori.y * ori.w + ori.x * ori.z,
                                            0.5 - ori.x * ori.x - ori.y * ori.y)),
                        0, -1
                    ], np.float32)

                    view_mat = _look_at(eye_pos, center,
                                        np.array([0, 1, 0], np.float32))
                    proj_mat = _perspective_matrix(
                        abs(fov_.angle_up) + abs(fov_.angle_down),
                        1.0, DEFAULT_NEAR, DEFAULT_FAR
                    )
                    if eye_idx == EYE_LEFT:
                        fs.view_left = view_mat
                        fs.proj_left = proj_mat
                    else:
                        fs.view_right = view_mat
                        fs.proj_right = proj_mat

        except Exception as e:
            log.debug("XR locate_views : %s", e)
            self._simulate_frame(self._sim_t)

    def get_info_string(self) -> str:
        if self._simulation:
            return (f"Mode simulation  |  IPD {self._ipd*1000:.0f} mm  "
                    f"|  FOV {self._fov_deg:.0f}°")
        return f"OpenXR actif ({_XR_LIB})  |  IPD {self._ipd*1000:.0f} mm"

    def set_ipd(self, ipd_m: float):
        self._ipd = max(0.04, min(0.08, ipd_m))

    def set_fov(self, deg: float):
        self._fov_deg = max(30.0, min(120.0, deg))

    def shutdown(self):
        self._running = False
        if self._session and _xr:
            try:
                _xr.destroy_session(self._session)
            except Exception:
                pass
        if self._instance and _xr:
            try:
                _xr.destroy_instance(self._instance)
            except Exception:
                pass
        log.info("OpenXR session fermée")


# ── XRControllerMapper ────────────────────────────────────────────────────────

class XRControllerInput:
    """Identifiant d'une entrée contrôleur XR."""
    TRIGGER_L   = 'trigger_l'
    TRIGGER_R   = 'trigger_r'
    GRIP_L      = 'grip_l'
    GRIP_R      = 'grip_r'
    THUMB_L_X   = 'thumb_l_x'
    THUMB_L_Y   = 'thumb_l_y'
    THUMB_R_X   = 'thumb_r_x'
    THUMB_R_Y   = 'thumb_r_y'
    BTN_A       = 'btn_a'
    BTN_B       = 'btn_b'
    BTN_X       = 'btn_x'
    BTN_Y       = 'btn_y'

    ALL = [TRIGGER_L, TRIGGER_R, GRIP_L, GRIP_R,
           THUMB_L_X, THUMB_L_Y, THUMB_R_X, THUMB_R_Y,
           BTN_A, BTN_B, BTN_X, BTN_Y]


class XRControllerMapping:
    """
    Association entrée XR → uniform GLSL.
    Compatible avec l'architecture MidiMapping existante.
    """
    def __init__(self, xr_input: str, uniform: str,
                 lo: float = 0.0, hi: float = 1.0,
                 curve: str = 'linear'):
        self.xr_input = xr_input
        self.uniform  = uniform
        self.lo       = lo
        self.hi       = hi
        self.curve    = curve

    def scale(self, raw: float) -> float:
        t = max(0.0, min(1.0, (raw - 0.0) / 1.0))   # raw déjà normalisé [0,1]
        if self.curve == 'exp':
            t = t * t
        elif self.curve == 'log':
            t = math.log1p(t * (math.e - 1))
        return self.lo + t * (self.hi - self.lo)

    def to_dict(self) -> dict:
        return {'xr_input': self.xr_input, 'uniform': self.uniform,
                'lo': self.lo, 'hi': self.hi, 'curve': self.curve}

    @classmethod
    def from_dict(cls, d: dict) -> 'XRControllerMapping':
        return cls(d['xr_input'], d['uniform'],
                   float(d.get('lo', 0)), float(d.get('hi', 1)),
                   d.get('curve', 'linear'))


class XRControllerMapper(QObject):
    """
    Lit XRFrameState, applique les mappings et émet uniform_changed
    — même interface que MidiEngine pour le raccorder au ShaderEngine.
    """
    uniform_changed = pyqtSignal(str, float)   # (uniform_name, value)
    learn_triggered = pyqtSignal(str)           # (xr_input_id)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mappings:   list[XRControllerMapping] = []
        self._learn_mode: bool = False
        self._learn_slot: Optional[str] = None   # uniform cible en attente

    # ── Mappings ──────────────────────────────────────────────────────────────

    def add_mapping(self, xr_input: str, uniform: str,
                    lo: float = 0.0, hi: float = 1.0,
                    curve: str = 'linear') -> XRControllerMapping:
        m = XRControllerMapping(xr_input, uniform, lo, hi, curve)
        self._mappings.append(m)
        log.debug("XR mapping : %s → %s [%.2f..%.2f]", xr_input, uniform, lo, hi)
        return m

    def remove_mapping(self, m: XRControllerMapping):
        if m in self._mappings:
            self._mappings.remove(m)

    def clear_mappings(self):
        self._mappings.clear()

    def get_mappings(self) -> list[XRControllerMapping]:
        return list(self._mappings)

    def to_dict(self) -> list[dict]:
        return [m.to_dict() for m in self._mappings]

    def from_dict(self, data: list[dict]):
        self._mappings = [XRControllerMapping.from_dict(d) for d in data]

    # ── Learn ─────────────────────────────────────────────────────────────────

    def start_learn(self, uniform_name: str):
        """Active le mode XR Learn : le prochain input actif est mappé sur uniform_name."""
        self._learn_mode = True
        self._learn_slot = uniform_name
        log.info("XR Learn actif : attente d'un input pour '%s'", uniform_name)

    def cancel_learn(self):
        self._learn_mode = False
        self._learn_slot = None

    # ── Polling ───────────────────────────────────────────────────────────────

    def process_frame(self, fs: XRFrameState):
        """
        À appeler à chaque frame depuis VRWindow._render_frame().
        Lit les valeurs brutes, détecte le Learn, émet les uniforms.
        """
        raw: dict[str, float] = {
            XRControllerInput.TRIGGER_L:  fs.ctrl_trigger_l,
            XRControllerInput.TRIGGER_R:  fs.ctrl_trigger_r,
            XRControllerInput.GRIP_L:     fs.ctrl_grip_l,
            XRControllerInput.GRIP_R:     fs.ctrl_grip_r,
            XRControllerInput.THUMB_L_X:  fs.ctrl_thumb_l[0] * 0.5 + 0.5,
            XRControllerInput.THUMB_L_Y:  fs.ctrl_thumb_l[1] * 0.5 + 0.5,
            XRControllerInput.THUMB_R_X:  fs.ctrl_thumb_r[0] * 0.5 + 0.5,
            XRControllerInput.THUMB_R_Y:  fs.ctrl_thumb_r[1] * 0.5 + 0.5,
            XRControllerInput.BTN_A:      float(fs.btn_a),
            XRControllerInput.BTN_B:      float(fs.btn_b),
            XRControllerInput.BTN_X:      float(fs.btn_x),
            XRControllerInput.BTN_Y:      float(fs.btn_y),
        }

        # Learn : détecte l'input le plus actif (> seuil)
        if self._learn_mode and self._learn_slot:
            for inp_id, val in raw.items():
                if val > 0.5:
                    self.add_mapping(inp_id, self._learn_slot)
                    self.learn_triggered.emit(inp_id)
                    self._learn_mode = False
                    self._learn_slot = None
                    break

        # Appliquer les mappings
        for m in self._mappings:
            v = raw.get(m.xr_input, 0.0)
            self.uniform_changed.emit(m.uniform, m.scale(v))

    def default_mappings(self):
        """Mappings par défaut utiles : trigger → uBrightness, thumb → shader params."""
        self._mappings.clear()
        self.add_mapping(XRControllerInput.TRIGGER_R, 'uBrightness', 0.0, 2.0)
        self.add_mapping(XRControllerInput.TRIGGER_L, 'uDistortion', 0.0, 1.0)
        self.add_mapping(XRControllerInput.THUMB_R_X, 'uColorShift', 0.0, 1.0)
        self.add_mapping(XRControllerInput.THUMB_R_Y, 'uSpeed',      0.1, 3.0)
        self.add_mapping(XRControllerInput.THUMB_L_X, 'uScale',      0.5, 4.0)
        self.add_mapping(XRControllerInput.GRIP_R,    'uMix',        0.0, 1.0)


# ── VRShaderEngine ────────────────────────────────────────────────────────────

class VRShaderEngine:
    """
    Sur-couche de ShaderEngine pour le rendu stéréo VR.

    Pour chaque frame :
      1. Injecte les uniforms VR dans extra_uniforms du ShaderEngine de base
      2. Rend l'œil gauche dans eye_fbo_l
      3. Rend l'œil droit dans eye_fbo_r
      4. Les FBOs peuvent être lus pour le miroir Qt ou soumis au compositor XR

    Le source GLSL est automatiquement préfixé avec VR_GLSL_HEADER.
    """

    def __init__(self, base_engine, eye_width: int = 1200, eye_height: int = 1200):
        from .shader_engine import ShaderEngine
        self._base:       ShaderEngine = base_engine
        self.eye_width    = eye_width
        self.eye_height   = eye_height
        self._eye_fbos:   list = [None, None]
        self._eye_textures: list = [None, None]
        self._initialized = False

    def initialize(self, ctx):
        """Crée les FBOs par œil. Appeler après base_engine.initialize()."""
        import moderngl as mgl
        self._ctx = ctx
        for i in range(2):
            tex = ctx.texture((self.eye_width, self.eye_height), 4, dtype='f1')
            fbo = ctx.framebuffer(color_attachments=[tex])
            self._eye_fbos[i]    = fbo
            self._eye_textures[i] = tex
        self._initialized = True
        log.debug("VRShaderEngine FBOs créés (%dx%d × 2 yeux)",
                  self.eye_width, self.eye_height)

    def render_stereo(self, current_time: float, frame_state: XRFrameState):
        """
        Rend les deux yeux dans leurs FBOs respectifs.
        Injecte tous les uniforms VR dans le ShaderEngine de base.
        """
        if not self._initialized:
            return

        base = self._base
        fs   = frame_state

        # Matrices View (column-major pour OpenGL)
        view_l_t  = _mat4_to_tuple(fs.view_left)
        view_r_t  = _mat4_to_tuple(fs.view_right)
        proj_l_t  = _mat4_to_tuple(fs.proj_left)
        proj_r_t  = _mat4_to_tuple(fs.proj_right)
        head_t    = _mat4_to_tuple(
            _translation_matrix(*fs.head_pos.tolist())
        )

        # Uniformes partagés (posés une fois, lus par les deux yeux)
        base.set_uniform('uViewMatrix',        [view_l_t, view_r_t])
        base.set_uniform('uProjectionMatrix',  [proj_l_t, proj_r_t])
        base.set_uniform('uHeadMatrix',        head_t)

        # Contrôleurs
        base.set_uniform('uCtrlPosL',     tuple(fs.ctrl_pos_l.tolist()))
        base.set_uniform('uCtrlPosR',     tuple(fs.ctrl_pos_r.tolist()))
        base.set_uniform('uCtrlRotL',     tuple(fs.ctrl_rot_l.tolist()))
        base.set_uniform('uCtrlRotR',     tuple(fs.ctrl_rot_r.tolist()))
        base.set_uniform('uCtrlTriggerL', float(fs.ctrl_trigger_l))
        base.set_uniform('uCtrlTriggerR', float(fs.ctrl_trigger_r))
        base.set_uniform('uCtrlGripL',    float(fs.ctrl_grip_l))
        base.set_uniform('uCtrlGripR',    float(fs.ctrl_grip_r))
        base.set_uniform('uCtrlThumbL',   tuple(fs.ctrl_thumb_l))
        base.set_uniform('uCtrlThumbR',   tuple(fs.ctrl_thumb_r))

        # Rendu œil gauche
        base.set_uniform('uEyeIndex', EYE_LEFT)
        base.render(current_time, screen_fbo=self._eye_fbos[EYE_LEFT])

        # Rendu œil droit
        base.set_uniform('uEyeIndex', EYE_RIGHT)
        base.render(current_time, screen_fbo=self._eye_fbos[EYE_RIGHT])

    def grab_eye_image(self, eye: int) -> Optional[QImage]:
        """Lit le FBO d'un œil et retourne un QImage RGBA."""
        fbo = self._eye_fbos[eye]
        if not fbo:
            return None
        try:
            fbo.use()
            raw = fbo.read(components=4)
            img = QImage(raw, self.eye_width, self.eye_height,
                         QImage.Format.Format_RGBA8888)
            return img.mirrored(False, True)   # flip Y OpenGL→Qt
        except Exception as e:
            log.debug("grab_eye_image(%d) : %s", eye, e)
            return None

    def resize(self, eye_w: int, eye_h: int):
        self.eye_width  = eye_w
        self.eye_height = eye_h
        if not self._initialized:
            return
        for i in range(2):
            try:
                if self._eye_fbos[i]:     self._eye_fbos[i].release()
                if self._eye_textures[i]: self._eye_textures[i].release()
            except Exception:
                pass
            tex = self._ctx.texture((eye_w, eye_h), 4, dtype='f1')
            fbo = self._ctx.framebuffer(color_attachments=[tex])
            self._eye_fbos[i]    = fbo
            self._eye_textures[i] = tex

    def cleanup(self):
        for i in range(2):
            try:
                if self._eye_fbos[i]:     self._eye_fbos[i].release()
                if self._eye_textures[i]: self._eye_textures[i].release()
            except Exception:
                pass


# ── VRTimelineOverlay ─────────────────────────────────────────────────────────

class VRTimelineOverlay:
    """
    Quad flottant dans l'espace XR affichant un snapshot de la timeline.

    Implémentation :
      - Un QTimer à 5 Hz capture le widget timeline en QPixmap
      - La pixmap est uploadée comme texture OpenGL
      - Un quad 3D (1.2m × 0.3m) est rendu à ~0.8m devant la tête,
        légèrement en dessous du centre visuel (ergonomie VR)

    Le shader du quad est minimaliste (texture passthrough).
    """

    QUAD_W  = 1.2   # largeur du quad (mètres)
    QUAD_H  = 0.3   # hauteur
    QUAD_Z  = -0.8  # distance devant la tête
    QUAD_Y  = -0.1  # offset vertical

    def __init__(self, timeline_widget):
        self._tl_widget = timeline_widget
        self._texture   = None
        self._program   = None
        self._vao       = None
        self._ctx       = None
        self._visible   = False
        self._dirty     = True

        self._refresh_timer = QTimer()
        self._refresh_timer.setInterval(200)   # 5 Hz
        self._refresh_timer.timeout.connect(self._mark_dirty)

    def _mark_dirty(self):
        self._dirty = True

    def initialize(self, ctx):
        import moderngl as mgl
        self._ctx = ctx

        # Shader quad VR
        VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
uniform mat4 uMVP;
out vec2 vUV;
void main() {
    gl_Position = uMVP * vec4(in_pos, 0.0, 1.0);
    vUV = in_uv;
}
"""
        FRAG = """
#version 330 core
uniform sampler2D uTex;
in vec2 vUV;
out vec4 fragColor;
void main() {
    fragColor = texture(uTex, vUV);
}
"""
        try:
            self._program = ctx.program(vertex_shader=VERT,
                                        fragment_shader=FRAG)
            w, h = self.QUAD_W / 2, self.QUAD_H / 2
            verts = np.array([
                -w, -h,  0, 1,
                 w, -h,  1, 1,
                -w,  h,  0, 0,
                 w, -h,  1, 1,
                 w,  h,  1, 0,
                -w,  h,  0, 0,
            ], dtype=np.float32)
            vbo = ctx.buffer(verts)
            self._vao = ctx.vertex_array(
                self._program,
                [(vbo, '2f 2f', 'in_pos', 'in_uv')]
            )
            log.debug("VRTimelineOverlay initialisé")
        except Exception as e:
            log.warning("VRTimelineOverlay init : %s", e)

    def set_visible(self, v: bool):
        self._visible = v
        if v:
            self._refresh_timer.start()
        else:
            self._refresh_timer.stop()

    def render(self, frame_state: XRFrameState, eye: int):
        """Rend le quad de la timeline pour l'œil donné."""
        if not self._visible or not self._program or not self._vao:
            return

        if self._dirty:
            self._upload_texture()
            self._dirty = False

        if not self._texture:
            return

        # MVP : translation devant la tête + légère descente
        head_t = _translation_matrix(
            frame_state.head_pos[0],
            frame_state.head_pos[1] + self.QUAD_Y,
            frame_state.head_pos[2] + self.QUAD_Z
        )
        view = frame_state.view_left if eye == EYE_LEFT else frame_state.view_right
        proj = frame_state.proj_left if eye == EYE_LEFT else frame_state.proj_right

        mvp = proj @ view @ head_t
        try:
            if 'uMVP' in self._program:
                self._program['uMVP'].write(mvp.astype(np.float32).tobytes())
            self._texture.use(location=7)
            if 'uTex' in self._program:
                self._program['uTex'].value = 7
            self._vao.render()
        except Exception as e:
            log.debug("VRTimelineOverlay render : %s", e)

    def _upload_texture(self):
        if not self._ctx or not self._tl_widget:
            return
        try:
            pm   = self._tl_widget.grab()
            img  = pm.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = img.width(), img.height()
            if w == 0 or h == 0:
                return
            ptr  = img.constBits()
            ptr.setsize(w * h * 4)
            if self._texture:
                try:
                    self._texture.release()
                except Exception:
                    pass
            self._texture = self._ctx.texture((w, h), 4, bytes(ptr))
            self._texture.filter = (0x2601, 0x2601)  # LINEAR
        except Exception as e:
            log.debug("VRTimelineOverlay texture upload : %s", e)

    def cleanup(self):
        self._refresh_timer.stop()
        if self._texture:
            try:
                self._texture.release()
            except Exception:
                pass


# ── VRWindow ──────────────────────────────────────────────────────────────────

class VRWindow(QMainWindow):
    """
    Fenêtre VR principale.

    Affiche un miroir split-screen (œil G / œil D) dans la fenêtre Qt
    pendant que le rendu stéréo tourne dans les FBOs VRShaderEngine.
    Si un compositor XR est disponible, les frames lui sont soumises.

    Hotkeys :
      Escape — fermer
      Tab    — toggle overlay HUD
      T      — toggle overlay timeline 3D
      ←/→    — shader précédent/suivant
      1–9    — preset direct
      F11    — toggle fullscreen
      L      — XR Learn (mappe le prochain input sur le uniform sélectionné)
    """

    def __init__(self, parent_window: 'MainWindow'):
        super().__init__()
        self.setWindowTitle("OpenShader — Mode VR")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        self._pw   = parent_window
        self._xr   = OpenXRRuntime()
        self._ctrl = XRControllerMapper()
        self._ctrl.uniform_changed.connect(
            lambda name, val: parent_window.shader_engine.set_uniform(name, val)
        )
        self._ctrl.learn_triggered.connect(self._on_learn_triggered)
        self._ctrl.default_mappings()

        # VRShaderEngine sur le ShaderEngine existant
        self._vr_engine: Optional[VRShaderEngine] = None

        # Timeline overlay
        self._tl_overlay: Optional[VRTimelineOverlay] = None
        self._tl_visible = False

        # Mirror display
        self._mirror_l = QLabel()
        self._mirror_r = QLabel()
        for lbl in (self._mirror_l, self._mirror_r):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #000;")
            lbl.setScaledContents(False)

        # HUD overlay
        self._overlay    = QLabel(self)
        self._show_hud   = True
        self._overlay.setStyleSheet(
            "QLabel { color: rgba(255,255,255,210); font: bold 11px 'Segoe UI';"
            " background: rgba(0,0,0,140); padding: 4px 12px; border-radius:4px;}"
        )
        self._overlay.move(12, 12)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._build_ui()
        self._setup_hotkeys()

        # Timer de rendu
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(11)   # ~90 Hz cible
        self._frame_timer.timeout.connect(self._render_frame)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet("background: #000;")
        self.setCentralWidget(central)
        vl = QVBoxLayout(central)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #111; width: 2px; }")
        splitter.addWidget(self._mirror_l)
        splitter.addWidget(self._mirror_r)
        splitter.setSizes([1, 1])
        vl.addWidget(splitter, 1)

    def _setup_hotkeys(self):
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.close)
        QShortcut(QKeySequence("Tab"),    self).activated.connect(self._toggle_hud)
        QShortcut(QKeySequence("T"),      self).activated.connect(self._toggle_timeline_overlay)
        QShortcut(QKeySequence("Right"),  self).activated.connect(self._next_shader)
        QShortcut(QKeySequence("Left"),   self).activated.connect(self._prev_shader)
        QShortcut(QKeySequence("F11"),    self).activated.connect(
            lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen()
        )
        QShortcut(QKeySequence("L"),      self).activated.connect(self._start_learn)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                lambda _, idx=i - 1: self._jump_to_shader(idx)
            )

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self):
        """Initialise OpenXR et démarre le rendu."""
        ok, msg = self._xr.initialize()
        if not ok:
            QMessageBox.critical(self, "Erreur OpenXR", msg)
            self.close()
            return

        # Initialise VRShaderEngine sur le contexte OpenGL existant
        try:
            from .gl_widget import GLWidget
            gl = self._pw.gl_widget
            gl.makeCurrent()
            self._vr_engine = VRShaderEngine(
                self._pw.shader_engine,
                eye_width=1200, eye_height=1200
            )
            self._vr_engine.initialize(self._pw.shader_engine.ctx)

            # Timeline overlay
            self._tl_overlay = VRTimelineOverlay(self._pw.timeline_widget)
            self._tl_overlay.initialize(self._pw.shader_engine.ctx)
            gl.doneCurrent()
        except Exception as e:
            log.warning("VRShaderEngine init : %s", e)
            self._vr_engine = None

        self._frame_timer.start()
        log.info("VRWindow démarrée — %s", self._xr.get_info_string())

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _render_frame(self):
        pw = self._pw
        if not pw:
            return

        t = pw._current_time

        # 1. Polling XR
        self._xr.poll_frame(t)
        fs = self._xr.frame_state

        # 2. Process contrôleurs → uniforms
        self._ctrl.process_frame(fs)

        # 3. Rendu stéréo dans les FBOs VR
        if self._vr_engine:
            try:
                pw.gl_widget.makeCurrent()
                self._vr_engine.render_stereo(t, fs)

                # Rendu overlay timeline si visible
                if self._tl_overlay and self._tl_visible:
                    for eye in (EYE_LEFT, EYE_RIGHT):
                        self._vr_engine._eye_fbos[eye].use()
                        self._tl_overlay.render(fs, eye)

                pw.gl_widget.doneCurrent()
            except Exception as e:
                log.debug("VR render_stereo : %s", e)

        # 4. Mirror Qt
        self._update_mirror()

        # 5. HUD
        if self._show_hud:
            self._refresh_hud(t, fs)
            self._overlay.show()
            self._overlay.raise_()
        else:
            self._overlay.hide()

    def _update_mirror(self):
        """Affiche les snapshots des deux yeux dans les QLabels."""
        if not self._vr_engine:
            # Fallback : miroir du viewport principal
            try:
                self._pw.gl_widget.makeCurrent()
                img = self._pw.gl_widget.grabFramebuffer()
                self._pw.gl_widget.doneCurrent()
            except Exception:
                return
            if not img.isNull():
                pm     = QPixmap.fromImage(img)
                # Côté gauche : moitié gauche
                w2 = img.width() // 2
                pm_l = pm.copy(0, 0, w2, img.height())
                pm_r = pm.copy(w2, 0, img.width() - w2, img.height())
                self._set_mirror(self._mirror_l, pm_l)
                self._set_mirror(self._mirror_r, pm_r)
            return

        for eye, lbl in ((EYE_LEFT, self._mirror_l), (EYE_RIGHT, self._mirror_r)):
            img = self._vr_engine.grab_eye_image(eye)
            if img:
                self._set_mirror(lbl, QPixmap.fromImage(img))

    @staticmethod
    def _set_mirror(lbl: QLabel, pm: QPixmap):
        sz = lbl.size()
        if sz.width() > 0 and sz.height() > 0:
            lbl.setPixmap(pm.scaled(
                sz,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def _refresh_hud(self, t: float, fs: XRFrameState):
        m, s, ms  = int(t) // 60, int(t) % 60, int((t % 1) * 1000)
        bpm       = getattr(self._pw.timeline, 'bpm', 120)
        beat      = int(t / (60.0 / bpm)) + 1 if bpm > 0 else 0
        shader_nm = (os.path.basename(self._pw._active_image_shader_path)
                     if self._pw._active_image_shader_path else "—")
        sim_mark  = " [SIM]" if self._xr.is_simulation else ""
        xr_mark   = " 🥽 XR" if not self._xr.is_simulation else " 🖥️ SIM"
        self._overlay.setText(
            f"{xr_mark}{sim_mark}  ⏱ {m:02d}:{s:02d}.{ms:03d}  |  "
            f"♩ {bpm:.0f} BPM  |  Beat {beat}  |  {shader_nm}  |  "
            f"TL: {'ON' if self._tl_visible else 'off'}  [Tab=HUD  T=TL  L=Learn]"
        )
        self._overlay.adjustSize()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _toggle_hud(self):
        self._show_hud = not self._show_hud

    def _toggle_timeline_overlay(self):
        self._tl_visible = not self._tl_visible
        if self._tl_overlay:
            self._tl_overlay.set_visible(self._tl_visible)

    def _start_learn(self):
        """Démarre le mode XR Learn sur l'uniform 'uLearnTarget'."""
        from PyQt6.QtWidgets import QInputDialog
        uniform, ok = QInputDialog.getText(
            self, "XR Learn", "Uniform cible (ex: uSpeed) :"
        )
        if ok and uniform.strip():
            self._ctrl.start_learn(uniform.strip())
            self._overlay.setText(f"XR Learn actif → '{uniform.strip()}'  "
                                  "Appuyez sur un bouton/axe contrôleur…")
            self._overlay.adjustSize()
            self._overlay.show()

    def _on_learn_triggered(self, xr_input: str):
        self._overlay.setText(f"✓ XR Learn : '{xr_input}' mappé")
        self._overlay.adjustSize()
        self._overlay.show()
        QTimer.singleShot(3000, self._overlay.hide)

    def _next_shader(self):
        if self._pw:
            self._pw._load_next_shader()

    def _prev_shader(self):
        if self._pw:
            self._pw._load_prev_shader()

    def _jump_to_shader(self, idx: int):
        if self._pw and hasattr(self._pw, '_vj_shader_paths'):
            paths = self._pw._vj_shader_paths
            if 0 <= idx < len(paths):
                self._pw._load_shader_file(paths[idx])

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._frame_timer.stop()
        if self._tl_overlay:
            self._tl_overlay.cleanup()
        if self._vr_engine:
            try:
                self._pw.gl_widget.makeCurrent()
                self._vr_engine.cleanup()
                self._pw.gl_widget.doneCurrent()
            except Exception:
                pass
        self._xr.shutdown()
        if self._pw:
            self._pw._vr_window = None
        log.info("VRWindow fermée")
        super().closeEvent(event)


import os  # noqa — nécessaire pour os.path.basename dans _refresh_hud
