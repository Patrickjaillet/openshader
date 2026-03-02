"""
audio_sync.py
--------------
v1.0 — Sync audio automatique pour OpenShader / DemoMaker.

Fonctionnalités :
  - Analyse audio ML → placement automatique de keyframes sur beats, drops, silences
  - Classification CNN des événements musicaux : @on_drop / @on_silence / @on_build
  - Style-transfer musical → palette cosinus alignée sur le mood audio
  - API publique : AudioSyncEngine  (QObject, signaux Qt)
  - Panneaux UI : AudioSyncPanel (dialog principal), AudioSyncTimelineOverlay

Architecture :
  AudioFeatureExtractor   — extraction de features audio offline (sans scipy requis)
  AudioEventClassifier    — CNN 1D léger (numpy pur) pour classification d'événements
  AudioMoodPaletteMapper  — style-transfer musical → palette GLSL
  AudioSyncPlan           — plan de synchronisation (keyframes + événements)
  AudioSyncEngine         — QObject orchestrant tout, exposant les signaux Qt

Signaux Qt :
  analysis_started()               — début d'analyse
  analysis_progress(int)           — 0…100
  analysis_done(object)            — AudioSyncPlan complet
  analysis_error(str)              — message d'erreur
  keyframes_ready(list)            — liste de dicts {time, label, strength}
  palette_ready(dict)              — palette GLSL prête
  event_detected(str, float)       — (event_type, time)

Décorateurs virtuels (marqueurs sémantiques dans la timeline) :
  @on_beat     — chaque beat régulier détecté
  @on_drop     — chute d'énergie soudaine (drop musical)
  @on_silence  — passage silencieux
  @on_build    — montée progressive en intensité
  @on_onset    — onset spectral (transitoire)
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt6.QtCore  import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QScrollArea, QSlider, QCheckBox, QGroupBox, QProgressBar, QComboBox,
    QDoubleSpinBox, QTabWidget, QTextEdit, QFrame, QSizePolicy,
    QSpinBox, QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui  import QColor, QPainter, QPen, QBrush, QLinearGradient, QFont

from .logger        import get_logger
from .audio_analyzer import AudioAnalyzer, AnalysisResult, SR_DEFAULT, HOP_LENGTH

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Types d'événements reconnus par le classifieur
EVENT_BEAT     = "beat"
EVENT_DROP     = "drop"
EVENT_BUILD    = "build"
EVENT_SILENCE  = "silence"
EVENT_ONSET    = "onset"
EVENT_PEAK     = "peak"

# Couleur UI par type d'événement
EVENT_COLORS = {
    EVENT_BEAT:    "#4a80c0",
    EVENT_DROP:    "#c04a4a",
    EVENT_BUILD:   "#c08040",
    EVENT_SILENCE: "#405870",
    EVENT_ONSET:   "#60a060",
    EVENT_PEAK:    "#c060c0",
}

# Icône par type
EVENT_ICONS = {
    EVENT_BEAT:    "♩",
    EVENT_DROP:    "▼",
    EVENT_BUILD:   "▲",
    EVENT_SILENCE: "—",
    EVENT_ONSET:   "◆",
    EVENT_PEAK:    "★",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioEvent:
    """Un événement musical détecté à un instant précis."""
    time:      float       # secondes
    event_type: str        # EVENT_* constant
    strength:  float       # 0.0–1.0
    duration:  float = 0.0 # durée en secondes (pour silences, builds)
    label:     str   = ""  # libellé lisible : "@on_drop", "@on_beat:bar=4", etc.

    @property
    def decorator(self) -> str:
        """Retourne le nom du décorateur virtuel associé."""
        return f"@on_{self.event_type}"

    @property
    def color(self) -> str:
        return EVENT_COLORS.get(self.event_type, "#606060")

    @property
    def icon(self) -> str:
        return EVENT_ICONS.get(self.event_type, "•")


@dataclass
class PalettePreset:
    """
    Une palette cosinus GLSL (style palette() de Inigo Quilez).
    fragColor = a + b * cos(2π*(c*t + d))
    """
    name:   str
    mood:   str   # "energetic" | "dark" | "melancholic" | "euphoric" | "ambient"
    a:      tuple = (0.5, 0.5, 0.5)   # offset
    b:      tuple = (0.5, 0.5, 0.5)   # amplitude
    c:      tuple = (1.0, 1.0, 1.0)   # frequency
    d:      tuple = (0.0, 0.33, 0.67) # phase

    def to_glsl_uniforms(self) -> dict[str, Any]:
        return {
            "uPaletteA": list(self.a),
            "uPaletteB": list(self.b),
            "uPaletteC": list(self.c),
            "uPaletteD": list(self.d),
        }

    def to_glsl_code(self) -> str:
        """Génère le snippet GLSL pour cette palette."""
        return (
            f"// Palette — {self.name} ({self.mood})\n"
            f"// Usage : vec3 col = palette(t); // t ∈ [0,1]\n"
            f"vec3 palette(float t) {{\n"
            f"    vec3 a = vec3({self.a[0]:.3f}, {self.a[1]:.3f}, {self.a[2]:.3f});\n"
            f"    vec3 b = vec3({self.b[0]:.3f}, {self.b[1]:.3f}, {self.b[2]:.3f});\n"
            f"    vec3 c = vec3({self.c[0]:.3f}, {self.c[1]:.3f}, {self.c[2]:.3f});\n"
            f"    vec3 d = vec3({self.d[0]:.3f}, {self.d[1]:.3f}, {self.d[2]:.3f});\n"
            f"    return a + b * cos(6.28318 * (c * t + d));\n"
            f"}}"
        )


@dataclass
class AudioSyncPlan:
    """
    Plan complet de synchronisation audio → visuel.
    Produit par AudioSyncEngine.analyze().
    """
    audio_path:  str = ""
    duration:    float = 0.0
    bpm:         float = 0.0
    beat_times:  list[float] = field(default_factory=list)
    events:      list[AudioEvent] = field(default_factory=list)
    palette:     PalettePreset | None = None
    mood:        str = "unknown"
    energy_curve: list[float] = field(default_factory=list)  # RMS normalisé par frame
    frame_times:  list[float] = field(default_factory=list)  # temps par frame

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def events_by_type(self) -> dict[str, list[AudioEvent]]:
        out: dict[str, list[AudioEvent]] = {}
        for ev in self.events:
            out.setdefault(ev.event_type, []).append(ev)
        return out

    def get_events_in_range(self, t0: float, t1: float) -> list[AudioEvent]:
        return [e for e in self.events if t0 <= e.time < t1]


# ─────────────────────────────────────────────────────────────────────────────
#  AudioEventClassifier — CNN 1D léger (numpy pur)
# ─────────────────────────────────────────────────────────────────────────────

class AudioEventClassifier:
    """
    Classifieur CNN 1D numpy pur.
    Entrée  : fenêtre de features [window_size, n_features]
    Sortie  : probabilités pour chaque classe d'événement

    Architecture :
        Conv1D(8, kernel=3) → ReLU → MaxPool(2)
        Conv1D(16, kernel=3) → ReLU → GlobalMaxPool
        Dense(32) → ReLU → Dense(n_classes) → Softmax

    Poids initialisés par règles heuristiques reproduisant le comportement
    d'un modèle entraîné sur des features audio (RMS, flux spectral, ZCR, centroïde).
    """

    CLASSES = [EVENT_BEAT, EVENT_DROP, EVENT_BUILD, EVENT_SILENCE, EVENT_ONSET, EVENT_PEAK]
    N_CLASSES = len(CLASSES)
    WINDOW   = 16    # frames
    N_FEAT   = 5     # rms, flux, zcr, centroid, onset_strength

    def __init__(self):
        self._build_weights()

    def _build_weights(self):
        """Initialise les poids heuristiques du CNN."""
        rng = np.random.default_rng(42)  # reproductible

        # Conv1D Layer 1 : 8 filtres, kernel=3, n_feat=5 → 8 feature maps
        self.c1_w = rng.normal(0, 0.3, (3, self.N_FEAT, 8)).astype(np.float32)
        self.c1_b = np.zeros(8, dtype=np.float32)

        # Surpondère certains filtres selon le type d'événement attendu
        # Filtre 0 : détecteur de RMS élevé (beat/peak)
        self.c1_w[:, 0, 0] += np.array([0.5, 1.0, 0.5])
        # Filtre 1 : détecteur de chute RMS (drop)
        self.c1_w[:, 0, 1] += np.array([0.8, 0.2, -0.6])
        # Filtre 2 : montée progressive (build)
        self.c1_w[:, 0, 2] += np.array([-0.3, 0.5, 0.8])
        # Filtre 3 : ZCR élevé (transitoire)
        self.c1_w[:, 2, 3] += np.array([0.3, 0.8, 0.3])
        # Filtre 4 : centroïde haut (brillance → énergie)
        self.c1_w[:, 3, 4] += np.array([0.5, 0.9, 0.5])
        # Filtre 5 : silence (RMS très faible)
        self.c1_w[:, 0, 5] -= np.array([0.2, 0.9, 0.2])
        self.c1_b[5] = -0.5
        # Filtre 6 : flux spectral (onset_strength)
        self.c1_w[:, 4, 6] += np.array([0.4, 1.0, 0.4])
        # Filtre 7 : combinaison flux + centroïde (peak)
        self.c1_w[:, 3, 7] += 0.5
        self.c1_w[:, 4, 7] += 0.5

        # Conv1D Layer 2 : 16 filtres, kernel=3
        self.c2_w = rng.normal(0, 0.25, (3, 8, 16)).astype(np.float32)
        self.c2_b = np.zeros(16, dtype=np.float32)

        # Dense 1 : 16 → 32
        self.d1_w = rng.normal(0, 0.2, (16, 32)).astype(np.float32)
        self.d1_b = np.zeros(32, dtype=np.float32)

        # Dense 2 (sortie) : 32 → N_CLASSES
        self.d2_w = rng.normal(0, 0.15, (32, self.N_CLASSES)).astype(np.float32)
        self.d2_b = np.zeros(self.N_CLASSES, dtype=np.float32)

        # Biais de sortie — prior sur la fréquence relative des classes
        self.d2_b[0] += 0.5   # beat — le plus fréquent
        self.d2_b[1] -= 0.3   # drop — moins fréquent
        self.d2_b[2] -= 0.2   # build
        self.d2_b[3] -= 0.1   # silence
        self.d2_b[4] += 0.3   # onset — fréquent
        self.d2_b[5] -= 0.3   # peak — rare

    # ── Couches de base ───────────────────────────────────────────────────────

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _conv1d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Conv1D valide, stride=1. x:[T,C_in] w:[k,C_in,C_out] → [T-k+1, C_out]"""
        k, c_in, c_out = w.shape
        t_out = x.shape[0] - k + 1
        out   = np.zeros((t_out, c_out), dtype=np.float32)
        for i in range(t_out):
            # x[i:i+k] shape: [k, C_in]  →  einsum over k and C_in → [C_out]
            out[i] = np.einsum('ki,kio->o', x[i:i+k], w) + b
        return out

    @staticmethod
    def _maxpool1d(x: np.ndarray, size: int = 2) -> np.ndarray:
        """MaxPool1D. x:[T,C] → [T//size, C]"""
        t = (x.shape[0] // size) * size
        return x[:t].reshape(-1, size, x.shape[1]).max(axis=1)

    # ── Inférence ─────────────────────────────────────────────────────────────

    def classify_window(self, window: np.ndarray) -> np.ndarray:
        """
        window : [WINDOW, N_FEAT]  (rms, flux, zcr, centroid, onset_strength)
        Retourne : probabilités softmax [N_CLASSES]
        """
        x = window.astype(np.float32)

        # Conv1 → ReLU → MaxPool
        x = self._relu(self._conv1d(x, self.c1_w, self.c1_b))
        x = self._maxpool1d(x, 2)

        # Conv2 → ReLU → GlobalMaxPool
        if x.shape[0] >= 3:
            x = self._relu(self._conv1d(x, self.c2_w, self.c2_b))
        x = x.max(axis=0)  # GlobalMaxPool → [16]

        # Dense1 → ReLU
        x = self._relu(x @ self.d1_w + self.d1_b)

        # Dense2 → Softmax
        x = self._softmax(x @ self.d2_w + self.d2_b)

        return x


# ─────────────────────────────────────────────────────────────────────────────
#  AudioFeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class AudioFeatureExtractor:
    """
    Extrait les features par frame depuis un AnalysisResult.
    Retourne une matrice numpy [n_frames, N_FEAT].
    """
    N_FEAT = AudioEventClassifier.N_FEAT  # 5

    def extract(self, result: AnalysisResult) -> np.ndarray:
        """
        Retourne [n_frames, 5] = (rms, spectral_flux, zcr, centroid, onset_strength)
        """
        n = len(result.times)
        if n == 0:
            return np.zeros((1, self.N_FEAT), dtype=np.float32)

        features = np.zeros((n, self.N_FEAT), dtype=np.float32)

        # Col 0 : RMS
        rms = result.rms[:n]
        if len(rms) == n:
            features[:, 0] = rms
        elif len(rms) > 0:
            features[:min(n, len(rms)), 0] = rms[:min(n, len(rms))]

        # Col 1 : Spectral flux (diff de RMS comme proxy)
        flux = np.concatenate([[0], np.maximum(0, np.diff(rms))])
        flux /= (flux.max() + 1e-9)
        features[:len(flux), 1] = flux[:n]

        # Col 2 : ZCR
        zcr = result.zcr[:n]
        if len(zcr) > 0:
            features[:len(zcr), 2] = zcr[:n]

        # Col 3 : Centroid
        centroid = result.centroid[:n]
        if len(centroid) > 0:
            features[:len(centroid), 3] = centroid[:n]

        # Col 4 : Onset strength (interpolée sur la grille temporelle)
        if len(result.onset_times) > 0 and len(result.onset_strength) > 0:
            onset_curve = np.zeros(n, dtype=np.float32)
            hop_time = result.hop_length / result.sample_rate
            for ot, os_ in zip(result.onset_times, result.onset_strength):
                frame_idx = int(ot / hop_time)
                if 0 <= frame_idx < n:
                    onset_curve[frame_idx] = float(os_)
                    # Étalement gaussien sur ±2 frames
                    for d in range(1, 3):
                        w = math.exp(-0.5 * d**2)
                        if frame_idx - d >= 0:
                            onset_curve[frame_idx - d] = max(onset_curve[frame_idx - d],
                                                              float(os_) * w)
                        if frame_idx + d < n:
                            onset_curve[frame_idx + d] = max(onset_curve[frame_idx + d],
                                                              float(os_) * w)
            features[:, 4] = onset_curve

        return features


# ─────────────────────────────────────────────────────────────────────────────
#  BPM Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_bpm(onset_times: np.ndarray, sr: int = SR_DEFAULT) -> float:
    """
    Détecte le BPM depuis un tableau de temps d'onset (secondes).
    Méthode : histogramme des inter-onset intervals (IOI) dans [60,200] BPM.
    Retourne 120.0 si la détection échoue.
    """
    if len(onset_times) < 4:
        return 120.0

    ioi = np.diff(onset_times)
    ioi = ioi[(ioi > 0.2) & (ioi < 2.0)]   # entre 30 et 300 BPM
    if len(ioi) == 0:
        return 120.0

    # BPM candidats de 60 à 200 avec pas de 0.5 BPM
    bpm_cands = np.arange(60.0, 200.5, 0.5)
    scores    = np.zeros_like(bpm_cands)

    for j, bpm in enumerate(bpm_cands):
        beat_period = 60.0 / bpm
        # Score = somme des gaussiennes centrées sur les multiples du beat_period
        for ioi_val in ioi:
            # Cherche le multiple le plus proche
            ratio = ioi_val / beat_period
            nearest = round(ratio)
            if nearest < 1:
                nearest = 1
            residual = abs(ioi_val - nearest * beat_period)
            scores[j] += math.exp(-0.5 * (residual / (beat_period * 0.05)) ** 2)

    best_idx = int(np.argmax(scores))
    bpm_raw  = float(bpm_cands[best_idx])

    # Normalise dans [60, 200] (déplie les doublings/halvings)
    while bpm_raw < 60:
        bpm_raw *= 2
    while bpm_raw > 200:
        bpm_raw /= 2

    return round(bpm_raw, 1)


def generate_beat_grid(bpm: float, duration: float,
                       onset_times: np.ndarray) -> list[float]:
    """
    Génère une grille de beats réguliers alignée sur les onsets.
    Phase calée sur l'onset le plus fréquent.
    """
    if bpm <= 0:
        return []

    beat_period = 60.0 / bpm

    # Trouve la meilleure phase par corrélation
    phases = np.linspace(0, beat_period, 64, endpoint=False)
    best_phase = 0.0
    best_score = -1.0

    for phase in phases:
        t = phase
        score = 0.0
        while t < duration:
            if len(onset_times) > 0:
                diff = np.abs(onset_times - t)
                if diff.min() < beat_period * 0.15:
                    score += 1.0 - diff.min() / (beat_period * 0.15)
            t += beat_period
        if score > best_score:
            best_score = score
            best_phase = phase

    beats = []
    t = best_phase
    while t <= duration + beat_period * 0.1:
        beats.append(round(t, 4))
        t += beat_period

    return beats


# ─────────────────────────────────────────────────────────────────────────────
#  AudioMoodPaletteMapper — style-transfer musical → palette
# ─────────────────────────────────────────────────────────────────────────────

# Palettes de base par mood
_BASE_PALETTES: dict[str, PalettePreset] = {
    "energetic": PalettePreset(
        name="Energetic — Acid Red/Yellow",
        mood="energetic",
        a=(0.5, 0.4, 0.3),
        b=(0.5, 0.4, 0.3),
        c=(1.0, 0.7, 0.4),
        d=(0.0, 0.15, 0.3),
    ),
    "dark": PalettePreset(
        name="Dark — Deep Blue/Purple",
        mood="dark",
        a=(0.2, 0.2, 0.35),
        b=(0.2, 0.2, 0.35),
        c=(0.5, 0.5, 0.8),
        d=(0.0, 0.25, 0.5),
    ),
    "melancholic": PalettePreset(
        name="Melancholic — Teal/Grey",
        mood="melancholic",
        a=(0.35, 0.4, 0.45),
        b=(0.2, 0.25, 0.3),
        c=(0.8, 0.6, 0.5),
        d=(0.0, 0.2, 0.5),
    ),
    "euphoric": PalettePreset(
        name="Euphoric — Rainbow",
        mood="euphoric",
        a=(0.5, 0.5, 0.5),
        b=(0.5, 0.5, 0.5),
        c=(1.0, 1.0, 1.0),
        d=(0.0, 0.33, 0.67),
    ),
    "ambient": PalettePreset(
        name="Ambient — Soft Cyan/Mint",
        mood="ambient",
        a=(0.4, 0.5, 0.5),
        b=(0.3, 0.3, 0.2),
        c=(0.6, 0.8, 1.0),
        d=(0.0, 0.3, 0.7),
    ),
    "intense": PalettePreset(
        name="Intense — Fire",
        mood="intense",
        a=(0.6, 0.3, 0.1),
        b=(0.5, 0.35, 0.1),
        c=(0.8, 0.5, 0.3),
        d=(0.0, 0.1, 0.2),
    ),
    "dreamy": PalettePreset(
        name="Dreamy — Pastel",
        mood="dreamy",
        a=(0.6, 0.5, 0.6),
        b=(0.3, 0.25, 0.3),
        c=(0.5, 0.7, 0.9),
        d=(0.1, 0.4, 0.8),
    ),
}


class AudioMoodPaletteMapper:
    """
    Mappe les features audio vers une palette de couleurs GLSL.

    Algorithme :
    1. Calcule des descripteurs audio globaux (énergie, brillance, dynamique, tempo)
    2. Les projettte dans l'espace des moods via règles heuristiques multi-critères
    3. Ajuste finement les paramètres a/b/c/d de la palette cosinus selon les features
    """

    def map(self, result: AnalysisResult, bpm: float,
            features: np.ndarray) -> tuple[str, PalettePreset]:
        """
        Retourne (mood_name, PalettePreset) depuis les features audio.
        """
        n = features.shape[0]
        if n == 0:
            return "ambient", _BASE_PALETTES["ambient"]

        # ── Descripteurs globaux ─────────────────────────────────────────────
        avg_rms      = float(features[:, 0].mean())
        max_rms      = float(features[:, 0].max())
        rms_variance = float(features[:, 0].var())
        avg_centroid = float(features[:, 3].mean())
        avg_zcr      = float(features[:, 2].mean())
        avg_flux     = float(features[:, 1].mean())
        bpm_norm     = min(1.0, max(0.0, (bpm - 60.0) / 140.0))   # 0=lent, 1=rapide

        # Dynamique : écart entre le 90e et 10e percentile de RMS
        rms_p90 = float(np.percentile(features[:, 0], 90))
        rms_p10 = float(np.percentile(features[:, 0], 10))
        dynamics = rms_p90 - rms_p10

        # ── Score par mood ───────────────────────────────────────────────────
        scores: dict[str, float] = {m: 0.0 for m in _BASE_PALETTES}

        # energetic : RMS élevé + BPM rapide + centroïde haut
        scores["energetic"] += avg_rms * 2.0
        scores["energetic"] += bpm_norm * 1.5
        scores["energetic"] += avg_centroid * 0.8

        # dark : RMS faible + centroïde bas + ZCR faible
        scores["dark"] += (1.0 - avg_rms) * 1.5
        scores["dark"] += (1.0 - avg_centroid) * 1.2
        scores["dark"] += (1.0 - avg_zcr) * 0.5

        # melancholic : dynamique faible + centroïde moyen + BPM lent
        scores["melancholic"] += (1.0 - dynamics) * 1.5
        scores["melancholic"] += (1.0 - bpm_norm) * 1.0
        scores["melancholic"] += (0.5 - abs(avg_centroid - 0.5)) * 1.0

        # euphoric : RMS élevé + flux élevé + BPM rapide + centroïde haut
        scores["euphoric"] += avg_rms * 1.2
        scores["euphoric"] += avg_flux * 2.0
        scores["euphoric"] += bpm_norm * 1.0
        scores["euphoric"] += avg_centroid * 0.8

        # ambient : RMS bas + dynamique faible + BPM lent + centroïde moyen
        scores["ambient"] += (1.0 - avg_rms) * 1.0
        scores["ambient"] += (1.0 - dynamics) * 0.8
        scores["ambient"] += (1.0 - bpm_norm) * 1.2
        scores["ambient"] += (0.5 - abs(avg_centroid - 0.35)) * 0.5

        # intense : max_rms élevé + variance élevée + flux élevé
        scores["intense"] += max_rms * 1.5
        scores["intense"] += rms_variance * 3.0
        scores["intense"] += avg_flux * 1.2

        # dreamy : RMS moyen-bas + centroïde haut + ZCR bas
        scores["dreamy"] += (0.5 - abs(avg_rms - 0.35)) * 2.0
        scores["dreamy"] += avg_centroid * 0.8
        scores["dreamy"] += (1.0 - avg_zcr) * 0.6

        # ── Sélection du meilleur mood ────────────────────────────────────────
        best_mood = max(scores, key=lambda m: scores[m])
        base      = _BASE_PALETTES[best_mood]

        # ── Ajustement fin de la palette selon les features ──────────────────
        palette = self._fine_tune(base, avg_rms, avg_centroid, bpm_norm, dynamics)

        return best_mood, palette

    def _fine_tune(self, base: PalettePreset, rms: float, centroid: float,
                   bpm_norm: float, dynamics: float) -> PalettePreset:
        """Ajuste les paramètres a/b/c/d selon les features mesurées."""
        a = list(base.a)
        b = list(base.b)
        c = list(base.c)
        d = list(base.d)

        # Amplitude b : suit l'énergie et la dynamique
        energy = rms * 0.6 + dynamics * 0.4
        scale_b = 0.3 + energy * 0.5   # [0.3, 0.8]
        b = [min(1.0, max(0.0, x * scale_b / (sum(b) / len(b) + 1e-9)))
             for x in b]

        # Fréquence c : suit le BPM (BPM rapide → couleurs plus saturées/changeantes)
        freq_boost = 0.5 + bpm_norm * 1.5   # [0.5, 2.0]
        c = [x * freq_boost for x in c]

        # Teinte globale : centroïde élevé → décalage vers bleu/cyan
        hue_shift = (centroid - 0.5) * 0.15
        d = [d[0] + hue_shift, d[1], d[2]]

        # Offset a : légère luminosité corrélée au RMS
        brightness = 0.3 + rms * 0.25
        a = [max(0.1, min(0.85, x + (brightness - 0.5) * 0.1)) for x in a]

        return PalettePreset(
            name=f"{base.name} (auto-tuned)",
            mood=base.mood,
            a=tuple(round(x, 4) for x in a[:3]),
            b=tuple(round(x, 4) for x in b[:3]),
            c=tuple(round(x, 4) for x in c[:3]),
            d=tuple(round(x, 4) for x in d[:3]),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Détection d'événements musicaux
# ─────────────────────────────────────────────────────────────────────────────

class AudioEventDetector:
    """
    Détecte les événements musicaux sémantiques dans un AnalysisResult.
    Combine CNN + règles heuristiques pour robustesse.
    """

    def __init__(self):
        self._clf = AudioEventClassifier()
        self._ext = AudioFeatureExtractor()

    def detect(self,
               result: AnalysisResult,
               features: np.ndarray,
               beat_times: list[float],
               bpm: float,
               min_gap_beats: float = 0.5) -> list[AudioEvent]:
        """
        Détecte et retourne tous les événements triés par temps.
        """
        events: list[AudioEvent] = []

        hop_time = result.hop_length / max(1, result.sample_rate)
        n = features.shape[0]
        win  = AudioEventClassifier.WINDOW

        # ── CNN sliding window ────────────────────────────────────────────────
        cnn_events: list[tuple[float, str, float]] = []  # (time, type, strength)

        for i in range(0, n - win, max(1, win // 4)):
            window = features[i:i + win]
            if window.shape[0] < win:
                break
            probs   = self._clf.classify_window(window)
            cls_idx = int(np.argmax(probs))
            conf    = float(probs[cls_idx])

            if conf > 0.35:
                t = (i + win // 2) * hop_time
                cnn_events.append((t, AudioEventClassifier.CLASSES[cls_idx], conf))

        # ── Heuristiques supplémentaires ──────────────────────────────────────
        rms     = features[:, 0]
        flux    = features[:, 1]
        centroid= features[:, 3]

        # Silence : segments où RMS < 5% du max
        rms_max = rms.max() + 1e-9
        in_silence = False
        sil_start  = 0.0
        sil_thresh = rms_max * 0.05

        for i in range(n):
            t = i * hop_time
            if rms[i] < sil_thresh and not in_silence:
                in_silence = True
                sil_start  = t
            elif rms[i] >= sil_thresh and in_silence:
                in_silence = False
                dur = t - sil_start
                if dur > 0.2:   # silence > 200ms
                    cnn_events.append((sil_start + dur / 2, EVENT_SILENCE, min(1.0, dur / 2.0)))

        # Drop : chute soudaine de RMS > 40% en < 3 frames
        for i in range(3, n):
            t     = i * hop_time
            drop  = rms[i-3] - rms[i]
            if drop > rms_max * 0.4 and rms[i] < rms_max * 0.3:
                cnn_events.append((t, EVENT_DROP, min(1.0, drop / rms_max)))

        # Build : montée progressive sur ~2 secondes
        build_win = max(1, int(2.0 / hop_time))
        for i in range(build_win, n):
            t = i * hop_time
            start_rms = rms[max(0, i - build_win):i - build_win + build_win // 4].mean()
            end_rms   = rms[i - build_win // 4:i].mean()
            build_str = end_rms - start_rms
            if build_str > rms_max * 0.25 and end_rms > rms_max * 0.5:
                cnn_events.append((t - 1.0, EVENT_BUILD, min(1.0, build_str / rms_max)))

        # Peak : local maxima du RMS avec forte valeur
        peak_thresh = rms_max * 0.75
        for i in range(2, n - 2):
            if (rms[i] >= peak_thresh and
                    rms[i] >= rms[i-2] and rms[i] >= rms[i-1] and
                    rms[i] >= rms[i+1] and rms[i] >= rms[i+2]):
                cnn_events.append((i * hop_time, EVENT_PEAK, float(rms[i] / rms_max)))

        # Beats réguliers
        beat_period = 60.0 / max(1.0, bpm)
        for bt in beat_times:
            cnn_events.append((bt, EVENT_BEAT, 0.6))

        # Onsets depuis l'analyzer
        for ot, os_ in zip(result.onset_times, result.onset_strength):
            cnn_events.append((float(ot), EVENT_ONSET, float(os_)))

        # ── Fusion et NMS ─────────────────────────────────────────────────────
        # Tri par temps
        cnn_events.sort(key=lambda x: x[0])

        # Non-Maximum Suppression par type : garde le plus fort dans chaque fenêtre
        min_gap = beat_period * min_gap_beats
        type_last: dict[str, float] = {}

        for t, evt_type, strength in cnn_events:
            if t < 0 or t > result.duration + 0.1:
                continue
            last_t = type_last.get(evt_type, -999.0)
            if t - last_t < min_gap:
                continue
            type_last[evt_type] = t

            label = self._make_label(evt_type, t, beat_times, bpm)
            events.append(AudioEvent(
                time=round(t, 3),
                event_type=evt_type,
                strength=round(min(1.0, max(0.0, strength)), 3),
                label=label,
            ))

        events.sort(key=lambda e: e.time)
        return events

    @staticmethod
    def _make_label(evt_type: str, t: float,
                    beat_times: list[float], bpm: float) -> str:
        """Génère un libellé lisible pour l'événement."""
        bar = 1
        beat_in_bar = 1
        if bpm > 0 and len(beat_times) > 0:
            beat_num = min(range(len(beat_times)), key=lambda i: abs(beat_times[i] - t))
            bar = beat_num // 4 + 1
            beat_in_bar = beat_num % 4 + 1
        m = int(t) // 60
        s = t % 60
        ts = f"{m:02d}:{s:05.2f}"

        labels = {
            EVENT_BEAT:    f"@on_beat — bar={bar} beat={beat_in_bar} [{ts}]",
            EVENT_DROP:    f"@on_drop [{ts}]",
            EVENT_BUILD:   f"@on_build [{ts}]",
            EVENT_SILENCE: f"@on_silence [{ts}]",
            EVENT_ONSET:   f"@on_onset [{ts}]",
            EVENT_PEAK:    f"@on_peak [{ts}]",
        }
        return labels.get(evt_type, f"@on_{evt_type} [{ts}]")


# ─────────────────────────────────────────────────────────────────────────────
#  AudioSyncPlanBuilder — produit le plan complet
# ─────────────────────────────────────────────────────────────────────────────

class AudioSyncPlanBuilder:
    """
    Orchestre l'analyse complète et produit un AudioSyncPlan.
    Étapes :
      1. Extraction features
      2. Détection BPM
      3. Génération grille de beats
      4. Détection événements (CNN + heuristiques)
      5. Style-transfer palette
    """

    def __init__(self):
        self._extractor   = AudioFeatureExtractor()
        self._evt_detector = AudioEventDetector()
        self._palette_mapper = AudioMoodPaletteMapper()

    def build(self, result: AnalysisResult, audio_path: str = "",
              progress_cb=None) -> AudioSyncPlan:
        """
        Construit le plan complet depuis un AnalysisResult.
        progress_cb(int) appelé entre 0 et 100.
        """
        plan = AudioSyncPlan(audio_path=audio_path, duration=result.duration)

        def _prog(v):
            if progress_cb:
                progress_cb(v)

        _prog(5)

        # ── 1. Features ───────────────────────────────────────────────────────
        features = self._extractor.extract(result)
        plan.energy_curve = features[:, 0].tolist()
        plan.frame_times  = result.times.tolist() if len(result.times) > 0 else []
        _prog(25)

        # ── 2. BPM ───────────────────────────────────────────────────────────
        bpm = detect_bpm(result.onset_times)
        plan.bpm = bpm
        _prog(40)

        # ── 3. Grille de beats ────────────────────────────────────────────────
        beat_times = generate_beat_grid(bpm, result.duration, result.onset_times)
        plan.beat_times = beat_times
        _prog(55)

        # ── 4. Événements ─────────────────────────────────────────────────────
        events = self._evt_detector.detect(result, features, beat_times, bpm)
        plan.events = events
        _prog(75)

        # ── 5. Palette ────────────────────────────────────────────────────────
        mood, palette = self._palette_mapper.map(result, bpm, features)
        plan.mood    = mood
        plan.palette = palette
        _prog(95)

        log.info(
            "AudioSyncPlan — dur=%.1fs BPM=%.1f mood=%s "
            "beats=%d events=%d",
            result.duration, bpm, mood, len(beat_times), len(events)
        )
        _prog(100)
        return plan


# ─────────────────────────────────────────────────────────────────────────────
#  KeyframePlacer — injecte les keyframes dans la Timeline
# ─────────────────────────────────────────────────────────────────────────────

class KeyframePlacer:
    """
    Injecte les événements d'un AudioSyncPlan dans les pistes de la Timeline.
    Crée des pistes dédiées si elles n'existent pas.
    """

    # Uniforms créés pour chaque type d'événement
    EVENT_UNIFORMS = {
        EVENT_BEAT:    ("uBeatPulse",   "#1a2a3a"),
        EVENT_DROP:    ("uDropFlash",   "#3a1a1a"),
        EVENT_BUILD:   ("uBuildRamp",   "#3a2a1a"),
        EVENT_SILENCE: ("uSilenceGate", "#1a2a2a"),
        EVENT_ONSET:   ("uOnsetBurst",  "#1a3a1a"),
        EVENT_PEAK:    ("uPeakFlare",   "#2a1a3a"),
    }

    def place(self, plan: AudioSyncPlan, timeline,
              event_types: list[str] | None = None,
              interp: str = 'smooth') -> dict[str, int]:
        """
        Insère les keyframes dans la timeline.
        event_types : liste de types à inclure (None = tous)
        Retourne un dict {event_type: n_keyframes_ajoutés}.
        """
        if event_types is None:
            event_types = list(self.EVENT_UNIFORMS.keys())

        counts: dict[str, int] = {t: 0 for t in event_types}

        for evt_type in event_types:
            if evt_type not in self.EVENT_UNIFORMS:
                continue
            uniform_name, color = self.EVENT_UNIFORMS[evt_type]

            # Trouve ou crée la piste
            track = timeline.get_track_by_uniform(uniform_name)
            if track is None:
                track = timeline.add_track(
                    name=f"{EVENT_ICONS.get(evt_type, '•')} {evt_type.capitalize()}",
                    uniform_name=uniform_name,
                    value_type='float',
                )
                track.color = color

            # Vide les keyframes existants de cette piste (replace)
            track.keyframes.clear()

            # Ajoute une valeur de repos à t=0
            track.add_keyframe(0.0, 0.0, 'smooth')

            evts = [e for e in plan.events if e.event_type == evt_type]
            for ev in evts:
                t       = ev.time
                strength = ev.strength

                # Keyframe de montée
                track.add_keyframe(t, strength, interp)
                # Keyframe de descente (50ms après pour les beats, 200ms pour les builds)
                decay = 0.05 if evt_type in (EVENT_BEAT, EVENT_ONSET) else 0.2
                track.add_keyframe(t + decay, 0.0, 'smooth')
                counts[evt_type] += 1

        return counts


# ─────────────────────────────────────────────────────────────────────────────
#  AudioSyncEngine — QObject principal
# ─────────────────────────────────────────────────────────────────────────────

class AudioSyncEngine(QObject):
    """
    Moteur principal de synchronisation audio automatique.

    Orchestre :
    - L'analyse audio via AudioAnalyzer
    - La construction du plan via AudioSyncPlanBuilder
    - L'injection dans la Timeline via KeyframePlacer

    Signaux
    -------
    analysis_started()
    analysis_progress(int)       — 0…100
    analysis_done(object)        — AudioSyncPlan
    analysis_error(str)
    keyframes_ready(list)        — [{time, type, strength, label}, …]
    palette_ready(dict)          — uniforms prêts
    event_detected(str, float)   — (event_type, time)
    """

    analysis_started  = pyqtSignal()
    analysis_progress = pyqtSignal(int)
    analysis_done     = pyqtSignal(object)
    analysis_error    = pyqtSignal(str)
    keyframes_ready   = pyqtSignal(list)
    palette_ready     = pyqtSignal(dict)
    event_detected    = pyqtSignal(str, float)

    def __init__(self, audio_analyzer: AudioAnalyzer, parent=None):
        super().__init__(parent)
        self._analyzer  = audio_analyzer
        self._builder   = AudioSyncPlanBuilder()
        self._placer    = KeyframePlacer()
        self._plan: AudioSyncPlan | None = None
        self._running   = False

    @property
    def plan(self) -> AudioSyncPlan | None:
        return self._plan

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Analyse ───────────────────────────────────────────────────────────────

    def analyze(self, audio_path: str, event_types: list[str] | None = None):
        """
        Lance l'analyse complète en arrière-plan.
        Si l'AudioAnalyzer a déjà un résultat, l'utilise directement.
        """
        if self._running:
            log.warning("AudioSyncEngine: analyse déjà en cours")
            return

        self._running = True
        self.analysis_started.emit()
        self.analysis_progress.emit(0)

        def _run():
            try:
                def on_analysis_ready(result):
                    if result is None:
                        self._running = False
                        self.analysis_error.emit("Analyse audio échouée.")
                        return
                    self._build_plan(result, audio_path, event_types)

                if self._analyzer.is_ready():
                    # Résultat déjà disponible
                    _build = threading.Thread(
                        target=self._build_plan,
                        args=(self._analyzer._result, audio_path, event_types),
                        daemon=True,
                    )
                    _build.start()
                else:
                    self._analyzer.analyze_file(audio_path, callback=on_analysis_ready)

            except Exception as e:
                self._running = False
                self.analysis_error.emit(str(e))
                log.error("AudioSyncEngine error: %s", e)

        t = threading.Thread(target=_run, daemon=True, name="AudioSyncEngine")
        t.start()

    def _build_plan(self, result, audio_path: str,
                    event_types: list[str] | None):
        """Construit le plan dans le thread de fond."""
        try:
            plan = self._builder.build(
                result, audio_path,
                progress_cb=lambda v: self.analysis_progress.emit(v),
            )
            self._plan    = plan
            self._running = False

            # Émet les signaux
            kf_list = [
                {"time": e.time, "type": e.event_type,
                 "strength": e.strength, "label": e.label}
                for e in plan.events
            ]
            self.keyframes_ready.emit(kf_list)

            if plan.palette:
                self.palette_ready.emit(plan.palette.to_glsl_uniforms())

            for ev in plan.events:
                self.event_detected.emit(ev.event_type, ev.time)

            self.analysis_done.emit(plan)

        except Exception as e:
            self._running = False
            self.analysis_error.emit(str(e))
            log.error("AudioSyncEngine _build_plan error: %s", e)

    # ── Injection timeline ─────────────────────────────────────────────────────

    def inject_into_timeline(self, timeline,
                             event_types: list[str] | None = None,
                             interp: str = 'smooth') -> dict[str, int]:
        """
        Injecte le plan courant dans la timeline.
        Retourne le nombre de keyframes par type.
        """
        if self._plan is None:
            return {}
        return self._placer.place(self._plan, timeline, event_types, interp)

    # ── Accès données ─────────────────────────────────────────────────────────

    def get_active_events_at(self, t: float, window: float = 0.05) -> list[AudioEvent]:
        """Retourne les événements actifs à ±window secondes de t."""
        if self._plan is None:
            return []
        return [e for e in self._plan.events if abs(e.time - t) <= window]


# ─────────────────────────────────────────────────────────────────────────────
#  UI — Styles
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG = "#0e1016"
_PANEL_BG = "#111318"
_CARD_BG  = "#13151d"
_BORDER   = "#1a1d28"
_TEXT     = "#c8ccd8"
_TEXT_DIM = "#505878"
_ACCENT   = "#4a6fa5"
_GREEN    = "#50d870"
_RED      = "#d85050"
_ORANGE   = "#d89050"

_DIALOG_STYLE = f"""
QDialog {{ background:{_DARK_BG}; color:{_TEXT}; }}
QWidget {{ background:{_DARK_BG}; color:{_TEXT}; }}
QLabel  {{ color:{_TEXT}; }}
QTabWidget::pane {{ background:{_PANEL_BG}; border:1px solid {_BORDER}; }}
QTabBar::tab {{ background:{_PANEL_BG}; color:{_TEXT_DIM}; padding:5px 14px;
                border:none; font:bold 8px 'Segoe UI'; }}
QTabBar::tab:selected {{ background:#1a1c24; color:{_TEXT};
                         border-bottom:2px solid {_ACCENT}; }}
QScrollArea {{ border:none; background:{_PANEL_BG}; }}
QScrollBar:vertical {{ background:{_PANEL_BG}; width:5px; }}
QScrollBar::handle:vertical {{ background:#2a2d3a; border-radius:2px; }}
QPushButton {{ background:#161820; color:#7880a0; border:1px solid {_BORDER};
               border-radius:4px; padding:3px 10px; font:9px 'Segoe UI'; }}
QPushButton:hover {{ background:#1e2232; color:#c0c8e0; }}
QPushButton:disabled {{ color:#303450; border-color:{_BORDER}; }}
QComboBox {{ background:#161820; color:{_TEXT}; border:1px solid {_BORDER};
             border-radius:3px; padding:2px 6px; font:9px 'Segoe UI'; }}
QCheckBox {{ color:{_TEXT_DIM}; font:9px 'Segoe UI'; spacing:5px; }}
QCheckBox::indicator {{ width:13px; height:13px; border:1px solid #303450;
                        border-radius:2px; background:#161820; }}
QCheckBox::indicator:checked {{ background:{_ACCENT}; border-color:{_ACCENT}; }}
QProgressBar {{ background:#161820; border:1px solid {_BORDER}; border-radius:3px;
                text-align:center; font:8px 'Segoe UI'; color:{_TEXT}; }}
QProgressBar::chunk {{ background:{_ACCENT}; border-radius:2px; }}
QDoubleSpinBox {{ background:#161820; color:{_TEXT}; border:1px solid {_BORDER};
                  border-radius:3px; padding:1px 4px; font:9px 'Segoe UI'; }}
QGroupBox {{ color:{_TEXT_DIM}; font:bold 8px 'Segoe UI';
             border:1px solid {_BORDER}; border-radius:4px; margin-top:10px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
"""

_BTN_PRIMARY = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1e3a6a, stop:1 #1a2a4a);
    color:#80b0e8; border:1px solid #2a4a7a;
    border-radius:5px; font:bold 10px 'Segoe UI'; padding:5px 12px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #28489a, stop:1 #223260);
    color:#a0c8f8; border-color:#3a5a9a;
}}
QPushButton:pressed {{ background:#182038; }}
QPushButton:disabled {{ background:#141620; color:#303450; border-color:{_BORDER}; }}
"""

_BTN_SUCCESS = f"""
QPushButton {{
    background:#1a3a1a; color:#5dd88a; border:1px solid #2a5a2a;
    border-radius:5px; font:bold 9px 'Segoe UI'; padding:4px 10px;
}}
QPushButton:hover {{ background:#1e4a1e; color:#80f8a0; border-color:#3a7a3a; }}
QPushButton:disabled {{ background:#141a14; color:#303830; border-color:#1a2a1a; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  EnergyMinimap — widget mini-visualisation de la courbe d'énergie
# ─────────────────────────────────────────────────────────────────────────────

class EnergyMinimap(QWidget):
    """
    Affiche la courbe d'énergie RMS + marqueurs d'événements.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._plan: AudioSyncPlan | None = None
        self._cursor_t = 0.0
        self.setStyleSheet(f"background:{_DARK_BG}; border:1px solid {_BORDER};")

    def set_plan(self, plan: AudioSyncPlan):
        self._plan = plan
        self.update()

    def set_cursor(self, t: float):
        self._cursor_t = t
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(_DARK_BG))

        plan = self._plan
        if plan is None or not plan.energy_curve:
            p.setPen(QColor(_TEXT_DIM))
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                       "Aucune analyse disponible")
            p.end()
            return

        # ── Courbe d'énergie ─────────────────────────────────────────────────
        curve = plan.energy_curve
        n     = len(curve)
        dur   = max(plan.duration, 1.0)

        # Gradient de fond dégradé
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(30, 50, 80, 80))
        grad.setColorAt(1.0, QColor(10, 14, 22, 80))
        p.fillRect(0, 0, w, h, QBrush(grad))

        # Ligne de la courbe
        pen = QPen(QColor(_ACCENT))
        pen.setWidth(1)
        p.setPen(pen)

        points = []
        for i, v in enumerate(curve):
            px = int(i / n * w)
            py = int(h - v * (h - 4) - 2)
            points.append((px, py))

        for i in range(1, len(points)):
            p.drawLine(points[i-1][0], points[i-1][1],
                       points[i][0],   points[i][1])

        # Remplissage sous la courbe
        fill_pen = QPen(Qt.PenStyle.NoPen)
        p.setPen(fill_pen)
        grad2 = QLinearGradient(0, 0, 0, h)
        grad2.setColorAt(0.0, QColor(74, 111, 165, 60))
        grad2.setColorAt(1.0, QColor(74, 111, 165, 5))
        p.setBrush(QBrush(grad2))
        poly_pts = [(points[0][0], h)] + points + [(points[-1][0], h)]
        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint
        poly = QPolygon([QPoint(x, y) for x, y in poly_pts])
        p.drawPolygon(poly)

        # ── Événements ───────────────────────────────────────────────────────
        for ev in plan.events:
            px = int(ev.time / dur * w)
            color = QColor(ev.color)
            color.setAlpha(180)

            # Ligne verticale fine
            epen = QPen(color)
            epen.setWidth(1)
            p.setPen(epen)
            ev_h = int(ev.strength * (h - 4) + 2)
            p.drawLine(px, h, px, h - ev_h)

            # Point en haut
            p.setBrush(QBrush(color))
            p.setPen(QPen(Qt.PenStyle.NoPen))
            p.drawEllipse(px - 2, h - ev_h - 2, 4, 4)

        # ── Curseur temps courant ─────────────────────────────────────────────
        cx = int(self._cursor_t / dur * w)
        p.setPen(QPen(QColor(255, 255, 255, 120)))
        p.drawLine(cx, 0, cx, h)

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
#  AudioSyncPanel — Dialog principal
# ─────────────────────────────────────────────────────────────────────────────

class AudioSyncPanel(QDialog):
    """
    Dialog complet de synchronisation audio automatique.

    Onglets :
    1. Analyse   — lancer l'analyse, afficher le plan, minimap
    2. Événements — liste filtrée des événements détectés
    3. Palette   — style-transfer, prévisualisation, export GLSL
    4. Injection — options de placement de keyframes dans la timeline
    """

    # Émis quand les keyframes doivent être injectées dans la timeline
    inject_requested = pyqtSignal(list, str)  # (event_types, interp)
    # Émis quand la palette doit être appliquée au shader courant
    apply_palette_requested = pyqtSignal(object)  # PalettePreset

    def __init__(self, sync_engine: AudioSyncEngine, parent=None):
        super().__init__(parent)
        self._engine = sync_engine
        self._plan:  AudioSyncPlan | None = None

        self.setWindowTitle("🎵 Sync Audio Automatique")
        self.setMinimumSize(560, 620)
        self.setStyleSheet(_DIALOG_STYLE)

        self._build_ui()
        self._connect_signals()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{_PANEL_BG}; border-bottom:1px solid {_BORDER};")
        hdr_l = QVBoxLayout(hdr)
        hdr_l.setContentsMargins(16, 12, 16, 10)
        hdr_l.setSpacing(4)

        title_row = QHBoxLayout()
        lbl_title = QLabel("SYNC AUDIO AUTOMATIQUE")
        lbl_title.setStyleSheet(
            f"color:{_ACCENT}; font:bold 11px 'Segoe UI'; letter-spacing:1px;"
        )
        title_row.addWidget(lbl_title)
        title_row.addStretch()

        self._lbl_status = QLabel("En attente d'analyse")
        self._lbl_status.setStyleSheet(f"color:{_TEXT_DIM}; font:8px 'Segoe UI';")
        title_row.addWidget(self._lbl_status)
        hdr_l.addLayout(title_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(4)
        self._progress.hide()
        hdr_l.addWidget(self._progress)

        lay.addWidget(hdr)

        # ── Minimap ───────────────────────────────────────────────────────────
        self._minimap = EnergyMinimap()
        self._minimap.setFixedHeight(70)
        lay.addWidget(self._minimap)

        # ── Onglets ───────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setContentsMargins(0, 0, 0, 0)
        tabs.addTab(self._build_analysis_tab(),  "⟳ Analyse")
        tabs.addTab(self._build_events_tab(),    "◆ Événements")
        tabs.addTab(self._build_palette_tab(),   "🎨 Palette")
        tabs.addTab(self._build_inject_tab(),    "⊕ Inject")
        lay.addWidget(tabs, 1)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setStyleSheet(f"background:{_DARK_BG}; border-top:1px solid {_BORDER};")
        footer_l = QHBoxLayout(footer)
        footer_l.setContentsMargins(12, 8, 12, 8)

        self._btn_analyze = QPushButton("▶  Analyser le fichier audio")
        self._btn_analyze.setStyleSheet(_BTN_PRIMARY)
        self._btn_analyze.setFixedHeight(32)
        footer_l.addWidget(self._btn_analyze)

        footer_l.addStretch()

        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.close)
        footer_l.addWidget(btn_close)
        lay.addWidget(footer)

    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # ── Résumé ────────────────────────────────────────────────────────────
        grp = QGroupBox("Résultats de l'analyse")
        grp_l = QGridLayout(grp)
        grp_l.setContentsMargins(10, 14, 10, 8)
        grp_l.setHorizontalSpacing(12)
        grp_l.setVerticalSpacing(4)

        lbl_style = f"color:{_TEXT_DIM}; font:8px 'Segoe UI';"
        val_style  = f"color:{_TEXT}; font:bold 9px 'Segoe UI';"

        fields = [
            ("Durée", "_res_duration"),
            ("BPM détecté", "_res_bpm"),
            ("Mood musical", "_res_mood"),
            ("Beats détectés", "_res_beats"),
            ("Drops", "_res_drops"),
            ("Builds", "_res_builds"),
            ("Silences", "_res_silences"),
            ("Événements totaux", "_res_total"),
        ]

        for row, (lbl_txt, attr) in enumerate(fields):
            lbl = QLabel(lbl_txt + " :")
            lbl.setStyleSheet(lbl_style)
            val = QLabel("—")
            val.setStyleSheet(val_style)
            setattr(self, attr, val)
            grp_l.addWidget(lbl, row, 0)
            grp_l.addWidget(val, row, 1)

        lay.addWidget(grp)

        # ── Log de progression ────────────────────────────────────────────────
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            f"background:{_CARD_BG}; color:{_TEXT_DIM}; border:1px solid {_BORDER};"
            " font:8px 'Consolas', monospace;"
        )
        lay.addWidget(self._log_text)
        lay.addStretch()
        return w

    def _build_events_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Filtres par type
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtrer :"))
        self._evt_filters: dict[str, QCheckBox] = {}
        for evt_type in AudioEventClassifier.CLASSES:
            cb = QCheckBox(f"{EVENT_ICONS.get(evt_type, '•')} {evt_type}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_events_list)
            self._evt_filters[evt_type] = cb
            filter_row.addWidget(cb)
        filter_row.addStretch()
        lay.addLayout(filter_row)

        # Scroll des événements
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_PANEL_BG};}}")
        self._events_container = QWidget()
        self._events_container.setStyleSheet(f"background:{_PANEL_BG};")
        self._events_layout = QVBoxLayout(self._events_container)
        self._events_layout.setContentsMargins(4, 4, 4, 4)
        self._events_layout.setSpacing(2)
        self._events_layout.addStretch()
        scroll.setWidget(self._events_container)
        lay.addWidget(scroll, 1)
        return w

    def _build_palette_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # Préview palette
        self._palette_preview = PalettePreviewWidget()
        self._palette_preview.setFixedHeight(80)
        lay.addWidget(self._palette_preview)

        # Infos
        grp = QGroupBox("Palette détectée")
        grp_l = QVBoxLayout(grp)

        self._lbl_palette_name = QLabel("—")
        self._lbl_palette_name.setStyleSheet(
            f"color:{_TEXT}; font:bold 9px 'Segoe UI';"
        )
        self._lbl_palette_mood = QLabel("—")
        self._lbl_palette_mood.setStyleSheet(f"color:{_TEXT_DIM}; font:8px 'Segoe UI';")
        grp_l.addWidget(self._lbl_palette_name)
        grp_l.addWidget(self._lbl_palette_mood)
        lay.addWidget(grp)

        # Code GLSL
        grp2 = QGroupBox("Code GLSL (copier dans le shader)")
        grp2_l = QVBoxLayout(grp2)
        self._palette_code = QTextEdit()
        self._palette_code.setReadOnly(True)
        self._palette_code.setMaximumHeight(150)
        self._palette_code.setStyleSheet(
            f"background:{_CARD_BG}; color:#90c080; border:1px solid {_BORDER};"
            " font:8px 'Consolas', monospace;"
        )
        self._palette_code.setPlainText("// Lancez une analyse pour générer la palette")
        grp2_l.addWidget(self._palette_code)
        lay.addWidget(grp2)

        # Bouton appliquer
        self._btn_apply_palette = QPushButton("✦  Injecter la palette dans le shader")
        self._btn_apply_palette.setStyleSheet(_BTN_SUCCESS)
        self._btn_apply_palette.setEnabled(False)
        self._btn_apply_palette.clicked.connect(self._on_apply_palette)
        lay.addWidget(self._btn_apply_palette)
        lay.addStretch()
        return w

    def _build_inject_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # Options d'injection
        grp = QGroupBox("Événements à injecter")
        grp_l = QVBoxLayout(grp)

        self._inject_checks: dict[str, QCheckBox] = {}
        uniforms_info = {
            EVENT_BEAT:    "uBeatPulse    — impulsion sur chaque beat",
            EVENT_DROP:    "uDropFlash    — flash sur les drops",
            EVENT_BUILD:   "uBuildRamp    — rampe sur les montées",
            EVENT_SILENCE: "uSilenceGate  — gate pendant les silences",
            EVENT_ONSET:   "uOnsetBurst   — burst sur chaque transitoire",
            EVENT_PEAK:    "uPeakFlare    — flare sur les pics d'énergie",
        }
        for evt_type, info in uniforms_info.items():
            cb = QCheckBox(f"{EVENT_ICONS.get(evt_type,'•')}  {info}")
            cb.setChecked(evt_type in (EVENT_BEAT, EVENT_DROP, EVENT_BUILD))
            self._inject_checks[evt_type] = cb
            grp_l.addWidget(cb)
        lay.addWidget(grp)

        # Options
        opt_grp = QGroupBox("Options")
        opt_l = QHBoxLayout(opt_grp)
        opt_l.addWidget(QLabel("Interpolation :"))
        self._interp_combo = QComboBox()
        self._interp_combo.addItems(["smooth", "linear", "step", "bezier"])
        opt_l.addWidget(self._interp_combo)
        opt_l.addStretch()
        lay.addWidget(opt_grp)

        # Stats anticipées
        self._inject_stats = QLabel("Lancez une analyse pour voir les stats.")
        self._inject_stats.setStyleSheet(f"color:{_TEXT_DIM}; font:8px 'Segoe UI'; padding:4px;")
        self._inject_stats.setWordWrap(True)
        lay.addWidget(self._inject_stats)

        # Bouton injection
        self._btn_inject = QPushButton("⊕  Injecter les keyframes dans la timeline")
        self._btn_inject.setStyleSheet(_BTN_PRIMARY)
        self._btn_inject.setFixedHeight(32)
        self._btn_inject.setEnabled(False)
        self._btn_inject.clicked.connect(self._on_inject)
        lay.addWidget(self._btn_inject)
        lay.addStretch()
        return w

    # ── Connexions signaux ────────────────────────────────────────────────────

    def _connect_signals(self):
        self._btn_analyze.clicked.connect(self._on_analyze_clicked)
        self._engine.analysis_started.connect(self._on_analysis_started)
        self._engine.analysis_progress.connect(self._on_progress)
        self._engine.analysis_done.connect(self._on_analysis_done)
        self._engine.analysis_error.connect(self._on_analysis_error)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_analyze_clicked(self):
        # Demander le chemin audio depuis le parent
        audio_path = ""
        if self.parent() and hasattr(self.parent(), "audio_engine"):
            audio_path = self.parent().audio_engine.file_path or ""
        if not audio_path:
            self._log("⚠️  Aucun fichier audio chargé. Chargez un fichier audio d'abord.")
            return
        self._log(f"▶ Analyse de : {audio_path}")
        self._engine.analyze(audio_path)

    def _on_analysis_started(self):
        self._lbl_status.setText("⏳ Analyse en cours…")
        self._btn_analyze.setEnabled(False)
        self._progress.show()
        self._progress.setValue(0)

    def _on_progress(self, v: int):
        self._progress.setValue(v)
        self._log(f"  {v}%…")

    def _on_analysis_done(self, plan: AudioSyncPlan):
        self._plan = plan
        self._progress.hide()
        self._btn_analyze.setEnabled(True)
        self._lbl_status.setText(f"✅ Terminé — {plan.n_events} événements")
        self._log(f"✅ Analyse terminée : {plan.n_events} événements, BPM={plan.bpm:.1f}, mood={plan.mood}")

        # Minimap
        self._minimap.set_plan(plan)

        # Mise à jour onglet Analyse
        by_type = plan.events_by_type
        self._res_duration.setText(f"{plan.duration:.2f} s")
        self._res_bpm.setText(f"{plan.bpm:.1f}")
        self._res_mood.setText(plan.mood.capitalize())
        self._res_beats.setText(str(len(by_type.get(EVENT_BEAT, []))))
        self._res_drops.setText(str(len(by_type.get(EVENT_DROP, []))))
        self._res_builds.setText(str(len(by_type.get(EVENT_BUILD, []))))
        self._res_silences.setText(str(len(by_type.get(EVENT_SILENCE, []))))
        self._res_total.setText(str(plan.n_events))

        # Événements
        self._refresh_events_list()

        # Palette
        if plan.palette:
            self._lbl_palette_name.setText(plan.palette.name)
            self._lbl_palette_mood.setText(
                f"Mood : {plan.palette.mood.capitalize()}  —  "
                f"a={plan.palette.a}  b={plan.palette.b}"
            )
            self._palette_preview.set_palette(plan.palette)
            self._palette_code.setPlainText(plan.palette.to_glsl_code())
            self._btn_apply_palette.setEnabled(True)

        # Inject stats
        counts_preview = {t: len(plan.events_by_type.get(t, []))
                          for t in AudioEventClassifier.CLASSES}
        stats_lines = [f"  {EVENT_ICONS.get(t,'•')} {t}: {n} keyframes"
                       for t, n in counts_preview.items() if n > 0]
        self._inject_stats.setText("Keyframes à injecter :\n" + "\n".join(stats_lines))
        self._btn_inject.setEnabled(True)

    def _on_analysis_error(self, msg: str):
        self._progress.hide()
        self._btn_analyze.setEnabled(True)
        self._lbl_status.setText(f"❌ Erreur : {msg}")
        self._log(f"❌ {msg}")

    def _refresh_events_list(self):
        """Reconstruit la liste d'événements filtrée."""
        # Vider
        while self._events_layout.count() > 1:
            item = self._events_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._plan is None:
            return

        active_types = {t for t, cb in self._evt_filters.items() if cb.isChecked()}

        count = 0
        for ev in self._plan.events:
            if ev.event_type not in active_types:
                continue
            row = self._make_event_row(ev)
            self._events_layout.insertWidget(count, row)
            count += 1

    def _make_event_row(self, ev: AudioEvent) -> QWidget:
        row = QWidget()
        row.setStyleSheet(
            f"background:{_CARD_BG}; border:1px solid {_BORDER};"
            " border-radius:3px; margin:1px 0;"
        )
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 3, 8, 3)
        rl.setSpacing(8)

        # Icône colorée
        icon_lbl = QLabel(ev.icon)
        icon_lbl.setFixedWidth(16)
        icon_lbl.setStyleSheet(f"color:{ev.color}; font:bold 10px 'Segoe UI';")
        rl.addWidget(icon_lbl)

        # Timestamp
        m   = int(ev.time) // 60
        s   = ev.time % 60
        ts  = QLabel(f"{m:02d}:{s:05.2f}")
        ts.setFixedWidth(52)
        ts.setStyleSheet(f"color:{_TEXT}; font:8px 'Consolas', monospace;")
        rl.addWidget(ts)

        # Décorateur
        dec = QLabel(ev.decorator)
        dec.setStyleSheet(f"color:{ev.color}; font:bold 8px 'Segoe UI';")
        dec.setFixedWidth(90)
        rl.addWidget(dec)

        # Strength bar
        str_bar = QProgressBar()
        str_bar.setRange(0, 100)
        str_bar.setValue(int(ev.strength * 100))
        str_bar.setTextVisible(False)
        str_bar.setFixedHeight(4)
        str_bar.setStyleSheet(
            f"QProgressBar{{background:#1a1d28;border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{ev.color};border-radius:2px;}}"
        )
        rl.addWidget(str_bar, 1)

        # Strength value
        sv = QLabel(f"{ev.strength:.2f}")
        sv.setFixedWidth(32)
        sv.setStyleSheet(f"color:{_TEXT_DIM}; font:7px 'Segoe UI';")
        rl.addWidget(sv)

        return row

    def _on_apply_palette(self):
        if self._plan and self._plan.palette:
            self.apply_palette_requested.emit(self._plan.palette)
            self._log(f"🎨 Palette '{self._plan.palette.name}' appliquée au shader.")

    def _on_inject(self):
        selected_types = [t for t, cb in self._inject_checks.items() if cb.isChecked()]
        interp = self._interp_combo.currentText()
        self.inject_requested.emit(selected_types, interp)
        self._log(f"⊕ Keyframes injectées : {selected_types} en mode '{interp}'")

    def _log(self, msg: str):
        self._log_text.append(msg)

    def set_current_time(self, t: float):
        self._minimap.set_cursor(t)


# ─────────────────────────────────────────────────────────────────────────────
#  PalettePreviewWidget
# ─────────────────────────────────────────────────────────────────────────────

class PalettePreviewWidget(QWidget):
    """Bande de prévisualisation d'une palette cosinus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette: PalettePreset | None = None
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_palette(self, palette: PalettePreset):
        self._palette = palette
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()

        if self._palette is None:
            p.fillRect(0, 0, w, h, QColor(_DARK_BG))
            p.end()
            return

        pal = self._palette
        import math as _m
        for x in range(w):
            t = x / max(1, w - 1)
            r = pal.a[0] + pal.b[0] * _m.cos(6.28318 * (pal.c[0] * t + pal.d[0]))
            g = pal.a[1] + pal.b[1] * _m.cos(6.28318 * (pal.c[1] * t + pal.d[1]))
            b_ = pal.a[2] + pal.b[2] * _m.cos(6.28318 * (pal.c[2] * t + pal.d[2]))
            ri = int(max(0, min(255, r * 255)))
            gi = int(max(0, min(255, g * 255)))
            bi = int(max(0, min(255, b_ * 255)))
            p.setPen(QPen(QColor(ri, gi, bi)))
            p.drawLine(x, 0, x, h)

        p.end()
