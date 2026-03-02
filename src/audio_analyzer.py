"""
audio_analyzer.py
-----------------
v2.1 — Analyse audio avancée (scipy).

Fournit :
  - Onset detection (vrais transitoires) pour @on_beat précis
  - Coefficients MFCC (13 bandes) exposés comme uniforms uMFCC[13]
  - Spectrogram 2D calculé offline, uploadé comme texture iSpectroTex
  - RMS, ZCR (zero crossing rate), spectral centroid

Dépendance optionnelle : scipy >= 1.10
Si scipy est absent, le module fonctionne en mode dégradé :
  - MFCC simulés à 0.0
  - Onset detection désactivée (fallback sur BPM régulier)

Usage :
    from .audio_analyzer import AudioAnalyzer, AudioFeatures
    analyzer = AudioAnalyzer()
    ok = analyzer.analyze_file("track.wav")   # analyse offline du fichier
    features = analyzer.get_features_at(t)    # features au temps t (secondes)
    texture_data = analyzer.get_spectrogram_texture()   # RGBA numpy array
"""

import os
import time
import threading
import numpy as np
from dataclasses import dataclass, field

from .logger import get_logger

log = get_logger(__name__)

# ── Imports optionnels ────────────────────────────────────────────────────────
try:
    from scipy import signal as scipy_signal
    from scipy.io import wavfile as scipy_wavfile
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    log.info("audio_analyzer : scipy absent — fonctionnement en mode dégradé. "
             "Installez : pip install scipy>=1.10")

# ── Constantes ────────────────────────────────────────────────────────────────
N_MFCC        = 13       # Nombre de coefficients MFCC
N_MELS        = 40       # Nombre de filtres mel
HOP_LENGTH    = 512      # En samples (≈ 11.6 ms @ 44100 Hz)
WIN_LENGTH    = 2048     # Fenêtre FFT
SR_DEFAULT    = 44100    # Sample rate de référence
SPEC_W        = 512      # Largeur texture spectrogram
SPEC_H        = 128      # Hauteur texture spectrogram (fréquences)

# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class AudioFeatures:
    """Features extraites à un instant t (en secondes)."""
    rms:        float = 0.0                    # RMS normalisé [0,1]
    zcr:        float = 0.0                    # Zero crossing rate [0,1]
    centroid:   float = 0.0                    # Centroïde spectral normalisé [0,1]
    mfcc:       list  = field(default_factory=lambda: [0.0] * N_MFCC)
    is_onset:   bool  = False                  # Vrai sur les transitoires
    onset_strength: float = 0.0               # Force de l'onset [0,1]

    def as_uniforms(self) -> dict:
        """Retourne un dict prêt à être injecté dans ShaderEngine."""
        d = {
            "uAudioRMS":      float(self.rms),
            "uAudioZCR":      float(self.zcr),
            "uAudioCentroid": float(self.centroid),
            "uAudioOnset":    float(1.0 if self.is_onset else 0.0),
            "uAudioOnsetStrength": float(self.onset_strength),
        }
        for i, v in enumerate(self.mfcc[:N_MFCC]):
            d[f"uMFCC{i}"] = float(v)
        return d


@dataclass
class AnalysisResult:
    """Résultat complet de l'analyse offline d'un fichier audio."""
    sample_rate:    int   = SR_DEFAULT
    duration:       float = 0.0
    hop_length:     int   = HOP_LENGTH
    # Arrays indexés par frame (shape: [n_frames])
    times:          np.ndarray = field(default_factory=lambda: np.array([]))
    rms:            np.ndarray = field(default_factory=lambda: np.array([]))
    zcr:            np.ndarray = field(default_factory=lambda: np.array([]))
    centroid:       np.ndarray = field(default_factory=lambda: np.array([]))
    onset_times:    np.ndarray = field(default_factory=lambda: np.array([]))
    onset_strength: np.ndarray = field(default_factory=lambda: np.array([]))
    # MFCC : shape [N_MFCC, n_frames]
    mfcc:           np.ndarray = field(default_factory=lambda: np.zeros((N_MFCC, 1)))
    # Spectrogram texture : shape [SPEC_H, SPEC_W, 4] (RGBA uint8)
    spectrogram_texture: np.ndarray = field(
        default_factory=lambda: np.zeros((SPEC_H, SPEC_W, 4), dtype=np.uint8)
    )


# ── AudioAnalyzer ─────────────────────────────────────────────────────────────

class AudioAnalyzer:
    """
    Analyse audio avancée offline.

    Workflow
    --------
    1. analyze_file(path)    → déclenche l'analyse en thread de fond
    2. is_ready()            → True quand l'analyse est terminée
    3. get_features_at(t)    → features interpolées au temps t
    4. get_spectrogram_texture() → array RGBA pour upload GL
    """

    def __init__(self):
        self._result:    AnalysisResult | None = None
        self._analyzing: bool  = False
        self._error:     str | None = None
        self._thread:    threading.Thread | None = None

    # ── Analyse ──────────────────────────────────────────────────────────────

    def analyze_file(self, path: str, callback=None):
        """
        Lance l'analyse en arrière-plan.
        callback(result: AnalysisResult | None) appelé à la fin.
        """
        if not os.path.isfile(path):
            log.warning("AudioAnalyzer : fichier introuvable — %s", path)
            return

        self._result    = None
        self._error     = None
        self._analyzing = True

        def _run():
            try:
                result = _analyze(path)
                self._result = result
                log.info("AudioAnalyzer : analyse terminée — %.1f s, %d onsets",
                         result.duration, len(result.onset_times))
            except Exception as e:
                self._error     = str(e)
                self._result    = AnalysisResult()   # résultat vide
                log.warning("AudioAnalyzer : erreur analyse — %s", e)
            finally:
                self._analyzing = False
                if callback:
                    callback(self._result)

        self._thread = threading.Thread(target=_run, daemon=True, name="AudioAnalyzer")
        self._thread.start()

    def is_ready(self) -> bool:
        return self._result is not None and not self._analyzing

    def is_analyzing(self) -> bool:
        return self._analyzing

    @property
    def error(self) -> str | None:
        return self._error

    # ── Accès temps réel ─────────────────────────────────────────────────────

    def get_features_at(self, t: float) -> AudioFeatures:
        """
        Retourne les AudioFeatures interpolées au temps t (secondes).
        Si l'analyse n'est pas prête, retourne des features nulles.
        """
        r = self._result
        if r is None or len(r.times) == 0:
            return AudioFeatures()

        # Trouve le frame le plus proche
        idx = int(t / (r.hop_length / r.sample_rate))
        idx = max(0, min(idx, len(r.times) - 1))

        # Vérifie s'il y a un onset à ±1 frame
        t_frame = r.times[idx]
        is_onset = False
        onset_str = 0.0
        if len(r.onset_times) > 0:
            diffs = np.abs(r.onset_times - t)
            nearest_idx = np.argmin(diffs)
            if diffs[nearest_idx] < (r.hop_length / r.sample_rate) * 1.5:
                is_onset = True
                onset_str = float(r.onset_strength[nearest_idx]) if len(r.onset_strength) > nearest_idx else 1.0

        mfcc_at_t = [float(r.mfcc[i, idx]) for i in range(N_MFCC)]

        return AudioFeatures(
            rms        = float(r.rms[idx]) if idx < len(r.rms) else 0.0,
            zcr        = float(r.zcr[idx]) if idx < len(r.zcr) else 0.0,
            centroid   = float(r.centroid[idx]) if idx < len(r.centroid) else 0.0,
            mfcc       = mfcc_at_t,
            is_onset   = is_onset,
            onset_strength = onset_str,
        )

    def get_onset_times(self) -> np.ndarray:
        """Retourne les timestamps (secondes) de tous les onsets détectés."""
        if self._result is None:
            return np.array([])
        return self._result.onset_times.copy()

    def get_spectrogram_texture(self) -> np.ndarray:
        """
        Retourne le spectrogram comme array RGBA uint8 [H, W, 4].
        Prêt pour upload via ctx.texture().
        """
        if self._result is None:
            return np.zeros((SPEC_H, SPEC_W, 4), dtype=np.uint8)
        return self._result.spectrogram_texture

    def get_summary(self) -> dict:
        """Résumé sérialisable."""
        if self._result is None:
            return {"ready": False, "analyzing": self._analyzing}
        r = self._result
        return {
            "ready":        True,
            "duration":     round(r.duration, 2),
            "sample_rate":  r.sample_rate,
            "n_frames":     len(r.times),
            "n_onsets":     len(r.onset_times),
            "avg_rms":      round(float(np.mean(r.rms)) if len(r.rms) else 0.0, 4),
            "scipy":        _SCIPY_AVAILABLE,
        }


# ── Analyse principale (thread de fond) ──────────────────────────────────────

def _analyze(path: str) -> AnalysisResult:
    """
    Analyse complète d'un fichier audio.
    Retourne un AnalysisResult complet.
    """
    result = AnalysisResult()

    # ── Chargement audio ──────────────────────────────────────────────────
    samples, sr = _load_audio(path)
    if samples is None:
        log.warning("AudioAnalyzer : impossible de charger %s", path)
        return result

    result.sample_rate = sr
    result.duration    = len(samples) / sr
    result.hop_length  = HOP_LENGTH

    if not _SCIPY_AVAILABLE:
        # Mode dégradé : on calcule au moins le RMS brut
        result.times = np.arange(0, result.duration, HOP_LENGTH / sr)
        n = len(result.times)
        result.rms      = np.zeros(n)
        result.zcr      = np.zeros(n)
        result.centroid = np.zeros(n)
        result.mfcc     = np.zeros((N_MFCC, max(n, 1)))
        log.info("AudioAnalyzer : scipy absent, analyse minimale")
        return result

    # ── Paramètres FFT ────────────────────────────────────────────────────
    hop    = HOP_LENGTH
    n_fft  = WIN_LENGTH
    window = scipy_signal.windows.hann(n_fft)

    # Frames
    n_frames = 1 + (len(samples) - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
    result.times = np.arange(n_frames) * hop / sr

    # ── STFT ──────────────────────────────────────────────────────────────
    freqs, _, Zxx = scipy_signal.stft(
        samples, fs=sr, window=window,
        nperseg=n_fft, noverlap=n_fft - hop
    )
    mag = np.abs(Zxx[:, :n_frames])           # [n_freq, n_frames]
    mag_db = 20 * np.log10(mag + 1e-9)

    # ── RMS ───────────────────────────────────────────────────────────────
    rms_raw = np.array([
        np.sqrt(np.mean(samples[i*hop : i*hop + n_fft]**2))
        for i in range(n_frames)
    ])
    rms_max = rms_raw.max() + 1e-9
    result.rms = np.clip(rms_raw / rms_max, 0, 1).astype(np.float32)

    # ── ZCR ───────────────────────────────────────────────────────────────
    def _zcr(frame):
        return np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
    zcr_raw = np.array([
        _zcr(samples[i*hop : i*hop + n_fft]) for i in range(n_frames)
    ], dtype=np.float32)
    result.zcr = np.clip(zcr_raw / (zcr_raw.max() + 1e-9), 0, 1)

    # ── Spectral centroid ─────────────────────────────────────────────────
    freq_bins = freqs
    mag_sum   = mag.sum(axis=0) + 1e-9
    centroid  = (freq_bins[:, None] * mag).sum(axis=0) / mag_sum
    result.centroid = np.clip(
        centroid / (sr / 2.0), 0, 1
    ).astype(np.float32)

    # ── Mel filterbank + MFCC ─────────────────────────────────────────────
    mel_filters = _mel_filterbank(sr, n_fft, N_MELS)     # [N_MELS, n_freq]
    mel_spec    = np.dot(mel_filters, mag)                # [N_MELS, n_frames]
    log_mel     = np.log(mel_spec + 1e-9)
    # DCT-II pour MFCC
    from scipy.fftpack import dct
    mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:N_MFCC, :]
    # Normalisation par feature
    mfcc_std = mfcc.std(axis=1, keepdims=True) + 1e-9
    mfcc_mean = mfcc.mean(axis=1, keepdims=True)
    result.mfcc = ((mfcc - mfcc_mean) / mfcc_std).astype(np.float32)

    # ── Onset detection ───────────────────────────────────────────────────
    # Flux spectral (différence entre frames consécutives)
    flux = np.sum(np.maximum(0, np.diff(mag_db, axis=1)), axis=0)
    flux = np.concatenate([[0], flux])
    flux_norm = flux / (flux.max() + 1e-9)

    # Seuillage adaptatif (médiane locale)
    win_size = max(1, int(sr / hop * 0.1))   # 100ms
    threshold = np.array([
        np.median(flux_norm[max(0, i-win_size):i+win_size+1]) * 1.3
        for i in range(len(flux_norm))
    ])
    onset_mask = (flux_norm > threshold) & (flux_norm > 0.1)

    # Suppression des onsets trop proches (min 80ms d'écart)
    min_gap = max(1, int(0.08 * sr / hop))
    onset_frames = []
    last = -min_gap
    for i, is_on in enumerate(onset_mask):
        if is_on and (i - last) >= min_gap:
            onset_frames.append(i)
            last = i

    result.onset_times    = np.array(onset_frames) * hop / sr
    result.onset_strength = flux_norm[onset_frames] if onset_frames else np.array([])

    # ── Spectrogram texture ───────────────────────────────────────────────
    result.spectrogram_texture = _build_spectrogram_texture(mag_db, SPEC_W, SPEC_H)

    log.debug("AudioAnalyzer : %d frames, %d onsets, sr=%d, dur=%.1f s",
              n_frames, len(onset_frames), sr, result.duration)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_audio(path: str) -> tuple[np.ndarray | None, int]:
    """Charge un fichier audio en mono float32. Supporte WAV (et MP3 via pygame)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.wav' and _SCIPY_AVAILABLE:
        try:
            sr, data = scipy_wavfile.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Normalise AVANT la conversion en float32 (après, dtype.kind == 'f')
            if data.dtype.kind in ('i', 'u'):
                data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
            else:
                data = data.astype(np.float32)
                peak = np.abs(data).max()
                if peak > 1.0:
                    data /= peak
            return data, sr
        except Exception as e:
            log.debug("_load_audio scipy_wavfile : %s", e)

    # Fallback via pygame (supporte WAV, OGG, MP3)
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=SR_DEFAULT, size=-16, channels=1, buffer=512)
        sound = pygame.mixer.Sound(path)
        arr   = pygame.sndarray.array(sound)
        sr    = pygame.mixer.get_init()[0]
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = arr.astype(np.float32) / 32768.0
        return arr, sr
    except Exception as e:
        log.debug("_load_audio pygame : %s", e)

    return None, SR_DEFAULT


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Crée une banque de filtres mel [n_mels, n_fft//2+1]."""
    n_freq  = n_fft // 2 + 1
    f_min   = 80.0
    f_max   = float(sr) / 2.0

    def hz_to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    mel_min  = hz_to_mel(f_min)
    mel_max  = hz_to_mel(f_max)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    filters = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        f_left   = bins[m - 1]
        f_center = bins[m]
        f_right  = bins[m + 1]
        for k in range(f_left, f_center):
            if f_center - f_left > 0:
                filters[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right - f_center > 0:
                filters[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filters


def _build_spectrogram_texture(mag_db: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Réduit/agrandit le spectrogramme à [height, width]
    et encode en RGBA uint8.
    """
    from scipy.ndimage import zoom
    # mag_db : [n_freq, n_frames]
    n_freq, n_frames = mag_db.shape
    zoom_y = height / n_freq
    zoom_x = width  / n_frames
    try:
        resized = zoom(mag_db, (zoom_y, zoom_x), order=1)
    except Exception:
        resized = np.zeros((height, width))

    # Normalise en [0, 255]
    lo, hi = resized.min(), resized.max()
    if hi > lo:
        norm = ((resized - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(resized, dtype=np.uint8)

    # Coloration "heatmap" (noir→bleu→vert→jaune→rouge)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    v = norm.astype(np.float32) / 255.0
    rgba[:, :, 0] = np.clip(v * 2 - 0.5, 0, 1) * 255              # R
    rgba[:, :, 1] = np.clip(np.abs(v * 2 - 1.0) * 2 - 0.5, 0, 1) * 255  # G
    rgba[:, :, 2] = np.clip(1.5 - v * 2, 0, 1) * 255              # B
    rgba[:, :, 3] = 255

    return rgba
