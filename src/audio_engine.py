"""
audio_engine.py
---------------
Moteur audio du DemoMaker.
Gère la lecture de fichiers .wav, .mp3 et .ogg via QtMultimedia.
La position de lecture est synchronisée avec le temps iTime des shaders.

Fix v2 :
  - QMediaPlayer charge de façon asynchrone ; la durée n'est pas disponible
    immédiatement après setSource(). On mémorise les demandes play/seek
    différées et on les exécute dans _on_media_status_changed dès que le
    média est prêt (LoadedMedia ou BufferedMedia).
  - Signal playback_ready(float) émis quand la durée est connue, pour que
    main_window puisse mettre à jour self.timeline.duration correctement.

Fix v5.0 :
  - Conversion automatique WAV pcm_f32le → pcm_s16le via FFmpeg.
    QtMultimedia (Windows/Linux) ne lit pas toujours les WAV float32 ;
    on détecte le format avant le chargement et on convertit silencieusement
    si nécessaire.
"""
import numpy as np
import os
import struct
import subprocess
from PyQt6.QtMultimedia import (QMediaPlayer, QAudioOutput, QAudioSource,
                                QMediaDevices, QAudioFormat)
from PyQt6.QtCore import QUrl, QObject, pyqtSignal

from .logger import get_logger

log = get_logger(__name__)


# ── WAV float32 → int16 conversion ───────────────────────────────────────────

def _wav_is_float32(filepath: str) -> bool:
    """
    Retourne True si le WAV est encodé en float32 (wFormatTag == 3).
    QtMultimedia ne lit pas ce format sur Windows/Linux.
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(44)
        if len(header) < 44:
            return False
        # Offset 20 : wFormatTag — 1=PCM int, 3=IEEE float
        fmt_tag = struct.unpack_from('<H', header, 20)[0]
        return fmt_tag == 3
    except (OSError, struct.error):
        return False


def _ffmpeg_available() -> bool:
    """Retourne True si FFmpeg est disponible dans le PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _needs_conversion(filepath: str) -> bool:
    """
    Retourne True si le fichier audio nécessite une conversion avant lecture.
    Cas concernés :
      - WAV pcm_f32le (float32) : non supporté par QtMultimedia Windows/Linux
      - MP3 : codec absent sur certaines installations Qt sans plugins GStreamer
      - OGG : idem
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.wav':
        return _wav_is_float32(filepath)
    return ext in ('.mp3', '.ogg', '.flac', '.aac', '.m4a')


def _convert_to_wav_s16(src: str) -> str | None:
    """
    Convertit n'importe quel fichier audio en WAV pcm_s16le 48 kHz via FFmpeg.
    Retourne le chemin du fichier converti (à côté de l'original, suffixe _tmp.wav),
    ou None si FFmpeg est absent ou si la conversion échoue.
    """
    if not _ffmpeg_available():
        log.warning("FFmpeg introuvable — conversion impossible pour : %s", os.path.basename(src))
        return None

    base = os.path.splitext(src)[0]
    dst  = base + '_tmp.wav'
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', src,
             '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', dst],
            capture_output=True, timeout=60
        )
        if result.returncode == 0 and os.path.isfile(dst):
            log.info("Converti automatiquement → %s", os.path.basename(dst))
            return dst
        log.warning("Conversion FFmpeg échouée (code %d) : %s",
                    result.returncode, result.stderr.decode(errors='replace')[:300])
        return None
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Erreur conversion : %s", e)
        return None


class AudioEngine(QObject):
    """
    Moteur audio basé sur QtMultimedia (QMediaPlayer).
    Expose get_position() pour synchroniser iTime avec la lecture audio.
    """

    amplitude_ready = pyqtSignal(float)
    fft_ready       = pyqtSignal(object)   # np.ndarray
    waveform_ready  = pyqtSignal(object)   # np.ndarray
    # Émis quand QMediaPlayer a fini de charger et que la durée est connue
    playback_ready  = pyqtSignal(float)    # durée en secondes

    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a')

    def __init__(self):
        super().__init__()
        self._player       = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)

        self._file_path    = None
        self._duration     = 0.0
        # Lecture / seek différés (demandés avant que le média soit chargé)
        self._pending_play = False
        self._pending_seek = None   # float | None

        self._audio_input  = None
        self._input_device = None

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._player.errorOccurred.connect(self._on_error)

    # ── Slots internes ───────────────────────────────────────────────────────

    def _on_duration_changed(self, d: int):
        if d > 0:
            self._duration = d / 1000.0
            self.playback_ready.emit(self._duration)

    def _on_media_status_changed(self, status):
        """Exécute les actions différées dès que le média est prêt."""
        ready_statuses = (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        )
        if status in ready_statuses:
            # Durée pas encore émise par durationChanged ? on la lit maintenant
            d = self._player.duration()
            if d > 0 and self._duration == 0.0:
                self._duration = d / 1000.0
                self.playback_ready.emit(self._duration)
            # Seek différé
            if self._pending_seek is not None:
                self._player.setPosition(int(self._pending_seek * 1000))
                self._pending_seek = None
            # Play différé
            if self._pending_play:
                self._pending_play = False
                self._player.play()

    def _on_error(self, error, error_string: str):
        log.warning(f"QMediaPlayer erreur {error}: {error_string}")

    # ── Chargement ──────────────────────────────────────────────────────────

    def load(self, filepath: str) -> tuple[bool, str]:
        """
        Charge un fichier audio.
        Retourne (succès, message).
        Le chargement est asynchrone : attendre le signal playback_ready
        avant d'utiliser self.duration.

        Fix v5.0 : si le fichier WAV est encodé en float32 (pcm_f32le),
        QtMultimedia ne peut pas le lire sur Windows/Linux. On le convertit
        automatiquement en pcm_s16le via FFmpeg avant le chargement.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return False, f"Format non supporté : {ext}"

        if not os.path.isfile(filepath):
            return False, f"Fichier introuvable : {filepath}"

        # Conversion automatique si le format n'est pas lu nativement par Qt
        # (WAV float32, MP3, OGG, FLAC, AAC, M4A selon la plateforme)
        actual_path = filepath
        if _needs_conversion(filepath):
            fmt_label = ext.upper().lstrip('.')
            if ext == '.wav':
                fmt_label = "WAV float32"
            log.info("%s détecté : %s — conversion en cours...", fmt_label, os.path.basename(filepath))
            converted = _convert_to_wav_s16(filepath)
            if converted:
                actual_path = converted
            else:
                return False, (
                    f"{fmt_label} : conversion impossible.\n"
                    "Installez FFmpeg pour la conversion automatique :\n"
                    "  https://ffmpeg.org/download.html\n"
                    "Ou convertissez manuellement :\n"
                    "  ffmpeg -i input.mp3 -acodec pcm_s16le output.wav"
                )

        try:
            self.stop()
            self._duration     = 0.0
            self._pending_play = False
            self._pending_seek = None
            self._file_path    = filepath          # on garde le chemin original affiché
            self._player.setSource(QUrl.fromLocalFile(actual_path))
            suffix = f" (converti {ext.upper().lstrip('.').replace('WAV','WAV-f32')}→WAV)" if actual_path != filepath else ""
            return True, f"Chargé : {os.path.basename(filepath)}{suffix}"
        except (OSError, RuntimeError, ValueError) as e:
            return False, f"Erreur chargement : {e}"

    # ── Transport ───────────────────────────────────────────────────────────

    def play(self):
        """Lance la lecture. Si le média n'est pas encore prêt, diffère."""
        if not self._file_path:
            return
        status = self._player.mediaStatus()
        not_ready = (
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.LoadingMedia,
        )
        if status in not_ready:
            self._pending_play = True
        else:
            self._player.play()

    def pause(self):
        """Met en pause."""
        if self._file_path:
            self._pending_play = False
            self._player.pause()

    def stop(self):
        """Arrête la lecture."""
        self._pending_play = False
        self._pending_seek = None
        if self._file_path:
            self._player.stop()

    def seek(self, position_seconds: float):
        """Déplace la tête de lecture. Diffère si le média n'est pas encore prêt."""
        if not self._file_path:
            return
        ms = int(position_seconds * 1000)
        status = self._player.mediaStatus()
        not_ready = (
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.LoadingMedia,
        )
        if status in not_ready:
            self._pending_seek = position_seconds
        else:
            self._player.setPosition(ms)

    # ── Enregistrement ──────────────────────────────────────────────────────

    def start_recording(self):
        """Démarre la capture depuis le microphone par défaut."""
        device = QMediaDevices.defaultAudioInput()
        if not device:
            log.warning("Aucun périphérique d'entrée audio trouvé.")
            return False

        fmt = QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        self._audio_input  = QAudioSource(device, fmt, self)
        self._input_device = self._audio_input.start()
        if self._input_device:
            self._input_device.readyRead.connect(self._on_mic_ready_read)
            return True
        return False

    def stop_recording(self):
        """Arrête la capture du microphone."""
        if self._audio_input:
            self._audio_input.stop()
            self._audio_input  = None
            self._input_device = None

    def _on_mic_ready_read(self):
        """Analyse un chunk de données audio et émet l'amplitude."""
        data      = self._input_device.readAll()
        raw_bytes = data.data()
        count     = len(data) // 2
        if count > 0:
            shorts    = struct.unpack(f'<{count}h', raw_bytes)
            amplitude = max(abs(s) for s in shorts) / 32767.0
            self.amplitude_ready.emit(amplitude)
        self._process_raw_audio(raw_bytes)

    def _process_raw_audio(self, raw_data: bytes):
        """Calcule le FFT sur des données audio brutes et émet le résultat."""
        sample_count = len(raw_data) // 2
        if sample_count < 256:
            return
        samples = np.frombuffer(raw_data, dtype=np.int16)
        window  = np.hanning(len(samples))
        samples = samples * window
        fft_data  = np.fft.rfft(samples)
        magnitude = np.abs(fft_data) / len(samples)
        magnitude = 20 * np.log10(magnitude + 1e-9)
        magnitude = np.clip((magnitude + 60) / 60, 0, 1)
        self.fft_ready.emit(magnitude)
        waveform = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
        self.waveform_ready.emit(waveform)

    # ── Temps synchronisé ───────────────────────────────────────────────────

    def get_position(self) -> float:
        """Retourne la position de lecture en secondes."""
        return self._player.position() / 1000.0

    # ── État ────────────────────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def has_file(self) -> bool:
        return self._file_path is not None

    @property
    def file_path(self) -> str | None:
        return self._file_path

    @property
    def file_name(self) -> str:
        if self._file_path:
            return os.path.basename(self._file_path)
        return ""

    def cleanup(self):
        """Libère les ressources audio."""
        self.stop_recording()
        self._player.stop()
