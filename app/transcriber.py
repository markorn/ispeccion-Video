import gc
import math
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from faster_whisper import WhisperModel
from app.config import (
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_SPANISH_MX_PROMPT,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_MODEL,
    get_ffmpeg_path,
    setup_cuda_environment,
)
from app.models import TranscriptSegment

setup_cuda_environment()


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    """Convierte segundos a formato MM:SS o HH:MM:SS(.ms)."""
    if seconds < 0:
        seconds = 0.0
    total_sec = int(seconds)
    ms = int(round((seconds - total_sec) * 100))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"
    return f"{base}.{ms:02d}" if include_ms else base


def has_audio_stream(media_path: Path) -> bool:
    """Verifica mediante FFmpeg si el archivo multimedia contiene pista de audio."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return True
    try:
        res = subprocess.run(
            [ffmpeg, "-i", str(media_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        output = (res.stderr or "") + (res.stdout or "")
        return "Audio:" in output
    except Exception:
        return True


def extract_wav_audio(media_path: Path, output_wav: Path, enhance_speech: bool = True) -> bool:
    """
    Extrae el audio a formato WAV 16kHz mono.
    Cuando enhance_speech=True, aplica filtros acústicos FFmpeg (highpass, lowpass, reducción de ruido
    y normalización dinámica) para mejorar la claridad de voces en grabaciones de calle/cámaras en Español (México).
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return False

    if enhance_speech:
        try:
            cmd_enhanced = [
                ffmpeg,
                "-y",
                "-i",
                str(media_path),
                "-vn",
                "-af",
                "highpass=f=85,lowpass=f=7600,afftdn=nf=-25,dynaudnorm=f=150:g=15",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_wav),
            ]
            res = subprocess.run(cmd_enhanced, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 1000:
                return True
        except Exception:
            pass

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 1000
    except Exception:
        return False


def build_spanish_mx_initial_prompt(language_mode: str, search_query: str = "", custom_vocab: str = "") -> Optional[str]:
    """
    Construye el prompt fonético y léxico inicial para orientar al decodificador de Faster-Whisper
    hacia el Español de México (es-MX), evitando anglicismos o confusiones fonéticas.
    """
    if language_mode == "auto":
        return custom_vocab.strip() or None

    parts = [DEFAULT_SPANISH_MX_PROMPT]
    if search_query.strip():
        parts.append(f"Contexto del video: {search_query.strip()}.")
    if custom_vocab.strip():
        parts.append(f"Términos y nombres adicionales: {custom_vocab.strip()}.")
    return " ".join(parts)


def transcribe_media(
    media_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = "es_mx",
    progress_callback: Optional[Callable[[str, int], None]] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    search_query: str = "",
    custom_vocab: str = "",
    enhance_speech: bool = True,
) -> Tuple[List[TranscriptSegment], str, str, str]:
    """
    Transcribe un archivo de audio o video usando Faster-Whisper optimizado para Español (México) en GPU CUDA.
    Devuelve: (segmentos, texto_completo, idioma_detectado, dispositivo_usado).
    """
    if progress_callback:
        progress_callback("Verificando pista de audio en el archivo...", 10)

    if not has_audio_stream(media_path):
        if progress_callback:
            progress_callback("El archivo de video no contiene pista de audio.", 35)
        return [], "", "sin_audio", "N/A (Sin audio)"

    temp_wav = media_path.parent / f"{media_path.stem}_temp16k.wav"
    target_audio = media_path

    if progress_callback:
        msg_audio = (
            "Normalizando voz y filtrando ruido ambiental a 16kHz (FFmpeg)..."
            if enhance_speech
            else "Extrayendo audio a 16kHz con FFmpeg..."
        )
        progress_callback(msg_audio, 15)

    if extract_wav_audio(media_path, temp_wav, enhance_speech=enhance_speech):
        target_audio = temp_wav

    device_pref = (compute_device or DEFAULT_COMPUTE_DEVICE).lower().strip()
    lang_mode = (language or "es_mx").lower().strip()
    whisper_lang = None if lang_mode == "auto" else "es"
    initial_prompt = build_spanish_mx_initial_prompt(lang_mode, search_query=search_query, custom_vocab=custom_vocab)

    model = None
    segments_list: List[TranscriptSegment] = []
    detected_lang = "es-MX" if lang_mode == "es_mx" else (whisper_lang or "es")
    actual_device_label = f"GPU CUDA ({DEFAULT_WHISPER_COMPUTE_TYPE})"

    def _run_whisper_inference(m: WhisperModel) -> Tuple[List[TranscriptSegment], str]:
        segs_iter, info = m.transcribe(
            str(target_audio),
            language=whisper_lang,
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=True,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.42,
                min_speech_duration_ms=200,
                min_silence_duration_ms=350,
                speech_pad_ms=250,
            ),
        )
        det = "es-MX" if lang_mode == "es_mx" else (info.language if info and info.language else "es")
        out_segs: List[TranscriptSegment] = []
        idx = 1
        for seg in segs_iter:
            text_clean = seg.text.strip()
            if not text_clean:
                continue
            prob = round(min(1.0, max(0.1, math.exp(seg.avg_logprob))), 2) if seg.avg_logprob else 0.92
            out_segs.append(
                TranscriptSegment(
                    id=idx,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    start_formatted=format_timestamp(seg.start, include_ms=True),
                    end_formatted=format_timestamp(seg.end, include_ms=True),
                    text=text_clean,
                    confidence=prob,
                )
            )
            idx += 1
        return out_segs, det

    try:
        if device_pref == "cpu":
            if progress_callback:
                progress_callback(f"Cargando Faster-Whisper ({model_size}) en CPU (int8)...", 20)
            model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            actual_device_label = "CPU (int8)"
        else:
            if progress_callback:
                progress_callback(
                    f"Cargando Faster-Whisper ({model_size}) en GPU NVIDIA CUDA ({DEFAULT_WHISPER_COMPUTE_TYPE})...",
                    20,
                )
            try:
                setup_cuda_environment()
                model = WhisperModel(
                    model_size,
                    device="cuda",
                    device_index=0,
                    compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
                )
                actual_device_label = f"GPU CUDA ({DEFAULT_WHISPER_COMPUTE_TYPE})"
            except Exception as cuda_init_err:
                model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                actual_device_label = f"CPU Fallback (int8) - {type(cuda_init_err).__name__}"

        if progress_callback:
            progress_callback(
                f"Transcribiendo audio en Español (México) con {model_size} [{actual_device_label}]...",
                25,
            )

        segments_list, detected_lang = _run_whisper_inference(model)

    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower() or "cudnn" in str(e).lower():
            if progress_callback:
                progress_callback("Reintentando transcripción en modo CPU (int8)...", 25)
            model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            actual_device_label = "CPU (int8)"
            segments_list, detected_lang = _run_whisper_inference(model)
        else:
            raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass

    full_text = " ".join(s.text for s in segments_list)
    if progress_callback:
        progress_callback(
            f"Transcripción Español (México) completada en {actual_device_label} ({len(segments_list)} segmentos).",
            40,
        )

    return segments_list, full_text, detected_lang, actual_device_label
