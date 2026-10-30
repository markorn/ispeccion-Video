import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np

from app.config import AUDIO_EXTENSIONS, FRAMES_DIR, get_ffmpeg_path
from app.models import VideoMetadata
from app.transcriber import format_timestamp


def ensure_web_compatible_video(file_path: Path, progress_callback: Optional[Callable[[str, int], None]] = None) -> Path:
    """
    Convierte o re-empaqueta (remux) instantáneamente contenedores como .avi, .mkv, .mov, .wmv
    a un contenedor .mp4 (H.264 + faststart) compatible con navegadores web HTML5 y corrige
    cabeceras de FPS corruptas (ej. 1000 FPS en grabadoras DVR/CCTV .avi).
    """
    ext = file_path.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return file_path

    # Verificar si ya es un .mp4 con FPS válido
    if ext == ".mp4":
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            if 1.0 <= fps <= 120.0:
                return file_path

    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return file_path

    out_mp4 = file_path.parent / f"{file_path.stem}_web.mp4"
    if out_mp4.exists() and out_mp4.stat().st_size > 10000:
        return out_mp4

    if progress_callback:
        progress_callback(f"Optimizando contenedor {ext.upper()} a MP4 compatible con navegador web y corrigiendo índice temporal...", 7)

    # 1. Intentar remux ultrarrápido (-c:v copy) si el stream de video ya es H.264 (tarda ~2s para 30 min)
    cmd_copy = [
        ffmpeg,
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(file_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    try:
        res = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 10000:
            return out_mp4
    except Exception:
        pass

    # 2. Si no era H.264, recodificar usando aceleración GPU NVIDIA (h264_nvenc) o libx264 ultrafast
    for vcodec in ("h264_nvenc", "libx264"):
        cmd_enc = [
            ffmpeg,
            "-y",
            "-i",
            str(file_path),
            "-c:v",
            vcodec,
            "-preset",
            "fast" if vcodec == "h264_nvenc" else "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        try:
            res = subprocess.run(cmd_enc, capture_output=True, text=True, timeout=300)
            if res.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 10000:
                return out_mp4
        except Exception:
            continue

    return file_path


def get_media_metadata(file_path: Path) -> VideoMetadata:
    """Obtiene metadatos precisos de un archivo de video o audio, validando la duración real con FFmpeg."""
    ext = file_path.suffix.lower()
    is_audio = ext in AUDIO_EXTENSIONS
    size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
    ffmpeg_dur = _probe_duration_ffmpeg(file_path)

    if not is_audio:
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()

            # Evitar cabeceras timebase de 1000 FPS típicas de archivos .avi de CCTV
            if fps > 240.0 and ffmpeg_dur > 0:
                fps = round(frame_count / ffmpeg_dur, 2) if frame_count > 0 else 15.0
                if fps > 120.0:
                    fps = 15.0

            cv_duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
            duration = ffmpeg_dur if ffmpeg_dur > 0 else cv_duration

            if width > 0 and height > 0:
                return VideoMetadata(
                    filename=file_path.name,
                    media_type="video",
                    duration_seconds=round(duration, 2),
                    duration_formatted=format_timestamp(duration),
                    width=width,
                    height=height,
                    fps=round(fps, 2),
                    file_size_mb=size_mb,
                )

    return VideoMetadata(
        filename=file_path.name,
        media_type="audio" if is_audio else "video",
        duration_seconds=round(ffmpeg_dur, 2),
        duration_formatted=format_timestamp(ffmpeg_dur),
        width=0,
        height=0,
        fps=0.0,
        file_size_mb=size_mb,
    )


def _probe_duration_ffmpeg(file_path: Path) -> float:
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return 0.0
    try:
        res = subprocess.run(
            [ffmpeg, "-i", str(file_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        output = (res.stderr or "") + (res.stdout or "")
        for line in output.splitlines():
            if "Duration:" in line:
                part = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = part.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def extract_key_frames(
    video_path: Path,
    job_id: str,
    sample_interval_sec: float = 2.0,
    max_frames: int = 20,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> List[Dict]:
    """
    Extrae fotogramas clave buscando por marca de tiempo real en milisegundos (CAP_PROP_POS_MSEC).
    Para videos largos (ej. 10 a 60+ minutos de cámaras CCTV):
    1. Realiza un escaneo rápido de actividad visual a lo largo de toda la duración para detectar
       en qué minutos/segundos hay movimiento o cambios en la escena.
    2. Combina los picos de mayor actividad motriz con puntos de anclaje cronológicos uniformes
       para cubrir todo el video sin quedarse estancado en el primer fotograma.
    """
    job_frames_dir = FRAMES_DIR / job_id
    job_frames_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration_ffmpeg(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    if duration <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps > 240.0:
            fps = 15.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 10.0

    max_frames = max(4, min(60, int(max_frames)))
    target_timestamps: List[float] = []

    if duration <= 90.0:
        # Videos cortos (<= 1.5 min): muestreo directo uniforme ajustado a max_frames
        step = max(sample_interval_sec, duration / float(max_frames))
        t = 0.0
        while t < duration and len(target_timestamps) < max_frames:
            target_timestamps.append(round(t, 2))
            t += step
    else:
        # Videos medianos/largos (ej. 30 min): Escaneo Inteligente en 2 Etapas
        if progress_callback:
            progress_callback(
                f"Escaneando actividad visual en video de {format_timestamp(duration)} para detectar eventos y movimiento...",
                43,
            )

        num_probes = min(180, max(60, int(duration / 10.0)))
        probe_step = duration / float(num_probes)
        prev_small = None
        probe_scores: List[tuple[float, float]] = []

        for i in range(num_probes):
            ts = round(i * probe_step, 2)
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, pframe = cap.read()
            if not ret or pframe is None:
                continue
            gray_s = cv2.resize(cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY), (160, 90))
            score = 0.0
            if prev_small is not None:
                diff = cv2.absdiff(gray_s, prev_small)
                _, th = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                score = float(np.count_nonzero(th)) / float(th.size)
            prev_small = gray_s
            probe_scores.append((ts, score))

        # 1) Anclas cronológicas uniformes (35% de max_frames) para cubrir inicio, desarrollo y fin
        num_anchors = max(4, int(round(max_frames * 0.35)))
        anchor_step = duration / float(max(1, num_anchors - 1))
        for a_idx in range(num_anchors):
            t_anchor = min(max(0.0, duration - 1.0), round(a_idx * anchor_step, 2))
            target_timestamps.append(t_anchor)

        # 2) Picos de mayor actividad visual / movimiento (65% restante de max_frames)
        min_separation = max(3.0, duration / float(max_frames * 2.5))
        sorted_by_motion = sorted(probe_scores, key=lambda x: x[1], reverse=True)
        for ts_cand, _ in sorted_by_motion:
            if len(target_timestamps) >= max_frames:
                break
            if all(abs(ts_cand - existing) >= min_separation for existing in target_timestamps):
                target_timestamps.append(ts_cand)

        target_timestamps.sort()

    # Extraer y guardar cada fotograma objetivo con su telemetría real
    extracted: List[Dict] = []
    prev_gray_saved = None

    for idx, ts in enumerate(target_timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray_curr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        motion_ratio = 0.0
        if prev_gray_saved is not None:
            diff = cv2.absdiff(gray_curr, prev_gray_saved)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        prev_gray_saved = gray_curr

        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / w
            frame_resized = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame

        saved_idx = len(extracted) + 1
        ts_fmt = format_timestamp(ts)
        filename = f"frame_{saved_idx:03d}_{int(ts * 100):07d}.jpg"
        out_path = job_frames_dir / filename
        cv2.imwrite(str(out_path), frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        telemetry = _compute_frame_telemetry(frame_resized, motion_ratio)
        extracted.append(
            {
                "frame_index": saved_idx,
                "timestamp": round(ts, 2),
                "timestamp_formatted": ts_fmt,
                "frame_path": str(out_path),
                "frame_url": f"/data/frames/{job_id}/{filename}",
                "telemetry": telemetry,
            }
        )

    cap.release()
    return extracted


def _compute_frame_telemetry(bgr_frame: np.ndarray, motion_ratio: float) -> Dict:
    """Calcula métricas visuales objetivas (iluminación, actividad motriz, colores predominantes)."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    if brightness < 60:
        lighting = "baja iluminación / escena nocturna u oscura"
    elif brightness > 190:
        lighting = "iluminación alta / clara"
    else:
        lighting = "iluminación media / normal"

    if motion_ratio > 0.15:
        motion_desc = "cambio o desplazamiento visual alto"
    elif motion_ratio > 0.04:
        motion_desc = "desplazamiento visual moderado"
    else:
        motion_desc = "escena estática o desplazamiento mínimo"

    return {
        "brightness": round(brightness, 1),
        "lighting": lighting,
        "motion_ratio": round(motion_ratio, 3),
        "motion_desc": motion_desc,
    }
