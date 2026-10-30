import asyncio
from datetime import datetime
import json
from pathlib import Path
import shutil
import uuid
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx

from app.config import (
    AUDIO_EXTENSIONS,
    DATA_DIR,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_MODEL,
    OLLAMA_BASE_URL,
    REPORTS_DIR,
    STATIC_DIR,
    UPLOADS_DIR,
    VIDEO_EXTENSIONS,
    get_ffmpeg_path,
    get_gpu_info,
    get_ollama_path,
)
from app.demo_generator import create_demo_forensic_video
from app.models import AnalysisReport
from app.report_generator import (
    generate_json_report,
    generate_markdown_report,
    generate_pdf_report,
)
from app.transcriber import transcribe_media
from app.video_processor import (
    ensure_web_compatible_video,
    extract_key_frames,
    get_media_metadata,
)
from app.vision_agent import (
    check_ollama_status,
    release_ollama_vram,
    run_visual_forensic_agent,
)

app = FastAPI(
    title="Analizador Pericial y Objetivo de Video/Audio con IA Local (CUDA + Español México)",
    description="Transcripción Faster-Whisper CUDA (es-MX) + Agente de Visión Local (Ollama Qwen2.5-VL CUDA) sin inferencias ni juicios de valor.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_JOBS: Dict[str, Dict] = {}


def _update_job(job_id: str, message: str, progress: int, status: str = "processing"):
    if job_id in ACTIVE_JOBS:
        ACTIVE_JOBS[job_id]["message"] = message
        ACTIVE_JOBS[job_id]["progress"] = progress
        ACTIVE_JOBS[job_id]["status"] = status


async def _execute_analysis_pipeline(
    job_id: str,
    file_path: Path,
    media_url: str,
    search_query: str,
    whisper_model: str,
    vision_model: str,
    sample_interval: float,
    max_frames: int,
    language: str,
    compute_device: str,
    custom_vocab: str = "",
    enhance_speech: bool = True,
):
    try:
        def sync_progress(msg: str, pct: int):
            _update_job(job_id, msg, pct)

        _update_job(job_id, "Extrayendo metadatos técnicos del archivo multimedia...", 5)
        playable_path = await asyncio.to_thread(ensure_web_compatible_video, file_path, sync_progress)
        playable_media_url = f"/data/uploads/{playable_path.name}"
        metadata = await asyncio.to_thread(get_media_metadata, playable_path)
        metadata.filename = file_path.name

        if compute_device.lower() in ("cuda", "auto"):
            await release_ollama_vram(vision_model)

        # 1. Transcripción de Audio con Faster-Whisper optimizada para Español (México)
        segments, full_text, detected_lang, whisper_device_used = await asyncio.to_thread(
            transcribe_media,
            playable_path,
            whisper_model,
            language,
            sync_progress,
            compute_device,
            search_query,
            custom_vocab,
            enhance_speech,
        )

        # 2. Extracción de Fotogramas Clave (si es video)
        extracted_frames = []
        if metadata.media_type == "video":
            _update_job(job_id, "Muestreando fotogramas clave y escaneando actividad visual...", 42)
            extracted_frames = await asyncio.to_thread(
                extract_key_frames,
                playable_path,
                job_id,
                sample_interval,
                max_frames,
                sync_progress,
            )

        # 3. Agente de IA Visual y Forense Objetivo en GPU CUDA
        (
            frames_analyzed,
            detected_situations,
            visual_summary,
            audio_summary,
            active_vision_model,
            audit_notes,
        ) = await run_visual_forensic_agent(
            extracted_frames=extracted_frames,
            transcript_segments=segments,
            search_query=search_query,
            preferred_vision_model=vision_model,
            sample_interval_sec=sample_interval,
            progress_callback=sync_progress,
            compute_device=compute_device,
        )

        audit_notes.append(
            f"Motor de transcripción Faster-Whisper ({whisper_model}) ejecutado en {whisper_device_used} con perfil lingüístico '{detected_lang}'."
        )
        if enhance_speech:
            audit_notes.append(
                "Pre-procesamiento acústico FFmpeg activo: filtro pasa-altos/pasa-bajos, reducción de ruido estacionario y normalización dinámica de voz."
            )

        # 4. Construcción del Reporte Consolidado
        _update_job(job_id, "Generando reportes exportables (PDF, Markdown y JSON)...", 96)
        report = AnalysisReport(
            job_id=job_id,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
            media_url=playable_media_url,
            search_query=search_query,
            whisper_model_used=f"{whisper_model} [{whisper_device_used}]",
            vision_model_used=active_vision_model,
            language_detected=detected_lang,
            full_transcript_text=full_text,
            transcript_segments=segments,
            frames_analyzed=frames_analyzed,
            detected_situations=detected_situations,
            objective_visual_summary=visual_summary,
            objective_audio_summary=audio_summary,
            neutrality_audit_notes=audit_notes,
        )

        await asyncio.to_thread(generate_json_report, report)
        await asyncio.to_thread(generate_markdown_report, report)
        await asyncio.to_thread(generate_pdf_report, report)

        ACTIVE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "message": "Análisis pericial en Español (México) finalizado correctamente.",
            "report": report.model_dump(),
        }

    except Exception as exc:
        ACTIVE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "error",
            "progress": 100,
            "message": f"Error durante el procesamiento: {str(exc)}",
            "report": None,
        }


@app.get("/api/system-status")
async def api_system_status():
    """Devuelve el estado del hardware local, CUDA, FFmpeg, Faster-Whisper y Ollama."""
    ffmpeg_path = get_ffmpeg_path()
    ollama_exe = get_ollama_path()
    gpu_info = get_gpu_info()
    ollama_status = await check_ollama_status()

    return {
        "gpu": gpu_info,
        "ffmpeg": {
            "available": ffmpeg_path is not None,
            "path": ffmpeg_path or "No encontrado",
        },
        "faster_whisper": {
            "available": True,
            "cuda_enabled": gpu_info.get("cuda_ready", False),
            "compute_type": DEFAULT_WHISPER_COMPUTE_TYPE,
            "default_device": DEFAULT_COMPUTE_DEVICE,
            "default_model": DEFAULT_WHISPER_MODEL,
            "language_profile": "es-MX (Español México)",
            "supported_models": ["tiny", "base", "small", "medium", "large-v3-turbo"],
        },
        "ollama": {
            "installed_binary": ollama_exe is not None,
            "binary_path": ollama_exe,
            **ollama_status,
        },
    }


@app.post("/api/ollama/pull")
async def api_ollama_pull(model_name: str = Form("qwen2.5vl:3b")):
    """Solicita al servidor local de Ollama la descarga de un modelo de visión."""
    ollama_status = await check_ollama_status()
    if not ollama_status["online"]:
        raise HTTPException(
            status_code=400,
            detail="El servidor local de Ollama no está en ejecución en http://127.0.0.1:11434.",
        )
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name, "stream": False},
            )
            if resp.status_code == 200:
                return {"status": "ok", "model": model_name}
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def api_analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    search_query: str = Form(""),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    vision_model: str = Form(DEFAULT_VISION_MODEL),
    sample_interval: float = Form(2.0),
    max_frames: int = Form(14),
    language: str = Form("es_mx"),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
    custom_vocab: str = Form(""),
    enhance_speech: bool = Form(True),
):
    """Recibe un archivo de video o audio y lanza el proceso de análisis objetivo."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido.")

    ext = Path(file.filename).suffix.lower()
    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{ext}' no soportado. Use archivos de video (.mp4, .avi, .mov, .mkv, .webm) o audio (.mp3, .wav, .m4a, .ogg).",
        )

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    safe_name = f"{job_id}{ext}"
    save_path = UPLOADS_DIR / safe_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    media_url = f"/data/uploads/{safe_name}"
    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 2,
        "message": "Archivo recibido. Inicializando transcripción Español (México) en CUDA...",
        "report": None,
    }

    background_tasks.add_task(
        _execute_analysis_pipeline,
        job_id,
        save_path,
        media_url,
        search_query.strip(),
        whisper_model,
        vision_model,
        max(0.5, float(sample_interval)),
        max(4, min(60, int(max_frames))),
        language,
        compute_device,
        custom_vocab,
        enhance_speech,
    )

    return {"job_id": job_id, "status": "processing"}


@app.post("/api/analyze-demo")
async def api_analyze_demo(
    background_tasks: BackgroundTasks,
    search_query: str = Form("Vehículo color rojo o acción de personas junto al vehículo"),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    vision_model: str = Form(DEFAULT_VISION_MODEL),
    sample_interval: float = Form(2.0),
    max_frames: int = Form(8),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
    language: str = Form("es_mx"),
    custom_vocab: str = Form(""),
    enhance_speech: bool = Form(True),
):
    """Genera el video sintético de vigilancia con locución en Español (México) y ejecuta el análisis completo en CUDA."""
    demo_path = await asyncio.to_thread(create_demo_forensic_video)
    job_id = "demo_" + datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
    media_url = f"/data/uploads/{demo_path.name}"

    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 3,
        "message": "Video de demostración (es-MX) generado. Iniciando transcripción y visión en GPU CUDA...",
        "report": None,
    }

    background_tasks.add_task(
        _execute_analysis_pipeline,
        job_id,
        demo_path,
        media_url,
        search_query.strip(),
        whisper_model,
        vision_model,
        max(0.5, float(sample_interval)),
        max(4, min(30, int(max_frames))),
        language,
        compute_device,
        custom_vocab,
        enhance_speech,
    )

    return {"job_id": job_id, "status": "processing"}


@app.post("/api/reanalyze/{job_id}")
async def api_reanalyze_existing(
    job_id: str,
    background_tasks: BackgroundTasks,
    search_query: str = Form(""),
    whisper_model: str = Form(DEFAULT_WHISPER_MODEL),
    vision_model: str = Form(DEFAULT_VISION_MODEL),
    sample_interval: float = Form(2.0),
    max_frames: int = Form(20),
    compute_device: str = Form(DEFAULT_COMPUTE_DEVICE),
    language: str = Form("es_mx"),
    custom_vocab: str = Form(""),
    enhance_speech: bool = Form(True),
):
    """Re-analiza un archivo ya subido previamente sin necesidad de volver a cargarlo."""
    candidates = [
        p for p in UPLOADS_DIR.glob(f"{job_id}.*") if not p.stem.endswith("_web")
    ]
    if not candidates:
        candidates = list(UPLOADS_DIR.glob(f"{job_id}*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="No se encontró el archivo original para re-analizar.")

    orig_file = candidates[0]
    media_url = f"/data/uploads/{orig_file.name}"

    ACTIVE_JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 3,
        "message": "Re-iniciando análisis con escaneo inteligente de fotogramas...",
        "report": None,
    }

    background_tasks.add_task(
        _execute_analysis_pipeline,
        job_id,
        orig_file,
        media_url,
        search_query.strip(),
        whisper_model,
        vision_model,
        max(0.5, float(sample_interval)),
        max(4, min(60, int(max_frames))),
        language,
        compute_device,
        custom_vocab,
        enhance_speech,
    )
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}")
async def api_get_job_status(job_id: str):
    """Consulta el progreso o resultado de un trabajo de análisis."""
    if job_id in ACTIVE_JOBS:
        return ACTIVE_JOBS[job_id]

    json_file = REPORTS_DIR / f"reporte_{job_id}.json"
    if json_file.exists():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "message": "Reporte cargado desde disco.",
            "report": data,
        }

    raise HTTPException(status_code=404, detail="Trabajo de análisis no encontrado.")


@app.get("/api/jobs")
async def api_list_jobs():
    """Lista los reportes generados previamente."""
    items = []
    for p in sorted(REPORTS_DIR.glob("reporte_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:15]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "job_id": data.get("job_id"),
                    "created_at": data.get("created_at"),
                    "filename": data.get("metadata", {}).get("filename"),
                    "media_type": data.get("metadata", {}).get("media_type"),
                    "search_query": data.get("search_query"),
                    "situations_count": len(data.get("detected_situations", [])),
                    "segments_count": len(data.get("transcript_segments", [])),
                }
            )
        except Exception:
            continue
    return {"jobs": items}


@app.get("/api/reports/{job_id}/{fmt}")
async def api_download_report(job_id: str, fmt: str):
    """Descarga el reporte pericial en formato pdf, md o json."""
    fmt_clean = fmt.lower().strip()
    ext_map = {
        "pdf": ("pdf", "application/pdf"),
        "md": ("md", "text/markdown; charset=utf-8"),
        "json": ("json", "application/json; charset=utf-8"),
    }
    if fmt_clean not in ext_map:
        raise HTTPException(status_code=400, detail="Formato inválido. Use pdf, md o json.")

    ext, mime = ext_map[fmt_clean]
    file_path = REPORTS_DIR / f"reporte_{job_id}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="El archivo de reporte solicitado no existe.")

    return FileResponse(
        path=str(file_path),
        media_type=mime,
        filename=f"reporte_objetivo_{job_id}.{ext}",
    )


app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
