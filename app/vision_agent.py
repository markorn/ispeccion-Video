import asyncio
import base64
import json
import os
import re
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import httpx
import numpy as np

from app.config import DEFAULT_COMPUTE_DEVICE, DEFAULT_VISION_MODEL, OLLAMA_BASE_URL, get_ollama_path
from app.models import (
    DetectedSituationInterval,
    FrameAnalysis,
    TranscriptSegment,
)
from app.transcriber import format_timestamp


FORENSIC_SYSTEM_PROMPT = """Eres un Agente de Análisis Visual Forense y Pericial estrictamente objetivo.
Tu única función es observar fotogramas de video y describir EXCLUSIVAMENTE lo que es visible de manera fáctica y verificable.

REGLAS OBLIGATORIAS DE OBJETIVIDAD (ANTI-INFERENCIAS):
1. DESCRIPCIÓN LITERAL Y VISUAL: Describe únicamente elementos observables: cantidad de personas, vestimenta (colores, tipo de prenda), vehículos (tipo, color, posición, dirección de desplazamiento), objetos visibles, entorno físico e iluminación.
2. CERO INFERENCIAS O SUPOSICIONES: NO inventes intenciones, emociones, pensamientos, relaciones entre personas ni antecedentes que no se vean en la imagen.
3. CERO JUICIOS DE VALOR O ETIQUETAS MORALES/LEGALES: Está estrictamente prohibido usar términos valorativos o especulativos como "sospechoso", "delincuente", "ladrón", "víctima", "actitud maliciosa", "culpable", "peligroso" o similares.
4. DETECCIÓN DE SITUACIONES ESPECÍFICAS (EJ. VEHÍCULOS, ROBO, VIOLENCIA):
   - Cuando el usuario solicite identificar una situación (p. ej. un vehículo específico, robo o violencia), evalúa si en el fotograma se observan los INDICADORES FÍSICOS Y MOTRICES de dicha situación (p. ej., presencia de un vehículo con las características descritas, contacto físico brusco entre personas, forcejeo, sujeción, toma y traslado rápido de un objeto, carrera o impacto físico).
   - Reporta la coincidencia describiendo la acción motriz exacta (ej. "Un individuo con sudadera negra extiende el brazo derecho hacia la ventanilla del vehículo blanco") sin emitir juicios de valor.
5. Responde SIEMPRE en español y en el formato JSON solicitado."""


async def ensure_ollama_running() -> bool:
    """Si Ollama está instalado pero no responde en el puerto 11434, lo inicia automáticamente con CUDA v12."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                return True
    except Exception:
        pass

    ollama_exe = get_ollama_path()
    if not ollama_exe:
        return False

    try:
        env = os.environ.copy()
        env["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [ollama_exe, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        for _ in range(8):
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=1.5) as client:
                    r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                    if r.status_code == 200:
                        return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def check_ollama_status() -> Dict:
    """Verifica si el servidor local de Ollama está en ejecución (iniciándolo si es necesario) y qué modelos están instalados."""
    await ensure_ollama_running()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                vision_models = [
                    m
                    for m in models
                    if any(v in m.lower() for v in ("vl", "llava", "moondream", "bakllava", "minicpm", "gemma3"))
                ]
                running_processor = "100% GPU (CUDA)"
                try:
                    ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
                    if ps_resp.status_code == 200:
                        ps_models = ps_resp.json().get("models", [])
                        if ps_models:
                            size_vram = ps_models[0].get("size_vram", 0)
                            size_tot = ps_models[0].get("size", 1) or 1
                            pct_gpu = int(round((size_vram / size_tot) * 100))
                            running_processor = f"{pct_gpu}% GPU (CUDA)"
                except Exception:
                    pass

                return {
                    "online": True,
                    "all_models": models,
                    "vision_models": vision_models,
                    "recommended_model": vision_models[0] if vision_models else DEFAULT_VISION_MODEL,
                    "processor": running_processor,
                }
    except Exception:
        pass
    return {
        "online": False,
        "all_models": [],
        "vision_models": [],
        "recommended_model": DEFAULT_VISION_MODEL,
        "processor": "Offline",
    }


async def release_ollama_vram(vision_model: str = DEFAULT_VISION_MODEL) -> None:
    """
    Libera temporalmente la VRAM ocupada por Ollama (keep_alive=0) antes de ejecutar Faster-Whisper en CUDA,
    evitando saturar los 6 GB de VRAM de la GTX 1060.
    """
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": vision_model, "keep_alive": 0},
            )
    except Exception:
        pass


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_dict(raw_text: str) -> Optional[Dict]:
    """Extrae un bloque JSON válido desde la respuesta del modelo."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


FORBIDDEN_TERMS_MAP = {
    r"\bsospechoso(s|a|as)?\b": "individuo observado",
    r"\bladr[oó]n(es|a|as)?\b": "sujeto",
    r"\bdelincuente(s)?\b": "sujeto",
    r"\bcriminal(es)?\b": "sujeto",
    r"\bmalintencionad[oa](s)?\b": "con desplazamiento directo",
    r"\bculpable(s)?\b": "interviniente",
}


def sanitize_objective_text(text: str) -> str:
    """Filtro determinista que reemplaza adjetivos valorativos por lenguaje neutro pericial."""
    if not text:
        return ""
    cleaned = text
    for pattern, replacement in FORBIDDEN_TERMS_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


async def analyze_frame_with_ollama(
    frame_info: Dict,
    search_query: str,
    vision_model: str,
    audio_context: str = "",
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
) -> Optional[FrameAnalysis]:
    """Analiza un fotograma individual usando el modelo de visión en Ollama con aceleración GPU CUDA."""
    img_b64 = _encode_image_b64(frame_info["frame_path"])
    query_instruction = (
        f'El usuario solicita identificar específicamente la siguiente situación o elemento: "{search_query}". '
        "Determina si en este fotograma se observan indicadores visuales directos de dicha situación."
        if search_query.strip()
        else "El usuario no especificó una situación puntual; realiza únicamente el inventario visual objetivo del fotograma."
    )

    user_prompt = f"""Analiza este fotograma capturado en el tiempo {frame_info["timestamp_formatted"]} ({frame_info["timestamp"]} s).
{query_instruction}
Contexto de audio transcrito en este instante (solo como referencia temporal): "{audio_context or 'Sin voz en este segundo'}".

Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
{{
  "objective_description": "Descripción estrictamente visual, literal y objetiva de lo que se ve en la imagen (escenario, personas, vestimenta, vehículos, colores, posiciones y acciones físicas visibles), sin inferencias ni juicios de valor.",
  "target_detected": true o false,
  "confidence_level": "alta" | "media" | "baja" | "ninguna",
  "target_evidence_detail": "Si target_detected es true, explica de forma fáctica qué elementos o movimientos visibles coinciden con '{search_query}'. Si es false, deja cadena vacía.",
  "observed_elements": ["elemento 1 (ej. vehículo sedán rojo)", "elemento 2 (ej. sujeto con camisa blanca)"]
}}"""

    # Si compute_device es "cpu", num_gpu=0; si es "cuda" o "auto", num_gpu=99 (100% de capas en GPU NVIDIA)
    num_gpu = 0 if (compute_device or "").lower() == "cpu" else 99

    payload = {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [img_b64]},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 450,
            "num_gpu": num_gpu,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            if resp.status_code != 200:
                return None
            content = resp.json().get("message", {}).get("content", "")
            parsed = _extract_json_dict(content)
            if not parsed:
                return None

            target_det = bool(parsed.get("target_detected", False)) if search_query.strip() else False
            conf = str(parsed.get("confidence_level", "ninguna")).lower()
            if conf not in ("alta", "media", "baja", "ninguna"):
                conf = "media" if target_det else "ninguna"
            if not target_det:
                conf = "ninguna"

            obj_desc = sanitize_objective_text(
                str(parsed.get("objective_description", "Fotograma analizado sin observaciones adicionales."))
            )
            ev_detail = sanitize_objective_text(str(parsed.get("target_evidence_detail", "")))
            elements = [sanitize_objective_text(str(x)) for x in parsed.get("observed_elements", []) if x]

            return FrameAnalysis(
                frame_index=frame_info["frame_index"],
                timestamp=frame_info["timestamp"],
                timestamp_formatted=frame_info["timestamp_formatted"],
                frame_url=frame_info["frame_url"],
                frame_path=frame_info["frame_path"],
                objective_description=obj_desc,
                target_detected=target_det,
                confidence_level=conf,  # type: ignore
                target_evidence_detail=ev_detail,
                observed_elements=elements,
            )
    except Exception:
        return None


def analyze_frame_opencv_fallback(
    frame_info: Dict,
    search_query: str,
    prev_bgr: Optional[np.ndarray] = None,
) -> Tuple[FrameAnalysis, np.ndarray]:
    """
    Analizador visual determinista basado en OpenCV cuando Ollama no está disponible.
    """
    bgr = cv2.imread(frame_info["frame_path"])
    if bgr is None:
        empty = np.zeros((90, 160, 3), dtype=np.uint8)
        return (
            FrameAnalysis(
                frame_index=frame_info["frame_index"],
                timestamp=frame_info["timestamp"],
                timestamp_formatted=frame_info["timestamp_formatted"],
                frame_url=frame_info["frame_url"],
                frame_path=frame_info["frame_path"],
                objective_description="No fue posible leer la matriz de píxeles del fotograma.",
                target_detected=False,
                confidence_level="ninguna",
            ),
            empty,
        )

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    telemetry = frame_info.get("telemetry", {})
    lighting_str = telemetry.get("lighting", "iluminación estándar")
    motion_ratio = float(telemetry.get("motion_ratio", 0.0))

    color_masks = {
        "rojo": cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)) | cv2.inRange(hsv, (165, 90, 70), (180, 255, 255)),
        "azul": cv2.inRange(hsv, (98, 80, 60), (132, 255, 255)),
        "amarillo": cv2.inRange(hsv, (20, 90, 80), (34, 255, 255)),
        "verde": cv2.inRange(hsv, (36, 70, 50), (85, 255, 255)),
        "blanco": cv2.inRange(hsv, (0, 0, 210), (180, 35, 255)),
    }

    detected_colors: List[str] = []
    color_regions: List[str] = []
    for cname, mask in color_masks.items():
        ratio = float(np.count_nonzero(mask)) / float(h * w)
        if ratio > 0.018:
            detected_colors.append(cname)
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                pos_x = "sector izquierdo" if cx < w * 0.35 else ("sector derecho" if cx > w * 0.65 else "sector central")
                color_regions.append(f"estructura/superficie color {cname} en {pos_x} ({round(ratio * 100, 1)}% del área)")

    moving_boxes = []
    if prev_bgr is not None and prev_bgr.shape == bgr.shape:
        diff = cv2.absdiff(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.008):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / float(max(1, ch))
                pos_x = "izquierda" if (x + cw / 2) < w * 0.35 else ("derecha" if (x + cw / 2) > w * 0.65 else "centro")
                if aspect >= 1.35:
                    moving_boxes.append(f"volumen horizontal en desplazamiento (proporción {aspect:.1f}:1, compatible con vehículo) en zona {pos_x}")
                elif aspect <= 0.85:
                    moving_boxes.append(f"silueta vertical en movimiento (proporción {aspect:.1f}:1, compatible con figura humana) en zona {pos_x}")
                else:
                    moving_boxes.append(f"elemento en movimiento activo en zona {pos_x}")

    elements: List[str] = [f"Encuadre {w}x{h}px con {lighting_str}"]
    elements.extend(color_regions[:3])
    elements.extend(moving_boxes[:3])

    desc_parts = [f"Escena con {lighting_str}."]
    if moving_boxes:
        desc_parts.append("Se registra desplazamiento físico de: " + "; ".join(moving_boxes[:2]) + ".")
    elif motion_ratio > 0.03:
        desc_parts.append(f"Se observa variación visual moderada entre fotogramas ({round(motion_ratio * 100, 1)}% de píxeles activos).")
    else:
        desc_parts.append("No se registran desplazamientos bruscos respecto al instante previo.")

    if color_regions:
        desc_parts.append("Elementos cromáticos visibles: " + ", ".join(color_regions[:2]) + ".")

    objective_desc = " ".join(desc_parts)

    target_detected = False
    conf_level = "ninguna"
    evidence_detail = ""
    q_low = search_query.lower().strip()

    if q_low:
        matched_reasons = []
        for cname in detected_colors:
            if cname in q_low:
                matched_reasons.append(f"presencia de color '{cname}' especificado en la consulta")

        if any(w in q_low for w in ("vehículo", "vehiculo", "auto", "coche", "camioneta", "carro", "moto")):
            horiz = [b for b in moving_boxes if "horizontal" in b or "vehículo" in b]
            if horiz:
                matched_reasons.append(horiz[0])

        if any(w in q_low for w in ("persona", "sujeto", "robo", "violencia", "forcejeo", "correr", "acción", "accion", "sustracción")):
            vert = [b for b in moving_boxes if "vertical" in b or "movimiento activo" in b]
            if vert and motion_ratio >= 0.04:
                matched_reasons.append(f"actividad motriz intensa ({round(motion_ratio * 100, 1)}%) con {len(vert)} silueta(s) en desplazamiento")
            elif len(moving_boxes) >= 2:
                matched_reasons.append(f"interacción/desplazamiento simultáneo de {len(moving_boxes)} elementos en escena")

        if matched_reasons:
            target_detected = True
            conf_level = "alta" if len(matched_reasons) >= 2 else "media"
            evidence_detail = "Coincidencia visual detectada: " + "; ".join(matched_reasons) + "."

    return (
        FrameAnalysis(
            frame_index=frame_info["frame_index"],
            timestamp=frame_info["timestamp"],
            timestamp_formatted=frame_info["timestamp_formatted"],
            frame_url=frame_info["frame_url"],
            frame_path=frame_info["frame_path"],
            objective_description=objective_desc,
            target_detected=target_detected,
            confidence_level=conf_level,  # type: ignore
            target_evidence_detail=evidence_detail,
            observed_elements=elements,
        ),
        bgr,
    )


async def synthesize_objective_summary_ollama(
    frames_analyzed: List[FrameAnalysis],
    transcript_segments: List[TranscriptSegment],
    search_query: str,
    vision_model: str,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
) -> Optional[Tuple[str, str]]:
    """
    Genera el resumen cronológico objetivo del video y del audio usando el LLM en Ollama en GPU CUDA.
    """
    if not frames_analyzed and not transcript_segments:
        return None

    visual_log = "\n".join(
        f"- [{f.timestamp_formatted}] {f.objective_description}"
        + (f" | Coincidencia con búsqueda: {f.target_evidence_detail}" if f.target_detected else "")
        for f in frames_analyzed
    )
    audio_log = "\n".join(
        f"- [{s.start_formatted} - {s.end_formatted}] \"{s.text}\"" for s in transcript_segments
    )

    prompt = f"""A partir del siguiente registro cronológico de fotogramas observados y segmentos de audio transcritos, redacta un informe descriptivo estrictamente objetivo.

REGISTRO VISUAL POR MARCA DE TIEMPO:
{visual_log or 'Sin fotogramas (archivo de solo audio).'}

REGISTRO DE AUDIO TRANSCRITO:
{audio_log or 'Sin locución detectada.'}

BÚSQUEDA ESPECÍFICA DEL USUARIO: "{search_query or 'Ninguna'}"

REGLAS ESTRICTAS:
- En "objective_visual_summary", explica ÚNICAMENTE lo que se ve en el video en orden cronológico, citando marcas de tiempo [MM:SS]. PROHIBIDO hacer inferencias sobre intenciones, causas ocultas o emitir juicios de valor.
- En "objective_audio_summary", resume de manera neutral y literal lo que se dice en la pista de audio (o indica si no hay voz).
- Devuelve ÚNICAMENTE un JSON válido con las claves "objective_visual_summary" y "objective_audio_summary"."""

    num_gpu = 0 if (compute_device or "").lower() == "cpu" else 99

    payload = {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 650, "num_gpu": num_gpu},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                parsed = _extract_json_dict(content)
                if parsed:
                    v_sum = sanitize_objective_text(str(parsed.get("objective_visual_summary", "")))
                    a_sum = sanitize_objective_text(str(parsed.get("objective_audio_summary", "")))
                    if v_sum:
                        return v_sum, a_sum
    except Exception:
        pass
    return None


def build_deterministic_summaries(
    frames_analyzed: List[FrameAnalysis],
    transcript_segments: List[TranscriptSegment],
    detected_intervals: List[DetectedSituationInterval],
    search_query: str,
) -> Tuple[str, str]:
    """Construye un resumen cronológico 100% fáctico y libre de juicios de valor a partir de las observaciones."""
    if not frames_analyzed:
        visual_summary = (
            "El archivo analizado corresponde únicamente a una pista de audio; "
            "no existen secuencias de video ni fotogramas para descripción visual."
        )
    else:
        lines = [
            f"Se analizaron objetivamente {len(frames_analyzed)} fotogramas clave distribuidos entre "
            f"el segundo {frames_analyzed[0].timestamp_formatted} y {frames_analyzed[-1].timestamp_formatted}. "
            "A continuación se expone la secuencia cronológica exclusivamente de los hechos visibles en pantalla:"
        ]
        for f in frames_analyzed:
            marker = " [HALLAZGO DE BÚSQUEDA]" if f.target_detected else ""
            detail = f" {f.target_evidence_detail}" if f.target_evidence_detail else ""
            lines.append(f"• [{f.timestamp_formatted}]{marker}: {f.objective_description}{detail}")

        if search_query.strip():
            if detected_intervals:
                lines.append(
                    f"\nRespecto a la situación consultada (\"{search_query}\"), se identificaron "
                    f"{len(detected_intervals)} intervalo(s) temporal(es) con indicadores visuales directos."
                )
            else:
                lines.append(
                    f"\nRespecto a la situación consultada (\"{search_query}\"), no se observaron "
                    "indicadores visuales directos en los fotogramas muestreados."
                )
        visual_summary = "\n".join(lines)

    if not transcript_segments:
        audio_summary = "No se registraron segmentos de voz o diálogo inteligible en la pista de audio."
    else:
        audio_lines = [
            f"Se transcribieron literalmente {len(transcript_segments)} segmento(s) de audio "
            f"entre [{transcript_segments[0].start_formatted}] y [{transcript_segments[-1].end_formatted}]:"
        ]
        for seg in transcript_segments[:12]:
            audio_lines.append(f"• [{seg.start_formatted} - {seg.end_formatted}]: \"{seg.text}\"")
        if len(transcript_segments) > 12:
            audio_lines.append(f"• (... y {len(transcript_segments) - 12} segmentos adicionales en la transcripción completa).")
        audio_summary = "\n".join(audio_lines)

    return visual_summary, audio_summary


def group_detected_situations(
    frames_analyzed: List[FrameAnalysis],
    transcript_segments: List[TranscriptSegment],
    sample_interval_sec: float = 2.0,
) -> List[DetectedSituationInterval]:
    """Agrupa fotogramas consecutivos donde se detectó la situación buscada."""
    positive_frames = [f for f in frames_analyzed if f.target_detected]
    if not positive_frames:
        return []

    intervals: List[DetectedSituationInterval] = []
    max_gap = max(3.5, sample_interval_sec * 1.8)
    current_group: List[FrameAnalysis] = [positive_frames[0]]

    for frame in positive_frames[1:]:
        if (frame.timestamp - current_group[-1].timestamp) <= max_gap:
            current_group.append(frame)
        else:
            intervals.append(_consolidate_group(len(intervals) + 1, current_group, transcript_segments, sample_interval_sec))
            current_group = [frame]

    if current_group:
        intervals.append(_consolidate_group(len(intervals) + 1, current_group, transcript_segments, sample_interval_sec))

    return intervals


def _consolidate_group(
    interval_id: int,
    group: List[FrameAnalysis],
    transcript_segments: List[TranscriptSegment],
    sample_interval_sec: float,
) -> DetectedSituationInterval:
    start_t = max(0.0, group[0].timestamp)
    end_t = group[-1].timestamp + round(sample_interval_sec * 0.5, 2)

    rank = {"alta": 3, "media": 2, "baja": 1, "ninguna": 0}
    best_frame = max(group, key=lambda f: rank.get(f.confidence_level, 0))
    best_conf = best_frame.confidence_level if best_frame.confidence_level in ("alta", "media", "baja") else "media"

    seen_elems = []
    for f in group:
        for el in f.observed_elements:
            if el not in seen_elems:
                seen_elems.append(el)

    descriptions = []
    for f in group:
        txt = f.target_evidence_detail or f.objective_description
        if txt and txt not in descriptions:
            descriptions.append(f"[{f.timestamp_formatted}] {txt}")

    overlapping_audio = [
        s.text
        for s in transcript_segments
        if (s.end >= start_t - 1.5) and (s.start <= end_t + 1.5)
    ]

    return DetectedSituationInterval(
        interval_id=interval_id,
        start_time=round(start_t, 2),
        end_time=round(end_t, 2),
        start_formatted=format_timestamp(start_t),
        end_formatted=format_timestamp(end_t),
        confidence_level=best_conf,  # type: ignore
        factual_description=" | ".join(descriptions),
        key_frame_url=best_frame.frame_url,
        key_frame_path=best_frame.frame_path,
        observed_elements=seen_elems[:8],
        related_audio_text=" ".join(overlapping_audio),
    )


async def run_visual_forensic_agent(
    extracted_frames: List[Dict],
    transcript_segments: List[TranscriptSegment],
    search_query: str,
    preferred_vision_model: str = DEFAULT_VISION_MODEL,
    sample_interval_sec: float = 2.0,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
) -> Tuple[List[FrameAnalysis], List[DetectedSituationInterval], str, str, str, List[str]]:
    """
    Ejecuta el Agente de IA para análisis de video en GPU CUDA (o CPU según configuración).
    """
    if not extracted_frames:
        v_sum, a_sum = build_deterministic_summaries([], transcript_segments, [], search_query)
        notes = [
            "Archivo de solo audio: se omitió el análisis de fotogramas.",
            "Transcripción literal sin inferencias ni juicios de valor.",
        ]
        return [], [], v_sum, a_sum, "Solo Audio (Sin VLM)", notes

    ollama_info = await check_ollama_status()
    use_ollama = ollama_info["online"] and len(ollama_info["vision_models"]) > 0
    active_model_name = preferred_vision_model
    accel_tag = "CPU" if (compute_device or "").lower() == "cpu" else "GPU CUDA (GTX 1060)"

    if use_ollama:
        if preferred_vision_model not in ollama_info["vision_models"]:
            active_model_name = ollama_info["vision_models"][0]
        display_model_name = f"{active_model_name} [{accel_tag}]"
    else:
        display_model_name = "OpenCV-Forensic-Local (Modo Respaldo Directo)"

    frames_analyzed: List[FrameAnalysis] = []
    prev_bgr: Optional[np.ndarray] = None
    total = len(extracted_frames)

    for idx, f_info in enumerate(extracted_frames):
        pct = 45 + int(((idx + 1) / total) * 40)
        ts_label = f_info["timestamp_formatted"]
        if progress_callback:
            progress_callback(
                f"Analizando fotograma {idx + 1}/{total} [{ts_label}] con {display_model_name}...",
                pct,
            )

        ts = f_info["timestamp"]
        near_audio = " ".join(
            s.text for s in transcript_segments if (s.start - 1.5) <= ts <= (s.end + 1.5)
        )

        analysis: Optional[FrameAnalysis] = None
        if use_ollama:
            analysis = await analyze_frame_with_ollama(
                frame_info=f_info,
                search_query=search_query,
                vision_model=active_model_name,
                audio_context=near_audio,
                compute_device=compute_device,
            )

        if analysis is None:
            analysis, prev_bgr = analyze_frame_opencv_fallback(
                frame_info=f_info,
                search_query=search_query,
                prev_bgr=prev_bgr,
            )
        else:
            prev_bgr = cv2.imread(f_info["frame_path"])

        frames_analyzed.append(analysis)

    if progress_callback:
        progress_callback("Consolidando marcas de tiempo de situaciones detectadas...", 88)

    detected_intervals = group_detected_situations(
        frames_analyzed=frames_analyzed,
        transcript_segments=transcript_segments,
        sample_interval_sec=sample_interval_sec,
    )

    if progress_callback:
        progress_callback("Redactando resumen objetivo y aplicando auditoría anti-inferencias...", 93)

    v_sum, a_sum = None, None
    if use_ollama:
        res_sum = await synthesize_objective_summary_ollama(
            frames_analyzed=frames_analyzed,
            transcript_segments=transcript_segments,
            search_query=search_query,
            vision_model=active_model_name,
            compute_device=compute_device,
        )
        if res_sum:
            v_sum, a_sum = res_sum

    if not v_sum:
        v_sum, a_sum = build_deterministic_summaries(
            frames_analyzed=frames_analyzed,
            transcript_segments=transcript_segments,
            detected_intervals=detected_intervals,
            search_query=search_query,
        )

    audit_notes = [
        "Protocolo de Descripción Fáctica activado: solo se reportan elementos visualmente verificables (formas, colores, posiciones espaciales y movimientos).",
        "Filtro Anti-Inferencias aplicado: se excluyeron suposiciones sobre intenciones, estados mentales, causas no visibles o calificativos morales/legales.",
        f"Motor de visión utilizado: {display_model_name}.",
    ]

    return frames_analyzed, detected_intervals, v_sum, a_sum or "", display_model_name, audit_notes
