from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(..., description="Segundo de inicio")
    end: float = Field(..., description="Segundo de fin")
    start_formatted: str = Field(..., description="Timestamp formateado MM:SS.ms")
    end_formatted: str = Field(..., description="Timestamp formateado MM:SS.ms")
    text: str = Field(..., description="Texto transcrito literal")
    confidence: float = Field(default=1.0, description="Nivel de confianza estimado (0-1)")


class FrameAnalysis(BaseModel):
    frame_index: int
    timestamp: float = Field(..., description="Segundo exacto del fotograma")
    timestamp_formatted: str = Field(..., description="Timestamp en formato MM:SS")
    frame_url: str = Field(..., description="URL relativa de la imagen capturada")
    frame_path: str = Field(..., description="Ruta local del archivo de imagen")
    objective_description: str = Field(
        ...,
        description="Descripción estrictamente visual y objetiva de lo que aparece en el fotograma, sin inferencias ni juicios de valor.",
    )
    target_detected: bool = Field(
        default=False,
        description="Indica si se observa la situación, vehículo o acción específica solicitada por el usuario.",
    )
    confidence_level: Literal["alta", "media", "baja", "ninguna"] = Field(
        default="ninguna",
        description="Grado de correspondencia visual con la situación buscada.",
    )
    target_evidence_detail: str = Field(
        default="",
        description="Detalle fáctico de los elementos visuales que coinciden con la búsqueda (ej. posición, color, postura física).",
    )
    observed_elements: List[str] = Field(
        default_factory=list,
        description="Lista de elementos físicos observables (personas, prendas, vehículos, objetos, acciones motrices).",
    )


class DetectedSituationInterval(BaseModel):
    interval_id: int
    start_time: float
    end_time: float
    start_formatted: str
    end_formatted: str
    confidence_level: Literal["alta", "media", "baja"]
    factual_description: str
    key_frame_url: str
    key_frame_path: str
    observed_elements: List[str] = Field(default_factory=list)
    related_audio_text: str = Field(default="", description="Audio transcrito que coincide en este rango temporal")


class VideoMetadata(BaseModel):
    filename: str
    media_type: Literal["video", "audio"]
    duration_seconds: float
    duration_formatted: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    file_size_mb: float = 0.0


class AnalysisReport(BaseModel):
    job_id: str
    created_at: str
    metadata: VideoMetadata
    media_url: str
    search_query: str = Field(default="", description="Situación específica buscada por el usuario")
    whisper_model_used: str
    vision_model_used: str
    language_detected: str = "es"
    full_transcript_text: str = ""
    transcript_segments: List[TranscriptSegment] = Field(default_factory=list)
    frames_analyzed: List[FrameAnalysis] = Field(default_factory=list)
    detected_situations: List[DetectedSituationInterval] = Field(default_factory=list)
    objective_visual_summary: str = Field(
        default="",
        description="Resumen cronológico y estrictamente descriptivo de lo que se observa en el video, sin inferencias ni juicios de valor.",
    )
    objective_audio_summary: str = Field(
        default="",
        description="Resumen objetivo del contenido verbal/auditivo transcrito.",
    )
    neutrality_audit_notes: List[str] = Field(
        default_factory=list,
        description="Garantías metodológicas aplicadas por el agente para evitar inferencias o juicios de valor.",
    )

