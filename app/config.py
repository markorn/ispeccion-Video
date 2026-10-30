import os
from pathlib import Path
import shutil
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
REPORTS_DIR = DATA_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

for directory in (DATA_DIR, UPLOADS_DIR, FRAMES_DIR, REPORTS_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def setup_cuda_environment() -> list[str]:
    """
    Registra automáticamente las librerías CUDA (cuBLAS / cuDNN) en Windows
    para que CTranslate2 (Faster-Whisper) y Ollama utilicen la GPU NVIDIA por CUDA.
    """
    registered_paths: list[str] = []
    local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    candidate_dirs = [
        local_app_data / "Programs/Ollama/lib/ollama/cuda_v12",
        local_app_data / "Programs/Ollama/lib/ollama/cuda_v13",
        local_app_data / "Programs/Ollama/lib/ollama",
    ]

    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_root.exists():
        for sub in sorted(cuda_root.glob("v*/bin"), reverse=True):
            candidate_dirs.append(sub)

    current_path = os.environ.get("PATH", "")
    for cdir in candidate_dirs:
        if cdir.exists() and cdir.is_dir():
            cdir_str = str(cdir)
            if cdir_str not in current_path:
                os.environ["PATH"] = cdir_str + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cdir_str)
                except Exception:
                    pass
            registered_paths.append(cdir_str)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cuda_v12")
    return registered_paths


REGISTERED_CUDA_PATHS = setup_cuda_environment()

# Configuración por defecto optimizada para Español (México) en NVIDIA GTX 1060 (6GB VRAM)
DEFAULT_COMPUTE_DEVICE = "cuda"  # Opciones: "cuda", "auto", "cpu"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_WHISPER_MODEL = "small"  # Opciones: tiny, base, small, medium, large-v3-turbo
DEFAULT_VISION_MODEL = "qwen2.5vl:3b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Prompt de contexto fonético y léxico para Español de México (es-MX)
DEFAULT_SPANISH_MX_PROMPT = (
    "Transcripción literal en español de México (es-MX), con ortografía, acentuación y puntuación precisas. "
    "Términos frecuentes: cámara de vigilancia, vehículo, automóvil, sedán, camioneta, pick-up, motocicleta, "
    "placas, carril, banqueta, camellón, avenida, calle, crucero, esquina, estacionamiento, local, mostrador, "
    "caja, persona, sujeto, individuo, chamarra, sudadera, playera, camisa, pantalón, gorra, casco, mochila, "
    "bolsa, celular, forcejeo, sustracción, robo, asalto, patrulla, policía, oficial, emergencia, reporte."
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def get_ffmpeg_path() -> str | None:
    """Localiza el ejecutable de FFmpeg en el sistema."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    winget_link = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
    if winget_link.exists():
        return str(winget_link)
    return None


def get_ollama_path() -> str | None:
    """Localiza el ejecutable de Ollama en Windows."""
    path = shutil.which("ollama")
    if path:
        return path
    local_prog = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
    if local_prog.exists():
        return str(local_prog)
    return None


def get_gpu_info() -> dict:
    """Obtiene información detallada de la GPU NVIDIA y el estado de CUDA en CTranslate2."""
    cuda_devices = 0
    try:
        import ctranslate2

        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_devices = 0

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
            return {
                "available": True,
                "cuda_ready": cuda_devices > 0,
                "cuda_device_count": cuda_devices,
                "compute_type": DEFAULT_WHISPER_COMPUTE_TYPE,
                "name": parts[0] if len(parts) > 0 else "NVIDIA GPU",
                "vram_total": parts[1] if len(parts) > 1 else "Desconocido",
                "vram_used": parts[2] if len(parts) > 2 else "Desconocido",
                "vram_free": parts[3] if len(parts) > 3 else "Desconocido",
                "driver_version": parts[4] if len(parts) > 4 else "Desconocido",
                "cuda_dll_paths": REGISTERED_CUDA_PATHS,
            }
    except Exception:
        pass
    return {
        "available": False,
        "cuda_ready": False,
        "cuda_device_count": 0,
        "compute_type": "int8",
        "name": "CPU (Sin GPU NVIDIA detectada)",
        "vram_total": "0 MiB",
        "vram_used": "0 MiB",
        "vram_free": "0 MiB",
        "driver_version": "N/A",
        "cuda_dll_paths": [],
    }
