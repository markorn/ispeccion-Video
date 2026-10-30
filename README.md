# ispeccion-Video — Transcripción y Análisis Pericial/Objetivo de Video y Audio con IA Local

Sistema Full-Stack en Python (**FastAPI + Faster-Whisper CUDA + Ollama Qwen2.5-VL CUDA**) diseñado para el análisis forense y objetivo de grabaciones de video y audio:

1. **Transcripción con Marcas de Tiempo (`[MM:SS.ms]`):**
   * Motor **Faster-Whisper** ejecutándose sobre GPU NVIDIA CUDA (cuantización `int8`).
   * Optimizado con perfil léxico y fonético para **Español de México (`es-MX`)** y reducción acústica de ruido ambiental con FFmpeg.

2. **Detección de Situaciones Específicas mediante Agente de Visión:**
   * Localiza automáticamente vehículos (por tipo o color), personas, forcejeos, acciones de sustracción/robo o violencia física según el criterio especificado por el usuario.
   * Extrae fotogramas de evidencia clave con sus intervalos temporales y marcas de tiempo exactas.

3. **Resumen Estrictamente Objetivo (Cero Inferencias / Cero Juicios de Valor):**
   * Protocolo pericial que describe exclusivamente los hechos observables en pantalla (colores, posiciones, movimientos y objetos) sin emitir juicios de valor, suposiciones morales ni inferencias de intención.

4. **Línea de Tiempo Interactiva y Reproductor:**
   * Remux instantáneo a MP4 compatible con navegador web HTML5 para contenedores CCTV/DVR (`.avi`, `.mkv`, `.mov`).
   * Salto directo al instante exacto del video al hacer clic en cualquier hallazgo o segmento de transcripción.

5. **Exportación de Reportes Periciales:**
   * Generación automática de informes en **PDF (con capturas de fotogramas incrustadas)**, **Markdown (`.md`)** y **JSON (`.json`)**.

---

## Requisitos de Hardware y Software

* **Sistema Operativo:** Windows 10 / 11
* **Python:** 3.11+ (administrado con `uv`)
* **FFmpeg:** Instalado y disponible en el sistema.
* **Aceleración GPU (Opcional pero recomendada):** GPU NVIDIA (ej. GTX 1060 6GB) con soporte CUDA.
* **Ollama:** Para el modelo multimodal `qwen2.5vl:3b`.

---

## Instalación y Puesta en Marcha

1. **Clonar el repositorio:**
   ```powershell
   git clone https://github.com/markorn/ispeccion-Video.git
   cd ispeccion-Video
   ```

2. **Instalar dependencias con `uv`:**
   ```powershell
   uv sync
   ```

3. **Iniciar el servidor local de Ollama y descargar el modelo de visión:**
   ```powershell
   ollama pull qwen2.5vl:3b
   ```

4. **Ejecutar la aplicación:**
   ```powershell
   .\run.bat
   ```
   O manualmente:
   ```powershell
   uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Abrir en el navegador:**
   Visita: [http://127.0.0.1:8000](http://127.0.0.1:8000)
