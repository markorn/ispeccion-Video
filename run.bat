@echo off
cd /d "%~dp0"
echo ===============================================================
echo   Iniciando VisionForense IA (FastAPI + Faster-Whisper + Qwen2.5-VL)
echo   Aceleracion CUDA v12 habilitada para NVIDIA GTX 1060 (CC 6.1)
echo   URL Local: http://127.0.0.1:8000
echo ===============================================================

set OLLAMA_LLM_LIBRARY=cuda_v12
set CUDA_VISIBLE_DEVICES=0

REM Iniciar el servidor de Ollama en segundo plano si está instalado y no está corriendo
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if "%ERRORLEVEL%"=="1" (
        echo Iniciando servidor local Ollama con CUDA v12...
        start "" /B "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >NUL 2>&1
    )
)

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
