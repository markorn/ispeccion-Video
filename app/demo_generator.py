import subprocess
from pathlib import Path
import cv2
import numpy as np

from app.config import UPLOADS_DIR, get_ffmpeg_path


def create_demo_forensic_video() -> Path:
    """
    Genera un video de prueba de cámara de seguridad (12 segundos, 640x360, 20 fps)
    con una pista de voz en Español de México (es-MX, Microsoft Sabina Desktop) generada localmente.
    """
    out_mp4 = UPLOADS_DIR / "demo_camara_vigilancia_mx.mp4"
    if out_mp4.exists() and out_mp4.stat().st_size > 15000:
        return out_mp4

    temp_raw_mp4 = UPLOADS_DIR / "temp_video_silent.mp4"
    temp_wav = UPLOADS_DIR / "temp_voice_es_mx.wav"

    width, height, fps, duration = 640, 360, 20, 12
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw_mp4), fourcc, fps, (width, height))

    for idx in range(total_frames):
        t = idx / float(fps)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Fondo urbano: cielo grisáceo superior y asfalto oscuro inferior
        frame[:130, :] = (78, 72, 65)
        frame[130:, :] = (52, 52, 54)
        cv2.line(frame, (0, 130), (width, 130), (130, 130, 130), 3)
        for lx in range(20, width, 80):
            cv2.rectangle(frame, (lx, 245), (lx + 40, 250), (160, 190, 200), -1)

        cv2.rectangle(frame, (40, 30), (170, 130), (95, 90, 85), -1)
        cv2.rectangle(frame, (440, 25), (590, 130), (88, 86, 92), -1)

        # Vehículo blanco estacionado fijo en sector derecho
        cv2.rectangle(frame, (455, 155), (585, 210), (235, 235, 235), -1)
        cv2.rectangle(frame, (475, 140), (560, 160), (210, 215, 220), -1)
        cv2.circle(frame, (480, 212), 13, (25, 25, 25), -1)
        cv2.circle(frame, (560, 212), 13, (25, 25, 25), -1)

        # Entre 02.5s y 09.5s: Vehículo sedán ROJO entra por la izquierda, se detiene en el centro y avanza
        if 2.5 <= t <= 9.5:
            if t < 4.5:
                car_x = int(-140 + ((t - 2.5) / 2.0) * 320)
            elif t <= 7.5:
                car_x = 180
            else:
                car_x = int(180 + ((t - 7.5) / 2.0) * 460)

            cv2.rectangle(frame, (car_x, 220), (car_x + 150, 278), (35, 30, 220), -1)
            cv2.rectangle(frame, (car_x + 25, 196), (car_x + 120, 222), (45, 40, 200), -1)
            cv2.rectangle(frame, (car_x + 32, 200), (car_x + 70, 218), (180, 200, 210), -1)
            cv2.rectangle(frame, (car_x + 76, 200), (car_x + 112, 218), (180, 200, 210), -1)
            cv2.circle(frame, (car_x + 32, 280), 14, (20, 20, 20), -1)
            cv2.circle(frame, (car_x + 118, 280), 14, (20, 20, 20), -1)

        # Entre 04.0s y 08.0s: Dos personas en movimiento rápido junto al vehículo central
        if 4.0 <= t <= 8.0:
            offset_p = int(np.sin((t - 4.0) * 6.0) * 10)
            px1 = 235 + offset_p
            py1 = 160
            cv2.circle(frame, (px1, py1), 9, (190, 200, 215), -1)
            cv2.rectangle(frame, (px1 - 10, py1 + 10), (px1 + 10, py1 + 48), (180, 80, 30), -1)
            cv2.rectangle(frame, (px1 - 8, py1 + 48), (px1 + 8, py1 + 80), (40, 40, 45), -1)

            px2 = 272 - offset_p
            py2 = 162
            cv2.circle(frame, (px2, py2), 9, (190, 200, 215), -1)
            cv2.rectangle(frame, (px2 - 10, py2 + 10), (px2 + 10, py2 + 48), (30, 210, 230), -1)
            cv2.rectangle(frame, (px2 - 8, py2 + 48), (px2 + 8, py2 + 80), (50, 50, 55), -1)

        cv2.rectangle(frame, (10, 8), (315, 34), (15, 15, 15), -1)
        cv2.putText(
            frame,
            f"CAM-01 NORTE | T+{t:05.2f}s",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()

    # Seleccionar explícitamente la voz es-MX (Microsoft Sabina Desktop) en Windows Speech Synthesizer
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { $synth.SelectVoice('Microsoft Sabina Desktop') } catch { "
        "$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [System.Globalization.CultureInfo]::GetCultureInfo('es-MX')) "
        "}; "
        f"$synth.SetOutputToWaveFile('{str(temp_wav)}'); "
        "$synth.Rate = 0; "
        "$synth.Speak('Reporte de cámara uno en avenida central. En el segundo tres ingresa un automóvil sedán color rojo por el carril izquierdo. "
        "En el segundo cinco se observan dos personas sobre la banqueta junto a la ventanilla del vehículo antes de que avance.'); "
        "$synth.Dispose();"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, timeout=20)
    except Exception:
        pass

    ffmpeg = get_ffmpeg_path()
    if ffmpeg and temp_wav.exists() and temp_wav.stat().st_size > 1000:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(temp_raw_mp4),
            "-i",
            str(temp_wav),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out_mp4),
        ]
        subprocess.run(cmd, capture_output=True, timeout=45)
    elif ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(temp_raw_mp4),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        subprocess.run(cmd, capture_output=True, timeout=45)
    else:
        temp_raw_mp4.replace(out_mp4)

    for tmp in (temp_raw_mp4, temp_wav):
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    return out_mp4
