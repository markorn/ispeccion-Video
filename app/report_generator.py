import json
from pathlib import Path
from fpdf import FPDF

from app.config import REPORTS_DIR
from app.models import AnalysisReport


def _safe_pdf_text(text: str, use_unicode_font: bool) -> str:
    """Normaliza caracteres especiales para evitar errores de codificación en PDF."""
    if not text:
        return ""
    replaced = (
        text.replace("•", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    if use_unicode_font:
        return replaced
    return replaced.encode("latin-1", errors="replace").decode("latin-1")


def generate_markdown_report(report: AnalysisReport) -> Path:
    """Genera un archivo Markdown (.md) con el reporte pericial objetivo completo."""
    md_path = REPORTS_DIR / f"reporte_{report.job_id}.md"

    lines = [
        "# REPORTE DE ANÁLISIS TÉCNICO Y OBJETIVO DE VIDEO / AUDIO",
        "",
        f"- **ID de Análisis:** `{report.job_id}`",
        f"- **Fecha y Hora (UTC/Local):** {report.created_at}",
        f"- **Archivo Analizado:** `{report.metadata.filename}` ({report.metadata.media_type.upper()})",
        f"- **Duración:** {report.metadata.duration_formatted} ({report.metadata.duration_seconds} s)",
    ]
    if report.metadata.width > 0:
        lines.append(f"- **Resolución y FPS:** {report.metadata.width}x{report.metadata.height} px @ {report.metadata.fps} fps")
    lines.extend(
        [
            f"- **Modelo de Transcripción (Local):** Faster-Whisper (`{report.whisper_model_used}`) — Idioma: `{report.language_detected}`",
            f"- **Modelo de Visión (Local):** `{report.vision_model_used}`",
            f"- **Situación / Criterio Buscado:** `{report.search_query or 'Análisis general sin búsqueda específica'}`",
            "",
            "---",
            "",
            "## 1. RESUMEN VISUAL ESTRICTAMENTE OBJETIVO (SIN INFERENCIAS NI JUICIOS DE VALOR)",
            "",
            report.objective_visual_summary or "Sin observaciones visuales.",
            "",
            "---",
            "",
            "## 2. SITUACIONES ESPECÍFICAS IDENTIFICADAS (TIMESTAMPS Y EVIDENCIAS)",
            "",
        ]
    )

    if not report.detected_situations:
        lines.append("_No se registraron intervalos con coincidencia directa para la situación especificada._")
        lines.append("")
    else:
        for item in report.detected_situations:
            lines.extend(
                [
                    f"### Hallazgo #{item.interval_id} — Intervalo `[{item.start_formatted} - {item.end_formatted}]` (Correspondencia: {item.confidence_level.upper()})",
                    f"- **Descripción Fáctica:** {item.factual_description}",
                    f"- **Elementos Observables:** {', '.join(item.observed_elements) if item.observed_elements else 'N/A'}",
                    f"- **Audio Simultáneo:** {item.related_audio_text or 'Sin locución en este intervalo'}",
                    f"- **Captura de Evidencia:** `{item.key_frame_path}`",
                    "",
                ]
            )

    lines.extend(
        [
            "---",
            "",
            "## 3. TRANSCRIPCIÓN LITERAL DE AUDIO CON MARCAS DE TIEMPO",
            "",
        ]
    )

    if not report.transcript_segments:
        lines.append("_No se detectó locución o pista de audio en el archivo._")
        lines.append("")
    else:
        lines.append("| # | Inicio | Fin | Confianza | Texto Transcrito |")
        lines.append("|---|--------|-----|-----------|------------------|")
        for seg in report.transcript_segments:
            safe_txt = seg.text.replace("|", "\\|")
            lines.append(
                f"| {seg.id} | `{seg.start_formatted}` | `{seg.end_formatted}` | {int(seg.confidence * 100)}% | {safe_txt} |"
            )
        lines.append("")

    if report.frames_analyzed:
        lines.extend(
            [
                "---",
                "",
                "## 4. BITÁCORA CRONOLÓGICA DE FOTOGRAMAS ANALIZADOS",
                "",
            ]
        )
        for f in report.frames_analyzed:
            flag = " **[COINCIDENCIA DETECTADA]**" if f.target_detected else ""
            lines.append(f"- **`[{f.timestamp_formatted}]` (Fotograma #{f.frame_index})**{flag}: {f.objective_description}")
            if f.target_evidence_detail:
                lines.append(f"  - *Detalle de coincidencia:* {f.target_evidence_detail}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. PROTOCOLO DE NEUTRALIDAD Y AUDITORÍA ANTI-INFERENCIAS",
            "",
        ]
    )
    for note in report.neutrality_audit_notes:
        lines.append(f"- {note}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def generate_json_report(report: AnalysisReport) -> Path:
    """Guarda el reporte completo en formato JSON estructurado."""
    json_path = REPORTS_DIR / f"reporte_{report.job_id}.json"
    json_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def generate_pdf_report(report: AnalysisReport) -> Path:
    """Genera un informe pericial en PDF con capturas de fotogramas incrustadas."""
    pdf_path = REPORTS_DIR / f"reporte_{report.job_id}.pdf"

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Intentar registrar fuente Arial de Windows para soporte Unicode completo
    font_family = "Helvetica"
    use_unicode = False
    arial_reg = Path("C:/Windows/Fonts/arial.ttf")
    arial_bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if arial_reg.exists() and arial_bold.exists():
        try:
            pdf.add_font("ArialUnicode", style="", fname=str(arial_reg))
            pdf.add_font("ArialUnicode", style="B", fname=str(arial_bold))
            font_family = "ArialUnicode"
            use_unicode = True
        except Exception:
            font_family = "Helvetica"
            use_unicode = False

    def txt(s: str) -> str:
        return _safe_pdf_text(s, use_unicode)

    pdf.add_page()

    # Encabezado institucional / pericial
    pdf.set_fill_color(18, 25, 38)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(font_family, "B", 13)
    pdf.cell(0, 11, txt("  INFORME DE ANÁLISIS OBJETIVO DE VIDEO Y AUDIO (IA LOCAL)"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(3)

    # Ficha técnica
    pdf.set_text_color(30, 35, 45)
    pdf.set_font(font_family, "B", 10)
    pdf.cell(0, 6, txt("1. FICHA TÉCNICA DEL ARCHIVO Y PARÁMETROS DE BÚSQUEDA"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_family, "", 9)

    meta_lines = [
        f"ID de Reporte: {report.job_id}   |   Fecha: {report.created_at}",
        f"Archivo: {report.metadata.filename} ({report.metadata.media_type.upper()})   |   Tamaño: {report.metadata.file_size_mb} MB",
        f"Duración: {report.metadata.duration_formatted} ({report.metadata.duration_seconds} s)"
        + (f"   |   Resolución: {report.metadata.width}x{report.metadata.height} px ({report.metadata.fps} fps)" if report.metadata.width else ""),
        f"Motor de Audio: Faster-Whisper ({report.whisper_model_used}) [Idioma: {report.language_detected}]",
        f"Motor de Visión: {report.vision_model_used}",
        f"Situación Específica Buscada: {report.search_query or 'Ninguna (Observación general)'}",
    ]
    for mline in meta_lines:
        pdf.multi_cell(0, 5, txt(mline), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. Resumen visual objetivo
    pdf.set_font(font_family, "B", 10)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(0, 7, txt(" 2. RESUMEN VISUAL OBJETIVO (EXCLUSIVAMENTE HECHOS OBSERVABLES)"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(1)
    pdf.set_font(font_family, "", 9)
    pdf.multi_cell(0, 5, txt(report.objective_visual_summary or "Sin observaciones visuales."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 3. Situaciones detectadas y capturas de evidencia
    pdf.set_font(font_family, "B", 10)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(0, 7, txt(" 3. SITUACIONES IDENTIFICADAS Y EVIDENCIA FOTOGRÁFICA"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    if not report.detected_situations:
        pdf.set_font(font_family, "", 9)
        pdf.multi_cell(
            0,
            5,
            txt("No se registraron intervalos con coincidencia visual directa para la situación especificada."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)
    else:
        for item in report.detected_situations:
            if pdf.get_y() > 215:
                pdf.add_page()
            pdf.set_font(font_family, "B", 9)
            pdf.set_text_color(180, 35, 35)
            pdf.cell(
                0,
                6,
                txt(f"Hallazgo #{item.interval_id} | Intervalo [{item.start_formatted} - {item.end_formatted}] | Nivel de correspondencia: {item.confidence_level.upper()}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_text_color(30, 35, 45)
            pdf.set_font(font_family, "", 9)
            pdf.multi_cell(0, 5, txt(f"Descripción fáctica: {item.factual_description}"), new_x="LMARGIN", new_y="NEXT")
            if item.observed_elements:
                pdf.multi_cell(0, 5, txt(f"Elementos observados: {', '.join(item.observed_elements)}"), new_x="LMARGIN", new_y="NEXT")
            if item.related_audio_text:
                pdf.multi_cell(0, 5, txt(f"Audio simultáneo: \"{item.related_audio_text}\""), new_x="LMARGIN", new_y="NEXT")

            img_file = Path(item.key_frame_path)
            if img_file.exists():
                try:
                    pdf.ln(1)
                    pdf.image(str(img_file), w=85)
                    pdf.ln(2)
                except Exception:
                    pass
            pdf.ln(2)

    # 4. Transcripción de Audio
    pdf.set_font(font_family, "B", 10)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(0, 7, txt(" 4. TRANSCRIPCIÓN LITERAL DE AUDIO CON MARCAS DE TIEMPO"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(1)
    pdf.set_font(font_family, "", 9)

    if not report.transcript_segments:
        pdf.multi_cell(0, 5, txt("No se detectó voz o pista de audio en el archivo."), new_x="LMARGIN", new_y="NEXT")
    else:
        for seg in report.transcript_segments:
            line_str = f"[{seg.start_formatted} -> {seg.end_formatted}] (Conf: {int(seg.confidence * 100)}%): {seg.text}"
            pdf.multi_cell(0, 5, txt(line_str), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 5. Garantías de neutralidad
    pdf.set_font(font_family, "B", 10)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(0, 7, txt(" 5. AUDITORÍA METODOLÓGICA DE OBJETIVIDAD (CERO JUICIOS DE VALOR)"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(1)
    pdf.set_font(font_family, "", 8)
    for note in report.neutrality_audit_notes:
        pdf.multi_cell(0, 4.5, txt(f"- {note}"), new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))
    return pdf_path

