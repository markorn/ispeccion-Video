let currentReport = null;
let selectedFile = null;
let pollTimer = null;
let lastSystemStatus = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzone();
  initChips();
  initPlayerEvents();
  initModals();
  fetchSystemStatus();
  fetchHistoryJobs();

  document.getElementById("analyzeForm").addEventListener("submit", onSubmitAnalyze);
  document.getElementById("btnRunDemo").addEventListener("click", onRunDemo);
  document.getElementById("btnRefreshJobs").addEventListener("click", fetchHistoryJobs);
  document.getElementById("transcriptSearch").addEventListener("input", onFilterTranscript);
  const btnRe = document.getElementById("btnReanalyzeCurrent");
  if (btnRe) btnRe.addEventListener("click", onReanalyzeCurrent);
});

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

function initDropzone() {
  const dz = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileNameLabel = document.getElementById("selectedFileName");

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      selectedFile = fileInput.files[0];
      fileNameLabel.textContent = `Archivo seleccionado: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
  });

  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("dragover");
  });

  dz.addEventListener("dragleave", () => {
    dz.classList.remove("dragover");
  });

  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectedFile = e.dataTransfer.files[0];
      fileInput.files = e.dataTransfer.files;
      fileNameLabel.textContent = `Archivo seleccionado: ${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
  });
}

function initChips() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("searchQuery").value = chip.dataset.query || "";
    });
  });
}

function getActiveMediaElement() {
  const v = document.getElementById("videoPlayer");
  const a = document.getElementById("audioPlayer");
  if (!v.classList.contains("hidden")) return v;
  if (!a.classList.contains("hidden")) return a;
  return null;
}

function seekTo(seconds) {
  const media = getActiveMediaElement();
  if (!media) return;
  media.currentTime = Math.max(0, seconds);
  media.play().catch(() => {});
}

function initPlayerEvents() {
  const v = document.getElementById("videoPlayer");
  const a = document.getElementById("audioPlayer");
  const playhead = document.getElementById("timelinePlayhead");
  const bar = document.getElementById("timelineBar");

  const onTimeUpdate = (mediaEl) => {
    if (!currentReport) return;
    const dur = mediaEl.duration || currentReport.metadata.duration_seconds || 1;
    const pct = Math.min(100, Math.max(0, (mediaEl.currentTime / dur) * 100));
    playhead.style.left = `${pct}%`;

    // Resaltar segmento de transcripción activo
    const cur = mediaEl.currentTime;
    document.querySelectorAll(".transcript-row").forEach((row) => {
      const st = parseFloat(row.dataset.start || "0");
      const en = parseFloat(row.dataset.end || "0");
      if (cur >= st && cur <= en) {
        row.classList.add("active-seg");
      } else {
        row.classList.remove("active-seg");
      }
    });
  };

  v.addEventListener("timeupdate", () => onTimeUpdate(v));
  a.addEventListener("timeupdate", () => onTimeUpdate(a));

  bar.addEventListener("click", (e) => {
    const media = getActiveMediaElement();
    if (!media || !currentReport) return;
    const rect = bar.getBoundingClientRect();
    const clickRatio = (e.clientX - rect.left) / rect.width;
    const dur = media.duration || currentReport.metadata.duration_seconds || 0;
    if (dur > 0) {
      seekTo(clickRatio * dur);
    }
  });
}

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/system-status");
    const data = await res.json();
    lastSystemStatus = data;

    const pillGpu = document.getElementById("pillGpu");
    if (data.gpu && data.gpu.available) {
      const cudaTag = data.gpu.cuda_ready ? "CUDA Activo" : "GPU Detectada";
      pillGpu.textContent = `${cudaTag}: ${data.gpu.name} (${data.gpu.vram_total})`;
      pillGpu.className = data.gpu.cuda_ready ? "sys-pill ok" : "sys-pill warn";
    } else {
      pillGpu.textContent = "Modo CPU";
      pillGpu.className = "sys-pill warn";
    }

    const pillWhisper = document.getElementById("pillWhisper");
    const wCuda = data.faster_whisper && data.faster_whisper.cuda_enabled ? "CUDA int8" : "CPU";
    pillWhisper.textContent = data.ffmpeg.available ? `Whisper (${wCuda}) + FFmpeg: Listo` : "FFmpeg no detectado";
    pillWhisper.className = data.ffmpeg.available ? "sys-pill ok" : "sys-pill warn";

    const pillOllama = document.getElementById("pillOllama");
    if (data.ollama && data.ollama.online && data.ollama.vision_models.length > 0) {
      const proc = data.ollama.processor || "100% GPU";
      pillOllama.textContent = `Visión: ${data.ollama.vision_models[0]} (${proc})`;
      pillOllama.className = "sys-pill ok";
    } else if (data.ollama && data.ollama.online) {
      pillOllama.textContent = "Ollama Activo (Sin VLM descargado — Respaldo OpenCV)";
      pillOllama.className = "sys-pill warn";
    } else {
      pillOllama.textContent = "Visión: Motor Local OpenCV (Ollama offline)";
      pillOllama.className = "sys-pill warn";
    }
  } catch (err) {
    console.error("Error consultando estado del sistema:", err);
  }
}

async function onSubmitAnalyze(e) {
  e.preventDefault();
  if (!selectedFile) {
    alert("Por favor selecciona un archivo de video o audio primero, o haz clic en '▶ Cargar Demo CCTV'.");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("search_query", document.getElementById("searchQuery").value);
  formData.append("compute_device", document.getElementById("computeDevice").value);
  formData.append("whisper_model", document.getElementById("whisperModel").value);
  formData.append("vision_model", document.getElementById("visionModel").value);
  formData.append("sample_interval", document.getElementById("sampleInterval").value);
  formData.append("max_frames", document.getElementById("maxFrames").value);
  formData.append("language", document.getElementById("languageSelect").value);
  formData.append("custom_vocab", document.getElementById("customVocab").value);
  formData.append("enhance_speech", document.getElementById("enhanceSpeech").checked ? "true" : "false");

  setProcessingUI(true, "Subiendo archivo e iniciando análisis...", 4);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Error al iniciar el análisis.");
    }
    const data = await res.json();
    startPollingJob(data.job_id);
  } catch (err) {
    setProcessingUI(false);
    alert(err.message);
  }
}

async function onReanalyzeCurrent() {
  if (!currentReport || !currentReport.job_id) return;
  const formData = new FormData();
  formData.append("search_query", document.getElementById("searchQuery").value);
  formData.append("compute_device", document.getElementById("computeDevice").value);
  formData.append("whisper_model", document.getElementById("whisperModel").value);
  formData.append("vision_model", document.getElementById("visionModel").value);
  formData.append("sample_interval", document.getElementById("sampleInterval").value);
  formData.append("max_frames", document.getElementById("maxFrames").value);
  formData.append("language", document.getElementById("languageSelect").value);
  formData.append("custom_vocab", document.getElementById("customVocab").value);
  formData.append("enhance_speech", document.getElementById("enhanceSpeech").checked ? "true" : "false");

  setProcessingUI(true, "Re-analizando video con escaneo de fotogramas y conversión MP4...", 5);
  try {
    const res = await fetch(`/api/reanalyze/${currentReport.job_id}`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Error al re-analizar.");
    }
    const data = await res.json();
    startPollingJob(data.job_id);
  } catch (err) {
    setProcessingUI(false);
    alert(err.message);
  }
}

async function onRunDemo() {
  const formData = new FormData();
  const q = document.getElementById("searchQuery").value.trim() || "Vehículo color rojo o acción de personas junto al vehículo";
  document.getElementById("searchQuery").value = q;

  formData.append("search_query", q);
  formData.append("compute_device", document.getElementById("computeDevice").value);
  formData.append("whisper_model", document.getElementById("whisperModel").value);
  formData.append("vision_model", document.getElementById("visionModel").value);
  formData.append("sample_interval", document.getElementById("sampleInterval").value);
  formData.append("max_frames", document.getElementById("maxFrames").value);
  formData.append("language", document.getElementById("languageSelect").value);
  formData.append("custom_vocab", document.getElementById("customVocab").value);
  formData.append("enhance_speech", document.getElementById("enhanceSpeech").checked ? "true" : "false");

  setProcessingUI(true, "Generando video CCTV de demostración con voz en español...", 5);

  try {
    const res = await fetch("/api/analyze-demo", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Error al lanzar demo.");
    }
    const data = await res.json();
    startPollingJob(data.job_id);
  } catch (err) {
    setProcessingUI(false);
    alert(err.message);
  }
}

function setProcessingUI(active, message = "", pct = 0) {
  const box = document.getElementById("progressBox");
  const btn = document.getElementById("btnSubmitAnalysis");
  const demoBtn = document.getElementById("btnRunDemo");

  if (active) {
    box.classList.remove("hidden");
    btn.disabled = true;
    demoBtn.disabled = true;
    document.getElementById("progressMessage").textContent = message;
    document.getElementById("progressPercent").textContent = `${pct}%`;
    document.getElementById("progressFill").style.width = `${pct}%`;
  } else {
    btn.disabled = false;
    demoBtn.disabled = false;
    setTimeout(() => box.classList.add("hidden"), 1200);
  }
}

function startPollingJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();

      setProcessingUI(true, data.message || "Procesando...", data.progress || 50);

      if (data.status === "completed" && data.report) {
        clearInterval(pollTimer);
        pollTimer = null;
        setProcessingUI(false, "¡Completado!", 100);
        renderReport(data.report);
        fetchHistoryJobs();
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        setProcessingUI(false);
        alert(data.message || "Ocurrió un error durante el análisis.");
      }
    } catch (e) {
      console.error("Error consultando trabajo:", e);
    }
  }, 1000);
}

function renderReport(report) {
  currentReport = report;

  // 1. Encabezado y botones de exportación
  document.getElementById("activeReportTitle").textContent =
    `${report.metadata.filename} — Duración: ${report.metadata.duration_formatted}`;
  document.getElementById("activeReportSubtitle").textContent =
    `Búsqueda: "${report.search_query || "General"}" | Audio: Faster-Whisper (${report.whisper_model_used}) | Visión: ${report.vision_model_used}`;

  const expBox = document.getElementById("exportButtons");
  expBox.classList.remove("hidden");
  document.getElementById("btnExportPdf").href = `/api/reports/${report.job_id}/pdf`;
  document.getElementById("btnExportMd").href = `/api/reports/${report.job_id}/md`;
  document.getElementById("btnExportJson").href = `/api/reports/${report.job_id}/json`;

  // 2. Configurar reproductor de video o audio
  const v = document.getElementById("videoPlayer");
  const a = document.getElementById("audioPlayer");
  const ph = document.getElementById("mediaPlaceholder");
  ph.classList.add("hidden");

  if (report.metadata.media_type === "video") {
    a.pause();
    a.classList.add("hidden");
    v.src = report.media_url;
    v.classList.remove("hidden");
  } else {
    v.pause();
    v.classList.add("hidden");
    a.src = report.media_url;
    a.classList.remove("hidden");
  }

  // 3. Renderizar Línea de Tiempo
  renderTimelineMarkers(report);

  // 4. Pestaña 1: Resumen Objetivo
  document.getElementById("visualSummaryText").textContent =
    report.objective_visual_summary || "Sin observaciones visuales.";
  document.getElementById("audioSummaryText").textContent =
    report.objective_audio_summary || "Sin transcripción de voz.";

  const auditUl = document.getElementById("auditNotesList");
  auditUl.innerHTML = "";
  (report.neutrality_audit_notes || []).forEach((note) => {
    const li = document.createElement("li");
    li.textContent = note;
    auditUl.appendChild(li);
  });

  // 5. Pestaña 2: Situaciones Detectadas
  const sitList = report.detected_situations || [];
  document.getElementById("countSituations").textContent = sitList.length;
  const sitContainer = document.getElementById("situationsContainer");
  sitContainer.innerHTML = "";

  if (sitList.length === 0) {
    sitContainer.innerHTML = `<p class="empty-muted">No se identificaron intervalos con coincidencia directa para "${escapeHtml(report.search_query || "la búsqueda")}". Revisa la pestaña 4 para ver todos los fotogramas analizados.</p>`;
  } else {
    sitList.forEach((item) => {
      const card = document.createElement("div");
      card.className = `situation-card conf-${item.confidence_level}`;
      const elemsHtml = (item.observed_elements || [])
        .map((el) => `<span class="elem-tag">${escapeHtml(el)}</span>`)
        .join("");

      card.innerHTML = `
        <div class="situation-img-wrap" title="Haz clic para reproducir en [${item.start_formatted}] o doble clic para ampliar imagen">
          <img src="${item.key_frame_url}" alt="Evidencia ${item.start_formatted}" />
          <span class="ts-badge">▶ ${item.start_formatted} - ${item.end_formatted}</span>
          <span class="conf-badge ${item.confidence_level}">Coincidencia ${item.confidence_level}</span>
        </div>
        <div class="situation-body">
          <div><strong>Descripción Objetiva:</strong> ${escapeHtml(item.factual_description)}</div>
          ${item.related_audio_text ? `<div><strong>Audio en este intervalo:</strong> "${escapeHtml(item.related_audio_text)}"</div>` : ""}
          <div class="elements-tags">${elemsHtml}</div>
        </div>
      `;

      const imgWrap = card.querySelector(".situation-img-wrap");
      imgWrap.addEventListener("click", () => seekTo(item.start_time));
      imgWrap.addEventListener("dblclick", () =>
        openImageModal(item.key_frame_url, `Evidencia [${item.start_formatted} - ${item.end_formatted}]`, item.factual_description)
      );
      sitContainer.appendChild(card);
    });
  }

  // 6. Pestaña 3: Transcripción con Timestamps
  renderTranscriptSegments(report.transcript_segments || []);

  // 7. Pestaña 4: Bitácora de Fotogramas
  const frames = report.frames_analyzed || [];
  document.getElementById("countFrames").textContent = frames.length;
  const framesGrid = document.getElementById("allFramesGrid");
  framesGrid.innerHTML = "";

  if (frames.length === 0) {
    framesGrid.innerHTML = `<p class="empty-muted">Este archivo no contiene secuencia de video.</p>`;
  } else {
    frames.forEach((f) => {
      const fc = document.createElement("div");
      fc.className = `frame-card ${f.target_detected ? "matched" : ""}`;
      fc.innerHTML = `
        <img src="${f.frame_url}" alt="Frame ${f.timestamp_formatted}" />
        <div class="frame-card-body">
          <div><strong>[${f.timestamp_formatted}]</strong> ${f.target_detected ? '<span style="color:#f87171">● Coincidencia</span>' : ""}</div>
          <div style="margin-top:4px;color:#cbd5e1">${escapeHtml(f.objective_description)}</div>
        </div>
      `;
      fc.addEventListener("click", () => seekTo(f.timestamp));
      fc.addEventListener("dblclick", () =>
        openImageModal(f.frame_url, `Fotograma #${f.frame_index} [${f.timestamp_formatted}]`, f.objective_description)
      );
      framesGrid.appendChild(fc);
    });
  }
}

function renderTimelineMarkers(report) {
  const bar = document.getElementById("timelineBar");
  const playhead = document.getElementById("timelinePlayhead");
  bar.innerHTML = "";
  bar.appendChild(playhead);

  const dur = Math.max(1, report.metadata.duration_seconds || 1);

  // Marcadores de fotogramas muestreados
  (report.frames_analyzed || []).forEach((f) => {
    const dot = document.createElement("div");
    dot.className = "timeline-marker frame-dot";
    dot.style.left = `${Math.min(99, (f.timestamp / dur) * 100)}%`;
    dot.title = `[${f.timestamp_formatted}] ${f.objective_description}`;
    dot.addEventListener("click", (e) => {
      e.stopPropagation();
      seekTo(f.timestamp);
    });
    bar.appendChild(dot);
  });

  // Marcadores de intervalos de situaciones detectadas
  (report.detected_situations || []).forEach((sit) => {
    const m = document.createElement("div");
    const cls = sit.confidence_level === "alta" ? "high" : "med";
    m.className = `timeline-marker ${cls}`;
    const leftPct = Math.min(98, (sit.start_time / dur) * 100);
    const widthPct = Math.max(1.8, ((sit.end_time - sit.start_time) / dur) * 100);
    m.style.left = `${leftPct}%`;
    m.style.width = `${Math.min(100 - leftPct, widthPct)}%`;
    m.title = `Hallazgo [${sit.start_formatted} - ${sit.end_formatted}]: ${sit.factual_description}`;
    m.addEventListener("click", (e) => {
      e.stopPropagation();
      seekTo(sit.start_time);
    });
    bar.appendChild(m);
  });
}

function renderTranscriptSegments(segments) {
  document.getElementById("countSegments").textContent = segments.length;
  const list = document.getElementById("transcriptList");
  list.innerHTML = "";

  if (segments.length === 0) {
    list.innerHTML = `<p class="empty-muted">No se detectaron diálogos o pista de voz en este archivo.</p>`;
    return;
  }

  segments.forEach((seg) => {
    const row = document.createElement("div");
    row.className = "transcript-row";
    row.dataset.start = seg.start;
    row.dataset.end = seg.end;
    row.dataset.text = seg.text.toLowerCase();
    row.innerHTML = `
      <span class="seg-time">▶ ${seg.start_formatted} → ${seg.end_formatted}</span>
      <span class="seg-text">${escapeHtml(seg.text)}</span>
      <span class="seg-conf">${Math.round((seg.confidence || 0.9) * 100)}%</span>
    `;
    row.addEventListener("click", () => seekTo(seg.start));
    list.appendChild(row);
  });
}

function onFilterTranscript(e) {
  const q = e.target.value.toLowerCase().trim();
  document.querySelectorAll(".transcript-row").forEach((row) => {
    const txt = row.dataset.text || "";
    row.style.display = !q || txt.includes(q) ? "flex" : "none";
  });
}

async function fetchHistoryJobs() {
  const container = document.getElementById("historyList");
  try {
    const res = await fetch("/api/jobs");
    const data = await res.json();
    const jobs = data.jobs || [];
    if (jobs.length === 0) {
      container.innerHTML = `<p class="empty-muted">Aún no hay análisis previos.</p>`;
      return;
    }
    container.innerHTML = "";
    jobs.forEach((j) => {
      const div = document.createElement("div");
      div.className = "history-item";
      div.innerHTML = `
        <div class="history-item-title">
          <span>${escapeHtml(j.filename || j.job_id)}</span>
          <span style="color:#60a5fa">${j.situations_count} hallazgos</span>
        </div>
        <div class="history-item-sub">
          ${escapeHtml(j.created_at || "")} • ${j.segments_count} segs audio
        </div>
      `;
      div.addEventListener("click", async () => {
        const r = await fetch(`/api/jobs/${j.job_id}`);
        if (r.ok) {
          const d = await r.json();
          if (d.report) renderReport(d.report);
        }
      });
      container.appendChild(div);
    });
  } catch (e) {
    console.error("Error cargando historial:", e);
  }
}

function initModals() {
  const diagModal = document.getElementById("diagModal");
  document.getElementById("btnOpenDiag").addEventListener("click", async () => {
    diagModal.classList.remove("hidden");
    await fetchSystemStatus();
    renderDiagModal();
  });
  document.getElementById("btnCloseDiag").addEventListener("click", () => {
    diagModal.classList.add("hidden");
  });

  const imgModal = document.getElementById("imageModal");
  document.getElementById("btnCloseImageModal").addEventListener("click", () => {
    imgModal.classList.add("hidden");
  });
}

function renderDiagModal() {
  const body = document.getElementById("diagModalBody");
  if (!lastSystemStatus) {
    body.innerHTML = "<p>No se pudo obtener el estado del servidor.</p>";
    return;
  }
  const s = lastSystemStatus;
  const vModels = (s.ollama && s.ollama.vision_models) ? s.ollama.vision_models.join(", ") : "Ninguno";
  const dllPaths = (s.gpu && s.gpu.cuda_dll_paths && s.gpu.cuda_dll_paths.length > 0)
    ? s.gpu.cuda_dll_paths.join("<br/>")
    : "Rutas del sistema";

  body.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px">
      <div>
        <strong>1. Aceleración GPU NVIDIA CUDA:</strong>
        ${s.gpu.available ? `<span style="color:#34d399">${s.gpu.name} (Total: ${s.gpu.vram_total} | Usada: ${s.gpu.vram_used || "N/A"} | Libre: ${s.gpu.vram_free} | Driver: ${s.gpu.driver_version || "OK"})</span>` : "Modo CPU"}
      </div>
      <div>
        <strong>2. Estado de CTranslate2 / Faster-Whisper en CUDA:</strong>
        ${s.gpu.cuda_ready ? `<span style="color:#34d399">Habilitado (${s.gpu.cuda_device_count} GPU detectada, cuantización ${s.gpu.compute_type} optimizada para Pascal GTX 1060)</span>` : "Ejecutando en CPU"}
      </div>
      <div>
        <strong>3. Servidor Multimodal Ollama (GPU Offload):</strong>
        ${s.ollama.online ? `<span style="color:#34d399">En línea — Aceleración: ${s.ollama.processor || "100% GPU (CUDA)"}</span>` : '<span style="color:#fbbf24">No detectado en ejecución</span>'}
      </div>
      <div><strong>Modelos de Visión instalados en Ollama:</strong> ${vModels || "Ninguno"}</div>
      <div style="background:#0e1420;padding:12px;border-radius:8px;border:1px solid #263248;font-size:0.78rem">
        <strong>Librerías CUDA (cuBLAS v12/v13) enlazadas automáticamente:</strong>
        <div style="margin-top:4px;color:#9aa7bd;font-family:monospace">${dllPaths}</div>
        <div style="margin-top:8px;color:#a7f3d0">
          ✓ Gestión inteligente de VRAM (6 GB): el sistema libera automáticamente la VRAM de Ollama antes de ejecutar Faster-Whisper en CUDA y viceversa, garantizando que ambos modelos corran al 100% en tu GTX 1060 sin desbordar memoria.
        </div>
      </div>
    </div>
  `;
}

function openImageModal(url, title, caption) {
  document.getElementById("imageModalTitle").textContent = title;
  document.getElementById("imageModalImg").src = url;
  document.getElementById("imageModalCaption").textContent = caption || "";
  document.getElementById("imageModal").classList.remove("hidden");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

