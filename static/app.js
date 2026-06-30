"use strict";

const $ = (sel) => document.querySelector(sel);
const STAGES = ["extract", "denoise", "transcribe", "diarize", "clean", "summarize"];
const STAGE_LABELS = {
  extract: "Extract", denoise: "Denoise", transcribe: "Transcribe",
  diarize: "Speakers", clean: "Clean up", summarize: "Summarize", done: "Done",
};

let currentJobId = null;
let pollTimer = null;
let editing = false;

// ---------- helpers ----------
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("application/json") ? r.json() : r.text();
}

function fmtTime(sec) {
  sec = Math.floor(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

// ---------- history ----------
async function loadHistory() {
  const jobs = await api("/api/jobs");
  const ul = $("#history");
  ul.innerHTML = "";
  jobs.forEach((j) => {
    const li = document.createElement("li");
    if (j.id === currentJobId) li.classList.add("active");

    const info = document.createElement("div");
    info.className = "h-info";
    info.innerHTML =
      `<span class="h-name">${escapeHtml(j.filename)}</span>` +
      `<span class="h-meta"><span class="dot ${j.status}"></span>${j.status}</span>`;
    info.onclick = () => openJob(j.id);

    const del = document.createElement("button");
    del.className = "h-del";
    del.title = "Delete transcript";
    del.textContent = "×";
    del.onclick = (e) => { e.stopPropagation(); deleteJob(j.id, j.filename); };

    li.append(info, del);
    ul.appendChild(li);
  });
}

async function deleteJob(id, name) {
  if (!confirm(`Delete "${name}"?\nThis removes the transcript and its audio for good.`)) return;
  try {
    await api(`/api/jobs/${id}`, { method: "DELETE" });
  } catch (e) {
    alert(`Couldn't delete: ${e.message}`);
    return;
  }
  if (id === currentJobId) {
    currentJobId = null;
    clearInterval(pollTimer);
    showView("upload");
  }
  loadHistory();
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------- upload ----------
async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("denoise", $("#opt-denoise").checked);
  fd.append("diarize", $("#opt-diarize").checked);
  const { job_id } = await api("/api/upload", { method: "POST", body: fd });
  await loadHistory();
  openJob(job_id);
}

// ---------- job view ----------
function showView(which) {
  $("#upload-view").classList.toggle("hidden", which !== "upload");
  $("#job-view").classList.toggle("hidden", which !== "job");
}

async function openJob(id) {
  currentJobId = id;
  editing = false;
  showView("job");
  clearInterval(pollTimer);
  await refreshJob();
  pollTimer = setInterval(refreshJob, 1200);
  loadHistory();
}

async function retryJob() {
  try {
    await api(`/api/jobs/${currentJobId}/retry`, { method: "POST" });
    openJob(currentJobId);   // restarts polling
  } catch (e) {
    $("#error-box").innerHTML = `<div>Retry failed: ${escapeHtml(e.message)}</div>`;
  }
}

async function refreshJob() {
  let job;
  try { job = await api(`/api/jobs/${currentJobId}`); }
  catch (e) { clearInterval(pollTimer); return; }

  $("#job-filename").textContent = job.filename;

  const done = job.status === "done";
  const failed = job.status === "failed";

  $("#error-box").classList.toggle("hidden", !failed);
  if (failed) {
    $("#error-box").innerHTML =
      `<div>Something went wrong: ${escapeHtml(job.error || "unknown error")}</div>` +
      `<button id="retry-btn" class="primary-btn" style="margin-top:12px">Retry</button>`;
    $("#retry-btn").onclick = retryJob;
  }

  $("#progress-box").classList.toggle("hidden", done || failed);
  $("#result").classList.toggle("hidden", !done);

  if (!done && !failed) renderProgress(job);
  if (done) { clearInterval(pollTimer); renderResult(job); loadHistory(); }
  if (failed) { clearInterval(pollTimer); loadHistory(); }
}

function renderProgress(job) {
  const stagesEl = $("#stages");
  const activeIdx = STAGES.indexOf(job.stage);
  stagesEl.innerHTML = STAGES.map((st, i) => {
    let cls = "stage";
    if (job.stage === "done" || (activeIdx > -1 && i < activeIdx)) cls += " done";
    else if (i === activeIdx) cls += " active";
    return `<span class="${cls}">${STAGE_LABELS[st]}</span>`;
  }).join("");
  $("#bar-fill").style.width = (job.progress || 0) + "%";
  const label = job.stage ? (STAGE_LABELS[job.stage] || job.stage) : "Queued…";
  $("#progress-label").textContent = `${label} — ${job.progress || 0}%`;
}

function renderResult(job) {
  renderExportButtons();
  renderTranscript(job.segments || []);
  renderSummary(job.summary);
}

function renderTranscript(segments) {
  const el = $("#transcript");
  el.innerHTML = segments.map((s) =>
    `<div class="seg" data-start="${s.start}" data-end="${s.end}">` +
    `<span class="ts">${fmtTime(s.start)}</span>` +
    `<span class="txt">${s.speaker ? `<span class="spk">${escapeHtml(s.speaker)}:</span> ` : ""}` +
    `${escapeHtml(s.text)}</span></div>`
  ).join("");
}

function renderSummary(summary) {
  const el = $("#summary");
  if (!summary) {
    el.innerHTML = `<p class="empty">No AI summary. Configure an AI provider in Settings to get TL;DR, key points, and action items.</p>`;
    return;
  }
  el.innerHTML = miniMarkdown(summary);
}

// Tiny markdown: headings, bullets, bold.
function miniMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", inList = false;
  for (let line of lines) {
    line = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (/^##\s+/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h2>${line.replace(/^##\s+/, "")}</h2>`;
    } else if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${line.replace(/^[-*]\s+/, "")}</li>`;
    } else if (line.trim() === "") {
      if (inList) { html += "</ul>"; inList = false; }
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function renderExportButtons() {
  const box = $("#export-actions");
  box.innerHTML = "";
  ["txt", "md", "srt", "vtt", "json"].forEach((fmt) => {
    const b = document.createElement("button");
    b.className = "exp-btn";
    b.textContent = fmt.toUpperCase();
    b.onclick = () => window.open(`/api/jobs/${currentJobId}/export?format=${fmt}`, "_blank");
    box.appendChild(b);
  });
}

// ---------- transcript editing ----------
function setEditing(on) {
  editing = on;
  const el = $("#transcript");
  el.setAttribute("contenteditable", on ? "true" : "false");
  $("#edit-btn").classList.toggle("hidden", on);
  $("#save-btn").classList.toggle("hidden", !on);
  if (on) el.focus();
}

async function saveTranscript() {
  const segs = [...$("#transcript").querySelectorAll(".seg")].map((d) => ({
    start: parseFloat(d.dataset.start),
    end: parseFloat(d.dataset.end),
    speaker: d.querySelector(".spk") ? d.querySelector(".spk").textContent.replace(/:$/, "") : null,
    text: d.querySelector(".txt").textContent.replace(/^[^:]*:\s*/, (m) =>
      d.querySelector(".spk") ? "" : m),
  }));
  await api(`/api/jobs/${currentJobId}/transcript`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments: segs }),
  });
  setEditing(false);
}

// ---------- settings ----------
async function openSettings() {
  const s = await api("/api/settings");
  $("#set-whisper").value = s.whisper_model;
  $("#set-provider").value = s.llm_provider;
  $("#set-ollama-host").value = s.ollama_host;
  $("#set-ollama-model").value = s.ollama_model;
  $("#set-anthropic-model").value = s.anthropic_model;
  $("#set-openai-model").value = s.openai_model;
  // secrets are blank (placeholders show whether they're set)
  $("#set-anthropic-key").placeholder = s.anthropic_api_key_set ? "•••••••  (saved)" : "sk-ant-…";
  $("#set-openai-key").placeholder = s.openai_api_key_set ? "•••••••  (saved)" : "sk-…";
  $("#set-hf").placeholder = s.hf_token_set ? "•••••••  (saved)" : "hf_…";
  syncProviderFields();
  $("#settings-modal").classList.remove("hidden");
}

function syncProviderFields() {
  const p = $("#set-provider").value;
  document.querySelectorAll(".provider-fields").forEach((el) =>
    el.classList.toggle("hidden", el.dataset.provider !== p));
}

async function saveSettings() {
  const body = {
    whisper_model: $("#set-whisper").value,
    llm_provider: $("#set-provider").value,
    ollama_host: $("#set-ollama-host").value,
    ollama_model: $("#set-ollama-model").value,
    anthropic_model: $("#set-anthropic-model").value,
    openai_model: $("#set-openai-model").value,
  };
  const ak = $("#set-anthropic-key").value, ok = $("#set-openai-key").value, hf = $("#set-hf").value;
  if (ak) body.anthropic_api_key = ak;
  if (ok) body.openai_api_key = ok;
  if (hf) body.hf_token = hf;
  await api("/api/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  $("#settings-modal").classList.add("hidden");
}

// ---------- wiring ----------
function init() {
  const dz = $("#dropzone"), input = $("#file-input");
  $("#pick-btn").onclick = () => input.click();
  input.onchange = () => input.files[0] && uploadFile(input.files[0]);
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("drag"); };
  dz.ondragleave = () => dz.classList.remove("drag");
  dz.ondrop = (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  };

  $("#new-btn").onclick = () => { currentJobId = null; clearInterval(pollTimer); showView("upload"); loadHistory(); };

  document.querySelectorAll(".tab").forEach((t) => {
    t.onclick = () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#tab-transcript").classList.toggle("hidden", t.dataset.tab !== "transcript");
      $("#tab-summary").classList.toggle("hidden", t.dataset.tab !== "summary");
    };
  });

  $("#edit-btn").onclick = () => setEditing(true);
  $("#save-btn").onclick = saveTranscript;

  $("#settings-btn").onclick = openSettings;
  $("#settings-cancel").onclick = () => $("#settings-modal").classList.add("hidden");
  $("#settings-save").onclick = saveSettings;
  $("#set-provider").onchange = syncProviderFields;

  loadHistory();
}

document.addEventListener("DOMContentLoaded", init);
