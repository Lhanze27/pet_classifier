// ============================================================
// The Breed Classifier — frontend logic
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const API_PREDICT = "/api/predict";
const API_BREEDS = "/api/breeds";
const STREAM_INTERVAL_MS = 280;

const state = {
  mode: "upload",        // 'upload' | 'webcam'
  currentImage: null,    // dataURL of last image to classify (upload)
  stream: null,
  streamTimer: null,
  inFlight: false,
};

// ---------- Boot ----------
window.addEventListener("DOMContentLoaded", async () => {
  bindModeTabs();
  bindUpload();
  bindActions();
  await loadBreeds();
});

// ---------- Mode tabs ----------
function bindModeTabs() {
  $$(".mode").forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });
}

async function switchMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;

  $$(".mode").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".stage__panel").forEach((p) => {
    p.hidden = p.dataset.panel !== mode;
  });

  if (mode === "webcam") {
    await startWebcam();
  } else {
    stopWebcam();
    $("#status").textContent = "";
  }
  refreshActions();
}

// ---------- Upload / Paste / Drop ----------
function bindUpload() {
  const dz = $("#dropzone");
  const file = $("#file");
  const browse = $("#browse");
  const preview = $("#preview");

  browse.addEventListener("click", () => file.click());

  file.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) loadFile(f);
  });

  dz.addEventListener("click", (e) => {
    if (e.target.closest(".link")) return;
    if (state.mode === "upload") file.click();
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); })
  );
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });

  window.addEventListener("paste", (e) => {
    if (state.mode !== "upload") return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) loadFile(item.getAsFile());
  });

  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      state.currentImage = reader.result;
      preview.src = reader.result;
      preview.hidden = false;
      $(".dropzone__inner").style.display = "none";
      refreshActions();
    };
    reader.readAsDataURL(file);
  }
}

// ---------- Webcam ----------
async function startWebcam() {
  const video = $("#video");
  const hint = $("#webcam-hint");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    hint.textContent = "Live capture in progress.";
    $("#status").textContent = "Streaming";
    startStreamLoop();
  } catch (err) {
    hint.textContent = "Camera permission denied.";
    console.error(err);
  }
}

function stopWebcam() {
  if (state.streamTimer) clearInterval(state.streamTimer);
  state.streamTimer = null;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  const video = $("#video");
  video.srcObject = null;
}

function startStreamLoop() {
  if (state.streamTimer) clearInterval(state.streamTimer);
  state.streamTimer = setInterval(captureAndPredict, STREAM_INTERVAL_MS);
}

function captureFrame() {
  const video = $("#video");
  if (!video.videoWidth) return null;
  const canvas = $("#canvas");
  const out = 320;  // square output edge — matches model's square input
  canvas.width = out; canvas.height = out;

  // Center-crop the largest square from the video frame so the model sees
  // a focused, undistorted view of the middle (where the subject usually is).
  const side = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - side) / 2;
  const sy = (video.videoHeight - side) / 2;

  const ctx = canvas.getContext("2d");
  ctx.save();
  // un-mirror for the model
  ctx.translate(out, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, sx, sy, side, side, 0, 0, out, out);
  ctx.restore();
  return canvas.toDataURL("image/jpeg", 0.82);
}

async function captureAndPredict() {
  if (state.inFlight) return;
  const data = captureFrame();
  if (!data) return;
  await predict(data, { silent: true });
}

// ---------- Actions ----------
function bindActions() {
  $("#classify").addEventListener("click", async () => {
    if (!state.currentImage) return;
    await predict(state.currentImage);
  });
  $("#reset").addEventListener("click", resetUpload);
}

function refreshActions() {
  const btn = $("#classify");
  const reset = $("#reset");
  if (state.mode === "upload") {
    btn.hidden = false;
    btn.disabled = !state.currentImage;
    reset.hidden = !state.currentImage;
  } else {
    btn.hidden = true;
    reset.hidden = true;
  }
}

function resetUpload() {
  state.currentImage = null;
  $("#preview").src = "";
  $("#preview").hidden = true;
  $(".dropzone__inner").style.display = "";
  $("#file").value = "";
  clearAnalysis();
  refreshActions();
}

// ---------- Predict ----------
async function predict(image, { silent = false } = {}) {
  if (state.inFlight) return;
  state.inFlight = true;
  if (!silent) $("#status").textContent = "Analyzing…";

  try {
    const res = await fetch(API_PREDICT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderAnalysis(data);
    if (!silent) $("#status").textContent = "Complete";
  } catch (err) {
    console.error(err);
    if (!silent) $("#status").textContent = "Error";
  } finally {
    state.inFlight = false;
  }
}

// ---------- Render ----------
function clearAnalysis() {
  $("#verdict").hidden = true;
  $("#ranks").innerHTML = '<li class="ranks__empty">No specimen yet. Submit one for analysis.</li>';
  $("#warn").hidden = true;
}

function renderAnalysis(data) {
  $("#verdict").hidden = false;
  $("#species-value").textContent = data.species;
  $("#top-value").textContent = data.top[0].pretty;

  const list = $("#ranks");
  list.innerHTML = "";
  data.top.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "rank" + (i === 0 ? " is-top" : "");
    li.style.animationDelay = `${i * 60}ms`;
    li.innerHTML = `
      <span class="rank__num">${String(p.rank).padStart(2, "0")}</span>
      <span class="rank__name">${p.pretty}<span class="species">${p.species}</span></span>
      <span class="rank__prob">${(p.prob * 100).toFixed(1)}%</span>
      <span class="rank__bar"><span data-w="${(p.prob * 100).toFixed(2)}"></span></span>
    `;
    list.appendChild(li);
  });

  // animate bars next frame
  requestAnimationFrame(() => {
    $$(".rank__bar > span").forEach((bar) => {
      bar.style.width = bar.dataset.w + "%";
    });
  });

  const warn = $("#warn");
  if (data.low_confidence) {
    warn.hidden = false;
    warn.innerHTML = `Low confidence &mdash; top prediction sits below
      ${(data.threshold * 100).toFixed(0)}%. The specimen may fall outside the
      thirty-seven catalogued breeds, or framing and lighting may be obscuring
      the signal.`;
  } else {
    warn.hidden = true;
  }
}

// ---------- Breeds index ----------
async function loadBreeds() {
  try {
    const res = await fetch(API_BREEDS);
    const data = await res.json();
    fill("#list-cats", data.cats);
    fill("#list-dogs", data.dogs);
    $("#count-cats").textContent = String(data.cats.length).padStart(2, "0");
    $("#count-dogs").textContent = String(data.dogs.length).padStart(2, "0");
  } catch (err) {
    console.error("Failed to load breeds", err);
  }
}

function fill(sel, items) {
  const ul = $(sel);
  ul.innerHTML = items.map((b) => `<li>${b.pretty}</li>`).join("");
}
