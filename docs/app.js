const QUESTION = "Will a person and a robot or forklift collide within the next few seconds?";
const player = document.querySelector("#player");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const playBtn = document.querySelector("#play");
const scrub = document.querySelector("#scrub");
const clock = document.querySelector("#clock");
const clipsEl = document.querySelector("#clips");
const chart = document.querySelector("#chart");
const probEl = document.querySelector("#prob");
const titleEl = document.querySelector("#clip-title");
const boxesEl = document.querySelector("#boxes");
const statusEl = document.querySelector("#status");
const modelSelect = document.querySelector("#model");
const fileInput = document.querySelector("#file");

let catalog = null;
let clip = null;
let objectUrl = null;

function setStatus(text) { statusEl.textContent = text; }

function pAt(probs, t) {
  if (!probs || !probs.length) return null;
  if (t <= probs[0].t) return probs[0].p;
  const last = probs[probs.length - 1];
  if (t >= last.t) return last.p;
  for (let i = 1; i < probs.length; i++) {
    if (t <= probs[i].t) {
      const a = probs[i - 1];
      const b = probs[i];
      const u = (t - a.t) / ((b.t - a.t) || 1);
      return a.p + (b.p - a.p) * u;
    }
  }
  return last.p;
}

function nearestFrame(frames, t) {
  if (!frames || !frames.length) return null;
  let best = frames[0];
  let bestD = Math.abs(best.t - t);
  for (const frame of frames) {
    const d = Math.abs(frame.t - t);
    if (d < bestD) { best = frame; bestD = d; }
  }
  return bestD > 0.2 ? null : best;
}

function contentRect() {
  const vw = player.videoWidth || clip.width || 16;
  const vh = player.videoHeight || clip.height || 9;
  const cw = player.clientWidth;
  const ch = player.clientHeight;
  const scale = Math.min(cw / vw, ch / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

function drawOverlay() {
  const width = player.clientWidth;
  const height = player.clientHeight;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  const frame = nearestFrame(clip && clip.frames, player.currentTime || 0);
  boxesEl.replaceChildren();
  if (!frame) return;
  const rect = contentRect();
  const srcW = clip.width || player.videoWidth;
  const srcH = clip.height || player.videoHeight;
  ctx.strokeStyle = "#d7e26a";
  ctx.fillStyle = "#d7e26a";
  ctx.lineWidth = 2;
  ctx.font = "13px Segoe UI, Helvetica, sans-serif";
  for (const box of frame.boxes) {
    const [x1, y1, x2, y2] = box.bbox;
    const x = rect.x + (x1 / srcW) * rect.w;
    const y = rect.y + (y1 / srcH) * rect.h;
    const w = ((x2 - x1) / srcW) * rect.w;
    const h = ((y2 - y1) / srcH) * rect.h;
    ctx.strokeRect(x, y, w, h);
    const label = `${box.label} ${box.confidence.toFixed(2)}`;
    ctx.fillText(label, x, Math.max(14, y - 4));
    const li = document.createElement("li");
    li.textContent = label;
    boxesEl.appendChild(li);
  }
}

function drawChart(t) {
  const probs = (clip && clip.probs) || [];
  const w = 320;
  const h = 140;
  const pad = 18;
  chart.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const add = (name, attrs) => {
    const node = document.createElementNS(ns, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    chart.appendChild(node);
    return node;
  };
  add("line", { x1: pad, y1: h - pad, x2: w - 8, y2: h - pad, stroke: "#2c3832" });
  add("line", { x1: pad, y1: pad, x2: pad, y2: h - pad, stroke: "#2c3832" });
  const yOf = (p) => h - pad - p * (h - pad * 2);
  const xOf = (time) => {
    const end = Math.max(player.duration || 5, probs.length ? probs[probs.length - 1].t : 5);
    return pad + (time / end) * (w - pad - 8);
  };
  add("line", { x1: pad, y1: yOf(0.5), x2: w - 8, y2: yOf(0.5), stroke: "#e0a080", "stroke-dasharray": "4 3" });
  if (probs.length) {
    const d = probs.map((pt, i) => `${i ? "L" : "M"}${xOf(pt.t).toFixed(1)},${yOf(pt.p).toFixed(1)}`).join(" ");
    add("path", { d, fill: "none", stroke: "#8ec5ff", "stroke-width": "2" });
  }
  add("line", { x1: xOf(t), y1: pad, x2: xOf(t), y2: h - pad, stroke: "#e7efe9", "stroke-width": "1" });
}

function renderReadout() {
  const t = player.currentTime || 0;
  const p = pAt(clip && clip.probs, t);
  probEl.textContent = p == null ? "—" : p.toFixed(2);
  clock.textContent = `${t.toFixed(1)}s`;
  if (player.duration) scrub.value = String(Math.round((t / player.duration) * 1000));
  drawChart(t);
  drawOverlay();
}

function loadClip(next) {
  clip = next;
  titleEl.textContent = next.title || next.id;
  if (objectUrl && next.video !== objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  if (player.src !== next.video) {
    player.src = next.video;
    player.load();
  }
  renderReadout();
  for (const button of clipsEl.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.id === next.id);
  }
  setStatus(next.note || "");
}

async function grab(video, t) {
  await new Promise((resolve) => {
    const onSeek = () => { video.removeEventListener("seeked", onSeek); resolve(); };
    video.addEventListener("seeked", onSeek);
    video.currentTime = Math.min(t, Math.max(0, (video.duration || t) - 0.04));
  });
  const canvas = document.createElement("canvas");
  const scale = 640 / video.videoWidth;
  canvas.width = 640;
  canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.7));
  const data = await blob.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(data);
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

async function scoreWindow(jpegA, jpegB, model) {
  const response = await fetch("api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, images: [jpegA, jpegB] }),
  });
  let body = {};
  try { body = await response.json(); } catch (err) { body = {}; }
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error("Live scoring is served with the local demo, not from GitHub Pages. Open http://127.0.0.1:8765/ on the machine running the demo.");
    }
    throw new Error(body.error || response.statusText || "Scoring failed");
  }
  return body.p;
}

function showUpload(file) {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  const label = document.querySelector("#file-label");
  if (label) label.textContent = file.name;
  loadClip({
    id: "upload",
    title: file.name,
    video: objectUrl,
    width: 0,
    height: 0,
    probs: [],
    frames: [],
    note: `Loaded ${file.name}. Press Score on W&B to measure it.`,
  });
  player.play().catch(() => {});
}

async function scoreUpload(file) {
  showUpload(file);
  if (location.hostname.endsWith("github.io")) {
    setStatus(`Loaded ${file.name}. GitHub Pages can play it, but live scoring runs at http://127.0.0.1:8765/.`);
    return;
  }
  const video = document.createElement("video");
  video.src = objectUrl;
  video.muted = true;
  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error("Could not read that video."));
  });
  const duration = Math.min(video.duration || 0, 5);
  const times = [];
  for (let t = 0; t + 0.1 < duration; t = Math.round((t + 0.4) * 10) / 10) times.push(t);
  if (!times.length) throw new Error("That clip is too short to score.");
  const probs = [];
  const model = (modelSelect && modelSelect.value) || "Qwen/Qwen3.6-27B";
  for (let i = 0; i < times.length; i++) {
    setStatus(`Loaded ${file.name}. Scoring window ${i + 1} of ${times.length}…`);
    const a = await grab(video, times[i]);
    const b = await grab(video, times[i] + 0.1);
    const p = await scoreWindow(a, b, model);
    probs.push({ t: times[i], p: Math.round(p * 1000) / 1000 });
    clip.probs = probs.slice();
    renderReadout();
  }
  clip.width = video.videoWidth;
  clip.height = video.videoHeight;
  setStatus(`Scored ${file.name} with ${model}. Uploads do not have YOLO boxes.`);
}

playBtn.addEventListener("click", () => {
  if (player.paused) player.play();
  else player.pause();
});
player.addEventListener("play", () => { playBtn.textContent = "Pause"; });
player.addEventListener("pause", () => { playBtn.textContent = "Play"; });
player.addEventListener("timeupdate", renderReadout);
player.addEventListener("loadeddata", renderReadout);
scrub.addEventListener("input", () => {
  if (!player.duration) return;
  player.currentTime = (Number(scrub.value) / 1000) * player.duration;
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) showUpload(file);
});
document.querySelector("#upload-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose a short mp4 or webm first.");
    return;
  }
  scoreUpload(file).catch((error) => setStatus(error.message));
});

fetch("data/catalog.json")
  .then((response) => response.json())
  .then((data) => {
    catalog = data;
    for (const item of data.clips) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.id = item.id;
      button.textContent = item.id;
      button.addEventListener("click", () => loadClip(item));
      clipsEl.appendChild(button);
    }
    loadClip(data.clips[0]);
    setStatus(data.model_note || "");
  })
  .catch(() => setStatus("Could not load the sample clips."));
