"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask-form"), input = $("question"), btn = $("ask-btn");
const answerBox = $("answer"), answerText = $("answer-text"), answerMeta = $("answer-meta");
const modeSel = $("mode"), modeHint = $("mode-hint");
const list = $("sources-list"), empty = $("sources-empty"), status = $("status"), statusText = $("status-text");

let raw = "";          // answer text with citation placeholders
let cited = new Set(); // hit indices cited so far
let controller = null;

const CITE = (i) => `\u0000${i}\u0000`;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe Markdown: paragraphs, "-" / "*" bullets, `code`, **bold**. Input is escaped first.
function render(text) {
  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\u0000(\d+)\u0000/g, (_, i) =>
      `<button class="cite" type="button" data-hit="${i}" aria-label="Source ${+i + 1}">${+i + 1}</button>`);
  const out = [];
  let bullets = [];
  const flush = () => { if (bullets.length) { out.push(`<ul>${bullets.map((b) => `<li dir="auto">${inline(b)}</li>`).join("")}</ul>`); bullets = []; } };
  for (const block of text.split(/\n{2,}/)) {
    for (const line of block.split("\n")) {
      const m = line.match(/^\s*[-*]\s+(.*)$/);
      if (m) { bullets.push(m[1]); continue; }
      flush();
      if (line.trim()) out.push(`<p dir="auto">${inline(line)}</p>`);
    }
    flush();
  }
  return out.join("");
}

function renderSources(hits) {
  list.innerHTML = "";
  empty.hidden = hits.length > 0;
  if (!hits.length) empty.textContent = "Nothing was retrieved for this question.";
  hits.forEach((h, i) => {
    const li = document.createElement("li");
    li.className = "src";
    li.id = `src-${i}`;
    const scores = [
      `score ${h.score.toFixed(4)}`,
      h.bm25_rank ? `bm25 #${h.bm25_rank}` : "bm25 –",
      h.dense_rank ? `dense #${h.dense_rank}` : "dense –",
      `cos ${h.dense.toFixed(3)}`,
    ];
    li.innerHTML = `
      <div class="src-head"><span class="src-num">[${i + 1}]</span><span class="src-label">${escapeHtml(h.label)}</span></div>
      <div class="src-title" dir="auto">${escapeHtml(h.title)}</div>
      <div class="src-scores">${scores.map((s) => `<span>${s}</span>`).join("")}</div>
      <details><summary>show chunk</summary><pre dir="auto">${escapeHtml(h.text)}</pre></details>`;
    list.appendChild(li);
  });
}

function markCited(indices) {
  for (const i of indices) {
    cited.add(i);
    $(`src-${i}`)?.classList.add("cited");
  }
}

function flashSource(i) {
  const el = $(`src-${i}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  el.querySelector("details").open = true;
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

answerText.addEventListener("click", (e) => {
  const b = e.target.closest(".cite");
  if (b) flashSource(+b.dataset.hit);
});

async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = frame.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

async function ask(question) {
  controller?.abort();
  controller = new AbortController();
  raw = ""; cited = new Set();
  answerBox.hidden = false;
  answerBox.classList.remove("abstained");
  answerText.innerHTML = "";
  answerText.classList.add("streaming");
  answerMeta.textContent = "retrieving…";
  btn.disabled = true;
  let gate = null;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: modeSel.value }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    for await (const ev of sseEvents(res)) {
      if (ev.type === "sources") {
        renderSources(ev.hits);
        gate = ev.gate;
        answerMeta.textContent = "generating…";
      } else if (ev.type === "token") {
        raw += ev.text;
        answerText.innerHTML = render(raw);
      } else if (ev.type === "cite") {
        raw += ev.hits.map(CITE).join("");
        markCited(ev.hits);
        answerText.innerHTML = render(raw);
      } else if (ev.type === "done") {
        if (ev.abstained) answerBox.classList.add("abstained");
        const parts = [`generator: ${ev.generator}`];
        if (gate) parts.push(`${gate.mode}`, `retrieval ${gate.retrieval_ms} ms`, `term coverage ${gate.coverage}`, `max cos ${gate.max_dense}`);
        if (ev.generation_ms) parts.push(`generation ${ev.generation_ms} ms`);
        if (ev.usage) parts.push(`tokens ${ev.usage.input_tokens} in / ${ev.usage.output_tokens} out`);
        parts.push(`${cited.size} source${cited.size === 1 ? "" : "s"} cited`);
        answerMeta.textContent = parts.join(" · ");
      } else if (ev.type === "error") {
        answerMeta.innerHTML = `<span class="error">${escapeHtml(ev.message)}</span>`;
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") answerMeta.innerHTML = `<span class="error">Request failed: ${escapeHtml(err.message)}</span>`;
  } finally {
    answerText.classList.remove("streaming");
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) ask(q);
});

const HINTS = {
  dense: "dense: bge-small embeddings, cosine similarity",
  hybrid: "hybrid: BM25 + embeddings, fused by rank (RRF)",
  bm25: "bm25: keyword ranking only",
};
function updateHint() { modeHint.textContent = HINTS[modeSel.value]; }
modeSel.addEventListener("change", updateHint);

// Theme toggle: same storage key as the portfolio site.
$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const isLight = root.dataset.theme === "light";
  if (isLight) delete root.dataset.theme; else root.dataset.theme = "light";
  try { localStorage.setItem("theme", isLight ? "dark" : "light"); } catch { /* storage blocked */ }
});

async function init() {
  try {
    const health = await (await fetch("/api/health")).json();
    const offline = health.generator !== "claude";
    statusText.textContent = `${health.files} files · ${health.chunks} chunks · ${health.embedder.split(":").pop()} · ` +
      (offline ? "offline mode (no LLM)" : "answers by Claude");
    status.classList.toggle("offline", offline);
    modeSel.value = health.mode;
    updateHint();
    const examples = await (await fetch("/api/examples")).json();
    const box = $("examples");
    for (const q of examples) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", () => { input.value = q; ask(q); });
      box.appendChild(b);
    }
  } catch {
    statusText.textContent = "server unreachable";
    status.classList.add("offline");
  }
}
init();
