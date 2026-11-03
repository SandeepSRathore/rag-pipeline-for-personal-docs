// Security note: every server string is inserted with textContent, never innerHTML,
// so model output can't inject markup, scripts or auto-loading images.
"use strict";

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const keyInput = $("api-key");
keyInput.value = sessionStorage.getItem("apiKey") || "";
keyInput.addEventListener("input", () => {
  sessionStorage.setItem("apiKey", keyInput.value.trim());
  $("role-badge").textContent = keyInput.value ? "key set" : "no key";
});
if (keyInput.value) $("role-badge").textContent = "key set";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}

function addMessage(kind, text) {
  const m = el("div", `msg ${kind}`);
  m.appendChild(el("div", "bubble", text));
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
  return m;
}

function renderAnswer(data) {
  const kind = data.blocked ? "blocked" : data.abstained ? "abstained" : "answer";
  const m = addMessage(`bot ${kind}`, data.answer);
  const meta = el("div", "meta");
  meta.appendChild(el("span", "badge", data.role));
  if (data.reason) meta.appendChild(el("span", `badge ${kind}`, data.reason));
  meta.appendChild(el("span", "rid", `request ${data.request_id}`));
  m.appendChild(meta);

  if (data.citations && data.citations.length) {
    const list = el("ol", "citations");
    for (const c of data.citations) {
      const li = el("li");
      li.appendChild(el("strong", null, `[${c.ref}] ${c.title}`));
      if (c.section) li.appendChild(el("span", null, ` › ${c.section}`));
      li.appendChild(el("code", null, c.path));
      list.appendChild(li);
    }
    m.appendChild(list);
  }

  if (data.debug) {
    const d = el("details", "debug");
    d.appendChild(el("summary", null, "retrieval & guardrail debug"));
    const t = data.debug.timings_ms || {};
    const llm = data.debug.llm || {};
    d.appendChild(el("p", null,
      `timings (ms): ${Object.entries(t).map(([k, v]) => `${k}=${v}`).join(", ")}` +
      (llm.usd !== undefined ? ` · tokens in/out ${llm.input_tokens}/${llm.output_tokens} · $${llm.usd}` : "")));
    const g = data.debug.input_guard || {};
    d.appendChild(el("p", null, `input guard: injection score ${g.injection_score ?? 0} ${JSON.stringify(g.injection_reasons || [])}`));
    if (data.debug.output_guard) {
      d.appendChild(el("p", null, `output guard: ${JSON.stringify(data.debug.output_guard)}`));
    }
    for (const r of data.debug.retrieved || []) {
      const box = el("div", `chunk ${r.kept ? "kept" : "dropped"}`);
      box.appendChild(el("div", "chunk-head",
        `${r.kept ? "✓ kept" : "✗ dropped"} · ${r.path} › ${r.section || "(intro)"}` +
        (r.fused_score !== null && r.fused_score !== undefined ? ` · rrf ${r.fused_score}` : "") +
        (r.rerank_score !== null ? ` · rerank ${r.rerank_score}` : "") +
        (r.superseded_by ? ` · superseded by ${r.superseded_by}` : "")));
      box.appendChild(el("pre", null, r.text));
      d.appendChild(box);
    }
    m.appendChild(d);
  }
  chat.scrollTop = chat.scrollHeight;
}

async function ask(question) {
  const key = keyInput.value.trim();
  if (!key) {
    addMessage("bot blocked",
      "Enter an API key in the \"API key\" box at the top right first. " +
      "Local dev keys: dev-employee-key, dev-hr-key, dev-admin-key (set in API_KEYS in .env).");
    keyInput.focus();
    keyInput.classList.add("needs-key");
    return;
  }
  keyInput.classList.remove("needs-key");
  addMessage("user", question);
  const pending = addMessage("bot pending", "Thinking…");
  $("send").disabled = true;
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": key },
      body: JSON.stringify({ question, debug: $("debug").checked }),
    });
    const data = await res.json();
    pending.remove();
    if (!res.ok) {
      addMessage("bot blocked", `Error ${res.status}: ${data.detail || "request failed"}`);
      return;
    }
    $("role-badge").textContent = `role: ${data.role}`;
    renderAnswer(data);
  } catch (e) {
    pending.remove();
    addMessage("bot blocked", "Network error.");
  } finally {
    $("send").disabled = false;
  }
}

$("ask-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  ask(q);
});
$("question").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("ask-form").requestSubmit(); }
});
$("key-form").addEventListener("submit", (ev) => ev.preventDefault());
