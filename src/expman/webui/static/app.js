"use strict";

const PAGE = document.body.dataset.page === "batch" ? "batch" : "list";
const BATCH_ID = (() => {
  const match = location.pathname.match(/^\/b\/(\d+)(?:\/|$)/);
  return match ? Number(match[1]) : null;
})();

const state = {
  snapshot: null,
  connected: false,
};

const KIND = { launch: "启动", finish: "结束", reuse: "复用", shed: "撤销", probe: "探测" };
const STATUS_TEXT = { running: "运行中", stopped: "已停止", finished: "已完成", failed: "有失败" };
const STATUS_CLASS = { running: "ok", stopped: "warn", finished: "", failed: "bad" };
// Each Gantt bar uses the status of its own attempt, including live attempts.
const BAR_CLASS = { succeeded: "ok", failed: "bad", cancelled: "warn" };
const SOURCE_TEXT = {
  last: "最近一次采样",
  archive: "持久化记录",
  missing: "记录不可读",
};

const app = document.getElementById("app");

function esc(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function p2(n) { return String(n).padStart(2, "0"); }
function ok(value) { return value != null && typeof value === "number" && Number.isFinite(value); }
function gib(kb) { return ok(kb) ? (kb / 1048576).toFixed(1) : "–"; }
function pctOf(value, total) { return ok(value) && ok(total) && total > 0 ? (100 * value) / total : 0; }
function clamp(value) { return Math.max(0, Math.min(100, ok(value) ? value : 0)); }
function secs(seconds, upper = false) {
  if (!ok(seconds)) return "??:??:??";
  const minutes = Math.max(0, seconds) / 60;
  const m = upper ? Math.ceil(minutes) : Math.floor(minutes);
  return p2(Math.floor(m / 1440)) + ":" + p2(Math.floor((m % 1440) / 60)) + ":" + p2(m % 60);
}
function compact(seconds) {
  if (!ok(seconds)) return "–";
  const s = Math.round(Math.max(0, seconds));
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + p2(s % 60) + "s";
  if (s < 86400) return Math.floor(s / 3600) + "h" + p2(Math.floor((s % 3600) / 60)) + "m";
  return Math.floor(s / 86400) + "d" + p2(Math.floor((s % 86400) / 3600)) + "h";
}
function span(lower, upper) { return ok(lower) && ok(upper) ? secs(lower) + "～" + secs(upper, true) : "–"; }
function bar(segments, envelope, cls) {
  const fills = segments.filter((s) => ok(s.size) && s.size > 0)
    .map((s) => '<span class="seg" style="left:' + clamp(s.from) + "%;width:" + clamp(s.size) + "%;background:" + s.color + '"></span>').join("");
  const box = envelope && envelope.size > 0 ? '<span class="env" style="left:' + clamp(envelope.from) + "%;width:" + clamp(envelope.size) + '%"></span>' : "";
  return '<div class="track ' + (cls || "") + '">' + fills + box + "</div>";
}
function memoryMark(kb, percent, cls) {
  if (!ok(kb) || !ok(percent)) return "";
  const label = gib(kb) + " GiB";
  return '<span class="memory-mark ' + cls + '" style="--at:' + clamp(percent) +
    '%;--label-width:' + label.length + 'ch">' + label + '</span>';
}
function pill(status) {
  return '<span class="pill ' + (STATUS_CLASS[status] || "") + '">' + esc(STATUS_TEXT[status] || status || "") + "</span>";
}
function section(title, body) {
  return '<h2 class="section">' + esc(title) + "</h2>" + body;
}
function unavailable(title, message) {
  return section(title, '<div class="card empty">' + esc(message) + "</div>");
}

function notices(s) {
  const parts = [];
  if (s.degraded) parts.push('<div class="notice">看板降级：' + esc(s.degraded) + "</div>");
  if (!state.connected) parts.push('<div class="notice warn">与看板的连接已断开，正在重连…</div>');
  const age = Date.now() / 1000 - (s.generated_at || 0);
  if (state.connected && age > 5) parts.push('<div class="notice warn">看板数据已 ' + Math.round(age) + " 秒未刷新</div>");
  return parts.join("");
}

function connectionPill() {
  return state.connected
    ? '<span class="pill ok"><span class="dot pulse"></span>实时</span>'
    : '<span class="pill bad"><span class="dot" style="background:var(--danger)"></span>已断开</span>';
}

function themeButton() {
  return '<button data-act="theme">' + (document.documentElement.dataset.theme === "dark" ? "亮色" : "暗色") + "</button>";
}

function sourceNote(s) {
  if (!s || s.source === "live") return "";
  if (!s.running) return '<div class="notice warn">批次已停止。</div>';
  const when = s.source === "last" && ok(s.age_seconds) ? "（" + compact(s.age_seconds) + " 前）" : "";
  return '<div class="notice warn">批次进程仍在运行。数据来源为' +
    esc(SOURCE_TEXT[s.source] || "持久化记录") + when + "。实时面板暂无数据。</div>";
}

/* ---------------------------------------------------------------- list page */

function batchRow(b) {
  const counts = b.counts || {};
  const total = b.experiments || 0;
  const done = counts.succeeded || 0;
  const failed = counts.failed || 0;
  const running = counts.running || 0;
  const segments = [
    { from: 0, size: pctOf(done, total), color: "var(--done)" },
    { from: pctOf(done, total), size: pctOf(failed, total), color: "var(--danger)" },
    { from: pctOf(done + failed, total), size: pctOf(running, total), color: "var(--running)" },
  ];
  const remaining = b.remaining || {};
  const remainingText = ok(remaining.lower_seconds) && ok(remaining.upper_seconds)
    ? compact(remaining.lower_seconds) + "～" + compact(remaining.upper_seconds)
    : "–";
  return '<div class="row batch-row">' +
    '<span class="mono muted">' + esc(b.id) + "</span>" +
    '<a class="name" href="/b/' + esc(b.id) + '/">' + esc(b.name || "batch") + "</a>" +
    pill(b.status) +
    "<div>" + bar(segments, null, "thin") + "</div>" +
    '<span class="mono">' + running + " / " + (counts.pending || 0) + "</span>" +
    '<span class="mono">' + secs(b.elapsed_seconds) + "</span>" +
    '<span class="mono">' + remainingText + "</span>" +
    "</div>";
}

function renderList(s) {
  const batches = s.batches || [];
  const head = '<div class="row batch-row head-row">' +
    ["ID", "批次", "状态", "进度", "运行/排队", "已运行", "预计剩余"]
      .map((t) => "<span>" + t + "</span>").join("") + "</div>";
  let html = notices(s);
  html += '<div class="top"><span class="title">expman 运行看板</span>' +
    '<span class="spacer"></span>' + themeButton() + connectionPill() + "</div>";
  if (!batches.length) {
    html += '<div class="card empty">当前目录下还没有批次。</div>';
  } else {
    html += '<div class="card rows scroll-x">' + head + batches.map(batchRow).join("") + "</div>";
  }
  app.innerHTML = html;
  wire();
}

/* -------------------------------------------------------------- detail page */

function topBar(detail) {
  const name = detail ? (detail.batch || {}).name || "batch" : "批次 " + BATCH_ID;
  const stopButton = detail && detail.running
    ? '<button class="danger" data-act="stop" data-name="' + esc(name) + '">停止实验</button>' : "";
  return '<div class="top">' +
    '<a class="pill" href="/">← 全部批次</a>' +
    '<span class="title">' + esc(name) + "</span>" +
    (detail ? pill(detail.status) : "") +
    '<span class="spacer"></span>' +
    themeButton() + stopButton + connectionPill() + "</div>";
}

function tiles(s) {
  const b = s.batch || {};
  const counts = b.counts || {};
  const remaining = b.remaining || {};
  const spanText = ok(remaining.lower_seconds) && ok(remaining.upper_seconds)
    ? span(remaining.lower_seconds, remaining.upper_seconds) : "??:??:??";
  return '<div class="grid tiles">' + [
    tile("已运行", secs(b.elapsed_seconds)),
    tile("预计剩余", spanText),
    tile("运行 / 排队", (counts.running || 0) + " / " + (counts.pending || 0)),
  ].join("") + "</div>";
}

function tile(label, value) {
  return '<div class="tile"><div class="label">' + esc(label) + '</div><div class="value mono">' + esc(value) +
    "</div></div>";
}

function stages(s) {
  const list = s.stages || [];
  if (!list.length) return "";
  const cards = list.map((stage) => {
    const total = stage.groups_total;
    const done = stage.groups_done;
    const running = s.running === false ? 0 : stage.groups_running;
    const pending = stage.groups_pending ?? (ok(total) && total === done ? 0 : null);
    const counts = [done, running, pending].map((value) => ok(value) ? value : "–").join("/");
    const segments = [
      { from: 0, size: pctOf(done, total), color: "var(--done)" },
      { from: pctOf(done, total), size: pctOf(running, total), color: "var(--running)" },
    ];
    return '<div class="card">' +
      '<div class="stage-head"><span class="stage-name">' + esc(stage.name || "Stage") + '</span><span class="mono">' + counts + "</span></div>" +
      '<div style="margin-top:8px">' + bar(segments) + "</div></div>";
  });
  return section("各阶段运行情况", '<div class="grid stages">' + cards.join("") + "</div>");
}

function workers(s) {
  const list = s.workers || [];
  if (!list.length) return '<div class="card empty">当前没有运行中的尝试</div>';
  const head = '<div class="row worker-row head-row">' +
    ["RUN", "STAGE", "#", "GPU", "已运行", "内存", "显存"].map((t) => "<span>" + t + "</span>").join("") + "</div>";
  return '<div class="card scroll-x">' + head + list.map(workerRow).join("") + "</div>";
}

function workerMemoryBar(used, budget) {
  const scale = Math.max(used ?? 0, budget ?? 0);
  const usedPercent = pctOf(used, scale);
  const budgetPercent = pctOf(budget, scale);
  const segments = [
    { from: 0, size: budgetPercent, color: "var(--other)" },
    { from: 0, size: usedPercent, color: "var(--actual)" },
  ];
  return '<div class="worker-meter">' +
    memoryMark(used, usedPercent, "worker-used") +
    bar(segments, null, "wide") +
    memoryMark(budget, budgetPercent, "worker-budget") + "</div>";
}

function workerRow(w) {
  const host = w.host || {};
  const gpu = w.gpu || {};
  return '<div class="row worker-row">' +
    '<span class="mono">' + esc(w.run) + "</span>" +
    "<span>" + esc(w.stage || "S" + w.stage_index) + "</span>" +
    '<span class="mono">a' + esc(w.attempt) + "</span>" +
    '<span class="mono">' + esc(w.device) + "</span>" +
    '<span class="mono">' + secs(w.elapsed_seconds) + "</span>" +
    workerMemoryBar(host.resident_kb, host.reserved_kb) +
    workerMemoryBar(gpu.used_kb, gpu.budget_kb) + "</div>";
}

function timeline(s) {
  const attempts = (s.gantt || []).filter((r) => !r.reused);
  const marks = (s.batch && s.batch.session_marks) || [];
  if (!attempts.length)
    return '<div class="card empty">暂无带起止时刻的尝试记录</div>';
  // Use batch elapsed time or the latest recorded end, whichever is greater.
  // Both exclude downtime between invocations. Fold marks show resume points.
  const span = Math.max(
    ...[attempts.map((r) => r.end_seconds), (s.batch || {}).elapsed_seconds || 0]
      .flat().filter(ok), 0,
  );
  if (!ok(span) || span <= 0)
    return '<div class="card empty">批次才开始，还没有可画的时间线</div>';
  // Attempts are sorted by start time. Reuse the first lane whose previous
  // attempt has ended, adding a lane when every existing one is occupied.
  const lanes = [];
  attempts.forEach((r) => {
    const start = Math.max(0, r.start_seconds);
    let lane = lanes.findIndex((free) => free <= start + 1e-9);
    if (lane === -1) { lane = lanes.length; lanes.push(0); }
    lanes[lane] = r.end_seconds;
    r._lane = lane;
  });
  const bars = attempts.map((r) => {
    const left = pctOf(r.start_seconds, span);
    const raw = pctOf(r.end_seconds - r.start_seconds, span);
    // Keep very short attempts visible.
    const width = Math.max(raw, 0.4);
    const cls = "gantt-bar " + (BAR_CLASS[r.status] || "");
    const label = esc(r.run);
    return '<span class="' + cls + '" style="left:' + left + "%;width:" + width + '%" title="' +
      esc(r.run) + " · " + esc(r.stage || "S" + r.stage_index) + " a" + esc(r.attempt) +
      " · " + secs(r.start_seconds) + "～" + secs(r.end_seconds) + '">' + label + "</span>";
  });
  const body = lanes.map((_, lane) => {
    const here = bars.filter((_, i) => attempts[i]._lane === lane).join("");
    return '<div class="gantt-lane">' + here + "</div>";
  }).join("");
  const folds = marks.filter((m) => m > 0 && m < span).map((m) =>
    '<span class="gantt-fold" style="left:' + pctOf(m, span) + '%" title="此处中断过，之后恢复"></span>'
  ).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const text = fraction === 0 ? "0" : compact(span * fraction);
    const place = fraction === 0
      ? ""
      : fraction === 1 ? ";transform:translateX(-100%)" : ";transform:translateX(-50%)";
    return '<span class="tl-tick" style="left:' + fraction * 100 + "%" + place + '">' + esc(text) + "</span>";
  }).join("");
  return '<div class="card"><div class="gantt">' + body +
    '<div class="gantt-axis">' + ticks + folds + "</div></div></div>";
}


function resourceBar(r) {
  const total = r.total_kb;
  const other = r.other_kb;
  const actual = r.expman_actual_kb;
  const budget = r.expman_budget_kb;
  if (!(total > 0) || !ok(other) || !ok(actual)) return '<span class="faint">–</span>';
  const start = pctOf(other, total);
  const track = bar([
    { from: 0, size: start, color: "var(--other)" },
    { from: start, size: pctOf(actual, total), color: "var(--actual)" },
  ], { from: start, size: pctOf(budget, total) }, "wide");
  const usedEnd = other + actual;
  const budgetEnd = other + budget;
  const hasBudget = ok(budget) && budget > 0;
  // The two lower labels share one line, even when both endpoints are close.
  const bottomSpace = hasBudget ? gib(budgetEnd).length + 5 : 0;
  return '<div class="resource-scale" style="--other-end:' + clamp(start) +
    '%;--other-width:' + (gib(other).length + 4) + 'ch;--budget-space:' + bottomSpace + 'ch">' +
    track + memoryMark(other, start, "resource-other") +
    memoryMark(usedEnd, pctOf(usedEnd, total), "resource-used") +
    (hasBudget ? memoryMark(budgetEnd, pctOf(budgetEnd, total), "resource-budget") : "") + '</div>';
}

function resourceRow(name, r) {
  const stale = ok(r.age_seconds) && r.age_seconds > 2.5 ? " stale" : "";
  const utilization = ok(r.utilization_percent) ? Math.round(r.utilization_percent) + "%" : "–";
  return '<div class="row resource-row"><span class="muted mono">' + esc(name) + '</span>' +
    '<div class="resource-reading' + stale + '">' + resourceBar(r) + '</div>' +
    '<span class="resource-utilization mono' + stale + '">' + utilization + '</span></div>';
}

function resources(s) {
  const r = s.resources || {};
  const rows = (r.devices || []).map(card => resourceRow("GPU " + card.index, card));
  rows.push(resourceRow("CPU", r.host || {}));
  return section("资源分配", '<div class="card">' + rows.join("") + "</div>");
}

function events(s) {
  const items = (s.events || []).slice().reverse();
  if (!items.length) return unavailable("调度事件", "暂无事件");
  const rows = items.map((e) => {
    const time = ok(e.at) ? new Date(e.at * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "??:??:??";
    const where = [
      e.run,
      e.stage_index != null ? "S" + e.stage_index : "",
      e.attempt != null ? "a" + e.attempt : "",
      e.device != null ? "GPU" + e.device : "",
    ].filter(Boolean).join(" ");
    let detail = e.detail || "";
    if (e.kind === "launch") detail = detail.replace(/\bhost \d+M gpu \d+M(?: MPS 请求上限 \d+M)?/, "").trim();
    if (e.kind === "finish") detail = detail.replace(/ 峰值 \d+M$/, "");
    return '<div><span class="t mono">' + time + "</span> " + esc(KIND[e.kind] || e.kind) +
      (where ? ' <span class="mono">' + esc(where) + "</span>" : "") +
      (e.status ? ' <span data-status="' + esc(e.status) + '">' + esc(e.status) + "</span>" : "") +
      (detail ? ' <span class="muted">' + esc(detail) + "</span>" : "") + "</div>";
  });
  return section("调度事件", '<div class="card events">' + rows.join("") + "</div>");
}

function statusBody(detail) {
  const live = detail.source === "live";
  let html = tiles(detail) + stages(detail);
  html += section("运行中的尝试",
    live ? workers(detail) : '<div class="card empty">没有实时 worker 数据</div>');
  html += section("并发时间线", timeline(detail));
  html += live ? resources(detail) : unavailable("资源分配", "没有实时采样");
  html += live ? events(detail) : unavailable("调度事件", "没有实时事件");
  return html;
}

function requestedBatchTab() {
  return location.pathname.match(/\/(results|plot)\/?$/)?.[1] || "status";
}

function renderBatchPanels(s, update) {
  const detail = s?.batch;
  const missing = s && Object.hasOwn(s, "batch") && !detail && !s.starting && !s.degraded;
  const name = missing ? "status" : requestedBatchTab();
  const panel = document.getElementById(name + "-panel");
  const entering = panel.hidden;
  const tabs = document.getElementById("batch-tabs");
  tabs.hidden = Boolean(missing);
  for (const tab of tabs.children) {
    const selected = tab.id === "tab-" + name;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  if (name !== "status") {
    if (entering || update) {
      const refresh = name === "plot" ? refreshResultPlot : refreshResultTable;
      refresh(panel, BATCH_ID, detail?.running ?? true, entering || update === "reconnect");
    }
  } else if (detail) {
    panel.innerHTML = sourceNote(detail) + statusBody(detail);
  } else {
    panel.innerHTML = '<div class="card empty">' + (missing
      ? "当前目录中没有批次 " + esc(BATCH_ID)
      : "正在读取批次 " + esc(BATCH_ID) + " 的快照，稍后自动刷新…") + "</div>";
  }
}

function wireBatchTabs() {
  const tabs = document.getElementById("batch-tabs");
  function select(tab) {
    if (tab.getAttribute("aria-selected") === "true") return;
    const name = tab.id.slice("tab-".length);
    const path = "/b/" + BATCH_ID + "/" + (name === "status" ? "" : name + "/");
    history.pushState(null, "", path);
    renderBatchPanels(state.snapshot);
  }
  tabs.onclick = (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) select(tab);
  };
  tabs.onkeydown = (event) => {
    const buttons = Array.from(tabs.children);
    const index = buttons.indexOf(event.target);
    if (index < 0) return;
    let target;
    switch (event.key) {
      case "Home": target = buttons[0]; break;
      case "End": target = buttons.at(-1); break;
      case "ArrowLeft": target = buttons[(index + buttons.length - 1) % buttons.length]; break;
      case "ArrowRight": target = buttons[(index + 1) % buttons.length]; break;
      default: return;
    }
    event.preventDefault();
    select(target);
    target.focus();
  };
  globalThis.onpopstate = () => renderBatchPanels(state.snapshot);
}

function renderBatch(s, update) {
  document.getElementById("batch-header").innerHTML = (s ? notices(s) : "") + topBar(s?.batch);
  renderBatchPanels(s, update);
  wire();
}

function render(update) {
  const s = state.snapshot;
  if (PAGE === "batch") renderBatch(s, update);
  else if (s) renderList(s);
  else app.innerHTML = '<p class="muted pad">等待第一个快照…</p>';
}

function wire() {
  document.querySelectorAll("[data-act]").forEach((node) => {
    const act = node.dataset.act;
    if (act === "theme") node.onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      render();
    };
    if (act === "stop") node.onclick = () => {
      const name = node.dataset.name;
      if (!window.confirm("停止批次 " + BATCH_ID + "（" + name + "）正在运行的实验？已完成的阶段会保留，可以之后 resume。")) return;
      fetch("/api/stop/" + BATCH_ID + "?name=" + encodeURIComponent(name), { method: "POST" }).then(() => setTimeout(() => location.reload(), 1500));
    };
  });
}

function connect() {
  const url = PAGE === "batch" && BATCH_ID !== null ? "/api/stream?batch=" + BATCH_ID : "/api/stream";
  const source = new EventSource(url);
  source.onopen = () => {
    state.connected = true;
    render(state.snapshot?.batch ? "reconnect" : undefined);
  };
  source.onerror = () => { state.connected = false; render(); };
  source.onmessage = (message) => {
    state.connected = true;
    const previousName = state.snapshot?.batch?.batch?.name;
    try { state.snapshot = JSON.parse(message.data); } catch (error) { return; }
    if (previousName && state.snapshot.batch?.batch?.name !== previousName) {
      location.reload();
      return;
    }
    render("snapshot");
  };
}

document.title = PAGE === "batch" ? "expman 看板 · 批次 " + BATCH_ID : "expman 运行看板";

if (PAGE === "batch") {
  wireBatchTabs();
  if (requestedBatchTab() !== "status") render();
}
connect();
