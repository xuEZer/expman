"use strict";

const state = {
  snapshot: null,
  paused: false,
  throttle: 1,
  lastPaint: 0,
  stopAllowed: false,
  connected: false,
};

const KIND = { launch: "启动", finish: "结束", reuse: "复用", shed: "撤销", probe: "探测" };
const STATE_TEXT = { running: "运行中", pending: "排队中", succeeded: "已完成", failed: "有失败", cancelled: "已停止" };

const app = document.getElementById("app");

function esc(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function p2(n) { return String(n).padStart(2, "0"); }
function ok(value) { return value != null && typeof value === "number" && Number.isFinite(value); }
function gib(kb) { return ok(kb) ? (kb / 1048576).toFixed(1) : "–"; }
function pctOf(value, total) { return ok(value) && ok(total) && total > 0 ? (100 * value) / total : 0; }
function clamp(value) { return Math.max(0, Math.min(100, ok(value) ? value : 0)); }
function secs(seconds) {
  if (!ok(seconds)) return "??:??:??";
  const m = Math.floor(Math.max(0, seconds) / 60);
  return p2(Math.floor(m / 1440)) + ":" + p2(Math.floor((m % 1440) / 60)) + ":" + p2(m % 60);
}
function compact(seconds) {
  if (!ok(seconds)) return "–";
  const s = Math.max(0, seconds);
  if (s < 60) return Math.round(s) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m" + p2(Math.round(s % 60)) + "s";
  if (s < 86400) return Math.floor(s / 3600) + "h" + p2(Math.floor((s % 3600) / 60)) + "m";
  return Math.floor(s / 86400) + "d" + p2(Math.floor((s % 86400) / 3600)) + "h";
}
function span(lower, upper) { return ok(lower) && ok(upper) ? secs(lower) + "～" + secs(upper) : "–"; }
function bar(segments, envelope, cap, cls) {
  const fills = segments.filter((s) => ok(s.size) && s.size > 0)
    .map((s) => '<span class="seg" style="left:' + clamp(s.from) + "%;width:" + clamp(s.size) + "%;background:" + s.color + '"></span>').join("");
  const box = ok(envelope) && envelope.size > 0 ? '<span class="env" style="left:' + clamp(envelope.from) + "%;width:" + clamp(envelope.size) + '%"></span>' : "";
  const mark = ok(cap) ? '<span class="cap" style="left:' + clamp(cap) + '%"></span>' : "";
  return '<div class="track ' + (cls || "") + '">' + fills + box + mark + "</div>";
}

function notices(s) {
  const parts = [];
  if (s.degraded) parts.push('<div class="notice">快照降级：' + esc(s.degraded) + "</div>");
  if (!state.connected) parts.push('<div class="notice warn">与看板的连接已断开，正在重连…</div>');
  const age = Date.now() / 1000 - (s.generated_at || 0);
  if (state.connected && age > 5) parts.push('<div class="notice warn">数据已 ' + Math.round(age) + " 秒未更新</div>");
  return parts.join("");
}

function top(s) {
  const b = s.batch || {};
  const counts = b.counts || {};
  const running = (s.workers || []).length;
  const key = running ? "running" : counts.pending ? "pending" : "succeeded";
  const pillClass = key === "running" ? "ok" : key === "pending" ? "warn" : "";
  const estimate = (s.schedule || {}).estimate;
  const buttons = [
    '<button data-act="pause">' + (state.paused ? "继续" : "暂停") + "</button>",
    '<select data-act="throttle">' +
      [1, 2, 5].map((v) => '<option value="' + v + '"' + (state.throttle === v ? " selected" : "") + ">" + (state.paused ? "—" : v + "s") + "</option>").join("") +
    "</select>",
    '<button data-act="theme">' + (document.documentElement.dataset.theme === "dark" ? "亮色" : "暗色") + "</button>",
  ];
  if (state.stopAllowed) buttons.push('<button class="danger" data-act="stop">停止实验</button>');
  const connection = state.connected
    ? '<span class="pill ok"><span class="dot pulse"></span>实时</span>'
    : '<span class="pill bad"><span class="dot" style="background:var(--danger)"></span>已断开</span>';
  return '<div class="top">' +
    '<span class="title">' + esc(b.name || "batch") + "</span>" +
    '<span class="pill mono">' + esc(b.dir || "") + "</span>" +
    '<span class="pill ' + pillClass + '">' + (STATE_TEXT[key] || key) + "</span>" +
    '<span class="pill mono">等价组 ' + esc(estimate ? estimate.groups : "–") + "</span>" +
    '<span class="spacer"></span>' + buttons.join("") + connection + "</div>";
}

function tiles(s) {
  const b = s.batch || {};
  const counts = b.counts || {};
  const remaining = b.remaining || {};
  const host = (s.resources || {}).host || {};
  const gate = (s.schedule || {}).gate;
  const coverage = ok(remaining.coverage) ? Math.round(remaining.coverage * 100) + "%" : "–";
  const spanText = ok(remaining.lower_seconds) && ok(remaining.upper_seconds)
    ? secs(remaining.lower_seconds) + "～" + secs(remaining.upper_seconds) : "??:??:??";
  const tight = host.tight ? "主机内存紧张" : gate === "open" ? "可接纳" : "等待内存";
  return '<div class="grid tiles">' + [
    tile("已运行", secs(b.elapsed_seconds), "共 " + esc(b.experiments) + " 个配置"),
    tile("预计剩余", spanText, "覆盖 " + coverage + " · 样本 " + esc(remaining.samples)),
    tile("运行 / 排队", (counts.running || 0) + " / " + (counts.pending || 0),
      "成功 " + (counts.succeeded || 0) + " · 失败 " + (counts.failed || 0) + " · 停止 " + (counts.cancelled || 0)),
    tile("调度门闸", gate === "open" ? "开放" : "关闭", tight + " · 排队 " + esc((s.schedule || {}).pending_runs) + " 个"),
  ].join("") + "</div>";
}

function tile(label, value, hint) {
  return '<div class="tile"><div class="label">' + esc(label) + '</div><div class="value mono">' + esc(value) +
    '</div><div class="label mono">' + (hint || "") + "</div></div>";
}

function stages(s) {
  const list = s.stages || [];
  if (!list.length) return "";
  const running = (s.schedule || {}).running_by_stage || {};
  const cards = list.map((stage) => {
    const known = stage.known !== false;
    const total = stage.groups_total || 0;
    const done = stage.groups_done || 0;
    const remaining = known ? Math.max(0, total - done) : null;
    const live = known ? Math.min(remaining, running[stage.index] || running[String(stage.index)] || 0) : 0;
    const samples = stage.samples || {};
    const width = (value) => pctOf(value, total);
    const segments = [
      { from: 0, size: width(done), color: "var(--done)" },
      { from: width(done), size: width(live), color: "var(--running)" },
    ];
    const unit = ok(stage.unit_lower_seconds) && ok(stage.unit_upper_seconds)
      ? compact(stage.unit_lower_seconds) + "～" + compact(stage.unit_upper_seconds)
      : "估计中";
    const serial = ok(stage.unit_lower_seconds) && ok(stage.unit_upper_seconds) && remaining
      ? compact(remaining * stage.unit_lower_seconds) + "～" + compact(remaining * stage.unit_upper_seconds)
      : "–";
    const count = remaining === null ? "–" : remaining;
    const subtitle = known ? "剩余组　共 " + total + " 组" : "等待调度器启动";
    return '<div class="card">' +
      '<div class="stage-head"><span class="stage-name">' + esc(stage.name || "Stage") + '</span><span class="faint mono">S' + esc(stage.index) + "</span></div>" +
      '<div style="margin-top:8px">' + bar(segments, null, null) + "</div>" +
      '<div class="stage-count"><b class="mono">' + count + '</b><span class="faint">' + subtitle + "</span></div>" +
      '<div class="stage-meta mono">单组中位 ' + esc(unit) + '</div>' +
      '<div class="stage-meta mono">串行合计 ' + esc(serial) + "</div>" +
      '<div class="stage-meta mono">样本 ' + (samples.measured || 0) + " 实测 · " + (samples.reused || 0) + " 复用 · " + (samples.censored || 0) + " 截断</div>" +
      "</div>";
  });
  return '<h2 class="section">各阶段剩余<span class="hint">实心 = 已完成 · 琥珀 = 运行中 · 底轨 = 待执行 · 串行合计不含并行加速</span></h2>' +
    '<div class="grid stages">' + cards.join("") + "</div>";
}

function workers(s) {
  const list = s.workers || [];
  const head = '<div class="row worker-row head-row">' +
    ["RUN", "STAGE", "#", "GPU", "已运行", "内存 / 显存"].map((t) => "<span>" + t + "</span>").join("") + "</div>";
  if (!list.length) return '<div class="card empty">当前没有运行中的尝试</div>';
  return '<div class="card">' + head + list.map(workerRow).join("") + "</div>";
}

function workerRow(w) {
  const host = w.host || {};
  const gpu = w.gpu || {};
  const hostRef = Math.max(host.cap_kb || 0, host.reserved_kb || 0, host.resident_kb || 0, 1);
  const gpuRef = Math.max(gpu.cap_kb || 0, gpu.budget_kb || 0, gpu.reserved_kb || 0, 1);
  const expected = w.expected ? span(w.expected.lower_seconds, w.expected.upper_seconds) : "–";
  const progress = w.progress && ok(w.progress.completed)
    ? '<div class="mono faint">进度 ' + w.progress.completed + (ok(w.progress.total) ? "/" + w.progress.total : "") + " " + esc(w.progress.unit || "") + "</div>"
    : "";
  return '<div class="row worker-row" style="margin-top:10px">' +
    '<span class="mono">' + esc(w.run) + "</span>" +
    "<span>" + esc(w.stage || "S" + w.stage_index) + "</span>" +
    '<span class="mono">a' + esc(w.attempt) + "</span>" +
    '<span class="mono">' + esc(w.device) + "</span>" +
    '<span class="mono">' + secs(w.elapsed_seconds) + "</span>" +
    "<div>" +
      '<div class="mono faint">host ' + gib(host.resident_kb) + " / " + gib(host.reserved_kb) + " GiB" +
        (host.reached_cap ? ' <span class="k-fail">触顶</span>' : "") + "</div>" +
      bar([{ from: 0, size: pctOf(host.resident_kb, hostRef), color: "var(--actual)" }],
        { from: 0, size: pctOf(host.reserved_kb, hostRef) },
        ok(host.cap_kb) ? pctOf(host.cap_kb, hostRef) : null, "thin") +
      '<div class="mono faint" style="margin-top:6px">gpu ' + gib(gpu.reserved_kb) + " / " + gib(gpu.budget_kb) + " GiB · 上限 " + gib(gpu.cap_kb) + " GiB" + "</div>" +
      bar([{ from: 0, size: pctOf(gpu.reserved_kb, gpuRef), color: "var(--actual)" }],
        { from: 0, size: pctOf(gpu.budget_kb, gpuRef) },
        ok(gpu.cap_kb) ? pctOf(gpu.cap_kb, gpuRef) : null, "thin") +
      '<div class="mono faint" style="margin-top:6px">预计 ' + expected + "</div>" +
      progress +
    "</div></div>";
}

function resources(s) {
  const r = s.resources || {};
  const rows = (r.devices || []).map(deviceRow);
  rows.push(hostRow(r.host || {}));
  return '<h2 class="section">资源：分配 vs 实际<span class="hint">悬停无用，数值见下方明细</span></h2>' +
    '<div class="card">' + rows.join("") +
    '<div class="legend">实心 = 实际占用 · 虚线框 = 调度分配（保留） · 红线 = 该尝试硬上限 · 右列 = 扣掉运行中尝试未来增长后可再接纳的量</div></div>';
}

function deviceRow(card) {
  const total = card.total_kb;
  const other = card.other_kb || 0;
  const actual = card.expman_actual_kb || 0;
  const budget = card.expman_budget_kb || 0;
  const start = pctOf(other, total);
  const stale = ok(card.age_seconds) && card.age_seconds > 2.5 ? " stale" : "";
  return '<div class="row" style="grid-template-columns:76px minmax(0,1fr) 92px">' +
      '<span class="muted mono">GPU ' + esc(card.index) + "</span>" +
      '<div class="' + stale.trim() + '">' + bar(
        [{ from: 0, size: start, color: "var(--other)" }, { from: start, size: pctOf(actual, total), color: "var(--actual)" }],
        { from: start, size: pctOf(budget, total) }, null, "wide") + "</div>" +
      '<span class="mono" style="text-align:right">' + gib(card.launchable_kb) + " GiB</span>" +
    "</div>" +
    '<div class="detail mono">总量 ' + gib(total) + " · 其他 " + gib(other) + " · 实际 " + gib(actual) + " · 分配 " + gib(budget) +
      (ok(card.launchable_kb) ? "" : " · 待首次采样") + "</div>";
}

function hostRow(host) {
  const total = host.total_kb;
  const other = host.other_kb || 0;
  const actual = host.expman_actual_kb || 0;
  const budget = host.expman_budget_kb || 0;
  const start = pctOf(other, total);
  const stale = ok(host.age_seconds) && host.age_seconds > 2.5 ? " stale" : "";
  return '<div class="row" style="grid-template-columns:76px minmax(0,1fr) 92px;margin-top:12px">' +
      '<span class="muted mono">主机</span>' +
      '<div class="' + stale.trim() + '">' + bar(
        [{ from: 0, size: start, color: "var(--other)" }, { from: start, size: pctOf(actual, total), color: "var(--actual)" }],
        { from: start, size: pctOf(budget, total) }, null, "wide") + "</div>" +
      '<span class="mono" style="text-align:right">' + gib(host.launchable_kb) + " GiB</span>" +
    "</div>" +
    '<div class="detail mono">总量 ' + gib(total) + " · 其他 " + gib(other) + " · 实际驻留 " + gib(actual) + " · 分配 " + gib(budget) +
      " · MemAvailable " + gib(host.available_kb) + " · 交换 " + gib(host.swap_free_kb) + "/" + gib(host.swap_total_kb) + "</div>";
}

function plan(s) {
  const schedule = s.schedule || {};
  const plan = schedule.plan;
  const estimate = schedule.estimate;
  const header = '<h2 class="section">本 tick 准入计划<span class="hint">' +
    (estimate && ok(estimate.makespan && estimate.makespan.lower_seconds)
      ? "预计完工 " + span(estimate.makespan.lower_seconds, estimate.makespan.upper_seconds)
      : "尚无完工估计") + "</span></h2>";
  const slots = Object.keys((estimate && estimate.device_slots) || {})
    .map((device) => "GPU" + device + "×" + estimate.device_slots[device]).join(" ");
  const summary = schedule.estimate === null
    ? "尚未产生调度估计"
    : "就绪候选 " + esc(plan ? plan.ready : "–") + " · 采纳 " + esc(plan ? (plan.launched || []).length : 0) +
      " · 剩余预算 host " + gib((plan && plan.capacity && plan.capacity.host_left_kb)) + " GiB" +
      (slots ? " · 并行位 " + slots : "");
  const rows = plan && plan.placements ? plan.placements.map((item) =>
    '<div class="row" style="grid-template-columns:62px 80px 56px minmax(0,1fr) 84px">' +
      '<span class="mono">' + esc(String(item.run_id || "").slice(0, 6)) + "</span>" +
      "<span>" + esc(item.stage || "") + "</span>" +
      '<span class="mono">GPU' + esc(item.device) + "</span>" +
      '<span class="mono faint">host ' + gib(item.host_reserve_kb) + " · gpu " + gib(item.gpu_reserve_kb) + " GiB</span>" +
      '<span class="mono" style="text-align:right">上限 ' + gib(item.gpu_cap_kb) + "</span>" +
    "</div>").join("") : "";
  return header + '<div class="card"><div class="mono faint">' + summary + "</div>" + rows + "</div>";
}

function events(s) {
  const items = (s.events || []).slice().reverse();
  const header = '<h2 class="section">调度事件<span class="hint">最近 ' + items.length + " 条</span></h2>";
  if (!items.length) return header + '<div class="card empty">暂无事件</div>';
  return header + '<div class="card events">' + items.map((e) => {
    const time = ok(e.at) ? new Date(e.at * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "??:??:??";
    const status = e.status ? ' <span class="' + (e.status === "succeeded" ? "k-done" : e.status === "failed" ? "k-fail" : "k-stop") + '">' + esc(e.status) + "</span>" : "";
    const where = [e.run, e.stage_index != null ? "S" + e.stage_index : null, e.attempt != null ? "a" + e.attempt : null, e.device != null ? "GPU" + e.device : null].filter(Boolean).join(" ");
    return '<div><span class="t mono">' + time + '</span> <span class="k">' + esc(KIND[e.kind] || e.kind) +
      '</span> <span class="mono">' + esc(where) + "</span>" + status +
      (e.detail ? ' <span class="faint">' + esc(e.detail) + "</span>" : "") + "</div>";
  }).join("") + "</div>";
}

function render(force) {
  if (state.paused && !force) return;
  const now = Date.now();
  if (!force && now - state.lastPaint < state.throttle * 1000 - 50) return;
  state.lastPaint = now;
  const s = state.snapshot;
  if (!s) { app.innerHTML = '<p class="muted pad">等待第一个快照…</p>'; return; }
  app.innerHTML = notices(s) + top(s) + (s.degraded ? "" : tiles(s) + stages(s) +
    '<div class="cols"><div>' +
      '<h2 class="section">运行中的尝试<span class="hint">' + (s.workers || []).length + " 个 worker</span></h2>" + workers(s) +
    "</div><div>" + resources(s) + "</div></div>" +
    '<div class="cols">' + plan(s) + events(s) + "</div>");
  wire();
}

function wire() {
  document.querySelectorAll("[data-act]").forEach((node) => {
    const act = node.dataset.act;
    if (act === "pause") node.onclick = () => { state.paused = !state.paused; render(true); };
    if (act === "throttle") node.onchange = () => { state.throttle = Number(node.value); render(true); };
    if (act === "theme") node.onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      render(true);
    };
    if (act === "stop") node.onclick = () => {
      if (!window.confirm("停止正在运行的实验？已完成的阶段会保留，可以之后 resume。")) return;
      fetch("/api/stop", { method: "POST" }).then(() => setTimeout(() => location.reload(), 1500));
    };
  });
}

function connect() {
  const source = new EventSource("/api/stream");
  source.onopen = () => { state.connected = true; render(); };
  source.onerror = () => { state.connected = false; render(); };
  source.onmessage = (message) => {
    state.connected = true;
    try { state.snapshot = JSON.parse(message.data); } catch (error) { return; }
    render();
  };
}

fetch("/api/health").then((response) => response.json()).then((health) => {
  state.stopAllowed = Boolean(health.stop);
  render(true);
}).catch(() => {});

connect();
