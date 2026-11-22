"use strict";

function* mergedCells(rows, id) {
  const valueAt = (i) => Object.hasOwn(rows[i], id) ? rows[i][id] : undefined;
  let start = 0;
  while (start < rows.length) {
    const value = valueAt(start);
    let end = start + 1;
    // HTML table cells can span at most 65,534 rows.
    while (end < rows.length && end - start < 65534 && valueAt(end) === value) end++;
    yield { start, span: end - start, value };
    start = end;
  }
}

function valueText(value, metric = false) {
  if (value === undefined) return metric ? "" : "–";
  if (value === "" || value === "–" || value === "null") return JSON.stringify(value);
  return String(value);
}

function textElement(tag, text, cls) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}

function resultTable(columns, rows) {
  const table = document.createElement("table");
  table.className = "results-table";
  table.setAttribute("aria-labelledby", "tab-results");
  const head = table.createTHead().insertRow();
  const number = textElement("th", "序号");
  number.scope = "col";
  head.append(number);
  const body = table.createTBody();
  const bodyRows = rows.map((_, index) => {
    const row = body.insertRow();
    const position = textElement("th", String(index + 1), "mono");
    position.scope = "row";
    row.append(position);
    return row;
  });
  for (const column of columns) {
    const metric = column.kind === "metric";
    const header = textElement("th", column.label, metric ? "metric-column" : "");
    header.scope = "col";
    head.append(header);
    for (const { start, span, value } of mergedCells(rows, column.id)) {
      const cell = bodyRows[start].insertCell();
      cell.textContent = valueText(value, metric);
      if (span > 1) cell.rowSpan = span;
      if (value === undefined) cell.className = "result-missing";
    }
  }
  return table;
}

const resultPanels = new WeakMap();

async function refreshResultTable(panel, batchId, running, force = false) {
  let view = resultPanels.get(panel);
  if (!view) {
    view = { running, revision: 0, loadedRevision: -1, pending: false, etag: null };
    resultPanels.set(panel, view);
  }
  // A tab entry or a run-state transition must be read even if it occurred
  // during a request that was already in flight.
  if (force || view.running !== running) view.revision++;
  view.running = running;
  if (view.pending || (!running && view.loadedRevision === view.revision)) return;
  const revision = view.revision;
  view.pending = true;
  try {
    const headers = view.etag ? { "If-None-Match": view.etag } : {};
    const response = await fetch("/api/batch/" + batchId + "/results", { headers, cache: "no-store" });
    if (response.ok) {
      const { columns, rows } = await response.json();
      const table = resultTable(columns, rows);
      let viewport = panel.firstElementChild;
      if (!viewport) {
        viewport = document.createElement("div");
        viewport.className = "card results-scroll";
        panel.append(viewport);
      }
      viewport.replaceChildren(table);
      view.etag = response.headers.get("ETag");
    } else if (response.status !== 304 || !view.etag) {
      throw new Error("HTTP " + response.status);
    }
    view.loadedRevision = revision;
  } catch (error) {
    console.warn("读取批次结果失败", error);
  } finally {
    view.pending = false;
  }
  if (revision !== view.revision) await refreshResultTable(panel, batchId, view.running);
}
