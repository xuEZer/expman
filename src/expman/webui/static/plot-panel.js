"use strict";

const plotPanels = new WeakMap();

function plotHistoryIds(data, x) {
  if (!data || (x !== "step" && !data.metrics.some(m => m.id === x && m.has_steps))) return [];
  return data.metrics.filter(m => m.has_steps).map(m => m.id).sort();
}

function plotXAxis({variables, metrics}, manualX, currentX) {
  const labels = plotColumnLabels(variables, metrics);
  const choices = [...variables, ...metrics].map(col => ({id: col.id, label: labels.get(col.id)}));
  const hasSteps = metrics.some(m => m.has_steps);
  if (hasSteps) choices.unshift({id: "step", label: "step"});
  const related = variables.filter(v => metrics.some(m => m.depends_on.includes(v.id)));
  const first = related.find(v => v.id === currentX) || related.find(v => v.label !== "seed") || related[0];
  const id = choices.some(c => c.id === manualX) ? manualX :
    first?.id || (hasSteps ? "step" : choices[0]?.id || "");
  return {choices, id};
}

function plotCheckbox(parent, text, key, checked, onChange, color) {
  const label = document.createElement("label");
  if (color) label.style.setProperty("--series-color", color);
  const input = document.createElement("input");
  input.type = "checkbox";
  input.value = key;
  input.checked = checked;
  input.onchange = () => onChange(input.checked);
  const caption = document.createElement("span");
  caption.textContent = text;
  label.append(input, caption);
  parent.append(label);
  return input;
}

function setPlotChoice(choices, id, defaults, value, checked) {
  const selected = choices.get(id) || new Set(defaults);
  if (checked) selected.add(value); else selected.delete(value);
  choices.set(id, selected);
}

function createPlotPanel(panel, batchId) {
  const xAxis = panel.querySelector("#plot-x");
  const yAxis = panel.querySelector("#plot-y");
  const filters = panel.querySelector("#plot-values");
  const filterField = panel.querySelector(".plot-filters");
  const output = panel.querySelector("#plot-output");
  const message = panel.querySelector("#plot-message");
  const metricChoices = new Map();
  const defaultMetrics = new Map();
  const selections = new Map();
  let data, loaded, pending, running, manualX, analysis;
  let axesKey, yKey, filterKey;
  let failed = false;

  function showMessage(text) {
    message.textContent = text;
    message.hidden = !text;
  }

  function query() {
    const ids = plotHistoryIds(data, xAxis.value);
    const params = new URLSearchParams(ids.map(id => ["metric", id])).toString();
    return {ids, url: "/api/batch/" + batchId + "/plot" + (params ? "?" + params : "")};
  }

  function updateAxes() {
    const {choices, id} = plotXAxis(data, manualX, xAxis.value);
    const key = JSON.stringify(choices);
    if (axesKey !== key) {
      xAxis.replaceChildren(...choices.map(col => new Option(col.label, col.id)));
      axesKey = key;
    }
    xAxis.value = id;
    xAxis.disabled = !choices.length;
  }

  function syncFilters(dimensions) {
    // Encoding the type keeps explicit null and an absent YAML field distinct.
    const valueKey = v => JSON.stringify([typeof v, v]);
    const key = JSON.stringify(dimensions.map(d => [d.id, d.label, d.values.map(valueKey)]));
    filterField.hidden = !dimensions.length;
    if (filterKey === key) return;
    filterKey = key;
    const focus = filters.contains(document.activeElement) ? document.activeElement : null;
    filters.replaceChildren();
    for (const {id, label, values} of dimensions) {
      const group = document.createElement("fieldset");
      const legend = document.createElement("legend");
      legend.textContent = label;
      const options = document.createElement("div");
      options.className = "plot-checks";
      group.append(legend, options);
      filters.append(group);
      for (const value of values) {
        const input = plotCheckbox(options, plotLabel(value), valueKey(value),
          !selections.has(id) || selections.get(id).has(value), checked => {
            manualX = xAxis.value;
            setPlotChoice(selections, id, values, value, checked);
            draw();
          });
        input.name = id;
        if (focus?.name === id && focus.value === input.value) input.focus();
      }
    }
  }

  function selectedMetrics() {
    const ids = analysis.metrics.map(col => col.id);
    const saved = metricChoices.get(xAxis.value);
    if (saved) return ids.filter(id => saved.has(id));
    if (!ids.length) return [];
    if (!ids.includes(defaultMetrics.get(xAxis.value))) defaultMetrics.set(xAxis.value, ids[0]);
    return [defaultMetrics.get(xAxis.value)];
  }

  function draw() {
    const ys = selectedMetrics();
    syncFilters(plotControls(analysis, ys));
    if (!ys.length) {
      output.replaceChildren();
      showMessage(analysis.metrics.length ? "请选择 Y 轴指标。" :
        (data.metrics.length ? "暂无随当前 X 轴变化的指标。" : "等待记录指标。"));
      return;
    }
    showMessage("");
    output.replaceChildren(renderPlot(buildPlot(analysis, ys, selections), output.clientWidth - 30));
  }

  function updateChart() {
    const x = xAxis.value;
    if (plotHistoryIds(data, x).some(id => !loaded.ids.includes(id))) {
      analysis = null;
      yAxis.closest("fieldset").hidden = true;
      filterField.hidden = true;
      output.replaceChildren();
      showMessage("正在读取 step 记录…");
      return;
    }
    analysis = analyzePlotAxis(data, x);
    yAxis.closest("fieldset").hidden = false;
    const selected = new Set(selectedMetrics());
    const labels = plotColumnLabels(data.variables, data.metrics);
    const key = JSON.stringify([x, analysis.metrics.map(m => [m.id, labels.get(m.id)]), [...selected]]);
    if (yKey !== key) {
      yKey = key;
      const focus = yAxis.contains(document.activeElement) ? document.activeElement.value : null;
      yAxis.replaceChildren();
      for (const [index, {id}] of analysis.metrics.entries()) {
        const input = plotCheckbox(yAxis, labels.get(id), id, selected.has(id), checked => {
          manualX = x;
          setPlotChoice(metricChoices, x, selectedMetrics(), id, checked);
          draw();
        }, plotSeriesColor(index));
        if (focus === id) input.focus();
      }
    }
    draw();
  }

  async function refresh(isRunning, force = false) {
    const transition = running !== isRunning;
    running = isRunning;
    const {url, ids} = query();
    if (pending?.url === url) {
      if (force || transition) pending.follow = true;
      return;
    }
    pending?.controller.abort();
    pending = null;
    if (!running && !force && !transition && !failed && loaded?.url === url) return;
    const request = {url, controller: new AbortController(), follow: false};
    pending = request;
    const headers = loaded?.url === url && loaded.etag ? {"If-None-Match": loaded.etag} : {};
    if (!data) showMessage("正在读取绘图数据…");
    try {
      const response = await fetch(url, {headers, signal: request.controller.signal, cache: "no-store"});
      if (panel.hidden || pending !== request) return;
      if (response.status === 304 && loaded?.url === url) {
        if (failed) updateChart();
      } else {
        if (!response.ok) throw new Error("HTTP " + response.status);
        const result = await response.json();
        if (panel.hidden || pending !== request) return;
        data = result;
        loaded = {url, ids, etag: response.headers.get("ETag")};
        updateAxes();
        updateChart();
      }
      failed = false;
    } catch (error) {
      if (panel.hidden || pending !== request || request.controller.signal.aborted) return;
      failed = true;
      showMessage("读取绘图数据失败。");
      console.warn("读取绘图数据失败", error);
    } finally {
      if (pending === request) {
        pending = null;
        if (!panel.hidden && (request.follow || (!failed && query().url !== url))) {
          await refresh(running, true);
        }
      }
    }
  }

  xAxis.onchange = () => {
    if (!data) return;
    manualX = xAxis.value;
    updateChart();
    if (!analysis || failed || (pending && pending.url !== query().url)) refresh(running);
  };
  return refresh;
}

function refreshResultPlot(panel, batchId, running, force = false) {
  if (!plotPanels.has(panel)) plotPanels.set(panel, createPlotPanel(panel, batchId));
  return plotPanels.get(panel)(running, force);
}
