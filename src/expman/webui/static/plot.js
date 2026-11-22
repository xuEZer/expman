"use strict";

function plotLabel(value) {
  if (value === undefined) return "–";
  if (typeof value === "string" && ["", "–", "null"].includes(value)) return JSON.stringify(value);
  return String(value);
}

function plotColumnLabels(variables, metrics) {
  const names = new Set(variables.map(col => col.label));
  return new Map([
    ...variables.map(col => [col.id, col.label === "step" ? "step（配置）" : col.label]),
    ...metrics.map(col => [col.id, names.has(col.label) || col.label === "step" ? col.label + "（指标）" : col.label]),
  ]);
}

function plotObservations(row, history, x, y, metricX) {
  if (x === "step") return history[y] || [];
  if (!Object.hasOwn(row, x) || !Number.isFinite(row[y])) return [];
  const xs = history[x], ys = history[y];
  if (metricX && xs?.length && ys?.length) {
    const atStep = new Map(ys);
    return xs.filter(([step]) => atStep.has(step)).map(([step, value]) => [value, atStep.get(step), step]);
  }
  return [[row[x], row[y]]];
}

function variesWithX(points) {
  return points.length > 1 && points.some(([x]) => x !== points[0][0])
    && points.some(([, y]) => y !== points[0][1]);
}

function analyzePlotAxis({variables, metrics, rows, steps = []}, xId) {
  const stepX = xId === "step";
  const metricX = metrics.some(col => col.id === xId);
  const controls = variables.filter(col => col.id !== xId);
  const levels = controls.map(() => new Map());
  const coordinates = rows.map(row => controls.map((col, i) => {
    const value = row[col.id], ids = levels[i];
    if (!ids.has(value)) ids.set(value, ids.size);
    return ids.get(value);
  }));
  // Group runs matching in every other setting. Coordinate IDs keep absent keys
  // distinct from null, and avoid serializing the full YAML values repeatedly.
  const contrasts = controls.map((_, i) => {
    const groups = new Map();
    return coordinates.map(point => {
      const key = point.slice(0, i).concat(point.slice(i + 1)).join(",");
      if (!groups.has(key)) groups.set(key, groups.size);
      return groups.get(key);
    });
  });

  const tracks = new Map();
  for (const metric of metrics) {
    if (metric.id === xId || (!stepX && !metricX && !metric.depends_on?.includes(xId))) continue;
    const measurements = rows.map((row, i) => plotObservations(row, steps[i] || {}, xId, metric.id, metricX));
    if ((stepX || metricX) && !measurements.some(variesWithX)) {
      if (!metricX) continue;
      // Hold the other settings fixed across runs. Compare history pairs at
      // the same step and latest-value pairs in their own group.
      const related = contrasts.some(groups => {
        const comparisons = new Map();
        for (const [i, points] of measurements.entries()) {
          if (!points.length) continue;
          if (!comparisons.has(groups[i])) comparisons.set(groups[i], new Map());
          const atStep = comparisons.get(groups[i]);
          for (const [x, y, step] of points) {
            if (!atStep.has(step)) atStep.set(step, {x, y, changingX: false, changingY: false});
            const seen = atStep.get(step);
            seen.changingX ||= seen.x !== x;
            seen.changingY ||= seen.y !== y;
            if (seen.changingX && seen.changingY) return true;
          }
        }
        return false;
      });
      if (!related) continue;
    }
    tracks.set(metric.id, measurements);
  }
  return {
    rows, controls, contrasts, coordinates, tracks, xId, stepX, metricX,
    metrics: metrics.filter(col => tracks.has(col.id)), labels: plotColumnLabels(variables, metrics),
    categories: stepX || metricX ? [] : [...new Set(rows.filter(row => Object.hasOwn(row, xId)).map(row => row[xId]))],
    controlsByMetric: new Map(),
  };
}

function metricControls(analysis, metricId) {
  if (analysis.controlsByMetric.has(metricId)) return analysis.controlsByMetric.get(metricId);
  const {controls, coordinates, contrasts} = analysis;
  const tracks = analysis.tracks.get(metricId);
  if (!controls.length || tracks.filter(points => points.length > 0).length < 2) {
    const none = new Set();
    analysis.controlsByMetric.set(metricId, none);
    return none;
  }
  const byX = new Map();
  tracks.forEach((points, run) => {
    for (const [x, y] of points) {
      if (!byX.has(x)) byX.set(x, new Map());
      const runs = byX.get(x);
      if (!runs.has(run)) runs.set(run, new Set());
      runs.get(run).add(y);
    }
  });
  // A trajectory can return to the same X with a different Y. Compare the
  // sets of Ys per run before attributing the difference to another setting.
  const conflicts = [];
  for (const runs of byX.values()) {
    if (runs.size < 2) continue;
    const entries = [...runs].map(([run, ys]) => [run, JSON.stringify([...ys].sort((a, b) => a - b))]);
    if (entries.some(([, value]) => value !== entries[0][1])) conflicts.push(entries);
  }

  const direct = new Set();
  for (const entries of conflicts) {
    controls.forEach((_, axis) => {
      if (direct.has(axis)) return;
      const first = new Map();
      for (const [run, value] of entries) {
        const group = contrasts[axis][run];
        if (!first.has(group)) first.set(group, [coordinates[run][axis], value]);
        const [level, previous] = first.get(group);
        if (level !== coordinates[run][axis] && previous !== value) {
          direct.add(axis);
          break;
        }
      }
    });
  }
  const required = new Set(direct);
  for (const entries of conflicts) {
    // With sparse/confounded grids, retain the settings that can distinguish
    // rows still disagreeing after all directly observed effects are fixed.
    const unresolved = new Map();
    for (const entry of entries) {
      const key = Array.from(direct, axis => coordinates[entry[0]][axis]).join(",");
      if (!unresolved.has(key)) unresolved.set(key, []);
      unresolved.get(key).push(entry);
    }
    for (const group of unresolved.values()) {
      if (!group.some(([, value]) => value !== group[0][1])) continue;
      controls.forEach((_, axis) => {
        if (required.has(axis)) return;
        const levels = new Set(group.map(([run]) => coordinates[run][axis]));
        const contexts = new Set(group.map(([run]) => contrasts[axis][run]));
        // A fully crossed axis with no direct effect cannot resolve this conflict.
        if (levels.size > 1 && group.length < levels.size * contexts.size) required.add(axis);
      });
    }
  }
  const ids = new Set(Array.from(required, axis => controls[axis].id));
  analysis.controlsByMetric.set(metricId, ids);
  return ids;
}

function plotControls(analysis, yIds) {
  const selected = yIds.filter(id => analysis.tracks.has(id));
  const required = new Set(selected.flatMap(id => [...metricControls(analysis, id)]));
  const measured = analysis.rows.filter((_, i) => selected.some(id => analysis.tracks.get(id)[i].length));
  return analysis.controls.filter(col => required.has(col.id)).map(col => ({
    id: col.id, label: analysis.labels.get(col.id), values: [...new Set(measured.map(row => row[col.id]))],
  }));
}

function plotPoint(x, values, allowError) {
  const first = values[0];
  if (values.every(value => value === first)) return {x, y: first, deviation: null, n: values.length};
  // Scaling before summation also handles large, finite metric values.
  const scale = values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
  const mean = values.reduce((sum, value) => sum + value / scale, 0) / values.length;
  const squared = allowError ? values.reduce((sum, value) => sum + (value / scale - mean) ** 2, 0) : 0;
  return {x, y: scale * mean, deviation: allowError ? scale * Math.sqrt(squared / (values.length - 1)) : null, n: values.length};
}

function buildPlot(analysis, yIds, selections = new Map()) {
  const {rows, metrics, labels, xId, stepX, metricX, categories} = analysis;
  const controls = plotControls(analysis, yIds);
  const numericX = categories.every(Number.isFinite);
  const order = new Map(categories.map((v, i) => [v, i]));
  const series = [];
  for (const [color, metric] of metrics.entries()) {
    if (!yIds.includes(metric.id)) continue;
    const tracks = analysis.tracks.get(metric.id).filter((points, run) => points.length && controls.every(col =>
      !selections.has(col.id) || selections.get(col.id).has(rows[run][col.id])));
    if (!tracks.length) continue;
    const first = tracks[0];
    let points;
    if (metricX && tracks.every(track => track === first || (track.length === first.length &&
      track.every(([x, y], i) => x === first[i][0] && y === first[i][1])))) {
      // Keep reversals and loops in a single observed trajectory.
      points = first.map(([x, y]) => ({x, y, deviation: null, n: tracks.length}));
    } else {
      const buckets = new Map();
      for (const track of tracks) {
        for (const [x, y] of track) {
          if (!buckets.has(x)) buckets.set(x, []);
          buckets.get(x).push(y);
        }
      }
      const relevant = metricControls(analysis, metric.id);
      const multiple = controls.some(col => relevant.has(col.id) &&
        col.values.filter(value => !selections.has(col.id) || selections.get(col.id).has(value)).length > 1);
      points = Array.from(buckets, ([x, ys]) => plotPoint(x, ys, multiple));
      points.sort(numericX ? (a, b) => a.x - b.x : (a, b) => order.get(a.x) - order.get(b.x));
    }
    series.push({label: labels.get(metric.id), color, points});
  }
  return {series, numericX, stepX, categories, xLabel: stepX ? "step" : labels.get(xId),
    yLabel: yIds.filter(id => analysis.tracks.has(id)).map(id => labels.get(id)).join(" / ")};
}

function plotSeriesColor(index) {
  const colors = ["var(--actual)", "var(--running)", "var(--done)", "#b782df", "var(--danger)", "#46a6aa"];
  return colors[index] || 'hsl(' + Math.round(index * 137.508 % 360) + ' 65% 56%)';
}

function renderPlot(chart, availableWidth = 800) {
  if (!chart.series.length) {
    const empty = document.createElement("p");
    empty.className = "plot-empty";
    empty.textContent = chart.yLabel ? "所选变量暂无可绘制的数据。" : "请选择 Y 轴变量。";
    return empty;
  }
  const figure = document.createElement("figure");
  figure.className = "plot-figure";
  const svgNS = "http://www.w3.org/2000/svg";
  function element(parent, tag, attributes, text) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  const width = Math.max(280, Math.min(1000, availableWidth));
  const svg = element(figure, "svg", {class: "plot-svg", viewBox: "0 0 " + width + " 330", role: "img",
    "aria-label": chart.yLabel + " 随 " + chart.xLabel + " 的变化"});
  const legend = document.createElement("figcaption");
  legend.className = "plot-legend";
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const line of chart.series) {
    for (const {x, y, deviation} of line.points) {
      if (chart.numericX) {
        xMin = Math.min(xMin, x);
        xMax = Math.max(xMax, x);
      }
      const low = y - (deviation || 0), high = y + (deviation || 0);
      yMin = Math.min(yMin, Number.isFinite(low) ? low : -Number.MAX_VALUE);
      yMax = Math.max(yMax, Number.isFinite(high) ? high : Number.MAX_VALUE);
    }
  }

  const left = width < 480 ? 62 : 82, right = width - 28, top = 25, bottom = 265;
  const levels = new Map(chart.categories.map((v, i) => [v, i]));
  function position(value, low, high) {
    if (low === high) return 0.5;
    if (Number.isFinite(high - low)) return (value - low) / (high - low);
    return (value / 2 - low / 2) / (high / 2 - low / 2);
  }
  const xPixel = chart.numericX ? x => left + (right - left) * position(x, xMin, xMax)
    : x => left + (right - left) * position(levels.get(x), 0, levels.size - 1);
  const yPixel = y => bottom - (bottom - top) * position(Math.max(yMin, Math.min(y, yMax)), yMin, yMax);
  function ticks(low, high, count) {
    if (low === high) return [low];
    return Array.from({length: count}, (_, i) => {
      const t = i / (count - 1);
      return low * (1 - t) + high * t;
    });
  }
  function shorten(label, length) {
    return label.length > length ? label.slice(0, length - 1) + "…" : label;
  }
  function tickText(value, range) {
    if (typeof value !== "number") return plotLabel(value);
    const precision = Math.min(16, Math.max(4, 3 + Math.ceil(Math.log10(Math.abs(value) / range) || 0)));
    return Number(value.toPrecision(precision)).toString();
  }
  const tickStyle = { fill: "var(--muted)", "font-size": 11 };
  const gridStyle = { stroke: "var(--border)", "stroke-width": 1 };
  for (const y of ticks(yMin, yMax, 5)) {
    const py = yPixel(y);
    element(svg, "line", {...gridStyle, x1: left, y1: py, x2: right, y2: py});
    element(svg, "text", {...tickStyle, x: left - 9, y: py + 4, "text-anchor": "end"}, tickText(y, yMax - yMin));
  }
  let xTicks;
  const tickCount = width < 480 ? 3 : 6;
  if (chart.numericX) {
    xTicks = ticks(xMin, xMax, tickCount);
    if (chart.stepX) xTicks = Array.from(new Set(xTicks.map(Math.round)));
  } else {
    const stride = Math.ceil(levels.size / tickCount);
    xTicks = chart.categories.filter((_, i) => i % stride === 0 || i === levels.size - 1);
  }
  for (const x of xTicks) {
    const px = xPixel(x);
    element(svg, "line", {...gridStyle, x1: px, y1: top, x2: px, y2: bottom + 4});
    const label = chart.numericX ? tickText(x, xMax - xMin) : plotLabel(x);
    const tick = element(svg, "text", {...tickStyle, x: px, y: bottom + 20,
      "text-anchor": px > right - 20 ? "end" : "middle"}, shorten(label, 14));
    element(tick, "title", {}, label);
  }
  element(svg, "path", {d: 'M' + left + ' ' + top + 'V' + bottom + 'H' + right, fill: "none", stroke: "var(--border-strong)"});
  const axisStyle = { fill: "var(--text)", "font-size": 13 };
  element(svg, "text", {...axisStyle, x: left, y: 14}, shorten(chart.yLabel, 84));
  element(svg, "text", {...axisStyle, x: (left + right) / 2, y: 317, "text-anchor": "middle"}, shorten(chart.xLabel, 80));

  chart.series.forEach(line => {
    const color = plotSeriesColor(line.color);
    const layer = element(svg, "g", {stroke: color, fill: color});
    if (line.points.length > 1) {
      element(layer, "polyline", {points: line.points.map(p => xPixel(p.x) + "," + yPixel(p.y)).join(" "),
        fill: "none", "stroke-width": 2, "stroke-linejoin": "round"});
    }
    // The line retains every vertex; draw individual markers only on sparse curves.
    const dense = line.points.length > 500;
    const bars = [];
    for (const {x, y, deviation, n} of line.points) {
      const px = xPixel(x), py = yPixel(y);
      if (deviation !== null) {
        const high = yPixel(y + deviation), low = yPixel(y - deviation);
        bars.push('M' + px + ',' + high + 'V' + low,
          'M' + (px - 4) + ',' + high + 'h8M' + (px - 4) + ',' + low + 'h8');
      }
      if (!dense) {
        const dot = element(layer, "circle", {cx: px, cy: py, r: 3});
        element(dot, "title", {}, line.label + "\n" + chart.xLabel + " = " + plotLabel(x) + "\n" + y
          + (deviation === null ? "" : " ± " + Number(deviation.toPrecision(5)) + " (n=" + n + ")"));
      }
    }
    if (bars.length) element(layer, "path", {d: bars.join(" "), fill: "none", "stroke-width": 1.3});
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = color;
    item.append(swatch, line.label);
    legend.append(item);
  });
  figure.append(legend);
  return figure;
}
