"""The ``vis`` page: one HTML template string with one placeholder.

A ``.py`` module rather than a ``.html`` package-data file, on purpose. The
template needs no resource loader, no sdist/wheel parity test, and no new
suffix in ``tools/check_independence.py`` -- a ``.py`` file is already inside
everything that scans this package. The cost is that ruff's line length applies
to the markup, which is fine for markup written by hand.

Everything here is inert text. ``insight_bench.vis`` substitutes
``__INSIGHT_BENCH_DATA__`` with an escaped JSON document and writes the result; the
page parses that document, and every value out of it reaches the DOM through
``textContent``. There is no ``innerHTML``, no ``eval``, no inline event
handler, and no remote reference of any kind -- no CDN, no font service, no
stylesheet, no script ``src`` -- because the page is opened over ``file://``,
where a network request is both unavailable and unwanted.

There is deliberately no CSP ``<meta>``: a ``file://`` document has an opaque
origin, so ``img-src 'self'`` matches nothing and would silently stop every
frame from loading with no console explanation.

Layout follows the information design of an internal run-result page; no code,
CSS or SVG from it is reproduced here.
"""

from __future__ import annotations

from typing import Final

PAGE_TEMPLATE: Final[str] = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>INSIGHT-Bench run report</title>
<style>
:root {
  --bg-0: #f6f7f9; --bg-1: #ffffff; --bg-2: #eef2f5;
  --text-0: #19212a; --text-1: #5e6a76; --text-2: #87919c;
  --border: #d8dee6; --accent: #197d72;
  --ok: #277247; --danger: #a33b42; --warn: #ca8a04;
  --panel: #14181d; --panel-2: #1b212a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg-0); color: var(--text-0);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
}
.num, .mono { font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
main { max-width: 1440px; margin: 0 auto; padding: 20px 18px 48px; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 13px; letter-spacing: .06em; text-transform: uppercase;
     color: var(--text-1); margin: 0 0 8px; }
section { margin-top: 22px; }
.card { background: var(--bg-1); border: 1px solid var(--border); border-radius: 8px; }
.pad { padding: 12px 14px; }
#identity { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; }
#identity .fp { color: var(--text-2); }
.pill { border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600;
        border: 1px solid var(--border); background: var(--bg-2); }
.pill.completed { color: var(--ok); border-color: #b6d5c2; background: #eef7f1; }
.pill.partial { color: var(--warn); border-color: #e6d9a8; background: #fbf6e6; }
.pill.failed { color: var(--danger); border-color: #e2c0c3; background: #fbeef0; }
.meta { color: var(--text-1); font-size: 13px; }
#warnings ul { margin: 0; padding-left: 20px; }
#warnings li { color: #6b5300; }
#warnings .card { background: #fdf9ec; border-color: #e6d9a8; }
#spec { display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 4px 10px;
        align-items: baseline; }
#spec dt { color: var(--text-2); font-size: 11px; letter-spacing: .08em; }
#spec dd { margin: 0; word-break: break-word; }
#spec dl { display: contents; }
.charts { display: flex; gap: 14px; flex-wrap: wrap; }
.chart { flex: 1 1 380px; min-width: 300px; }
.chart svg, #matrix svg { display: block; width: 100%; height: auto; }
#matrix { overflow-x: auto; }
#matrix svg { min-width: 640px; }
.legend { color: var(--text-1); font-size: 12px; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th { color: var(--text-1); font-weight: 600; position: sticky; top: 0; background: var(--bg-1);
     padding-right: 15px; cursor: pointer; vertical-align: top; }
/* A column dragged narrower than its content clips it rather than spilling into
   the next cell. Only the cells: a header has to let its dropdown out. */
td { overflow: hidden; text-overflow: ellipsis; }
th .lab { display: flex; align-items: baseline; gap: 4px; }
th .lab .t { overflow: hidden; text-overflow: ellipsis; min-width: 0; }
/* Fixed width, so the glyph can change without the label beside it moving. */
th .ind { flex: none; width: 9px; text-align: center; font-size: 10px; color: var(--text-2); }
th.sorted .ind { color: var(--accent); }
th .grip { position: absolute; top: 0; right: 0; width: 9px; height: 100%; cursor: col-resize; }
th .grip::after { content: ""; position: absolute; top: 4px; bottom: 4px; right: 3px;
                  width: 3px; border-radius: 2px; background: transparent; }
th .grip:hover::after, th .grip.on::after { background: var(--accent); }
th details { position: relative; font-weight: 400; }
th summary { cursor: pointer; list-style: none; font-size: 11px; margin-top: 3px;
             display: inline-block; padding: 0 5px; border: 1px solid var(--border);
             border-radius: 4px; background: var(--bg-1); color: var(--text-2); }
th summary::-webkit-details-marker { display: none; }
th details[open] summary, th details.on summary { border-color: var(--accent);
                                                  color: var(--accent); }
th .opts { position: absolute; z-index: 5; top: calc(100% + 3px); left: 0; padding: 4px;
           min-width: 190px; max-height: 264px; overflow: auto; background: var(--bg-1);
           border: 1px solid var(--border); border-radius: 6px; font-weight: 400;
           box-shadow: 0 6px 18px rgba(20, 24, 29, .12); }
/* Hung off the header's right edge instead, for a column near the right of the
   scrolling table, where opening rightwards would be half cut off. */
th .opts.left { left: auto; right: 0; }
th .opts label { display: flex; align-items: center; gap: 7px; padding: 2px 6px;
                 white-space: nowrap; cursor: pointer; color: var(--text-0); }
th .opts label:hover { background: var(--bg-2); }
th .opts .n { margin-left: auto; padding-left: 12px; color: var(--text-2);
              font-variant-numeric: tabular-nums; }
tbody tr { content-visibility: auto; contain-intrinsic-size: auto 27px; cursor: pointer; }
tbody tr:hover { background: var(--bg-2); }
tbody tr.sel { background: #e5eef4; }
td.r-passed { color: var(--ok); } td.r-failed { color: var(--danger); }
td.r-error { color: var(--warn); }
#bottom { display: flex; gap: 14px; align-items: stretch; }
#tablepane { flex: 1 1 auto; min-width: 0; }
#tablewrap { max-height: 70vh; overflow: auto; }
#filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
#filters input[type=search] { font: inherit; padding: 3px 6px; border: 1px solid var(--border);
                              border-radius: 5px; background: var(--bg-1); color: inherit; }
#inspector { flex: 0 0 440px; background: var(--panel); color: #e8edf2; border-radius: 8px;
             padding: 12px; display: flex; flex-direction: column; gap: 10px; }
#inspector h2 { color: #9fb0c0; }
#inspector .sub { color: #93a3b3; font-size: 12px; }
.shot { position: relative; background: #000; border-radius: 6px; overflow: hidden;
        aspect-ratio: 16 / 9; cursor: pointer; }
.shot img { width: 100%; height: 100%; object-fit: contain; display: block; }
.shot .empty { position: absolute; inset: 0; display: flex; align-items: center;
               justify-content: center; color: #7d8b99; font-size: 12px; text-align: center;
               padding: 0 16px; }
.caveat { color: #b9a06a; font-size: 11px; }
.map { background: var(--panel-2); border-radius: 6px; }
.map svg { display: block; width: 100%; height: auto; }
#transport { display: flex; align-items: center; gap: 6px; background: var(--panel-2);
             border-radius: 6px; padding: 6px 8px; }
#transport button, #nav button { font: inherit; color: #e8edf2; background: #2a323d;
                                 border: 1px solid #3a4552; border-radius: 5px; padding: 2px 8px;
                                 cursor: pointer; }
#transport button:disabled, #nav button:disabled { opacity: .4; cursor: default; }
#transport input[type=range] { flex: 1 1 auto; min-width: 0; }
#nav { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
#nav .id { font-size: 12px; word-break: break-all; }
.badge { font-size: 11px; border-radius: 4px; padding: 1px 6px; border: 1px solid #3a4552; }
.badge.passed { color: #7fd6a2; } .badge.failed { color: #f0a1a7; }
.badge.error { color: #e9cf7a; }
.diag { font-size: 11px; color: #93a3b3; word-break: break-word; }
.hint { font-size: 11px; color: #7d8b99; }
.empty-note { color: var(--text-1); font-size: 13px; }
@media (max-width: 1180px) {
  #bottom { flex-direction: column; }
  #inspector { flex: 1 1 auto; }
}
@media (max-width: 900px) { .chart { flex: 1 1 100%; } }
</style>
</head>
<body>
<main>
  <section id="identity"></section>
  <section id="warnings"></section>
  <section id="spec" class="card pad"></section>
  <section id="breakdown"></section>
  <section id="bottom">
    <div id="tablepane" class="card pad">
      <h2>Episodes</h2>
      <div id="filters"></div>
      <div id="tablewrap"><table id="table"><thead></thead><tbody></tbody></table></div>
      <div id="tableempty" class="empty-note" hidden>No matching episodes.</div>
    </div>
    <aside id="inspector"></aside>
  </section>
</main>
<script type="application/json" id="insight-bench-data">__INSIGHT_BENCH_DATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("insight-bench-data").textContent);

  var SVGNS = "http://www.w3.org/2000/svg";
  var PLAY_MS = 250;
  // Hard stop on top-down grid lines per axis. niceStep already keeps the count
  // near GRID_LINES_TARGET for any finite span; this is what makes the loop
  // bounded rather than merely usually small.
  var GRID_LINES_MAX = 40, GRID_LINES_TARGET = 12;
  // Narrowest a column can be dragged. Wide enough to keep the resize handle
  // reachable and a couple of characters of the header visible, so a column
  // dragged to its limit can still be dragged back.
  var MIN_COLUMN_PX = 46;

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function sv(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag), key;
    for (key in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) {
        node.setAttribute(key, String(attrs[key]));
      }
    }
    return node;
  }
  function button(label, aria, onClick) {
    var node = el("button", null, label);
    node.type = "button";
    node.setAttribute("aria-label", aria);
    node.addEventListener("click", onClick);
    return node;
  }
  function svtext(x, y, value, attrs) {
    var node = sv("text", attrs || {});
    node.setAttribute("x", String(x));
    node.setAttribute("y", String(y));
    node.textContent = String(value);
    return node;
  }
  // Colours reach the DOM only as constants from this file or as hex strings
  // computed from numbers upstream. This is the gate that keeps it true.
  function hex(value) {
    return /^#[0-9a-f]{6}$/i.test(String(value)) ? String(value) : "#eef2f5";
  }
  // Every header filter, so one can close the others and an outside click can
  // close them all.
  var OPEN_FILTERS = [];
  document.addEventListener("pointerdown", function (event) {
    OPEN_FILTERS.forEach(function (box) {
      if (box.open && !box.contains(event.target)) { box.open = false; }
    });
  }, true);
  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") { return; }
    OPEN_FILTERS.forEach(function (box) { box.open = false; });
  });

  function pct(value, digits) {
    return value === null || value === undefined
      ? "\\u2014" : (value * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }
  function num(value, digits) {
    if (value === null || value === undefined || value === "") { return "\\u2014"; }
    if (typeof value !== "number") { return String(value); }
    return value.toFixed(digits === undefined ? 3 : digits);
  }
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  // A 1-2-5 step so a span of any size lands on about GRID_LINES_TARGET lines.
  // Never below 1 m: an indoor episode keeps the metre grid it always had.
  function niceStep(span) {
    if (!isFinite(span) || span <= 0) { return 1; }
    var raw = span / GRID_LINES_TARGET;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var mult = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
    return Math.max(1, mult * mag);
  }

  // ---------------------------------------------------------------- identity
  function renderIdentity() {
    var host = document.getElementById("identity");
    var run = DATA.run, derived = DATA.derived;
    host.appendChild(el("h1", null, "INSIGHT-Bench run report"));
    host.appendChild(el("span", "mono", run.run_id || "\\u2014"));
    host.appendChild(el("span", "mono fp", (run.run_fingerprint || "").slice(0, 12)));
    host.appendChild(el("span", "pill " + (run.status || ""), run.status || "unknown"));
    // The run's own rate belongs here rather than in a card. Without it a run
    // recorded with no categories has no breakdown either, and the page would
    // then carry no headline number at all.
    if (derived.success_rate !== null && derived.success_rate !== undefined) {
      host.appendChild(el("span", "meta num", "SR " + pct(derived.success_rate)
        + " (" + derived.successes + "/" + derived.total + ")"));
    }
    host.appendChild(el("span", "meta num",
      derived.total + " episodes on this page"));
    // Beside the status, in the status's own colours: a partial run's one-line
    // reason has to travel with the numbers it qualifies.
    if (run.episode_subset) {
      host.appendChild(el("span", "pill " + (run.status || ""), run.episode_subset));
    }
    var replayed = Object.keys(DATA.replay).length;
    host.appendChild(el("span", "meta num",
      replayed + " of " + derived.total + " episodes carry replay"));
  }

  function renderWarnings() {
    var host = document.getElementById("warnings");
    if (!DATA.warnings.length) { return; }
    var card = el("div", "card pad"), list = el("ul");
    DATA.warnings.forEach(function (line) { list.appendChild(el("li", null, line)); });
    card.appendChild(list);
    host.appendChild(card);
  }

  // -------------------------------------------------------------- spec block
  function specRow(host, label, value) {
    var group = el("dl");
    group.appendChild(el("dt", null, label));
    group.appendChild(el("dd", "mono", value));
    host.appendChild(group);
  }
  function renderSpec() {
    var host = document.getElementById("spec"), run = DATA.run, att = DATA.attestation;
    specRow(host, "MODEL", (run.model_id || "\\u2014") + "  \\u00b7  " + (run.adapter_id || ""));
    specRow(host, "BENCH", run.benchmark || "\\u2014");
    specRow(host, "SEED", run.seed === null ? "\\u2014" : String(run.seed));
    // The file these numbers were read from. `vis` accepts a directory, so this
    // is the only thing on the page that says which run result it picked.
    specRow(host, "SOURCE", run.source || "\\u2014");
    if (!att) {
      specRow(host, "RUNTIME", "no attestation recorded");
      return;
    }
    var parts = ["backend_id", "backend_status", "runtime_kind", "os", "driver_version"]
      .map(function (key) { return key + "=" + (att[key] === undefined ? "?" : att[key]); });
    parts.push("runtime_publishable=" + (att.publishable === true));
    specRow(host, "RUNTIME", parts.join("  "));
  }

  // ------------------------------------------------------- breakdown: radars
  function radar(host, axes, title) {
    var wrap = el("div", "chart card pad");
    wrap.appendChild(el("h2", null, title));
    if (!axes.length) {
      wrap.appendChild(el("div", "empty-note", "No categories recorded."));
      host.appendChild(wrap);
      return;
    }
    var edge = DATA.breakdown.edge;
    var cx = 490, cy = 300, radius = 205;
    var svg = sv("svg", {
      viewBox: "0 0 980 660", role: "img", "aria-label": title,
      "font-family": "system-ui, sans-serif", "font-size": "13"
    });
    var grid = css("--border"), ink = css("--text-0"), soft = css("--text-1");
    var faint = css("--text-2");
    function point(index, frac) {
      var angle = -Math.PI / 2 + index * 2 * Math.PI / axes.length;
      return [cx + Math.cos(angle) * radius * frac, cy + Math.sin(angle) * radius * frac];
    }
    function ring(frac, attrs) {
      var pts = axes.map(function (_, index) {
        var p = point(index, frac);
        return p[0].toFixed(1) + "," + p[1].toFixed(1);
      }).join(" ");
      return sv("polygon", Object.assign({ points: pts, fill: "none" }, attrs));
    }
    var level;
    for (level = 0.2; level < edge - 1e-9; level += 0.2) {
      svg.appendChild(ring(level / edge, { stroke: grid, "stroke-width": 1 }));
      var mark = point(0, level / edge);
      svg.appendChild(svtext(mark[0] + 6, mark[1] + 4, pct(level, 0), { fill: faint }));
    }
    svg.appendChild(ring(1, { stroke: grid, "stroke-width": 1.5 }));
    var rim = point(0, 1);
    svg.appendChild(svtext(rim[0] + 6, rim[1] + 4, pct(edge, 0), { fill: faint }));
    axes.forEach(function (_, index) {
      var p = point(index, 1);
      svg.appendChild(sv("line", {
        x1: cx, y1: cy, x2: p[0].toFixed(1), y2: p[1].toFixed(1), stroke: grid
      }));
    });
    svg.appendChild(ring(DATA.breakdown.reference_frac, {
      stroke: "#9ca3af", "stroke-width": 1.5, "stroke-dasharray": "4 3"
    }));
    var line = axes.map(function (axis, index) {
      var p = point(index, axis.frac);
      return p[0].toFixed(1) + "," + p[1].toFixed(1);
    }).join(" ");
    svg.appendChild(sv("polygon", {
      points: line, fill: "none", stroke: "#2a78d6", "stroke-width": 2.5
    }));
    axes.forEach(function (axis, index) {
      var p = point(index, axis.frac);
      svg.appendChild(sv("circle", {
        cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 3.5, fill: "#2a78d6"
      }));
      var label = point(index, 1.14);
      var anchor = label[0] > cx + 6 ? "start" : (label[0] < cx - 6 ? "end" : "middle");
      var dim = axis.episodes === 0;
      svg.appendChild(svtext(label[0].toFixed(1), label[1].toFixed(1), axis.label, {
        "text-anchor": anchor, fill: dim ? faint : ink, "font-weight": "600"
      }));
      svg.appendChild(svtext(label[0].toFixed(1), (label[1] + 16).toFixed(1),
        "n=" + axis.episodes + (axis.rate === null ? "" : "  " + pct(axis.rate)),
        { "text-anchor": anchor, fill: dim ? faint : soft }));
    });
    wrap.appendChild(svg);
    wrap.appendChild(el("div", "legend",
      "Solid line: this run's success rate per category. Dashed ring: this run's overall "
      + pct(DATA.breakdown.overall) + ". Rim = " + pct(edge, 0)
      + ". One series only -- a result page has no other run to compare against."));
    host.appendChild(wrap);
  }

  // ------------------------------------------------------- breakdown: matrix
  function matrixCell(svg, cell, x, y, w, h, outline) {
    var fill = hex(cell.color);
    svg.appendChild(sv("rect", {
      x: x, y: y, width: w - 2, height: h - 2, rx: 3, fill: fill,
      stroke: outline ? css("--text-1") : "none", "stroke-width": outline ? 1 : 0
    }));
    var ink = css("--text-0");
    if (cell.episodes === 0) {
      svg.appendChild(svtext(x + w / 2 - 1, y + h / 2 + 4, "\\u2014",
        { "text-anchor": "middle", fill: css("--text-2") }));
      return;
    }
    svg.appendChild(svtext(x + w / 2 - 1, y + h / 2 - 3, pct(cell.rate),
      { "text-anchor": "middle", fill: ink, "font-weight": "600", "font-size": "14" }));
    svg.appendChild(svtext(x + w / 2 - 1, y + h / 2 + 14,
      cell.successes + "/" + cell.episodes,
      { "text-anchor": "middle", fill: ink, "font-size": "11", opacity: "0.85" }));
  }
  function renderMatrix(host) {
    var m = DATA.breakdown.matrix;
    var wrap = el("div", "card pad");
    wrap.setAttribute("id", "matrix");
    wrap.appendChild(el("h2", null, "Scene class x instruction type"));
    var cols = m.columns.length + 1, rows = m.rows.length + 1;
    var left = 150, top = 62, cw = Math.max(96, (1004 - left) / cols), ch = 74;
    var width = left + cw * cols, height = top + ch * rows;
    var svg = sv("svg", {
      viewBox: "0 0 " + Math.round(width) + " " + Math.round(height),
      role: "img", "aria-label": "success rate by scene class and instruction type",
      "font-family": "system-ui, sans-serif", "font-size": "13"
    });
    m.columns.concat([{ label: "Total", episodes: m.grand_total.episodes }])
      .forEach(function (column, index) {
        var x = left + cw * index + cw / 2 - 1;
        svg.appendChild(svtext(x, 24, column.label,
          { "text-anchor": "middle", fill: css("--text-0"), "font-weight": "600" }));
        svg.appendChild(svtext(x, 42, "n=" + column.episodes,
          { "text-anchor": "middle", fill: css("--text-2"), "font-size": "11" }));
      });
    m.rows.forEach(function (row, r) {
      var y = top + ch * r;
      svg.appendChild(svtext(left - 10, y + ch / 2 + 4, row.label,
        { "text-anchor": "end", fill: css("--text-0") }));
      row.cells.forEach(function (cell, c) {
        matrixCell(svg, cell, left + cw * c, y, cw, ch, false);
      });
      matrixCell(svg, row.total, left + cw * m.columns.length, y, cw, ch, true);
    });
    var lastY = top + ch * m.rows.length;
    svg.appendChild(svtext(left - 10, lastY + ch / 2 + 4, "Total",
      { "text-anchor": "end", fill: css("--text-0"), "font-weight": "600" }));
    m.totals.forEach(function (cell, c) {
      matrixCell(svg, cell, left + cw * c, lastY, cw, ch, true);
    });
    matrixCell(svg, m.grand_total, left + cw * m.columns.length, lastY, cw, ch, true);
    wrap.appendChild(svg);
    wrap.appendChild(el("div", "legend",
      "Colour is the cell's success rate minus this run's overall "
      + pct(DATA.breakdown.overall) + ": warm below, cool above, saturating at "
      + Math.round(DATA.breakdown.span * 100)
      + " points either way. Every cell also carries its rate and its k/n, so the colour is "
      + "never the only encoding. Totals carry a thin outline; the bottom-right cell is the "
      + "run total, whose deviation is zero by construction."));
    host.appendChild(wrap);
  }

  function renderBreakdown() {
    var host = document.getElementById("breakdown");
    host.appendChild(el("h2", null, "Breakdown"));
    if (!DATA.breakdown) {
      host.appendChild(el("div", "card pad empty-note",
        "Needs a run with --vis: the instruction type and scene class of an episode live in the "
        + "episode dataset, and only --vis copies them next to the trace as episode.json."));
      return;
    }
    var charts = el("div", "charts");
    radar(charts, DATA.breakdown.instr_type, "Success rate by instruction type");
    radar(charts, DATA.breakdown.scene_class, "Success rate by scene class");
    host.appendChild(charts);
    renderMatrix(host);
  }

  // --------------------------------------------------------- episode table
  var state = {
    index: -1, playing: false, timer: null, cursor: 0,
    // Which column the table is sorted by and which way: 1 ascending, -1
    // descending, 0 the order the run result lists its episodes in.
    sort: { column: null, dir: 0 },
    // Set by a resize drag and read by the header click that follows it, which
    // is otherwise indistinguishable from a click asking for a sort.
    dragged: false,
    // Whether the columns' measured widths have been written down, which the
    // first resize does once so that every drag after it is exact.
    frozen: false
  };

  // The table as one list of columns, so the header, the cells, the filters and
  // the sort all read one spec rather than three copies of it. #, Episode and
  // Frames belong to the page skeleton; everything between them is DATA.columns
  // in the order the column spec sets, which is why Result is not hard-coded
  // third any more.
  var COLUMNS = [
    { key: "#", label: "#", source: "pos" },
    { key: "episode_id", label: "Episode", source: "eid" }
  ].concat(DATA.columns, [{ key: "frames", label: "Frames", source: "frames" }]);

  // The status, the run position and the frame count are on the row rather than
  // in cells, so the table, the filters and the sort go through here rather than
  // each reaching into one or the other.
  function cellValue(row, column) {
    if (column.source === "status") { return row.status; }
    if (column.source === "pos") { return row.pos; }
    if (column.source === "eid") { return row.episode_id; }
    if (column.source === "frames") { return row.frames; }
    return row.cells[column.key];
  }
  function cellText(row, column) {
    var value = cellValue(row, column);
    if (value === null || value === undefined || value === "") { return "\\u2014"; }
    return typeof value === "number" ? num(value) : String(value);
  }
  // A column sorts as a number when it holds numbers: the run position, the
  // frame count and any metric. Everything else sorts as text.
  function isNumeric(column) {
    return column.source === "pos" || column.source === "frames"
      || column.source === "metric";
  }

  function cellNode(row, column) {
    if (column.source === "status") {
      var result = el("td", "r-" + row.status, row.status);
      // Why it failed, on the cell that says it failed.
      if (row.note) { result.title = row.note; }
      return result;
    }
    if (column.source === "pos") { return el("td", "num", row.pos); }
    if (column.source === "eid") { return el("td", "mono", row.episode_id); }
    if (column.source === "frames") {
      var frames = el("td", "num", row.frames === null ? "\\u2014" : row.frames);
      if (row.frames === null) {
        frames.title = row.no_trace
          ? "no trace.json beside this episode"
          : "replay data is not inlined for this episode (page limit)";
      }
      return frames;
    }
    return el("td", column.source === "metric" ? "num" : null, cellText(row, column));
  }

  function renderTable() {
    var head = document.querySelector("#table thead");
    var body = document.querySelector("#table tbody");
    // The run position is the row's own, not the row's place in the table, so
    // it is fixed here rather than recomputed after a sort.
    DATA.episodes.forEach(function (row, index) { row.pos = index + 1; });
    var hrow = el("tr");
    var filters = [];
    COLUMNS.forEach(function (column) { hrow.appendChild(header(column, filters)); });
    head.appendChild(hrow);
    DATA.episodes.forEach(function (row, index) {
      var tr = el("tr");
      COLUMNS.forEach(function (column) { tr.appendChild(cellNode(row, column)); });
      tr.addEventListener("click", function () { select(index); });
      body.appendChild(tr);
      row.node = tr;
    });
    state.body = body;
    renderControls(filters);
    applyFilters();
  }

  // ------------------------------------------------------- header: one cell
  // The label with its sort state, the column's own filter when it has one, and
  // the handle that resizes it. The filter is here rather than in a row of
  // dropdowns above the table so that a reader filters where they are looking.
  function header(column, filters) {
    var th = el("th");
    var lab = el("div", "lab");
    lab.appendChild(el("span", "t", column.label));
    column.ind = el("span", "ind", "\\u2195");
    lab.appendChild(column.ind);
    th.appendChild(lab);
    th.setAttribute("aria-sort", "none");
    if (column.filter) { filters.push(checkFilter(th, column, applyFilters)); }
    th.appendChild(grip(th, column));
    // A stale drag flag cannot eat a later sort: every press on a header clears
    // it, and only a pointer that then moves sets it again.
    th.addEventListener("pointerdown", function () { state.dragged = false; });
    th.addEventListener("click", function (event) {
      // A click that came out of the filter control or off the resize handle is
      // not a request to sort, and neither is the click that ends a drag.
      if (event.target.closest(".flt") || event.target.closest(".grip")) { return; }
      if (state.dragged) { state.dragged = false; return; }
      sortBy(column);
    });
    column.th = th;
    return th;
  }

  // ------------------------------------------------------------------- sort
  // Ascending, then descending, then back to the run's own order -- which is
  // what the table shows before anything is clicked, so the third click is an
  // undo rather than a fourth state to find a way out of.
  function sortBy(column) {
    var sort = state.sort;
    sort.dir = sort.column === column ? (sort.dir === 1 ? -1 : sort.dir === -1 ? 0 : 1) : 1;
    sort.column = sort.dir === 0 ? null : column;
    COLUMNS.forEach(function (item) {
      var dir = item === sort.column ? sort.dir : 0;
      item.ind.textContent = dir === 1 ? "\\u2191" : dir === -1 ? "\\u2193" : "\\u2195";
      item.th.setAttribute("aria-sort",
        dir === 1 ? "ascending" : dir === -1 ? "descending" : "none");
      item.th.classList.toggle("sorted", dir !== 0);
    });
    applySort();
  }

  function sortKey(row, column, numeric) {
    var value = cellValue(row, column);
    if (value === null || value === undefined || value === "") { return null; }
    if (!numeric) { return String(value).toLowerCase(); }
    var number = typeof value === "number" ? value : Number(value);
    return isFinite(number) ? number : null;
  }

  // Sorting moves the rows it already has rather than rebuilding them: this
  // table holds a thousand of them, and a rebuild would drop the filters' hidden
  // flags and the selected row on the floor. One fragment, so the whole reorder
  // costs one insertion.
  function applySort() {
    var sort = state.sort, order = DATA.episodes;
    if (sort.dir !== 0) {
      var numeric = isNumeric(sort.column);
      // From the run's order, and Array.prototype.sort is stable, so rows the
      // sorted column cannot separate stay in the order the run listed them.
      order = DATA.episodes.slice().sort(function (a, b) {
        var x = sortKey(a, sort.column, numeric), y = sortKey(b, sort.column, numeric);
        // Missing last in both directions: an episode that recorded no distance
        // is not the nearest to the goal, and it is not the furthest either.
        if (x === null || y === null) { return x === y ? 0 : (x === null ? 1 : -1); }
        var cmp = x < y ? -1 : x > y ? 1 : 0;
        return sort.dir < 0 ? -cmp : cmp;
      });
    }
    var frag = document.createDocumentFragment();
    order.forEach(function (row) { frag.appendChild(row.node); });
    state.body.appendChild(frag);
  }

  // ----------------------------------------------------------- header: size
  // Drag the right edge of a header to set that column's width. The table is
  // auto-laid-out until the first drag, so the page opens at the widths the
  // content asked for; writing those measured widths down at that moment is what
  // makes every later drag exact rather than a hint the auto algorithm may
  // ignore. Widths are per view: nothing about them is persisted.
  function freezeWidths(table) {
    if (state.frozen) { return; }
    var widths = COLUMNS.map(function (column) {
      return column.th.getBoundingClientRect().width;
    });
    COLUMNS.forEach(function (column, index) {
      column.th.style.width = widths[index] + "px";
    });
    table.style.tableLayout = "fixed";
    state.frozen = true;
    sizeTable(table);
  }
  // Once the columns are fixed the table is exactly as wide as they are, so
  // narrowing one narrows the table instead of having the space handed back to
  // its neighbours by the 100% width the table starts at.
  function sizeTable(table) {
    var total = 0;
    COLUMNS.forEach(function (column) { total += parseFloat(column.th.style.width) || 0; });
    table.style.width = total + "px";
  }

  function grip(th, column) {
    var handle = el("span", "grip");
    handle.setAttribute("aria-hidden", "true");
    handle.title = "drag to resize " + column.label;
    var table = document.getElementById("table");
    var from = 0, width = 0;
    function move(event) {
      // Any movement at all makes this a drag, so the click it ends with is not
      // a request to sort.
      state.dragged = true;
      th.style.width = Math.max(MIN_COLUMN_PX, width + (event.clientX - from)) + "px";
      sizeTable(table);
    }
    function done() {
      handle.classList.remove("on");
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", done);
      document.removeEventListener("pointercancel", done);
    }
    handle.addEventListener("pointerdown", function (event) {
      // Not a text selection, and not the header's own press-then-sort either.
      event.preventDefault();
      event.stopPropagation();
      freezeWidths(table);
      from = event.clientX;
      width = th.getBoundingClientRect().width;
      state.dragged = false;
      handle.classList.add("on");
      // On the document, not on the handle: the pointer leaves a nine-pixel
      // handle at once, and a release anywhere -- another column, the page, off
      // the window -- has to end the drag rather than leave it live.
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", done);
      document.addEventListener("pointercancel", done);
    });
    return handle;
  }

  // Every distinct value of one column with its count, highest count first --
  // counted over the whole run, so the numbers beside the options do not move
  // as you tick boxes and cannot mislead you about what a filter will leave.
  function tally(column) {
    var counts = {}, order = [], i;
    for (i = 0; i < DATA.episodes.length; i += 1) {
      var value = cellValue(DATA.episodes[i], column);
      var text = value === null || value === undefined ? "" : String(value);
      if (counts[text] === undefined) { counts[text] = 0; order.push(text); }
      counts[text] += 1;
    }
    return order.map(function (text) {
      return { value: text, count: counts[text] };
    }).sort(function (a, b) {
      return b.count - a.count || (a.value < b.value ? -1 : a.value > b.value ? 1 : 0);
    });
  }

  // A checkbox dropdown per filterable column: the affordance the internal
  // episode browser uses. Nothing ticked and everything ticked are the same
  // state -- not filtering -- so ticking the last box clears the filter instead
  // of leaving one that excludes nothing.
  function checkFilter(host, column, onChange) {
    var box = el("details", "flt");
    var head = el("summary");
    var panel = el("div", "opts");
    var ticks = [];
    function relabel() {
      var on = chosen();
      // The header cell this control sits in already carries the column's name,
      // so the summary only has to say how much of the column is getting
      // through -- and it has to say that with the dropdown shut, or a filter
      // left on is a table quietly missing rows.
      head.textContent = (on === null ? "all" : on.length + " of " + ticks.length)
        + "  \\u25be";
      box.classList.toggle("on", on !== null);
    }
    function chosen() {
      var on = ticks.filter(function (tick) { return tick.checked; });
      return on.length === 0 || on.length === ticks.length
        ? null : on.map(function (tick) { return tick.value; });
    }
    tally(column).forEach(function (item) {
      var line = el("label");
      var tick = el("input");
      tick.type = "checkbox";
      tick.value = item.value;
      tick.addEventListener("change", function () { relabel(); onChange(); });
      line.appendChild(tick);
      line.appendChild(el("span", null, item.value === "" ? "\\u2014" : item.value));
      line.appendChild(el("span", "n", item.count));
      panel.appendChild(line);
      ticks.push(tick);
    });
    // The table scrolls sideways when the episode ids are long, and this panel
    // is inside that scroller: a column near the right edge would open a
    // dropdown with half its options cut off. Measured on open rather than
    // decided by column, because which columns are near the edge depends on the
    // widths, which the reader can now drag.
    box.addEventListener("toggle", function () {
      if (!box.open) { return; }
      // One dropdown at a time, and a click anywhere else closes it. A <details>
      // otherwise stays open until its own summary is clicked again, which leaves
      // a panel covering the rows the reader just filtered down to.
      OPEN_FILTERS.forEach(function (other) { if (other !== box) { other.open = false; } });
      panel.classList.remove("left");
      var edge = document.getElementById("tablewrap").getBoundingClientRect().right;
      if (panel.getBoundingClientRect().right > edge - 4) { panel.classList.add("left"); }
    });
    OPEN_FILTERS.push(box);
    box.appendChild(head);
    box.appendChild(panel);
    host.appendChild(box);
    relabel();
    return {
      chosen: chosen,
      column: column,
      clear: function () {
        ticks.forEach(function (tick) { tick.checked = false; });
        relabel();
      }
    };
  }

  // What stays above the table: the id search, the reset, and the count. The
  // per-column filters are in the header cells, which is where a reader is
  // already looking when they want to narrow one column; a search over ids and
  // a count of what survived belong to the table as a whole, not to a column.
  function renderControls(filters) {
    var host = document.getElementById("filters");
    var search = el("input");
    search.type = "search";
    search.placeholder = "search episode id\\u2026";
    search.setAttribute("aria-label", "search episode id");
    search.addEventListener("input", applyFilters);
    host.appendChild(search);
    var clear = button("Clear filters", "clear filters", function () {
      filters.forEach(function (filter) { filter.clear(); });
      search.value = "";
      applyFilters();
    });
    clear.hidden = true;
    host.appendChild(clear);
    var count = el("span", "meta num");
    host.appendChild(count);
    state.filters = { list: filters, search: search, clear: clear, count: count };
  }

  function applyFilters() {
    var f = state.filters, visible = [];
    var needle = f.search.value.trim().toLowerCase();
    // Only the filters that actually narrow anything, resolved once rather than
    // per episode: this loop runs over every episode in the run on every
    // keystroke.
    var active = [];
    f.list.forEach(function (filter) {
      var on = filter.chosen();
      if (on !== null) { active.push({ column: filter.column, wanted: on }); }
    });
    DATA.episodes.forEach(function (row, index) {
      var keep = !needle || row.episode_id.toLowerCase().indexOf(needle) !== -1;
      for (var i = 0; keep && i < active.length; i += 1) {
        var value = cellValue(row, active[i].column);
        keep = active[i].wanted.indexOf(
          value === null || value === undefined ? "" : String(value)) !== -1;
      }
      row.node.hidden = !keep;
      if (keep) { visible.push(index); }
    });
    f.clear.hidden = !(needle || active.length);
    f.count.textContent = visible.length + " of " + DATA.episodes.length + " shown";
    document.getElementById("tableempty").hidden = visible.length !== 0;
  }

  // ------------------------------------------------------- replay inspector
  function replayFor(index) {
    var row = DATA.episodes[index];
    return row ? DATA.replay[row.episode_id] || null : null;
  }

  function select(index) {
    if (index < 0 || index >= DATA.episodes.length) { return; }
    state.index = index;
    state.cursor = 0;
    DATA.episodes.forEach(function (row) {
      if (row.node) { row.node.classList.remove("sel"); }
    });
    var row = DATA.episodes[index];
    if (row.node) { row.node.classList.add("sel"); }
    location.hash = "ep=" + encodeURIComponent(row.episode_id);
    renderInspector();
    if (replayFor(index)) { play(true); }
  }

  function step(delta) {
    var entry = replayFor(state.index);
    if (!entry) { return; }
    var next = state.cursor + delta;
    if (next < 0) { next = 0; }
    if (next >= entry.step.length) { next = entry.step.length - 1; play(false); }
    state.cursor = next;
    paint();
  }

  function play(on) {
    state.playing = on;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    // Play requested from the last state means "again, from the start". step()
    // clamps at the end and stops there, so without this the request starts a
    // timer that immediately re-reaches the end and switches itself back off:
    // the button changes glyph for 250 ms and nothing moves, once per click,
    // for ever.
    var entry = replayFor(state.index);
    if (on && entry && state.cursor >= entry.step.length - 1) {
      state.cursor = 0;
      paint();
    }
    if (on) { state.timer = setInterval(function () { step(1); }, PLAY_MS); }
    if (state.controls) { state.controls.play.textContent = on ? "\\u275a\\u275a" : "\\u25b6"; }
  }

  function renderInspector() {
    var host = document.getElementById("inspector");
    host.textContent = "";
    state.controls = null;
    if (state.index < 0) {
      host.appendChild(el("h2", null, "Replay"));
      host.appendChild(el("div", "sub", "Select an episode to inspect its replay."));
      return;
    }
    var row = DATA.episodes[state.index], entry = replayFor(state.index);
    var nav = el("div");
    nav.id = "nav";
    var back = button("\\u2190", "previous episode", function () { select(state.index - 1); });
    var forward = button("\\u2192", "next episode", function () { select(state.index + 1); });
    back.disabled = state.index === 0;
    forward.disabled = state.index === DATA.episodes.length - 1;
    nav.appendChild(back);
    nav.appendChild(forward);
    nav.appendChild(el("span", "mono id", row.episode_id));
    nav.appendChild(el("span", "badge " + row.status, row.status));
    host.appendChild(nav);
    if (!entry) {
      host.appendChild(el("div", "sub", row.no_trace
        ? "There is no trace.json beside this episode, so there is nothing to replay. A run "
          + "writes one per episode when you pass --run-dir."
        : "Replay data is not inlined for this episode. This page inlines per-step replay for "
          + DATA.limits.max_replay_episodes + " episodes, error and failed first."));
      if (row.note) { host.appendChild(el("div", "diag", row.note)); }
      return;
    }
    host.appendChild(el("div", "sub", entry.instruction || "(no instruction recorded)"));
    var shot = el("div", "shot");
    var img = el("img");
    img.alt = "camera frame the policy was shown for this decision";
    shot.appendChild(img);
    var blank = el("div", "empty");
    shot.appendChild(blank);
    shot.addEventListener("click", function () { play(!state.playing); });
    host.appendChild(shot);
    var map = el("div", "map");
    host.appendChild(map);
    var transport = el("div");
    transport.id = "transport";
    var first = button("\\u23ee", "first state", function () {
      play(false);
      state.cursor = 0;
      paint();
    });
    var prev = button("\\u25c0", "previous state", function () { play(false); step(-1); });
    var toggle = button("\\u25b6", "play or pause", function () { play(!state.playing); });
    var next = button("\\u25b8", "next state", function () { play(false); step(1); });
    var last = button("\\u23ed", "last state", function () {
      play(false);
      state.cursor = entry.step.length - 1;
      paint();
    });
    var slider = el("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(entry.step.length - 1);
    slider.value = "0";
    slider.setAttribute("aria-label", "replay position");
    slider.addEventListener("input", function () {
      play(false);
      state.cursor = Number(slider.value);
      paint();
    });
    var readout = el("span", "mono");
    [first, prev, toggle, next, last, slider, readout].forEach(function (node) {
      transport.appendChild(node);
    });
    host.appendChild(transport);
    var diag = el("div", "diag");
    host.appendChild(diag);
    host.appendChild(el("div", "hint",
      "A / D previous or next episode \\u00b7 Space play or pause \\u00b7 click the image to "
      + "toggle \\u00b7 replay is sampled every " + entry.stride + " of "
      + entry.total_steps + " trace steps"));
    state.controls = {
      img: img, blank: blank, map: map,
      slider: slider, readout: readout, diag: diag, play: toggle, entry: entry
    };
    paint();
  }

  // ------------------------------------------------------------ frame panel
  // This panel used to draw a crosshair and a goal ring over the frame, decoding
  // a pointing policy's apos/opos tokens as a 48x27 cell grid and a 0..999 bin
  // pair into image coordinates. That convention was never checked against a
  // real pointing policy, so the markers could be confidently wrong about where
  // the policy had pointed. Marker, decode and the token values it read are all
  // gone: the frame is the frame, and `run-result.json` keeps every field.

  function column(entry, name, index) {
    var values = entry[name];
    return values === undefined ? null : values[index];
  }

  function paintFrame(controls, entry, index) {
    var frame = column(entry, "frame", index);
    if (frame) {
      controls.img.setAttribute("src", frame);
      controls.img.hidden = false;
      controls.blank.hidden = true;
    } else {
      controls.img.removeAttribute("src");
      controls.img.hidden = true;
      controls.blank.hidden = false;
      controls.blank.textContent = entry.frame === undefined
        ? "no frames for this episode (run without --vis)"
        : "no frame for this step";
    }
    // The frame carries no text. A pointing policy's apos/opos fields used to be
    // printed over it as `apos_id=106 apos_xy=[2.1,-3.1]` and so on, next to a
    // marker drawn from them through an unverified image-space convention. The
    // marker went because the convention was never checked; the numbers went
    // with it, because the sentence that explained them went too and six bare
    // token values over a camera image explain nothing on their own.
    return [];
  }

  // --------------------------------------------------------- top-down panel
  function paintMap(controls, entry, index) {
    var W = 420, H = 300, pad = 26;
    var svg = sv("svg", {
      viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "top-down trajectory", "font-family": "system-ui, sans-serif",
      "font-size": "10"
    });
    var xs = [], ys = [], i;
    for (i = 0; i < entry.to.length; i += 1) {
      xs.push(entry.to[i][0]);
      ys.push(entry.to[i][1]);
    }
    xs.push(entry.from[0][0]);
    ys.push(entry.from[0][1]);
    (entry.reference_path || []).forEach(function (point) {
      xs.push(point[0]);
      ys.push(point[1]);
    });
    var radius = entry.goal_radius_m || 0;
    if (entry.goal) {
      xs.push(entry.goal[0] - radius, entry.goal[0] + radius);
      ys.push(entry.goal[1] - radius, entry.goal[1] + radius);
    }
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var spanX = Math.max(1, maxX - minX), spanY = Math.max(1, maxY - minY);
    var scale = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY) * 0.9;
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    function px(x, y) {
      return [W / 2 + (x - cx) * scale, H / 2 - (y - cy) * scale];
    }
    var grid = "#2c3540";
    // The grid step is chosen from the span, never fixed at 1 m: positions come
    // out of an untrusted trace, and a unit mix-up or a teleport puts a span of
    // 500 km in here. paint() rebuilds this SVG four times a second, so "one
    // line per metre" is not a slow page, it is a dead tab. GRID_LINES_MAX is
    // the hard stop for whatever the arithmetic still does not anticipate.
    var gridStep = niceStep(Math.max(spanX, spanY));
    var drawnX = 0, drawnY = 0;
    for (i = Math.ceil(minX / gridStep); i * gridStep <= maxX; i += 1) {
      if (drawnX >= GRID_LINES_MAX) { break; }
      var gx = px(i * gridStep, 0)[0];
      svg.appendChild(sv("line", { x1: gx, y1: 0, x2: gx, y2: H, stroke: grid }));
      drawnX += 1;
    }
    for (i = Math.ceil(minY / gridStep); i * gridStep <= maxY; i += 1) {
      if (drawnY >= GRID_LINES_MAX) { break; }
      var gy = px(0, i * gridStep)[1];
      svg.appendChild(sv("line", { x1: 0, y1: gy, x2: W, y2: gy, stroke: grid }));
      drawnY += 1;
    }
    function polyline(points, attrs) {
      var text = points.map(function (point) {
        var p = px(point[0], point[1]);
        return p[0].toFixed(1) + "," + p[1].toFixed(1);
      }).join(" ");
      svg.appendChild(sv("polyline", Object.assign({ points: text, fill: "none" }, attrs)));
    }
    if ((entry.reference_path || []).length > 1) {
      polyline(entry.reference_path, {
        stroke: "#3f9c92", "stroke-dasharray": "5 4", "stroke-width": 1.5
      });
    }
    if (entry.goal) {
      var goal = px(entry.goal[0], entry.goal[1]);
      if (radius > 0) {
        svg.appendChild(sv("circle", {
          cx: goal[0].toFixed(1), cy: goal[1].toFixed(1), r: (radius * scale).toFixed(1),
          fill: "none", stroke: "#4f9d6b", "stroke-dasharray": "3 3"
        }));
      }
      svg.appendChild(sv("circle", {
        cx: goal[0].toFixed(1), cy: goal[1].toFixed(1), r: 4, fill: "#4f9d6b"
      }));
    }
    var walked = [entry.from[0]].concat(entry.to.slice(0, index + 1));
    var ahead = entry.to.slice(index);
    if (ahead.length > 1) {
      polyline(ahead, { stroke: "#4a5764", "stroke-width": 1.5 });
    }
    if (walked.length > 1) {
      polyline(walked, { stroke: "#c3ced9", "stroke-width": 2 });
    }
    var origin = px(entry.from[0][0], entry.from[0][1]);
    svg.appendChild(sv("rect", {
      x: (origin[0] - 3).toFixed(1), y: (origin[1] - 3).toFixed(1), width: 6, height: 6,
      fill: "#8b98a5"
    }));
    function triangle(pose, filled) {
      var here = px(pose[0], pose[1]);
      var yaw = pose[2], size = 9;
      var pts = [0, 2.4, -2.4].map(function (offset) {
        var angle = yaw + offset;
        var scaleFor = offset === 0 ? size : size * 0.66;
        return [
          (here[0] + Math.cos(angle) * scaleFor).toFixed(1),
          (here[1] - Math.sin(angle) * scaleFor).toFixed(1)
        ].join(",");
      }).join(" ");
      svg.appendChild(sv("polygon", {
        points: pts, fill: filled ? "#e8edf2" : "none", stroke: "#e8edf2",
        "stroke-width": 1.4
      }));
    }
    var from = entry.from[index], to = entry.to[index];
    var a = px(from[0], from[1]), b = px(to[0], to[1]);
    svg.appendChild(sv("line", {
      x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1),
      stroke: "#e8edf2", "stroke-dasharray": "2 2"
    }));
    var hfov = entry.hfov_deg || (DATA.camera ? DATA.camera.camera_hfov_deg : null);
    if (hfov) {
      var half = hfov * Math.PI / 360, reach = 2.5 * scale;
      var wedge = [
        a[0].toFixed(1) + "," + a[1].toFixed(1),
        (a[0] + Math.cos(from[2] + half) * reach).toFixed(1) + ","
          + (a[1] - Math.sin(from[2] + half) * reach).toFixed(1),
        (a[0] + Math.cos(from[2] - half) * reach).toFixed(1) + ","
          + (a[1] - Math.sin(from[2] - half) * reach).toFixed(1)
      ].join(" ");
      svg.appendChild(sv("polygon", {
        points: wedge, fill: "rgba(232,237,242,.10)", stroke: "none"
      }));
    }
    triangle(from, false);
    triangle(to, true);
    var waypoint = column(entry, "waypoint", index);
    if (waypoint) {
      var wx = from[0] + Math.cos(from[2]) * waypoint[0] - Math.sin(from[2]) * waypoint[1];
      var wy = from[1] + Math.sin(from[2]) * waypoint[0] + Math.cos(from[2]) * waypoint[1];
      var target = px(wx, wy);
      svg.appendChild(sv("circle", {
        cx: target[0].toFixed(1), cy: target[1].toFixed(1), r: 3.5, fill: "none",
        stroke: "#2a78d6", "stroke-width": 1.5
      }));
    }
    // The bar is one grid step wide, and says which, so the scale stays honest
    // when the step is not 1 m.
    svg.appendChild(sv("line", {
      x1: 12, y1: H - 12, x2: 12 + gridStep * scale, y2: H - 12, stroke: "#8b98a5",
      "stroke-width": 2
    }));
    svg.appendChild(svtext(16 + gridStep * scale, H - 8, gridStep + " m", { fill: "#8b98a5" }));
    if (!entry.goal) {
      svg.appendChild(svtext(12, 16, "no goal recorded for this episode", { fill: "#8b98a5" }));
    }
    controls.map.textContent = "";
    controls.map.appendChild(svg);
  }

  function paint() {
    var controls = state.controls;
    if (!controls) { return; }
    var entry = controls.entry, index = state.cursor;
    var notes = paintFrame(controls, entry, index);
    paintMap(controls, entry, index);
    controls.slider.value = String(index);
    controls.readout.textContent = (index + 1) + "/" + entry.step.length;
    controls.diag.textContent = notes.join(" \\u00b7 ");
  }

  // -------------------------------------------------------------- keyboard
  document.addEventListener("keydown", function (event) {
    var tag = (event.target && event.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tag === "SUMMARY") {
      return;
    }
    if (event.key === "a" || event.key === "A") { select(state.index - 1); }
    if (event.key === "d" || event.key === "D") { select(state.index + 1); }
    if (event.key === " ") {
      event.preventDefault();
      play(!state.playing);
    }
  });

  renderIdentity();
  renderWarnings();
  renderSpec();
  renderBreakdown();
  renderTable();
  var wanted = decodeURIComponent((location.hash.match(/ep=(.*)$/) || [])[1] || "");
  var start = 0, index;
  for (index = 0; index < DATA.episodes.length; index += 1) {
    if (DATA.episodes[index].episode_id === wanted) { start = index; }
  }
  select(start);
}());
</script>
</body>
</html>
"""
