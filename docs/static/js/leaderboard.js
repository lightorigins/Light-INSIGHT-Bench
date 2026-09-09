(function () {
  "use strict";

  var DATA = null;
  var sortKey = "avg", sortDesc = true, showScene = false;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  // Leaf columns in display order. `group` is null for the ones that stand
  // outside the two metric blocks.
  function columns() {
    var cols = [{ id: "method", label: "Method", group: null }];
    DATA.instruction_types.forEach(function (k, i) {
      cols.push({ id: "i:" + k, key: k, block: "instruction", label: k, first: i === 0 });
    });
    DATA.scene_classes.forEach(function (k, i) {
      cols.push({ id: "s:" + k, key: k, block: "scene", label: k, first: i === 0 });
    });
    cols.push({ id: "avg", label: "Avg.", group: null, first: true });
    return cols.filter(function (c) { return showScene || c.block !== "scene"; });
  }

  function value(row, col) {
    if (col.id === "method") return row.method;
    if (col.id === "avg") return row.avg;
    return row[col.block][col.key];
  }

  // Best and second best per numeric column, so the table carries the same
  // emphasis the paper table does. Computed over every row, not the visible
  // ones, because hiding a block must not change what "best" means.
  function marksFor(col) {
    var vals = DATA.rows.map(function (r) { return value(r, col); });
    var sorted = vals.slice().sort(function (a, b) { return b - a; });
    return { best: sorted[0], second: sorted.find(function (v) { return v < sorted[0]; }) };
  }

  function sortable(th, id, numeric) {
    th.className = (th.className ? th.className + " " : "") + "sortable";
    if (sortKey === id) th.appendChild(el("span", "arrow", sortDesc ? " ▾" : " ▴"));
    th.addEventListener("click", function () {
      if (sortKey === id) sortDesc = !sortDesc;
      else { sortKey = id; sortDesc = numeric; }
      render();
    });
  }

  function buildHead(cols) {
    var head = document.getElementById("lb-head");
    head.textContent = "";
    var top = el("tr"), leaf = el("tr");

    var rank = el("th", "rank", "#"); rank.rowSpan = 2; top.appendChild(rank);
    var method = el("th", null, "Method"); method.rowSpan = 2;
    sortable(method, "method", false); top.appendChild(method);

    [["instruction", "Instruction type", DATA.instruction_types],
     ["scene", "Scene class", DATA.scene_classes]].forEach(function (blk) {
      var members = cols.filter(function (c) { return c.block === blk[0]; });
      if (!members.length) return;
      var th = el("th", "group blockhead", blk[1]);
      th.colSpan = members.length;
      top.appendChild(th);
      members.forEach(function (c) {
        var lt = el("th", c.first ? "group" : null, c.label);
        sortable(lt, c.id, true);
        leaf.appendChild(lt);
      });
    });

    var avg = el("th", "group", "Avg."); avg.rowSpan = 2;
    sortable(avg, "avg", true); top.appendChild(avg);

    head.appendChild(top);
    head.appendChild(leaf);
  }

  function buildBody(cols) {
    var body = document.getElementById("lb-body");
    body.textContent = "";

    var marks = {};
    cols.forEach(function (c) { if (c.id !== "method") marks[c.id] = marksFor(c); });

    var sortCol = cols.filter(function (c) { return c.id === sortKey; })[0] ||
                  cols[cols.length - 1];
    var rows = DATA.rows.slice().sort(function (a, b) {
      var x = value(a, sortCol), y = value(b, sortCol);
      var d = (typeof x === "string") ? x.localeCompare(y) : x - y;
      return sortDesc ? -d : d;
    });

    rows.forEach(function (r) {
      var tr = el("tr", r.method === DATA.reference ? "ours" : null);
      tr.appendChild(el("td", "rank", r.rank));
      cols.forEach(function (col) {
        if (col.id === "method") {
          var td = el("td");
          var a = el("a", null, r.method);
          a.href = r.code;
          a.title = "code repository";
          td.appendChild(a);
          if (r.torch) td.appendChild(el("span", "links", "  · torch " + r.torch));
          var tag = el("span", "tag " + r.provenance,
            r.provenance === "evidence-pack" ? "verified" : "published");
          tag.title = r.provenance === "evidence-pack"
            ? "computed from an evidence pack that passed insight-bench verify"
            : "reported in " + (r.reference || "the paper") + ", not recomputed here";
          td.appendChild(tag);
          tr.appendChild(td);
          return;
        }
        var v = value(r, col);
        var cls = col.first ? "group" : "";
        var m = marks[col.id];
        if (v === m.best) cls += " best";
        else if (v === m.second) cls += " second";
        tr.appendChild(el("td", cls.trim() || null, v.toFixed(1)));
      });
      body.appendChild(tr);
    });
  }

  function buildToggle() {
    var host = document.getElementById("lb-controls");
    if (!host) return;
    host.textContent = "";
    var b = el("button", "btn small", showScene ? "Hide scene classes" : "Show scene classes");
    b.type = "button";
    b.addEventListener("click", function () { showScene = !showScene; render(); });
    host.appendChild(b);
    host.appendChild(el("span", "hint",
      showScene ? "  11 metric columns" : "  5 scene-class columns hidden"));
  }

  function render() {
    var cols = columns();
    buildHead(cols);
    buildBody(cols);
    buildToggle();
  }

  function boot(d) {
    DATA = d;
    document.getElementById("lb-caption").textContent =
      d.note + " " + d.rows.length + " policies, " + d.episodes + " episodes in " + d.scenes +
      " scenes, coordinate " + d.coordinate + ". Updated " + d.updated + ".";
    render();
  }

  fetch("data/leaderboard.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(boot)
    .catch(function (e) {
      document.getElementById("lb-caption").textContent =
        "Could not load data/leaderboard.json: " + e.message;
    });

  var copy = document.getElementById("copybib");
  if (copy) {
    copy.addEventListener("click", function () {
      navigator.clipboard.writeText(document.getElementById("bib").textContent).then(function () {
        copy.textContent = "Copied";
        setTimeout(function () { copy.textContent = "Copy BibTeX"; }, 1600);
      });
    });
  }
})();
