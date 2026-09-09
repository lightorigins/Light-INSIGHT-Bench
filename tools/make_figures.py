#!/usr/bin/env python3
"""Regenerate the README image assets for the INSIGHT-Bench evaluation SDK.

Outputs (written next to this file, in ``../assets``):

  assets/insight_bench_matrix.png
      Joint scene x instruction capability matrix for LightNav-0
      (paper Fig. 12).
  assets/insight_radar.png
      Two radar charts -- success rate by instruction type and by scene
      type, one polygon per model, shared 0-70 scale (blog Fig. 10B).

Run:  uv run --with matplotlib --with numpy --with pillow python tools/make_figures.py

This script is self-contained: every number below is hard-coded and it
reads nothing outside this repository at runtime.

Provenance of every number
==========================

Matrix (Fig. 12) -- ported from the paper's own generator,
``figure/insight_bench_matrix.py`` in the LightNav-0 Overleaf source,
which emits a PDF.  This port emits a PNG for GitHub; the data,
colours, layout metrics and consistency assertions are unchanged.

  DEN, the per-cell episode denominators
      paper table/insight_bench_taxonomy.tex (Episode distribution of
      the INSIGHT-Bench evaluation split).
  NUM, the per-cell successful episodes for LightNav-0
      blog run ``v2-open-models-brief-unified-radius`` (unified 2.0 m
      indoor / 3.0 m outdoor success radius), as recorded in the paper
      generator.
  ROW_TOT_DEN / COL_TOT_DEN, the marginal episode counts
      Total row and Episodes column of insight_bench_taxonomy.tex.
  Marginal success rates implied by NUM/DEN reproduce the LightNav-0
      row of paper table/insight_bench_breakdown.tex exactly:
      Base 45.1, Direction 57.7, Relation 37.8, Extremum 37.2,
      Ordinal 38.2; Apartment 61.1, House 50.5, Commercial 42.1,
      Institution 29.2, Outdoor 34.2.
  GRAND_NUM / GRAND_DEN = 479 / 1,097 = 43.66 %
      overall aggregate, same blog run; 43.7 % is the Avg. column of
      insight_bench_breakdown.tex.

Radar (blog Fig. 10B) -- the seven-model comparison.

  MODELS[*]["instruction"] and MODELS[*]["scene"], all in percent
      paper table/insight_bench_breakdown.tex (Fine-grained success
      rate on INSIGHT-Bench), identical to the blog's own figure data.
  AXIS_N, the episodes behind each axis
      Total row of insight_bench_taxonomy.tex:
      Base 237, Direction 239, Relation 193, Extremum 250, Ordinal 178;
      Apartment 239, House 216, Commercial 195, Institution 219,
      Outdoor 228.  Both sets sum to 1,097.
  RMAX = 70 and the 14/28/42/56/70 rings
      the shared 0-70 scale the blog uses for both panels.
  Per-model colours and dash patterns
      the blog stylesheet's ``.insight-radar-series.model-*`` rules.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Inter",
    "Helvetica Neue",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]

TARGET_W = 1600  # px, the width both README assets are exported at


# =========================================================== Fig. 12: matrix
ROWS = ["Apartment", "House", "Commercial", "Institution", "Outdoor"]
COLS = ["Base", "Direction", "Relation", "Extremum", "Ordinal"]

# fmt: off
# successes (blog: v2-open-models-brief-unified-radius, unified 2.0/3.0 m radius)
NUM = np.array([
    [34, 36, 27, 33, 16],
    [30, 37,  9, 16, 17],
    [16, 21, 12, 17, 16],
    [14, 22, 10,  9,  9],
    [13, 22, 15, 18, 10],
], dtype=float)

# episodes (table/insight_bench_taxonomy.tex)
DEN = np.array([
    [50, 50, 50, 50, 39],
    [50, 50, 31, 50, 35],
    [42, 40, 30, 50, 33],
    [50, 49, 32, 50, 38],
    [45, 50, 50, 50, 33],
], dtype=float)
# fmt: on

ROW_TOT_DEN = np.array([239, 216, 195, 219, 228], dtype=float)
COL_TOT_DEN = np.array([237, 239, 193, 250, 178], dtype=float)
GRAND_NUM, GRAND_DEN = 479.0, 1097.0

# consistency guard: never let the figure drift from the tables
assert NUM.sum(1).tolist() == [146, 109, 82, 64, 78]
assert NUM.sum(0).tolist() == [107, 138, 73, 93, 68]
assert DEN.sum(1).tolist() == ROW_TOT_DEN.tolist()
assert DEN.sum(0).tolist() == COL_TOT_DEN.tolist()
assert NUM.sum() == GRAND_NUM and DEN.sum() == GRAND_DEN

OVERALL = 100.0 * GRAND_NUM / GRAND_DEN  # 43.6645... -> 43.66 %

# full 6x6 grid (5 scene rows + Total row) x (5 instruction cols + Total col)
G_NUM = np.zeros((6, 6))
G_DEN = np.zeros((6, 6))
G_NUM[:5, :5], G_DEN[:5, :5] = NUM, DEN
G_NUM[:5, 5], G_DEN[:5, 5] = NUM.sum(1), ROW_TOT_DEN
G_NUM[5, :5], G_DEN[5, :5] = NUM.sum(0), COL_TOT_DEN
G_NUM[5, 5], G_DEN[5, 5] = GRAND_NUM, GRAND_DEN
G_SR = 100.0 * G_NUM / G_DEN

WARM, COOL = "#ED8179", "#6A9DDF"  # colours declared by the paper
CMAP = LinearSegmentedColormap.from_list("insight", [WARM, "#FFFFFF", COOL])
VMIN, VMAX = G_SR.min(), G_SR.max()  # 18.0 .. 74.0
NORM = TwoSlopeNorm(vmin=VMIN, vcenter=OVERALL, vmax=VMAX)

INK, SUB, RULE, HAIR = "#0B0B0B", "#55534F", "#8A8A8A", "#D8D8D8"


def make_matrix(path: Path) -> None:
    """Port of figure/insight_bench_matrix.py, rasterised to PNG.

    Every unit below is a PostScript point; the axes is 1:1 with the canvas.
    """
    PAD = 7.0  # target margin on all four sides
    LBL_W = 84.0  # row-label column
    LBL_GAP = 9.0
    CELL_W, CELL_H = 89.0, 40.0
    GAP = 5.0
    HDR_H = 22.0
    CB_GAP, CB_W, CB_TICKW, CB_LABEL = 15.0, 13.0, 30.0, 17.0

    GRID_X0 = PAD + LBL_W + LBL_GAP
    GRID_W = 6 * CELL_W + 5 * GAP
    GRID_H = 6 * CELL_H + 5 * GAP
    W = GRID_X0 + GRID_W + CB_GAP + CB_W + CB_TICKW + CB_LABEL + PAD
    H = PAD + HDR_H + GRID_H + PAD

    FS_PCT, FS_FRAC, FS_HDR, FS_CB = 15.0, 11.0, 13.0, 11.0

    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.patch.set_visible(False)

    col_x = [GRID_X0 + c * (CELL_W + GAP) for c in range(6)]
    row_y = [H - PAD - HDR_H - (r + 1) * CELL_H - r * GAP for r in range(6)]

    # column headers
    for c, name in enumerate([*COLS, "Total"]):
        ax.text(
            col_x[c] + CELL_W / 2,
            H - PAD - HDR_H / 2 - 1.0,
            name,
            ha="center",
            va="center",
            fontsize=FS_HDR,
            fontweight="bold",
            color=INK,
        )

    # cells
    for r in range(6):
        for c in range(6):
            sr = G_SR[r, c]
            is_margin = (r == 5) or (c == 5)
            is_grand = (r == 5) and (c == 5)
            if is_grand:
                edge, lw = "#3A3A3A", 1.3
            elif is_margin:
                edge, lw = RULE, 0.9
            else:
                edge, lw = HAIR, 0.5
            ax.add_patch(
                FancyBboxPatch(
                    (col_x[c] + 2.5, row_y[r] + 2.5),
                    CELL_W - 5.0,
                    CELL_H - 5.0,
                    boxstyle="round,pad=2.5,rounding_size=5",
                    facecolor=CMAP(NORM(sr)),
                    edgecolor=edge,
                    linewidth=lw,
                    mutation_aspect=1.0,
                    zorder=1,
                )
            )
            cx, cy = col_x[c] + CELL_W / 2, row_y[r] + CELL_H / 2
            pct = f"{sr:.2f}%" if is_grand else f"{sr:.1f}%"
            ax.text(
                cx,
                cy + 7.5,
                pct,
                ha="center",
                va="center",
                fontsize=FS_PCT,
                fontweight="bold",
                color=INK,
                zorder=2,
            )
            ax.text(
                cx,
                cy - 9.0,
                f"{int(G_NUM[r, c])}/{int(G_DEN[r, c]):,}",
                ha="center",
                va="center",
                fontsize=FS_FRAC,
                color=SUB,
                zorder=2,
            )

    # row labels
    for r, name in enumerate([*ROWS, "Total"]):
        ax.text(
            PAD + LBL_W,
            row_y[r] + CELL_H / 2,
            name,
            ha="right",
            va="center",
            fontsize=FS_HDR,
            fontweight="bold",
            color=INK,
        )

    # colorbar
    cb_x0 = GRID_X0 + GRID_W + CB_GAP
    cax = fig.add_axes([cb_x0 / W, row_y[5] / H, CB_W / W, GRID_H / H])
    cbar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax)
    ticks = [VMIN, 30.0, OVERALL, 55.0, 65.0, VMAX]
    assert VMIN < ticks[1] < OVERALL < ticks[3] < ticks[4] < VMAX
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.2f}" if t is OVERALL else f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(labelsize=FS_CB, length=2.5, width=0.7, pad=2.5, color=INK)
    cbar.outline.set(edgecolor=RULE, linewidth=0.8)
    cbar.set_label("Success rate (%)", fontsize=FS_CB + 1, color=INK, labelpad=3)
    # mark the overall SR, i.e. the neutral point of the diverging scale
    cbar.ax.axhline(NORM(OVERALL), color=INK, linewidth=1.1, zorder=5)

    # Equal margins: rasterise once, find the true ink bbox, crop to it with
    # an equal pad, then downsample to the README width.
    render_dpi = 2.0 * TARGET_W / (W / 72.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=render_dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    arr = np.asarray(img).astype(int)
    ink = np.abs(arr - 255).sum(2) > 12
    ys, xs = np.nonzero(ink)
    pad_px = round(PAD / 72.0 * render_dpi)
    x0, y0 = int(xs.min()) - pad_px, int(ys.min()) - pad_px
    x1, y1 = int(xs.max()) + 1 + pad_px, int(ys.max()) + 1 + pad_px
    # Ink can sit closer to the canvas edge than PAD -- the colorbar's bottom
    # tick label does. Compose onto a white sheet of the full padded size
    # rather than clamping the crop, which would thin that margin instead.
    sheet = Image.new("RGB", (x1 - x0, y1 - y0), "white")
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, img.width), min(y1, img.height)
    sheet.paste(img.crop((cx0, cy0, cx1, cy1)), (cx0 - x0, cy0 - y0))
    img = sheet
    img = img.resize((TARGET_W, max(1, round(img.height * TARGET_W / img.width))), Image.LANCZOS)
    img.save(path)
    print(f"wrote {path}  {img.width} x {img.height} px")


# ============================================================ Fig. 14: radar
# Success rate (%) per model, per axis (table/insight_bench_breakdown.tex).
# fmt: off
MODELS = [
    {"name": "JanusVLN",     "color": "#16181d", "dash": (0, ()),
     "instruction": [32.5, 28.5, 28.5, 22.0, 25.8],
     "scene":       [39.7, 35.2, 26.7, 18.3, 16.7]},
    {"name": "NaVid",        "color": "#267f87", "dash": (0, (8, 4)),
     "instruction": [37.6, 29.7, 18.7, 22.4, 24.2],
     "scene":       [37.7, 39.8, 24.6, 18.3, 13.6]},
    {"name": "Uni-NaVid",    "color": "#7561a8", "dash": (0, (2, 4)),
     "instruction": [32.9, 29.3, 21.8, 16.8, 19.7],
     "scene":       [39.3, 28.7, 23.1, 15.5, 14.0]},
    {"name": "TAMP-Nav",     "color": "#667d4f", "dash": (0, (10, 3, 2, 3)),
     "instruction": [18.1, 14.2, 17.6, 10.8, 20.8],
     "scene":       [26.8, 18.5, 15.9,  9.6,  8.3]},
    {"name": "InternVLA-N1", "color": "#4f7597", "dash": (0, (4, 5)),
     "instruction": [18.1, 13.0,  9.8,  8.4,  7.9],
     "scene":       [12.1, 19.0, 12.3,  7.8,  7.5]},
    {"name": "StreamVLN",    "color": "#a45474", "dash": (0, (1, 4)),
     "instruction": [15.6,  9.6, 14.5,  8.0, 10.7],
     "scene":       [19.7, 15.7, 12.8,  4.1,  5.3]},
    {"name": "LightNav-0",   "color": "#315bda", "dash": (0, ()),
     "instruction": [45.1, 57.7, 37.8, 37.2, 38.2],
     "scene":       [61.1, 50.5, 42.1, 29.2, 34.2]},
]
# fmt: on
HERO = "LightNav-0"

# Episodes behind each axis (Total row of table/insight_bench_taxonomy.tex).
# fmt: off
AXIS_N = {
    "instruction": {"Base": 237, "Direction": 239, "Relation": 193,
                    "Extremum": 250, "Ordinal": 178},
    "scene":       {"Apartment": 239, "House": 216, "Commercial": 195,
                    "Institution": 219, "Outdoor": 228},
}
# fmt: on
assert sum(AXIS_N["instruction"].values()) == int(GRAND_DEN)
assert sum(AXIS_N["scene"].values()) == int(GRAND_DEN)

RMAX = 70.0  # shared 0-70 scale, as the blog does
RINGS = [14.0, 28.0, 42.0, 56.0, RMAX]
GRIDC, TICKC = "#C9CBD2", "#8A8D96"


def _vertices(values):
    """Value list -> unit-radius xy vertices, first axis up, going clockwise."""
    out = []
    for i, v in enumerate(values):
        a = np.pi / 2.0 - i * 2.0 * np.pi / len(values)
        d = v / RMAX
        out.append((np.cos(a) * d, np.sin(a) * d))
    return out


def _draw_panel(ax, key, labels, title):
    ax.set_aspect("equal")
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.12, 1.62)
    ax.set_axis_off()

    n = len(labels)
    for ring in RINGS:
        ax.add_patch(
            Polygon(
                _vertices([ring] * n),
                closed=True,
                fill=False,
                edgecolor=GRIDC,
                linewidth=0.8,
                zorder=1,
            )
        )

    for i, label in enumerate(labels):
        a = np.pi / 2.0 - i * 2.0 * np.pi / n
        cs, sn = np.cos(a), np.sin(a)
        ax.plot([0, cs], [0, sn], color=GRIDC, linewidth=0.8, zorder=1)
        ha = "left" if cs > 0.28 else "right" if cs < -0.28 else "center"
        va = "bottom" if sn > 0.3 else "top" if sn < -0.3 else "center"
        lx, ly = cs * 1.13, sn * 1.13
        # the episode count always sits directly under the axis name
        name_y = ly + 0.078 if va == "bottom" else ly
        n_y = ly if va == "bottom" else ly - 0.078
        ax.text(lx, name_y, label, ha=ha, va=va, fontsize=12.5, color=INK, zorder=4)
        ax.text(lx, n_y, f"n={AXIS_N[key][label]}", ha=ha, va=va, fontsize=9, color=TICKC, zorder=4)

    for model in MODELS:
        hero = model["name"] == HERO
        verts = _vertices(model[key])
        if hero:
            # translucent wash first, so the stroke over it stays saturated
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=model["color"],
                    alpha=0.10,
                    edgecolor="none",
                    zorder=5,
                )
            )
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor="none",
                edgecolor=model["color"],
                alpha=1.0 if hero else 0.68,
                linewidth=2.8 if hero else 1.4,
                linestyle=model["dash"],
                zorder=6 if hero else 3,
            )
        )
        if hero:
            xs, ys = zip(*verts, strict=True)
            ax.plot(
                xs,
                ys,
                linestyle="none",
                marker="o",
                markersize=5.4,
                markerfacecolor="white",
                markeredgecolor=model["color"],
                markeredgewidth=1.7,
                zorder=7,
            )

    # ring values last, so they read over whatever crosses the centre spoke
    for ring in RINGS:
        ax.text(
            0.026,
            ring / RMAX,
            f"{int(ring)}",
            ha="left",
            va="center",
            fontsize=8,
            color=TICKC,
            zorder=9,
            bbox={
                "boxstyle": "square,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )

    ax.text(0, 1.54, title, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    # \u2013 is an en dash, spelled as an escape so the source stays ASCII
    ax.text(
        0,
        1.44,
        "success rate (%) \u00b7 shared 0\u201370 scale",
        ha="center",
        va="center",
        fontsize=9.5,
        color=SUB,
    )


def make_radar(path: Path) -> None:
    fig = plt.figure(figsize=(TARGET_W / 100.0, 7.9), dpi=100)
    fig.patch.set_facecolor("white")

    ax_l = fig.add_axes([0.005, 0.085, 0.49, 0.90])
    ax_r = fig.add_axes([0.505, 0.085, 0.49, 0.90])
    _draw_panel(ax_l, "instruction", list(AXIS_N["instruction"]), "By instruction type")
    _draw_panel(ax_r, "scene", list(AXIS_N["scene"]), "By scene type")

    handles = [
        Line2D(
            [],
            [],
            color=m["color"],
            linestyle=m["dash"],
            linewidth=2.6 if m["name"] == HERO else 1.5,
            alpha=1.0 if m["name"] == HERO else 0.75,
            marker="o" if m["name"] == HERO else None,
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=m["color"],
            markeredgewidth=1.6,
            label=m["name"],
        )
        for m in MODELS
    ]
    # LightNav-0 first, as the blog legend does
    handles = [handles[-1], *handles[:-1]]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=11.5,
        handlelength=2.6,
        columnspacing=2.0,
        bbox_to_anchor=(0.5, 0.012),
    )
    ordered = [MODELS[-1], *MODELS[:-1]]
    for text, m in zip(leg.get_texts(), ordered, strict=True):
        text.set_color(INK)
        if m["name"] == HERO:
            text.set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    # flatten to RGB: the page is opaque white, and a matching mode keeps both
    # README assets identical in kind.
    img = Image.open(buf).convert("RGB")
    img.save(path)
    print(f"wrote {path}  {img.width} x {img.height} px")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_matrix(ASSETS / "insight_bench_matrix.png")
    make_radar(ASSETS / "insight_radar.png")
