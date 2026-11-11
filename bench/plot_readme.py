"""Render the README figures from the committed result files (light and dark variants).

    .venv/Scripts/python bench/plot_readme.py

Reads bench/results/*.json and phase0/results_1.5b_holdout/summary.json; writes docs/img/*.png.
"""
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", full="#2a78d6", plain="#eb6834", rec="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", full="#3987e5", plain="#d95926", rec="#199e70"),
}


def figure(t, title, subtitle, note):
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig = plt.figure(figsize=(8, 4.4), dpi=200, facecolor=t["surface"])
    ax = fig.add_axes([0.085, 0.19, 0.72, 0.54], facecolor=t["surface"])
    fig.text(0.03, 0.035, note, color=t["muted"], fontsize=8.5, va="bottom")
    fig.text(0.03, 0.95, title, color=t["ink"], fontsize=14, fontweight="bold", va="top")
    fig.text(0.03, 0.875, subtitle, color=t["ink2"], fontsize=9.5, va="top")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def legend(fig, t, items):
    x, r = 0.03, fig.canvas.get_renderer()
    for label, color in items:
        fig.patches.append(plt.Rectangle((x, 0.782), 0.022, 0.008, transform=fig.transFigure, color=color))
        txt = fig.text(x + 0.03, 0.786, label, color=t["ink2"], fontsize=9, va="center")
        x += 0.03 + txt.get_window_extent(r).width / fig.bbox.width + 0.035


def line(ax, t, x, y, color):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, markerfacecolor=color,
            markeredgecolor=t["surface"], markeredgewidth=1.5, clip_on=False, zorder=3)


def latency(t, name):
    res = json.load(open(ROOT / "bench/results/scaling_Qwen2.5-0.5B-Instruct.json"))["results"]
    lengths = ["1000", "2000", "4000", "8000"]  # 16k hit VRAM spill: invalid, left out
    med = {k: [statistics.median(res[L]["res"][k]["ttft"]) for L in lengths]
           for k in ("full", "tread_r0.0", "tread_r0.2")}
    fig, ax = figure(t, "Reuse pulls ahead as the context grows",
                     "Time to first token when the same documents come back in a different order.",
                     "Qwen2.5-0.5B fp16, RTX 3050 laptop, median of 13 questions per length. "
                     "Plain reuse is fastest but least accurate; see the quality figure.")
    x = np.arange(len(lengths))
    series = [("Full prefill", "full", t["full"]), ("tread, 20% recompute", "tread_r0.2", t["rec"]),
              ("tread, plain reuse", "tread_r0.0", t["plain"])]
    for label, k, c in series:
        line(ax, t, x, med[k], c)
        end = med[k][-1]
        txt = f"{end:.0f} ms" + ("" if k == "full" else f"  ·  {med['full'][-1] / end:.1f}× faster")
        ax.annotate(txt, (x[-1], end), xytext=(9, 0), textcoords="offset points", va="center",
                    color=t["ink2"], fontsize=9, annotation_clip=False)
    legend(fig, t, [(s[0], s[2]) for s in series])
    ax.set_xticks(x, ["1k", "2k", "4k", "8k"])
    ax.set_xlim(-0.15, len(x) - 0.85)
    ax.set_ylim(0, 900)
    ax.set_yticks(range(0, 901, 300), [f"{v} ms" if v else "0" for v in range(0, 901, 300)])
    ax.set_xlabel("context tokens", color=t["muted"], fontsize=9)
    fig.savefig(OUT / f"latency-{name}.png", facecolor=t["surface"])
    plt.close(fig)
    return {k: [round(v) for v in vs] for k, vs in med.items()}


def quality(t, name):
    rows = json.load(open(ROOT / "phase0/results_1.5b_holdout/summary.json"))
    pts = sorted((0.0 if r["method"] == "naive" else r["ratio"], r["kl"], *r["ci"]["kl"]) for r in rows
                 if r["method"] in ("naive", "deviation"))
    pts.append((1.0, 0.0, 0.0, 0.0))  # 100% recompute reproduces full prefill
    r, kl, lo, hi = (np.array(c) for c in zip(*pts))
    fig, ax = figure(t, "A little recompute buys back most of the quality",
                     "Divergence from a full prefill (KL, lower is better) by share of document tokens recomputed.",
                     "Qwen2.5-1.5B fp16, 300 held-out HotpotQA questions, 4 shuffled documents each.")
    ax.fill_between(r * 100, lo, hi, color=t["full"], alpha=0.18, linewidth=0)
    line(ax, t, r * 100, kl, t["full"])
    for i, dy in ((0, 10), (4, 10), (6, 10)):
        ax.annotate(f"{kl[i]:.3f}", (r[i] * 100, kl[i]), xytext=(6, dy), textcoords="offset points",
                    color=t["ink2"], fontsize=9)
    ax.annotate("exact at 100%", (100, 0), xytext=(-6, 10), textcoords="offset points", ha="right",
                color=t["ink2"], fontsize=9)
    fig.text(0.03, 0.786, "Shaded band: 95% confidence interval", color=t["ink2"], fontsize=9, va="center")
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 0.33)
    ax.set_xticks([0, 10, 20, 30, 50, 100], ["none", "10%", "20%", "30%", "50%", "100%"])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("document tokens recomputed", color=t["muted"], fontsize=9)
    fig.savefig(OUT / f"quality-{name}.png", facecolor=t["surface"])
    plt.close(fig)
    return [(float(a), round(float(b), 3)) for a, b in zip(r, kl)]


def spill(t, name):
    res = json.load(open(ROOT / "bench/results/k8_vram_spill.json"))["res"]
    full, plain = res["full"]["ttft"], res["tread_r0.0"]["ttft"]
    mf, mp = statistics.median(full), statistics.median(plain)
    fig, ax = figure(t, f"The benchmark that lied: a fake {mf / mp:.0f}× speedup",
                     "Out of VRAM, Windows silently spilled to system RAM and slowed only the big passes.",
                     "Each dot is one request. Qwen2.5-1.5B fp16 on a 4 GB card with 19 MiB free. "
                     "Invalid run, kept as a warning.")
    rng = np.random.default_rng(0)
    for y, (vals, c, m) in enumerate(((plain, t["plain"], mp), (full, t["full"], mf))):
        ax.scatter(vals, y + rng.uniform(-0.16, 0.16, len(vals)), s=38, color=c, alpha=0.85,
                   edgecolors=t["surface"], linewidths=1, zorder=3, clip_on=False)
        ax.plot([m, m], [y - 0.3, y + 0.3], color=t["ink"], linewidth=1.5, zorder=4)
        ax.annotate(f"median {m:,.0f} ms", (m, y + 0.3), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=t["ink2"], fontsize=9)
    legend(fig, t, [("Full prefill (spilled, invalid)", t["full"]), ("tread, plain reuse", t["plain"])])
    ax.set_xscale("log")
    ax.set_xlim(60, 4000)
    ax.set_xticks([100, 300, 1000, 3000], ["100 ms", "300 ms", "1 s", "3 s"])
    ax.minorticks_off()
    ax.set_ylim(-0.6, 1.7)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_xlabel("time to first token (log scale)", color=t["muted"], fontsize=9)
    fig.savefig(OUT / f"vram-spill-{name}.png", facecolor=t["surface"])
    plt.close(fig)
    return dict(full_median=round(mf), plain_median=round(mp), full_min=round(min(full)), full_max=round(max(full)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        out = latency(t, name), quality(t, name), spill(t, name)
    for o in out:
        print(o)
