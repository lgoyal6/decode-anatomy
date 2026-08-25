"""Render the roofline plot and the category-breakdown figure."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAT_COLORS = {
    "proj_gemm": "#c0392b", "lm_head": "#e67e22", "attention": "#2980b9",
    "norm_elem": "#27ae60", "mem_move": "#8e44ad", "sampling": "#7f8c8d",
    "gqa_expand": "#16a085",
}
MARKERS = {128: "o", 512: "s", 2048: "^", 4096: "D"}


def roofline(rl, sweep, out):
    peak_bw = rl["summary"]["peak_hbm_gbs"]
    peak_tf = rl["summary"]["peak_bf16_tflops"]
    ridge = rl["summary"]["ridge_point_flop_per_byte"]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ai = np.logspace(-1.3, 4.2, 400)
    ax.plot(ai, np.minimum(peak_bw * 1e9 * ai / 1e12, peak_tf),
            "k-", lw=2.2, zorder=5,
            label=f"empirical roofline ({peak_bw:.0f} GB/s, {peak_tf:.0f} TFLOP/s)")
    spec_bw, spec_tf = rl["spec"]["mem_bw_gbs"], rl["spec"]["bf16_tensor_tflops"]
    ax.plot(ai, np.minimum(spec_bw * 1e9 * ai / 1e12, spec_tf),
            "k--", lw=1.2, alpha=0.45,
            label=f"spec sheet ({spec_bw:.0f} GB/s, {spec_tf:.0f} TFLOP/s)")
    ax.axvline(ridge, color="k", ls=":", lw=1, alpha=0.5)
    ax.text(ridge * 1.12, peak_tf * 0.055,
            f"ridge\n{ridge:.0f} FLOP/byte", fontsize=8.5, alpha=0.8)

    # measured square GEMMs, to show where a well-fed kernel actually lands
    g = [r for r in rl["gemm"] if r["dtype"] == "bf16"]
    ax.scatter([r["arith_intensity"] for r in g], [r["tflops"] for r in g],
               marker="x", s=45, c="k", alpha=0.55, zorder=6,
               label="bf16 square GEMM (N=512..16384)")

    # decode categories
    seen = set()
    for p in sweep["points"]:
        for c in p["categories"]:
            if c["achieved_tflops"] <= 0:
                continue
            lab = None
            if c["category"] not in seen:
                seen.add(c["category"]); lab = c["category"]
            ax.scatter(c["arith_intensity"], c["achieved_tflops"],
                       marker=MARKERS.get(p["cache_len"], "o"),
                       s=26 + 3.2 * np.log2(p["batch"] + 1) ** 2,
                       facecolor=CAT_COLORS.get(c["category"], "#333"),
                       edgecolor="white", lw=0.5, alpha=0.85, zorder=7,
                       label=lab)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte of HBM traffic)")
    ax.set_ylabel("achieved throughput (TFLOP/s)")
    ax.set_title("RTX A6000 (sm_86): decode-step kernel categories on the "
                 "empirical roofline\nQwen2.5-7B-Instruct bf16, batch 1-64, "
                 "context 128-4096", fontsize=10.5)
    ax.grid(True, which="both", alpha=0.16)
    ax.set_ylim(1e-4, peak_tf * 2.2)
    ax.legend(fontsize=7.6, loc="lower right", ncol=2, framealpha=0.93)
    fig.text(0.012, 0.012,
             "marker shape = context length (o 128, s 512, ^ 2048, D 4096); "
             "marker size grows with batch. Points far below the roofline are "
             "latency-bound: neither ceiling explains them.",
             fontsize=7, alpha=0.75)
    fig.tight_layout(rect=(0, 0.032, 1, 1))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def breakdown(sweep, out):
    pts = sweep["points"]
    cls = sorted({p["cache_len"] for p in pts})
    fig, axes = plt.subplots(2, len(cls), figsize=(4.1 * len(cls), 7.6),
                             sharey="row")
    if len(cls) == 1:
        axes = axes.reshape(2, 1)
    cats = ["proj_gemm", "lm_head", "attention", "gqa_expand", "norm_elem",
            "mem_move", "sampling"]

    for j, cl in enumerate(cls):
        sub = sorted([p for p in pts if p["cache_len"] == cl],
                     key=lambda r: r["batch"])
        xs = np.arange(len(sub))
        labels = [str(p["batch"]) for p in sub]

        # row 0: where wall time goes (the four-way split)
        ax = axes[0, j]
        keys = ["memory-bound", "compute-bound", "latency-bound", "non-kernel"]
        cols = ["#c0392b", "#f39c12", "#27ae60", "#34495e"]
        bot = np.zeros(len(sub))
        for k, c in zip(keys, cols):
            v = np.array([p["split_pct_of_span"][k] for p in sub])
            ax.bar(xs, v, 0.72, bottom=bot, color=c,
                   label=k if j == 0 else None)
            bot += v
        ax.set_title(f"context {cl}", fontsize=10)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel("% of profiled GPU span")
            ax.legend(fontsize=7.4, loc="lower left")

        # row 1: category share of GPU busy time
        ax = axes[1, j]
        bot = np.zeros(len(sub))
        for c in cats:
            v = np.array([next((x["pct_gpu"] for x in p["categories"]
                                if x["category"] == c), 0.0) for p in sub])
            ax.bar(xs, v, 0.72, bottom=bot, color=CAT_COLORS[c],
                   label=c if j == 0 else None)
            bot += v
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("batch size"); ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel("% of GPU busy time")
            ax.legend(fontsize=7.4, loc="lower left")

    fig.suptitle("Where a decode step goes, Qwen2.5-7B-Instruct bf16 on "
                 "RTX A6000\ntop: profiled span by roofline verdict   "
                 "bottom: GPU time by kernel category", fontsize=11)
    fig.text(0.012, 0.004,
             "Top row denominator is the profiled GPU span, so the four buckets "
             "sum to 100%. The profiler inflates that span 1.8x at batch 1 and "
             "1.03x at batch 64, so the non-kernel bar is an upper bound; the "
             "CUDA-graph measurement (38.3% at batch 1) is authoritative.",
             fontsize=7.4, alpha=0.78)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def bandwidth(rl, sweep, out):
    """Achieved bandwidth per category, as a fraction of the measured peak.

    The roofline plot cannot show `gqa_expand` or `mem_move`: both do zero
    arithmetic, so they have no position on a FLOP/s axis -- yet `gqa_expand` is
    53.5% of GPU time at batch 64 / context 4096.  Bandwidth utilization is the
    axis on which every category, arithmetic or not, can be compared.
    """
    peak_bw = rl["summary"]["peak_hbm_gbs"]
    pts = sweep["points"]
    cats = ["proj_gemm", "lm_head", "attention", "gqa_expand", "norm_elem",
            "mem_move", "sampling"]
    cls = sorted({p["cache_len"] for p in pts})
    fig, axes = plt.subplots(1, len(cls), figsize=(3.9 * len(cls), 4.4),
                             sharey=True)
    if len(cls) == 1:
        axes = [axes]
    for ax, cl in zip(axes, cls):
        sub = sorted([p for p in pts if p["cache_len"] == cl],
                     key=lambda r: r["batch"])
        xs = [p["batch"] for p in sub]
        for c in cats:
            ys = []
            for p in sub:
                m = next((x for x in p["categories"]
                          if x["category"] == c), None)
                ys.append(100.0 * m["achieved_gbs"] / peak_bw
                          if m and m["gpu_ms"] > 0 else float("nan"))
            ax.plot(xs, ys, "o-", ms=4, lw=1.4,
                    color=CAT_COLORS[c], label=c)
        ax.axhline(100, color="k", ls="-", lw=1.6, alpha=0.8)
        ax.axhline(65, color="k", ls=":", lw=1, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(f"context {cl}", fontsize=10)
        ax.set_xlabel("batch size")
        ax.grid(True, alpha=0.16)
        ax.set_ylim(0, 118)
    axes[0].set_ylabel("achieved bandwidth (% of measured 711 GB/s)")
    axes[0].legend(fontsize=7.2, loc="center left", ncol=2)
    fig.suptitle("Bandwidth utilization per kernel category "
                 "(solid line = measured peak, dotted = 65% bound-verdict "
                 "threshold)", fontsize=10.5)
    fig.text(0.012, 0.012,
             "Byte counts are as-implemented, not compulsory: gqa_expand and "
             "attention sustain 80-96% of peak while moving up to 8x the "
             "traffic the arithmetic requires.", fontsize=7.4, alpha=0.78)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", default="results/01_roofline.json")
    ap.add_argument("--sweep", default="results/03_sweep.json")
    ap.add_argument("--outdir", default="plots")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rl = json.load(open(a.roofline))
    sw = json.load(open(a.sweep))
    roofline(rl, sw, os.path.join(a.outdir, "roofline.png"))
    breakdown(sw, os.path.join(a.outdir, "breakdown.png"))
    bandwidth(rl, sw, os.path.join(a.outdir, "bandwidth.png"))


if __name__ == "__main__":
    main()
