"""Step 1.4 -- sweep batch x cache length and watch the bound move.

The question this answers: of a decode step's wall time, what fraction is
explained by the memory ceiling, what fraction by the compute ceiling, what
fraction by neither (latency/occupancy), and what fraction is not on the GPU
at all -- and how do those four fractions move as batch and context grow.
"""
import argparse
import gc
import json

import torch

from common import env_report, fix_seeds, save
from decode_point import run_point
import model_harness as mh

BATCHES = [1, 2, 4, 8, 16, 32, 64]
CACHE_LENS = [128, 512, 2048, 4096]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", default="results/01_roofline.json")
    ap.add_argument("--out", default="results/03_sweep.json")
    ap.add_argument("--trace-dir", default="results/traces")
    ap.add_argument("--batches", type=int, nargs="*", default=BATCHES)
    ap.add_argument("--cache-lens", type=int, nargs="*", default=CACHE_LENS)
    args = ap.parse_args()

    with open(args.roofline) as f:
        rl = json.load(f)["summary"]
    peak_bw, peak_tf = rl["peak_hbm_gbs"], rl["peak_bf16_tflops"]
    ridge = rl["ridge_point_flop_per_byte"]
    print(f"empirical roofline: {peak_bw:.0f} GB/s, {peak_tf:.1f} TFLOP/s bf16, "
          f"ridge {ridge:.1f} FLOP/byte\n")

    fix_seeds()
    model = mh.load()
    cfgs = mh.config_summary(model.config)
    print(f"{mh.MODEL}: {sum(p.numel() for p in model.parameters())/1e9:.3f}B "
          f"params, KV {cfgs['kv_bytes_per_token']/1024:.1f} KiB/token\n")

    points = []
    for cl in args.cache_lens:
        for b in args.batches:
            try:
                r = run_point(model, b, cl, peak_bw, peak_tf, args.trace_dir)
            except torch.cuda.OutOfMemoryError:
                print(f"  b={b:3d} s={cl:5d}  OOM, skipping")
                gc.collect(); torch.cuda.empty_cache()
                continue
            points.append(r)
            s = r["split_pct_of_span"]
            print(f"  b={b:3d} s={cl:5d}  wall {r['wall_ms']:7.2f} ms  "
                  f"{r['tokens_per_s']:8.1f} tok/s  AI {r['total_arith_intensity']:6.2f}  "
                  f"mem {s['memory-bound']:5.1f}%  cmp {s['compute-bound']:5.1f}%  "
                  f"lat {s['latency-bound']:5.1f}%  nonkern {s['non-kernel']:5.1f}%  "
                  f"ampl {r['byte_amplification']:4.1f}x  "
                  f"launches {r['n_launches_per_step']:.0f}")
            gc.collect(); torch.cuda.empty_cache()

    # ---- tables -----------------------------------------------------------
    print("\n\n=== where decode time goes (% of wall) ===")
    print(f"{'batch':>6}{'ctx':>6}{'wall_ms':>9}{'tok/s':>9}{'AI':>7}"
          f"{'mem%':>7}{'cmp%':>7}{'lat%':>7}{'nonk%':>7}{'GB/s':>7}"
          f"{'ampl':>7}{'prof':>6}")
    for r in points:
        s = r["split_pct_of_span"]
        print(f"{r['batch']:>6}{r['cache_len']:>6}{r['wall_ms']:>9.2f}"
              f"{r['tokens_per_s']:>9.1f}{r['total_arith_intensity']:>7.2f}"
              f"{s['memory-bound']:>7.1f}{s['compute-bound']:>7.1f}"
              f"{s['latency-bound']:>7.1f}{s['non-kernel']:>7.1f}"
              f"{r['achieved_gbs_total']:>7.0f}"
              f"{r['byte_amplification']:>6.1f}x{r['span_over_wall']:>6.2f}")

    cats = sorted({c["category"] for r in points for c in r["categories"]})
    for cl in args.cache_lens:
        sub = [r for r in points if r["cache_len"] == cl]
        if not sub:
            continue
        print(f"\n=== category share of GPU time, ctx={cl} (% of GPU busy) ===")
        print(f"{'batch':>6}" + "".join(f"{c:>12}" for c in cats)
              + f"{'gpu_ms':>9}")
        for r in sub:
            m = {c["category"]: c["pct_gpu"] for c in r["categories"]}
            print(f"{r['batch']:>6}"
                  + "".join(f"{m.get(c, 0.0):>12.1f}" for c in cats)
                  + f"{r['gpu_busy_ms']:>9.2f}")

    print("\n=== attention vs weights: which traffic dominates ===")
    print(f"{'batch':>6}{'ctx':>6}{'weight_GiB':>12}{'kv_GiB':>9}"
          f"{'kv_share%':>11}{'attn%gpu':>10}{'gqa%gpu':>9}{'ampl':>7}")
    for r in points:
        m = {c["category"]: c for c in r["categories"]}
        wb = m["proj_gemm"]["bytes"] + m["lm_head"]["bytes"]
        kv = m["attention"]["bytes"]
        print(f"{r['batch']:>6}{r['cache_len']:>6}{wb/2**30:>12.2f}"
              f"{kv/2**30:>9.2f}{100*kv/(wb+kv):>11.1f}"
              f"{m['attention']['pct_gpu']:>10.1f}"
              f"{m.get('gqa_expand',{}).get('pct_gpu',0.0):>9.1f}"
              f"{r['byte_amplification']:>6.1f}x")

    save({"env": env_report(), "roofline": rl, "model": mh.MODEL,
          "config": cfgs, "points": points}, args.out)


if __name__ == "__main__":
    main()
