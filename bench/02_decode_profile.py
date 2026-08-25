"""Step 1.2/1.3 -- profile one decode step and place each category on the roofline.

A single (batch, cache_len) point, in detail.  The batch x context sweep is
03_sweep.py; both call the same `decode_point.run_point`, so there is exactly
one copy of the measurement and classification logic.
"""
import argparse
import json

import torch

from common import env_report, fix_seeds, save
from decode_point import run_point
import model_harness as mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--roofline", default="results/01_roofline.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default="results/traces")
    args = ap.parse_args()

    with open(args.roofline) as f:
        rl = json.load(f)["summary"]
    peak_bw, peak_tf = rl["peak_hbm_gbs"], rl["peak_bf16_tflops"]
    print(f"empirical roofline: {peak_bw:.0f} GB/s, {peak_tf:.1f} TFLOP/s bf16, "
          f"ridge {rl['ridge_point_flop_per_byte']:.1f} FLOP/byte\n")

    fix_seeds()
    model = mh.load()
    print(f"loaded {mh.MODEL}: "
          f"{sum(p.numel() for p in model.parameters())/1e9:.3f}B params, "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB resident\n")

    r = run_point(model, args.batch, args.cache_len, peak_bw, peak_tf,
                  args.trace_dir)

    print(f"=== decode step, batch={r['batch']} cache_len={r['cache_len']} ===")
    print(f"  wall (clean)      {r['wall_ms']:8.3f} ms  "
          f"({r['tokens_per_s']:.1f} tok/s)")
    print(f"  GPU busy          {r['gpu_busy_ms']:8.3f} ms")
    print(f"  GPU span (profiled){r['gpu_span_ms']:7.3f} ms  "
          f"= {r['span_over_wall']:.2f}x wall (profiler inflation)")
    print(f"  gap within span   {r['gap_ms']:8.3f} ms  ({r['gap_pct']:.1f}%)")
    print(f"  wall - busy       {r['wall_ms']-r['gpu_busy_ms']:8.3f} ms  "
          f"({100*(r['wall_ms']-r['gpu_busy_ms'])/r['wall_ms']:.1f}% of wall)")
    print(f"  host-side time    {r['cpu_time_ms']:8.3f} ms  "
          f"cpu_bound={r['cpu_bound']} (host/GPU = {r['cpu_over_gpu']:.2f})")
    print(f"  launches/step     {r['n_launches_per_step']:8.0f}")
    print(f"  analytic: {r['total_flops']/1e9:.1f} GFLOP, "
          f"{r['total_bytes']/2**30:.2f} GiB moved, "
          f"{r['compulsory_bytes']/2**30:.2f} GiB compulsory "
          f"({r['byte_amplification']:.2f}x amplification), "
          f"AI={r['total_arith_intensity']:.2f} FLOP/byte")
    print(f"  achieved: {r['achieved_gbs_total']:.0f} GB/s on moved bytes, "
          f"{r['compulsory_gbs_total']:.0f} GB/s on compulsory bytes "
          f"(peak {peak_bw:.0f})")
    if r["other_bucket_ms"] > 0:
        print(f"  UNCLASSIFIED: {r['other_bucket_ms']:.3f} ms")

    hdr = (f"\n  {'category':<11}{'ms':>8}{'%gpu':>7}{'kern':>6}{'AI':>9}"
           f"{'GB/s':>8}{'TFLOP/s':>9}{'%ceil':>7}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for c in r["categories"]:
        print(f"  {c['category']:<11}{c['gpu_ms']:8.3f}{c['pct_gpu']:7.1f}"
              f"{c['n_kernels']:6.0f}{c['arith_intensity']:9.2f}"
              f"{c['achieved_gbs']:8.0f}{c['achieved_tflops']:9.2f}"
              f"{100*c['frac_of_ceiling']:7.0f}  {c['verdict']}")

    print("\n  four-way split (% of profiled span):")
    for k, v in r["split_pct_of_span"].items():
        print(f"    {k:<15}{v:6.1f}%")

    out = args.out or f"results/02_decode_b{args.batch}_s{args.cache_len}.json"
    save({"env": env_report(), "roofline": rl,
          "model": mh.MODEL, "config": mh.config_summary(model.config),
          "point": r}, out)


if __name__ == "__main__":
    main()
