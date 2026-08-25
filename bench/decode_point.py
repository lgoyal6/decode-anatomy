"""Measure one (batch, cache_len) decode point. Shared by 02 and 03."""
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from common import time_loop
import model_harness as mh
from trace_parse import Trace, summarize


def bound_verdict(ai, achieved_tflops, peak_bw_gbs, peak_tflops, tol=0.65,
                  achieved_gbs=None, does_arithmetic=True):
    """Classify a category against the empirical roofline.

    The ceiling at intensity `ai` is min(peak_compute, bandwidth * ai).  A
    category counts as memory- or compute-bound only if it reaches `tol` of
    that ceiling.  Otherwise neither ceiling explains its runtime and it is
    latency-bound -- too little work per launch to fill the card.  That third
    verdict is the whole reason for the tolerance: without it, every
    low-intensity kernel gets filed as "memory-bound" regardless of whether it
    moves any bytes at a respectable rate.

    Pure data-movement categories (the GQA expansion, the KV cache write) do
    *zero* arithmetic.  Comparing their FLOP/s against a ceiling therefore
    always yields zero and files them as latency-bound no matter how fast they
    move bytes -- which mislabelled `gqa_expand`, 53% of the GPU time at batch
    64 / context 4096, even though it sustains ~575 GB/s.  Those are judged on
    achieved bandwidth against peak bandwidth instead, and `frac` then means
    "fraction of the bandwidth ceiling" rather than "fraction of the roofline".
    """
    if not does_arithmetic:
        frac = (achieved_gbs / peak_bw_gbs) if achieved_gbs else 0.0
        return ("memory-bound" if frac >= tol else "latency-bound"), frac, 0.0

    mem_ceiling = peak_bw_gbs * 1e9 * ai / 1e12
    ceiling = min(peak_tflops, mem_ceiling)
    frac = achieved_tflops / ceiling if ceiling else 0.0
    ridge = peak_tflops * 1e12 / (peak_bw_gbs * 1e9)
    if frac < tol:
        return "latency-bound", frac, ceiling
    return ("memory-bound" if ai < ridge else "compute-bound"), frac, ceiling


def _reps_for(batch, cache_len):
    """Fewer iterations where a step is expensive, to bound total runtime."""
    if batch >= 32 and cache_len >= 2048:
        return 8
    if batch >= 16 or cache_len >= 2048:
        return 15
    return 30


def run_point(model, batch, cache_len, peak_bw, peak_tf,
              trace_dir="results/traces", n_prof_steps=3):
    cfg = model.config
    ids, cache, pos, reset = mh.make_decode_state(model, batch, cache_len)
    reps = _reps_for(batch, cache_len)

    def step():
        reset()          # rewind cache write pointer; see make_decode_state
        return mh.decode_step(model, ids, cache, pos)

    # --- wall time, profiler detached (the profiler itself perturbs) -------
    t = time_loop(step, inner=reps, warmup=10, reps=5)
    t_reset = time_loop(reset, inner=reps, warmup=10, reps=5)
    wall_s = t["min_s"] - t_reset["min_s"]

    # --- host-side time: Python + dispatch + launch, no trailing sync ------
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    cpu_s = (time.perf_counter() - t0) / reps
    torch.cuda.synchronize()

    # --- profile ----------------------------------------------------------
    os.makedirs(trace_dir, exist_ok=True)
    handles = mh.annotate(model)
    path = os.path.join(trace_dir, f"trace_b{batch}_s{cache_len}.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(n_prof_steps):
            step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    mh.remove(handles)

    tr = Trace(path)
    ks = tr.kernels()
    rows, total_us = summarize(ks)
    gpu_busy_s = total_us / 1e6 / n_prof_steps
    # Span comes from the same trace as the busy time, so span - busy is a real
    # non-negative idle time.  Comparing profiled kernel sums against a clean
    # wall from a different run let the difference go negative once profiling
    # inflated the kernels.
    gpu_span_s = tr.gpu_span_us() / 1e6 / n_prof_steps
    n_launches = len(ks) / n_prof_steps

    # --- analytic cost + roofline placement -------------------------------
    cost = mh.analytic_cost(cfg, batch, cache_len)
    by_cat = {r["category"]: r for r in rows}
    cats = []
    for cat, c in cost.items():
        if cat == "TOTAL":
            continue
        meas = by_cat.get(cat)
        gpu_s = (meas["gpu_us"] / 1e6 / n_prof_steps) if meas else 0.0
        tflops = (c["flops"] / gpu_s / 1e12) if gpu_s > 0 else 0.0
        gbs = (c["bytes"] / gpu_s / 1e9) if gpu_s > 0 else 0.0
        verdict, frac, ceil = bound_verdict(
            c["arith_intensity"], tflops, peak_bw, peak_tf,
            achieved_gbs=gbs, does_arithmetic=c["flops"] > 0)
        cats.append({
            "category": cat,
            "gpu_ms": gpu_s * 1e3,
            "pct_gpu": 100.0 * gpu_s / gpu_busy_s if gpu_busy_s else 0.0,
            "n_kernels": (meas["n_kernels"] / n_prof_steps) if meas else 0.0,
            "flops": c["flops"], "bytes": c["bytes"],
            "arith_intensity": c["arith_intensity"],
            "achieved_tflops": tflops, "achieved_gbs": gbs,
            "pct_of_hbm_peak": 100.0 * gbs / peak_bw,
            "roofline_ceiling_tflops": ceil,
            "frac_of_ceiling": frac,
            "verdict": verdict,
        })
    cats.sort(key=lambda r: -r["gpu_ms"])

    # --- the headline split: how much of the step is explained by what -----
    # Denominator is the profiled span, so the four buckets sum to 100% by
    # construction.  `span_over_wall` below reports how much the profiler
    # inflated that span relative to the clean wall time.
    split = {"memory-bound": 0.0, "compute-bound": 0.0, "latency-bound": 0.0}
    for c in cats:
        split[c["verdict"]] += c["gpu_ms"]
    wall_ms = wall_s * 1e3
    span_ms = gpu_span_s * 1e3
    split["non-kernel"] = span_ms - gpu_busy_s * 1e3
    frac = {k: 100.0 * v / span_ms for k, v in split.items()} if span_ms else {}

    other = by_cat.get("other")
    return {
        "batch": batch, "cache_len": cache_len, "reps": reps,
        "wall_ms": wall_ms,
        "gpu_busy_ms": gpu_busy_s * 1e3,
        "gpu_span_ms": span_ms,
        "span_over_wall": span_ms / wall_ms if wall_ms else 0.0,
        "gap_ms": span_ms - gpu_busy_s * 1e3,
        "gap_pct": 100.0 * (span_ms - gpu_busy_s * 1e3) / span_ms if span_ms else 0.0,
        "cpu_time_ms": cpu_s * 1e3,
        "cpu_bound": cpu_s > gpu_busy_s,
        "cpu_over_gpu": cpu_s / gpu_busy_s if gpu_busy_s else 0.0,
        "n_launches_per_step": n_launches,
        "us_per_launch_gap": ((span_ms - gpu_busy_s * 1e3) * 1e3 / n_launches
                              if n_launches else 0.0),
        "tokens_per_s": batch / wall_s,
        "total_flops": cost["TOTAL"]["flops"],
        "total_bytes": cost["TOTAL"]["bytes"],
        "total_arith_intensity": cost["TOTAL"]["arith_intensity"],
        "compulsory_bytes": cost["TOTAL"]["compulsory_bytes"],
        "byte_amplification": cost["TOTAL"]["byte_amplification"],
        "achieved_gbs_total": cost["TOTAL"]["bytes"] / wall_s / 1e9,
        "compulsory_gbs_total": cost["TOTAL"]["compulsory_bytes"] / wall_s / 1e9,
        "achieved_tflops_total": cost["TOTAL"]["flops"] / wall_s / 1e12,
        "split_ms": split,
        "split_pct_of_span": frac,
        "categories": cats,
        "other_bucket_ms": (other["gpu_us"] / 1e6 / n_prof_steps) if other else 0.0,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "trace": path,
    }


def free(cache):
    del cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
