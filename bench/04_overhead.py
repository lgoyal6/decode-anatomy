"""Step 1.5 -- measure the non-kernel slice directly.

03_sweep infers non-kernel time as (wall - sum of kernel durations).  That is
the right total but it does not say what the time *is*.  This script attributes
it, four ways:

  1. floor      -- cost of launching a do-nothing kernel, host side and device
                   side.  This is the per-launch price of admission.
  2. dispatch   -- cost of one trivial aten op (Python + dispatcher + launch)
                   against the raw launch floor.  The difference is framework.
  3. sync       -- how many host synchronizations and device-to-host copies a
                   decode step actually performs, counted from the trace, and
                   what an empty synchronize costs.
  4. removable  -- capture the whole decode step in a CUDA graph and replay it.
                   Eager minus graph is the overhead that is *actually
                   removable* without touching a single kernel, which is the
                   number that matters for deciding whether to optimize kernels
                   or optimize issue.
"""
import argparse
import json
import time

import torch
from torch.profiler import ProfilerActivity, profile

from common import env_report, fix_seeds, save, time_loop
import model_harness as mh
from trace_parse import Trace


# ---------------------------------------------------------------------------
# 1 + 2: launch floor and dispatch cost
# ---------------------------------------------------------------------------
def launch_floor(n=2000):
    """Host-side and device-side cost of a minimal kernel launch."""
    a = torch.zeros(1, device="cuda")

    # Host-side issue cost: enqueue without ever waiting on the GPU.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        a.add_(1.0)
    host_issue_s = (time.perf_counter() - t0) / n
    torch.cuda.synchronize()

    # Device-side cost of the same stream of launches: with a 1-element kernel
    # the kernel body is free, so this is launch latency plus inter-kernel gap.
    start, end = (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True))
    start.record()
    for _ in range(n):
        a.add_(1.0)
    end.record()
    end.synchronize()
    device_s = start.elapsed_time(end) / 1e3 / n

    # An empty synchronize, for reference: the price of any host/device
    # rendezvous even when there is nothing to wait for.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        torch.cuda.synchronize()
    sync_s = (time.perf_counter() - t0) / n

    return {
        "host_issue_us": host_issue_s * 1e6,
        "device_per_kernel_us": device_s * 1e6,
        "empty_synchronize_us": sync_s * 1e6,
        "queue_bound": host_issue_s > device_s,
    }


# ---------------------------------------------------------------------------
# 3: synchronizations inside a real decode step
# ---------------------------------------------------------------------------
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
              "cudaEventSynchronize", "cudaStreamWaitEvent",
              "cudaMemcpyAsync", "cudaMemcpy", "cudaHostAlloc",
              "cudaStreamQuery", "cudaEventQuery")


def sync_census(trace_path, n_steps):
    """Count host/device rendezvous points per decode step, from the trace."""
    with open(trace_path) as f:
        evs = json.load(f)["traceEvents"]
    counts, dur = {}, {}
    d2h_bytes = 0
    for e in evs:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        if e.get("cat") == "cuda_runtime" and name in SYNC_NAMES:
            counts[name] = counts.get(name, 0) + 1
            dur[name] = dur.get(name, 0.0) + e.get("dur", 0)
        if e.get("cat") == "gpu_memcpy" and "DtoH" in name:
            d2h_bytes += (e.get("args") or {}).get("bytes", 0) or 0
    return {
        "per_step": {k: v / n_steps for k, v in sorted(counts.items())},
        "us_per_step": {k: v / n_steps for k, v in sorted(dur.items())},
        "total_sync_us_per_step": sum(dur.values()) / n_steps,
        "d2h_bytes_per_step": d2h_bytes / n_steps,
    }


# ---------------------------------------------------------------------------
# 4: how much of the overhead is removable
# ---------------------------------------------------------------------------
def cuda_graph_step(model, ids, cache, pos, reset, reps=30):
    """Capture the decode step in a CUDA graph; return eager vs replay times.

    A graph replay issues the whole step as one command, so it pays the launch
    and dispatch cost exactly once instead of ~1500 times.  Eager minus replay
    is therefore the removable overhead, measured rather than modelled.
    """
    def step():
        reset()
        return mh.decode_step(model, ids, cache, pos)

    eager = time_loop(step, inner=reps, warmup=10, reps=5)
    t_reset = time_loop(reset, inner=reps, warmup=10, reps=5)
    eager_s = eager["min_s"] - t_reset["min_s"]

    # Capture requires a warmed-up side stream so that lazy allocations,
    # cuBLAS workspaces and autotuning all happen before capture begins.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    reset()
    try:
        with torch.cuda.graph(g):
            out = mh.decode_step(model, ids, cache, pos)
    except Exception as e:
        return {"captured": False, "error": f"{type(e).__name__}: {e}",
                "eager_ms": eager_s * 1e3}

    def replay():
        reset()
        g.replay()

    rep = time_loop(replay, inner=reps, warmup=10, reps=5)
    replay_s = rep["min_s"] - t_reset["min_s"]
    return {
        "captured": True,
        "eager_ms": eager_s * 1e3,
        "graph_ms": replay_s * 1e3,
        "removable_ms": (eager_s - replay_s) * 1e3,
        "removable_pct": 100.0 * (eager_s - replay_s) / eager_s,
        "speedup": eager_s / replay_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--out", default="results/04_overhead.json")
    args = ap.parse_args()

    fix_seeds()
    print("=== 1. launch floor (no model) ===")
    floor = launch_floor()
    for k, v in floor.items():
        print(f"  {k}: {v}")

    model = mh.load()
    results = []
    for b in args.batches:
        print(f"\n=== batch {b}, ctx {args.cache_len} ===")
        ids, cache, pos, reset = mh.make_decode_state(model, b, args.cache_len)

        # sync census from a profiled run
        handles = mh.annotate(model)
        path = f"results/traces/overhead_b{b}_s{args.cache_len}.json"
        n_steps = 3
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n_steps):
                reset()
                mh.decode_step(model, ids, cache, pos)
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)
        mh.remove(handles)
        census = sync_census(path, n_steps)
        n_kernels = len(Trace(path).kernels()) / n_steps

        print(f"  kernels/step: {n_kernels:.0f}")
        print(f"  host/device rendezvous per step: {census['per_step']}")
        print(f"  time in those calls: {census['total_sync_us_per_step']:.1f} us/step")
        print(f"  D2H bytes/step: {census['d2h_bytes_per_step']:.0f}")

        cg = cuda_graph_step(model, ids, cache, pos, reset)
        if cg["captured"]:
            print(f"  eager      {cg['eager_ms']:8.3f} ms")
            print(f"  cuda graph {cg['graph_ms']:8.3f} ms")
            print(f"  removable  {cg['removable_ms']:8.3f} ms "
                  f"({cg['removable_pct']:.1f}% of eager, "
                  f"{cg['speedup']:.2f}x speedup)")
            print(f"  implied per-launch overhead: "
                  f"{cg['removable_ms']*1e3/n_kernels:.2f} us over "
                  f"{n_kernels:.0f} launches")
        else:
            print(f"  CUDA graph capture FAILED: {cg['error']}")

        results.append({"batch": b, "cache_len": args.cache_len,
                        "n_kernels_per_step": n_kernels,
                        "sync_census": census, "cuda_graph": cg})
        del cache
        torch.cuda.empty_cache()

    save({"env": env_report(), "launch_floor": floor, "points": results},
         args.out)


if __name__ == "__main__":
    main()
