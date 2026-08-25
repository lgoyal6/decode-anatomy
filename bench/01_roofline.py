"""Step 1.1 -- empirical roofline for this card.

Two axes, measured before anything is profiled:

  * achieved memory bandwidth  -- device-to-device copy, scale, read, and a
    STREAM-style triad, swept over buffer sizes from inside-L2 to 4 GiB.
  * achieved compute throughput -- large square GEMMs in bf16 / fp16 / tf32 /
    fp32, swept over N.

Both are reported against the vendor spec sheet.  The gap is part of the
result, so we also log SM clock and board power at each point: a compute
number taken while the card is power-limited is a different number than one
taken at base clock, and the roofline needs to say which it is.
"""
import argparse
import subprocess

import torch

from common import SPEC, env_report, fix_seeds, save, time_loop

MiB = 2 ** 20
GiB = 2 ** 30


def clocks():
    """Instantaneous SM clock (MHz), board power (W), and throttle reasons."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,"
             "clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        sm, pw, temp, reasons = [x.strip() for x in out.split(",")]
        return {"sm_mhz": float(sm), "power_w": float(pw),
                "temp_c": float(temp), "throttle": reasons}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Memory bandwidth
# ---------------------------------------------------------------------------
def bandwidth_sweep():
    """Bytes moved per second for four access patterns.

    `bytes_moved` counts compulsory traffic only (what the kernel must read
    plus what it must write), which is the standard STREAM convention.  Sizes
    below the 6 MB L2 are kept in the sweep on purpose: they measure L2, not
    HBM, and the step between the two is a useful sanity check that the large
    sizes really are missing cache.
    """
    sizes = [1 * MiB, 4 * MiB, 8 * MiB, 32 * MiB, 128 * MiB,
             256 * MiB, 512 * MiB, 1 * GiB, 2 * GiB, 4 * GiB]
    rows = []
    for nbytes in sizes:
        n = nbytes // 4  # fp32 elements
        try:
            a = torch.empty(n, dtype=torch.float32, device="cuda")
            b = torch.randn(n, dtype=torch.float32, device="cuda")
            c = torch.randn(n, dtype=torch.float32, device="cuda")
        except torch.cuda.OutOfMemoryError:
            print(f"  {nbytes/GiB:.2f} GiB  -- OOM, stopping sweep")
            break

        # (name, callable, compulsory bytes touched)
        kernels = [
            ("copy",  lambda: a.copy_(b),                    2 * nbytes),
            ("scale", lambda: torch.mul(b, 2.0, out=a),      2 * nbytes),
            ("triad", lambda: torch.add(b, c, alpha=2.0, out=a), 3 * nbytes),
            ("read",  lambda: torch.sum(b),                  1 * nbytes),
        ]
        # Big buffers make each call expensive, so fewer inner iterations are
        # enough to amortize launch overhead.
        inner = 20 if nbytes <= 128 * MiB else 5
        for name, fn, moved in kernels:
            t = time_loop(fn, inner=inner, warmup=3, reps=5)
            gbs = moved / t["min_s"] / 1e9
            rows.append({
                "kernel": name,
                "buf_bytes": nbytes,
                "buf_mib": nbytes / MiB,
                "bytes_moved": moved,
                "time_s": t["min_s"],
                "spread_pct": round(t["spread_pct"], 2),
                "gbs": gbs,
                "pct_of_spec": 100.0 * gbs / SPEC["mem_bw_gbs"],
                "in_l2": nbytes <= SPEC["l2_mb"] * 1e6,
                **clocks(),
            })
            print(f"  {name:6s} buf={nbytes/MiB:8.1f} MiB  "
                  f"{gbs:8.1f} GB/s  ({100*gbs/SPEC['mem_bw_gbs']:5.1f}% of spec)")
        del a, b, c
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Compute throughput
# ---------------------------------------------------------------------------
def gemm_sweep():
    """Square GEMM throughput per dtype.

    fp16 is included next to bf16 because GA102 is documented to run tensor
    core matmul at a different rate for fp16-with-fp16-accumulate than for
    fp32-accumulate, and torch always accumulates in fp32.  If the achieved
    bf16 number lands near half the datasheet figure, that is the reason, and
    it changes the roofline ridge point by 2x -- worth resolving rather than
    assuming.
    """
    Ns = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    configs = [
        ("bf16", torch.bfloat16, False),
        ("fp16", torch.float16, False),
        ("tf32", torch.float32, True),
        ("fp32", torch.float32, False),
    ]
    rows = []
    for name, dtype, tf32 in configs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for N in Ns:
            # fp32 at large N is slow and not the point; cap it.
            if dtype == torch.float32 and N > 8192:
                continue
            try:
                x = torch.randn(N, N, dtype=dtype, device="cuda")
                w = torch.randn(N, N, dtype=dtype, device="cuda")
                out = torch.empty(N, N, dtype=dtype, device="cuda")
            except torch.cuda.OutOfMemoryError:
                print(f"  {name} N={N} -- OOM, skipping")
                continue
            flops = 2.0 * N ** 3
            inner = 20 if N <= 4096 else 5
            t = time_loop(lambda: torch.mm(x, w, out=out),
                          inner=inner, warmup=5, reps=5)
            tflops = flops / t["min_s"] / 1e12
            # Arithmetic intensity of a square GEMM: 2N^3 flops over the
            # 3N^2 elements that must cross HBM at least once.
            ai = flops / (3 * N * N * x.element_size())
            rows.append({
                "dtype": name, "N": N, "flops": flops,
                "time_s": t["min_s"],
                "spread_pct": round(t["spread_pct"], 2),
                "tflops": tflops,
                "arith_intensity": ai,
                "pct_of_spec": (100.0 * tflops / SPEC["bf16_tensor_tflops"]
                                if name in ("bf16", "fp16") else
                                100.0 * tflops / (SPEC["tf32_tensor_tflops"]
                                                  if name == "tf32"
                                                  else SPEC["fp32_tflops"])),
                **clocks(),
            })
            print(f"  {name:5s} N={N:6d}  {tflops:8.2f} TFLOP/s  "
                  f"AI={ai:8.1f}  sm={rows[-1].get('sm_mhz','?')}MHz "
                  f"pwr={rows[-1].get('power_w','?')}W")
            del x, w, out
            torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/01_roofline.json")
    args = ap.parse_args()

    fix_seeds()
    env = env_report()
    print("environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    print("\nmemory bandwidth sweep:")
    bw = bandwidth_sweep()
    print("\nGEMM throughput sweep:")
    gemm = gemm_sweep()

    peak_bw = max(r["gbs"] for r in bw if not r["in_l2"])
    peak_bf16 = max(r["tflops"] for r in gemm if r["dtype"] == "bf16")
    ridge = peak_bf16 * 1e12 / (peak_bw * 1e9)

    summary = {
        "peak_hbm_gbs": peak_bw,
        "peak_hbm_pct_of_spec": 100.0 * peak_bw / SPEC["mem_bw_gbs"],
        "peak_bf16_tflops": peak_bf16,
        "peak_bf16_pct_of_spec": 100.0 * peak_bf16 / SPEC["bf16_tensor_tflops"],
        "ridge_point_flop_per_byte": ridge,
    }
    print("\n--- empirical roofline ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")
    print(f"  interpretation: a kernel needs > {ridge:.1f} FLOP per byte of "
          f"HBM traffic to be compute-bound on this card.")

    save({"spec": SPEC, "env": env, "summary": summary,
          "bandwidth": bw, "gemm": gemm}, args.out)


if __name__ == "__main__":
    main()
