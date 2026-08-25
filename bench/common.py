"""Shared timing + device utilities.

All timings use CUDA events around a K-iteration inner loop so that per-call
Python and launch overhead amortizes out of the reported number.  Where the
overhead itself is the measurement (04_overhead.py) we say so explicitly.
"""
import json
import os
import platform
import subprocess
import statistics
import sys

import torch

# ---------------------------------------------------------------------------
# Vendor spec sheet numbers for the RTX A6000 (GA102), used only as the
# denominator in "achieved / spec" columns.  Source: NVIDIA RTX A6000
# datasheet, "Peak" figures, non-sparse.
# ---------------------------------------------------------------------------
SPEC = {
    "name": "NVIDIA RTX A6000",
    "arch": "GA102 / sm_86",
    "mem_bw_gbs": 768.0,          # 384-bit GDDR6 @ 16 Gbps
    "l2_mb": 6.0,
    "sm_count": 84,
    "fp32_tflops": 38.7,
    "tf32_tensor_tflops": 77.4,
    "bf16_tensor_tflops": 154.8,  # dense; 309.7 with structured sparsity
    "int8_tensor_tops": 309.7,
}


def env_report():
    """Everything needed to reproduce a number, collected in one dict."""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
             "clocks.max.sm,clocks.max.memory,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        smi = f"unavailable: {e}"
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_smi": smi,
        "gpu_name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_mem_gib": round(props.total_memory / 2**30, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "seed": SEED,
    }


SEED = 0


def fix_seeds(seed=SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def time_loop(fn, inner=20, warmup=5, reps=7):
    """Median seconds per call of `fn`.

    `fn` is called `inner` times between one pair of CUDA events, so launch
    overhead is amortized.  We take the median of `reps` such measurements and
    also return the min, since the min is the cleanest estimate of achievable
    throughput and the median-vs-min gap flags a noisy machine.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3 / inner)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "spread_pct": 100.0 * (max(samples) - min(samples)) / statistics.median(samples),
    }


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\nwrote {path}")
