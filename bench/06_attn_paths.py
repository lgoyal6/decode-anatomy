"""Step 1 addendum -- cost the attention path choice at the primitive level.

05_gqa.py tried to flip the branch from outside by passing
`attention_mask=None` to `forward`.  That produced bit-identical logits and
identical timings: `Qwen2Model.forward` builds its own mask whenever the cache
is a padded StaticCache, so the caller cannot turn mask construction off.  That
is a real negative result and it means the counterfactual has to be measured on
`scaled_dot_product_attention` directly.

Four ways to compute the same decode attention at GQA shapes:

  A  repeat_kv + additive mask   -- what transformers actually runs
  B  enable_gqa=True, no mask    -- broadcast inside the kernel
  C  repeat_kv, no mask          -- isolates the mask from the expansion
  D  enable_gqa=True + mask      -- the combination the transformers guard
                                    avoids, included to show why

All four are checked against A for numerical agreement before any timing is
quoted, and the dispatched backend is recorded, since the whole reason D is
guarded against is that a mask plus enable_gqa falls back to the math kernel.
"""
import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from common import env_report, fix_seeds, save, time_loop

# Qwen2.5-7B-Instruct attention geometry
NH, NKV, HD = 28, 4, 128
N_REP = NH // NKV
E = 2  # bf16


def repeat_kv(x, n):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def backend_of(fn):
    """Which SDPA kernel actually ran, read out of a one-shot trace."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in p.key_averages():
        n = e.key
        if any(t in n.lower() for t in ("fmha", "flash", "attention", "cutlass",
                                        "softmax", "bmm", "gemm")):
            if getattr(e, "self_device_time_total", 0) > 0:
                names.append((n, e.self_device_time_total))
    names.sort(key=lambda x: -x[1])
    return [n for n, _ in names[:3]]


def run(batch, ctx, reps=20):
    q = torch.randn(batch, NH, 1, HD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, NKV, ctx, HD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, NKV, ctx, HD, device="cuda", dtype=torch.bfloat16)
    # A fully-permissive additive mask: same shape and dtype transformers
    # builds for a full StaticCache, so the comparison isolates the *presence*
    # of a mask rather than what it masks out.
    mask = torch.zeros(batch, 1, 1, ctx, device="cuda", dtype=torch.bfloat16)
    scale = HD ** -0.5

    variants = {
        "A_repeat_kv+mask": lambda: F.scaled_dot_product_attention(
            q, repeat_kv(k, N_REP), repeat_kv(v, N_REP),
            attn_mask=mask, scale=scale),
        "B_enable_gqa": lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, scale=scale, enable_gqa=True),
        "C_repeat_kv": lambda: F.scaled_dot_product_attention(
            q, repeat_kv(k, N_REP), repeat_kv(v, N_REP),
            attn_mask=None, scale=scale),
        "D_enable_gqa+mask": lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=True),
    }

    ref = variants["A_repeat_kv+mask"]().float()
    out = {}
    # Compulsory KV traffic vs what each variant moves.
    kv_compulsory = 2 * batch * NKV * HD * ctx * E
    kv_expanded = 2 * batch * NH * HD * ctx * E
    for name, fn in variants.items():
        try:
            got = fn().float()
            err = (got - ref).abs().max().item()
            t = time_loop(fn, inner=reps, warmup=10, reps=5)["min_s"]
            moved = (kv_compulsory + 2 * kv_expanded
                     if "repeat_kv" in name else kv_compulsory)
            out[name] = {
                "ms": t * 1e3,
                "max_abs_diff_vs_A": err,
                "bytes": moved,
                "gbs": moved / t / 1e9,
                "backend": backend_of(fn),
            }
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    del q, k, v, mask
    torch.cuda.empty_cache()
    return {"batch": batch, "ctx": ctx,
            "kv_compulsory_gib": kv_compulsory / 2**30,
            "kv_expanded_gib": kv_expanded / 2**30,
            "variants": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/06_attn_paths.json")
    a = ap.parse_args()
    fix_seeds()
    print(f"decode attention, nh={NH} nkv={NKV} n_rep={N_REP} hd={HD}, bf16\n")

    pts = [(1, 512), (1, 4096), (8, 2048), (8, 4096),
           (32, 2048), (64, 2048), (64, 4096)]
    rows = []
    for b, c in pts:
        try:
            r = run(b, c)
        except torch.cuda.OutOfMemoryError:
            print(f"  b={b} ctx={c}: OOM"); torch.cuda.empty_cache(); continue
        rows.append(r)
        vs = r["variants"]
        A = vs["A_repeat_kv+mask"]; B = vs["B_enable_gqa"]
        print(f"  b={b:3d} ctx={c:5d}  A {A.get('ms',0):7.3f} ms  "
              f"B {B.get('ms',0):7.3f} ms  "
              f"B is {A.get('ms',1)/B.get('ms',1):5.2f}x faster   "
              f"err(B vs A) {B.get('max_abs_diff_vs_A', float('nan')):.2e}")

    names = ["A_repeat_kv+mask", "B_enable_gqa", "C_repeat_kv",
             "D_enable_gqa+mask"]
    print("\n=== per-layer decode attention, ms ===")
    print(f"{'batch':>6}{'ctx':>6}" + "".join(f"{n.split('_',1)[0]:>10}"
                                              for n in names)
          + f"{'A/B':>7}{'kv_GiB':>8}{'expanded':>10}")
    for r in rows:
        vs = r["variants"]
        line = f"{r['batch']:>6}{r['ctx']:>6}"
        for n in names:
            line += f"{vs[n].get('ms', float('nan')):>10.3f}"
        line += (f"{vs['A_repeat_kv+mask']['ms']/vs['B_enable_gqa']['ms']:>7.2f}"
                 f"{r['kv_compulsory_gib']:>8.3f}{r['kv_expanded_gib']:>10.3f}")
        print(line)

    print("\n=== achieved bandwidth on the bytes each variant moves ===")
    print(f"{'batch':>6}{'ctx':>6}" + "".join(f"{n.split('_',1)[0]+' GB/s':>13}"
                                              for n in names))
    for r in rows:
        vs = r["variants"]
        print(f"{r['batch']:>6}{r['ctx']:>6}"
              + "".join(f"{vs[n].get('gbs', float('nan')):>13.0f}"
                        for n in names))

    print("\n=== dispatched backend (largest device kernel) ===")
    last = rows[-1]
    for n in names:
        print(f"  {n:<20} {last['variants'][n].get('backend')}")

    print("\n=== numerical agreement vs variant A ===")
    for n in names[1:]:
        worst = max(r["variants"][n].get("max_abs_diff_vs_A", 0) for r in rows)
        print(f"  {n:<20} max abs diff {worst:.3e}")

    save({"env": env_report(), "geometry": {"nh": NH, "nkv": NKV, "hd": HD},
          "points": rows}, a.out)


if __name__ == "__main__":
    main()
