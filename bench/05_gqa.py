"""Step 1 addendum -- what the GQA materialization actually costs.

The sweep showed `aten::copy_` inside the attention module dominating the step
at long context.  That copy is transformers' `repeat_kv`: it expands the 4 KV
heads out to 28 query heads before calling sdpa, because
`use_gqa_in_sdpa()` requires `attention_mask is None`, and StaticCache always
supplies a mask to hide its padding.  sdpa's own `enable_gqa=True` would do the
broadcast inside the kernel for free.

This measures the difference on identical shapes.  Passing no mask is only
*correct* when every slot in the cache is real, which is exactly the state
`make_decode_state` builds (cache_len == max_cache_len, fully populated), so
both paths must produce the same logits -- and that is asserted here before any
timing is reported.  Without that check a speedup would be meaningless.
"""
import argparse

import torch

from common import env_report, fix_seeds, save, time_loop
import model_harness as mh


@torch.no_grad()
def compare(model, batch, cache_len, reps=15):
    ids, cache, pos, reset = mh.make_decode_state(model, batch, cache_len)

    def masked():
        reset()
        return model(input_ids=ids, past_key_values=cache, use_cache=True,
                     cache_position=pos).logits

    def unmasked():
        # attention_mask=None lets sdpa_attention_forward take the
        # enable_gqa=True branch instead of calling repeat_kv.
        reset()
        return model(input_ids=ids, past_key_values=cache, use_cache=True,
                     cache_position=pos, attention_mask=None).logits

    # --- correctness first ------------------------------------------------
    a = masked().float()
    b = unmasked().float()
    max_abs = (a - b).abs().max().item()
    rel = max_abs / a.abs().max().item()
    same_argmax = bool((a.argmax(-1) == b.argmax(-1)).all().item())

    # --- then timing ------------------------------------------------------
    t_reset = time_loop(reset, inner=reps, warmup=10, reps=5)["min_s"]
    t_m = time_loop(masked, inner=reps, warmup=10, reps=5)["min_s"] - t_reset
    t_u = time_loop(unmasked, inner=reps, warmup=10, reps=5)["min_s"] - t_reset

    cost_m = mh.analytic_cost(model.config, batch, cache_len, gqa_materialized=True)
    cost_u = mh.analytic_cost(model.config, batch, cache_len, gqa_materialized=False)

    del cache
    torch.cuda.empty_cache()
    return {
        "batch": batch, "cache_len": cache_len,
        "masked_ms": t_m * 1e3, "gqa_ms": t_u * 1e3,
        "speedup": t_m / t_u, "saved_ms": (t_m - t_u) * 1e3,
        "masked_tok_s": batch / t_m, "gqa_tok_s": batch / t_u,
        "bytes_masked_gib": cost_m["TOTAL"]["bytes"] / 2**30,
        "bytes_gqa_gib": cost_u["TOTAL"]["bytes"] / 2**30,
        "gbs_masked": cost_m["TOTAL"]["bytes"] / t_m / 1e9,
        "gbs_gqa": cost_u["TOTAL"]["bytes"] / t_u / 1e9,
        "max_abs_diff": max_abs, "max_rel_diff": rel,
        "argmax_identical": same_argmax,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/05_gqa.json")
    ap.add_argument("--points", default="1:512,1:4096,8:2048,8:4096,"
                                        "32:2048,64:2048,64:4096")
    a = ap.parse_args()
    fix_seeds()
    model = mh.load()
    print(f"{mh.MODEL}: GQA n_rep = "
          f"{model.config.num_attention_heads // model.config.num_key_value_heads}\n")

    rows = []
    for spec in a.points.split(","):
        b, cl = (int(x) for x in spec.split(":"))
        try:
            r = compare(model, b, cl)
        except torch.cuda.OutOfMemoryError:
            print(f"  b={b} s={cl}: OOM"); torch.cuda.empty_cache(); continue
        rows.append(r)
        print(f"  b={b:3d} s={cl:5d}  repeat_kv {r['masked_ms']:7.2f} ms  "
              f"enable_gqa {r['gqa_ms']:7.2f} ms  "
              f"{r['speedup']:5.2f}x  saved {r['saved_ms']:7.2f} ms   "
              f"max|d|={r['max_abs_diff']:.2e} argmax_same={r['argmax_identical']}")

    print("\n=== cost of materializing the GQA expansion ===")
    print(f"{'batch':>6}{'ctx':>6}{'repeat_kv':>11}{'enable_gqa':>12}"
          f"{'speedup':>9}{'tok/s':>9}{'->tok/s':>9}{'GiB':>7}{'->GiB':>7}"
          f"{'GB/s':>7}{'->GB/s':>8}")
    for r in rows:
        print(f"{r['batch']:>6}{r['cache_len']:>6}{r['masked_ms']:>11.2f}"
              f"{r['gqa_ms']:>12.2f}{r['speedup']:>9.2f}"
              f"{r['masked_tok_s']:>9.1f}{r['gqa_tok_s']:>9.1f}"
              f"{r['bytes_masked_gib']:>7.2f}{r['bytes_gqa_gib']:>7.2f}"
              f"{r['gbs_masked']:>7.0f}{r['gbs_gqa']:>8.0f}")

    bad = [r for r in rows if not r["argmax_identical"]]
    print(f"\ncorrectness: {len(rows)-len(bad)}/{len(rows)} points produce an "
          f"identical argmax; max abs logit diff "
          f"{max((r['max_abs_diff'] for r in rows), default=0):.3e}")
    if bad:
        print("  NOTE: argmax differs at "
              + ", ".join(f"b{r['batch']}/s{r['cache_len']}" for r in bad))

    save({"env": env_report(), "model": mh.MODEL, "points": rows}, a.out)


if __name__ == "__main__":
    main()
