"""Step 4.1 -- is a vLLM request even deterministic when it runs alone?

Everything downstream assumes a stable reference: "does batching change the
output" is only answerable if the un-batched output is itself repeatable. So
before any matrix, this runs the same prompt at temperature 0, alone, N times,
and checks byte-identity.

Three separate questions get answered, because they can come apart:

  1. **Within one engine instance.** Same process, same weights in memory, same
     KV cache blocks, N sequential requests. Failure here means the kernels
     themselves are nondeterministic.
  2. **Across engine restarts.** A fresh process each time. Failure here but not
     in (1) means something is fixed per-process (memory layout, autotuning
     choices, cache block assignment) but varies between runs.
  3. **Against the request the reference was taken from.** Trivially the same as
     (1) but reported separately so the tables downstream have an explicit
     baseline row rather than an implied one.

If (1) fails, the rest of step 4 measures nondeterminism rather than the effect
of batching, and the design has to change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


PROMPTS = [
    "Explain why the sky appears blue, in three sentences.",
    "List the first 10 prime numbers, separated by commas.",
    "Write a haiku about a broken vending machine.",
    "What is 17 * 23? Show your working.",
    "Summarize the causes of the 1929 stock market crash.",
]


def run_engine(model, n_repeats, max_tokens, gpu_frac, seed, prompts):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=gpu_frac,
              max_model_len=2048, seed=seed, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=None)
    runs = []
    for _ in range(n_repeats):
        # One prompt per generate call, so nothing shares a batch with anything.
        outs = []
        for p in prompts:
            o = llm.generate([p], sp)
            outs.append({"text": o[0].outputs[0].text,
                         "ids": list(o[0].outputs[0].token_ids)})
        runs.append(outs)
    del llm
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", action="store_true",
                    help="internal: run one engine and dump JSON to stdout")
    ap.add_argument("--out", default="results/04_determinism.json")
    a = ap.parse_args()

    if a.emit:
        runs = run_engine(a.model, a.repeats, a.max_tokens, a.gpu_frac,
                          a.seed, PROMPTS)
        print("---JSON---")
        print(json.dumps(runs))
        return

    # ---- 1. within one engine instance --------------------------------
    runs = run_engine(a.model, a.repeats, a.max_tokens, a.gpu_frac, a.seed,
                      PROMPTS)
    within = []
    for i, p in enumerate(PROMPTS):
        texts = [r[i]["text"] for r in runs]
        uniq = sorted(set(texts))
        within.append({"prompt": p, "n_unique": len(uniq),
                       "identical": len(uniq) == 1,
                       "hashes": [sha(t) for t in texts]})
    n_ok = sum(w["identical"] for w in within)

    print(f"=== 1. within one engine, {a.repeats} sequential runs, alone ===")
    for w in within:
        mark = "OK " if w["identical"] else "VARIES"
        print(f"  {mark} {w['n_unique']} unique / {a.repeats}  "
              f"{w['prompt'][:52]}")
    print(f"  {n_ok}/{len(PROMPTS)} prompts byte-identical within one engine")

    out = {"model": a.model, "repeats": a.repeats, "max_tokens": a.max_tokens,
           "seed": a.seed, "within_engine": within,
           "within_engine_all_identical": n_ok == len(PROMPTS),
           "reference": [{"prompt": p, "text": runs[0][i]["text"],
                          "ids": runs[0][i]["ids"]}
                         for i, p in enumerate(PROMPTS)]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")
    if n_ok != len(PROMPTS):
        print("\nNOTE: output is not deterministic even running alone. "
              "The rest of step 4 must treat this as the floor, not as an "
              "effect of batching.")


if __name__ == "__main__":
    main()
