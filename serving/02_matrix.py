"""Step 4.2/4.3 -- does the serving stack change the answer?

The baseline (01_determinism_baseline.py) established that a solo request at
temperature 0 is byte-identical across repeats, so any divergence measured here
is caused by the serving configuration and not by the kernels being unstable.

Divergence is measured at three levels because they can disagree, and the
interesting outcomes live in the disagreement:

  * **byte-identical rate** -- did the exact same string come out.
  * **first divergent token** -- where it stopped matching. A divergence at
    token 3 is a different answer; a divergence at token 150 is usually the same
    answer worded differently past a tie.
  * **task-level equivalence** -- did the extracted final answer stay the same.

GSM8K is the task because its answers are a single number, so task-level
equivalence is a string compare on an extracted integer rather than a judgement
call. A benchmark scored by another model could not separate "the serving stack
changed the answer" from "the judge is noisy".

Conditions vary prefix caching, chunked prefill, and whether a request is
scheduled alone or alongside others. Batch *composition* is varied separately
from batch *size*, because reduction order inside a batched kernel depends on
who else is in the batch, not just how many.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re


ANSWER_RE = re.compile(r"final answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)


def extract_answer(text: str) -> str | None:
    """Prefer the explicit answer sentence; fall back to the last number.

    "Last number in the response" is not good enough: a correct response ending
    "Kylar needs to pay $64 for 16 glasses" extracts 16. The prompt therefore
    asks for a fixed closing sentence and this reads that, which also makes a
    truncated response extract nothing rather than extract a wrong intermediate
    value and be scored as a confident error.
    """
    m = None
    for m in ANSWER_RE.finditer(text):
        pass
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token id, or -1 if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


# (name, enable_prefix_caching, enable_chunked_prefill, batch_size, shuffle_seed)
# batch_size 1 means each request is submitted on its own.
CONDITIONS = [
    ("reference",        False, False, 1,  None),
    ("prefix_cache",     True,  False, 1,  None),
    ("chunked_prefill",  False, True,  1,  None),
    ("batch8_a",         False, False, 8,  0),
    ("batch8_b",         False, False, 8,  1),
    ("batch32",          False, False, 32, 0),
    ("batch8_prefix",    True,  False, 8,  0),
    ("batch8_chunked",   False, True,  8,  0),
    ("all_on_batch32",   True,  True,  32, 2),
]


def load_prompts(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for ex in list(ds)[:n]:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"question": ex["question"], "gold": gold})
    return out


def run_condition(model, cond, prompts, max_tokens, gpu_frac, seed):
    from vllm import LLM, SamplingParams
    name, prefix, chunked, bs, shuf = cond
    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=gpu_frac,
              max_model_len=2048, seed=seed, disable_log_stats=True,
              enable_prefix_caching=prefix,
              enable_chunked_prefill=chunked)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    texts = [p["text"] for p in prompts]
    order = list(range(len(texts)))
    if shuf is not None:
        # Changing submission order changes which requests are co-resident in a
        # batch, which is the variable of interest, not just the batch size.
        random.Random(shuf).shuffle(order)

    results = [None] * len(texts)
    for s in range(0, len(order), bs):
        idxs = order[s:s + bs]
        outs = llm.generate([texts[i] for i in idxs], sp)
        for i, o in zip(idxs, outs):
            results[i] = {"text": o.outputs[0].text,
                          "ids": list(o.outputs[0].token_ids)}
    del llm
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=200)
    # 192 truncated the longer chains mid-sentence; 512 lets every response in
    # the sample reach its closing answer sentence.
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--out", default="results/04_matrix.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    raw = load_prompts(a.n_prompts)
    prompts = []
    for r in raw:
        text = tok.apply_chat_template(
            [{"role": "user",
              "content": r["question"]
              + "\n\nEnd your response with exactly: "
                "The final answer is <number>"}],
            tokenize=False, add_generation_prompt=True)
        prompts.append({"text": text, "gold": r["gold"],
                        "question": r["question"]})
    print(f"{a.model} | {len(prompts)} GSM8K prompts | "
          f"max_tokens {a.max_tokens}\n")

    conds = [c for c in CONDITIONS
             if a.conditions is None or c[0] in a.conditions]
    ref = None
    rows = []
    for cond in conds:
        name = cond[0]
        res = run_condition(a.model, cond, prompts, a.max_tokens, a.gpu_frac,
                            a.seed)
        if ref is None:
            ref = res
        ident = div = same_ans = correct = truncated = 0
        divs = []
        for i, (r0, r1) in enumerate(zip(ref, res)):
            same = r0["ids"] == r1["ids"]
            ident += same
            d = first_divergence(r0["ids"], r1["ids"])
            if not same:
                divs.append(d if d >= 0 else min(len(r0["ids"]), len(r1["ids"])))
            a0, a1 = extract_answer(r0["text"]), extract_answer(r1["text"])
            same_ans += (a0 == a1)
            if a1 is None:
                truncated += 1
            correct += (a1 == prompts[i]["gold"])
        n = len(prompts)
        row = {
            "condition": name,
            "prefix_caching": cond[1], "chunked_prefill": cond[2],
            "batch_size": cond[3], "shuffle_seed": cond[4],
            "byte_identical_pct": 100.0 * ident / n,
            "answer_identical_pct": 100.0 * same_ans / n,
            "accuracy_pct": 100.0 * correct / n,
            "n_diverged": n - ident,
            "n_no_answer": truncated,
            "median_first_divergence": (sorted(divs)[len(divs) // 2]
                                        if divs else None),
            "min_first_divergence": min(divs) if divs else None,
        }
        rows.append(row)
        print(f"  {name:<18} byte-identical {row['byte_identical_pct']:6.1f}%  "
              f"answer-identical {row['answer_identical_pct']:6.1f}%  "
              f"accuracy {row['accuracy_pct']:5.1f}%  "
              f"first-div median "
              f"{row['median_first_divergence'] if divs else '-'}")

    print(f"\n{'condition':<18}{'prefix':>8}{'chunked':>9}{'batch':>7}"
          f"{'byte-id %':>11}{'answer-id %':>13}{'accuracy %':>12}"
          f"{'med 1st div':>13}")
    for r in rows:
        print(f"{r['condition']:<18}{str(r['prefix_caching']):>8}"
              f"{str(r['chunked_prefill']):>9}{r['batch_size']:>7}"
              f"{r['byte_identical_pct']:>11.1f}{r['answer_identical_pct']:>13.1f}"
              f"{r['accuracy_pct']:>12.1f}"
              f"{str(r['median_first_divergence']):>13}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "n_prompts": len(prompts),
                   "max_tokens": a.max_tokens, "rows": rows}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
