# Step 4: does the serving stack change the answer?

Prefix caching and continuous batching are sold as output-neutral. Floating
point reduction order is not invariant to batch composition, so that claim is
testable rather than definitional.

Measured on the same card as the rest of this repo.

| | |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB, sm_86, driver 595.71.05 |
| vLLM | 0.19.1 |
| torch | 2.10.0+cu128 |
| model | `Qwen/Qwen2.5-7B-Instruct`, float16 |
| task | GSM8K test, first 200 prompts, greedy (temperature 0), max 512 new tokens |
| seed | 0 |

These four rows are not decoration. A determinism result without the model, the
serving version and the GPU pinned is not reproducible by anyone, including its
author a month later.

---

## First: is a solo request even deterministic?

Everything downstream compares against a reference, so the reference has to be
stable. Same prompt, temperature 0, run alone, 5 times, across 5 prompts:

**5/5 byte-identical.** So divergence measured below is caused by the serving
configuration, not by unstable kernels. Had this failed, the rest of the study
would have been measuring nondeterminism rather than the effect of batching, and
the design would have had to change.

---

## The matrix

200 prompts per condition. `byte-id` is exact token-sequence equality against
the reference; `answer-id` is equality of the extracted final number;
`med 1st div` is the median index of the first differing token among the
requests that diverged.

| condition | prefix cache | chunked prefill | batch | byte-id % | answer-id % | accuracy % | med 1st div |
|---|---|---|---|---|---|---|---|
| reference | off | off | 1 | 100.0 | 100.0 | 88.5 | - |
| prefix_cache | **on** | off | 1 | **97.0** | 100.0 | 88.5 | 97 |
| chunked_prefill | off | **on** | 1 | **100.0** | 100.0 | 88.5 | - |
| batch8_a | off | off | 8 | 95.5 | 100.0 | 88.5 | 117 |
| batch8_b | off | off | 8 | 95.0 | **99.0** | 88.5 | 91 |
| batch32 | off | off | 32 | 95.0 | **99.0** | 88.5 | 117 |
| batch8_prefix | on | off | 8 | 94.0 | 100.0 | 88.5 | 89 |
| batch8_chunked | off | on | 8 | 95.5 | 100.0 | 88.5 | 117 |
| all_on_batch32 | on | on | 32 | **92.5** | 99.5 | 88.5 | **47** |

### Which optimizations are output-neutral on this hardware

**Chunked prefill is.** 100.0% byte-identical on its own, and it adds nothing
to batching's divergence either (batch8 95.5% with it, 95.5% without). Not
"close enough" -- not one differing byte in 200 requests.

**Prefix caching is not.** 3.0% of requests change at batch 1, where nothing
else is varying. This is the cleanest single-variable result in the table.

**Batching is not.** 4.5-5.0% of requests change.

They compound rather than overlap: with everything on at batch 32, byte-identity
falls to 92.5% and the median first divergence moves from ~117 to token **47**,
so divergence starts earlier in the response as well as more often.

### Batch composition matters, not just batch size

`batch8_a` and `batch8_b` differ **only** in submission order -- same batch
size, same flags, same prompts, different neighbours in each scheduled batch.
One preserves every answer (100.0%) and the other does not (99.0%).

Meanwhile going from batch 8 to batch 32 does not make byte-identity worse
(95.0% both). So *who a request is batched with* affects the output more than
*how many* it is batched with, which is what a reduction-order explanation
predicts and a naive "bigger batch, more noise" model does not.

### The three levels genuinely disagree

This is the point of measuring three things instead of one:

- **byte-identical: 92.5-100%.** Collapses readily.
- **answer-identical: 99.0-100%.** Mostly holds. 1% of requests reach a
  *different final answer*.
- **accuracy: 88.5% in every single condition.** Never moves at all.

A 1% answer change with unmoved aggregate accuracy means the changes cancel:
roughly as many requests flip right-to-wrong as wrong-to-right. An A/B test
reading only aggregate accuracy would see a perfectly clean result while 1% of
individual users got a different answer.

So the honest summary is neither "batching is output-neutral" nor "batching
changes the answer". It is: **batching changes individual outputs, does not
change aggregate quality, and the difference between those two statements is
where the risk lives.**

---

## Caveats worth stating

**Divergence rate depends on generation length.** An earlier run at
`max_tokens=192` showed 100% byte-identity at batch 8 -- because responses were
truncated before they had a chance to diverge. At 512 the same configuration
shows 95.5%. Any determinism claim is implicitly a claim about how long the
generation is, and a short-generation benchmark will report cleaner
determinism than the system actually delivers.

**One model, one GPU, one vLLM version.** Reduction order depends on kernel
selection, which depends on all three. Nothing here transfers without
re-measuring.

**n=200 per condition.** A 1.0% answer-identity difference is 2 requests. That
is enough to demonstrate the effect exists and not enough to rank conditions by
it; the byte-identity differences (5-8 percentage points, 10-16 requests) are
firmer.

**Accuracy is a blunt instrument at this n.** 88.5% is 177/200. Identical
accuracy across conditions is consistent with "no quality change" and also with
"changes that cancel"; the answer-identity column is what distinguishes them,
and it says the second is happening.

---

## Reproduce

```bash
python serving/01_determinism_baseline.py
python serving/02_matrix.py --n-prompts 200
```

Results in `results/04_determinism.json` and `results/04_matrix.json`.
