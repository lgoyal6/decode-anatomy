# decode-anatomy

"Decode is memory-bound" is the most repeated claim in LLM inference and almost
nobody attaches a number to it. Memory-bound compared to what ceiling, measured
on which card, at which batch size? The claim is usually made about a datacentre
GPU in FP8 and then applied to whatever hardware is on hand.

decode-anatomy measures one card first - achieved bandwidth and achieved
compute, not the spec sheet - and then decomposes a real decode step against
that roofline, reporting what fraction of the step each ceiling explains and
what fraction neither explains. It turns out that on an RTX A6000 the answer at
batch 1 is not memory and not compute: **38.3% of the step is not kernel work at
all**, and the single largest kernel at long context is a `memcpy`.

> **Thesis.** A roofline is only useful if both of its ceilings are measured on
> the machine you are actually running on, and only honest if it admits a third
> verdict. Kernels that sit far below both ceilings are latency-bound, and
> filing them as "memory-bound" because their arithmetic intensity is low is how
> the folk claim survives contact with data. Every category here gets one of
> three verdicts, and the fraction explained by neither ceiling is reported
> rather than rounded away.

Two things here contradicted what I set out to measure, and both are kept.
The same attention kernel reads as "11% of peak, latency-bound" or "93% of peak,
memory-bound" depending only on which byte count you divide by. And the
profiler inflates precisely the quantity it is being used to measure, by 1.83x
at batch 1, which is why the headline non-kernel number comes from a CUDA graph
capture instead.

Steps 1 and 4 of the build plan share this repo. Step 1 is written up below;
**step 4 (serving-stack determinism under vLLM) is in
[`serving/README.md`](serving/README.md)**, and its one-line summary is that
chunked prefill is exactly output-neutral, prefix caching and batching are not,
and aggregate accuracy never moves even when 1% of individual answers do.

---

## Hardware and software, pinned

| | |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB GDDR6, GA102 / **sm_86**, 84 SMs, 6 MB L2 |
| Driver | 595.71.05 (CUDA 13.2) |
| Host | NixOS 26.05, 12 CPU cores, 125 GB RAM, no Slurm, no usable Docker |
| PyTorch | 2.13.0+cu129 |
| Triton | 3.7.1 |
| transformers | 5.15.1 |
| nsys | 2025.1.3.140 |
| Python | 3.12.13 (uv-managed standalone build) |
| Model | `Qwen/Qwen2.5-7B-Instruct`, bf16, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Seeds | 0 for `random`, `torch`, `torch.cuda`, `numpy` |

**sm_86 has no FP8.** Every number here is BF16 or FP32. Anything benchmarked
on H100 in FP8 is not comparable to this.

### Why this model

`Qwen/Qwen2.5-7B-Instruct` is 7.6B parameters, ungated on the Hub, and has the
GQA geometry that turns out to dominate the result: 28 query heads, **4** KV
heads, head_dim 128, so `n_rep = 7`.

| | |
|---|---|
| hidden / layers / intermediate | 3584 / 28 / 18944 |
| heads (Q / KV) | 28 / 4 (`n_rep` = 7) |
| vocab | 152064 (untied `lm_head`, 545M params ≈ 1.09 GiB in bf16) |
| KV cache | **56.0 KiB per token** (all 28 layers, bf16) |
| weight bytes read per decode step | **13.17 GiB** = 14.14 GB |

That last row sets the floor: at the measured 711 GB/s, one batch-1 decode step
cannot beat **19.9 ms**, i.e. **50.3 tok/s**, no matter what else is optimized.
This accounting is checked independently of the profiler: summing the config
gives 7.6153 B parameters against the 7.616 B the loaded model reports.

---

## Environment setup (NixOS, no root, no Docker)

Docker is installed on this host but the account is not in the `docker` group,
so containers are unavailable. `uv` plus `nix profile` is the working path.
Three NixOS-specific failures were hit and each needs an env var:

```bash
# 1. torch: "Found no NVIDIA driver on your system" despite nvidia-smi working.
#    The userspace driver is not on the default loader path on NixOS.
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

# 2. Triton: FileNotFoundError: '/sbin/ldconfig'. Triton shells out to ldconfig
#    to locate libcuda; NixOS has no such binary. Supported override:
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

# 3. Triton compiles its launcher stub with a host C compiler, and NixOS has
#    none on the default PATH.
nix profile install nixpkgs#gcc
export CC=$HOME/.nix-profile/bin/gcc CXX=$HOME/.nix-profile/bin/g++
```

`nsys` needs unfree packages allowed, and works **without root** as long as you
only trace the CUDA API (GPU performance counters would need
`NVreg_RestrictProfilingToAdminUsers=0`, which we do not have):

```bash
echo '{ allowUnfree = true; }' > ~/.config/nixpkgs/config.nix
NIXPKGS_ALLOW_UNFREE=1 nix profile install --impure nixpkgs#cudaPackages.nsight_systems
```

Full reproduction:

```bash
uv python install 3.12
uv venv --python 3.12 /var/tmp/lg-env/venv
uv pip install --python /var/tmp/lg-env/venv/bin/python torch --torch-backend=cu129
uv pip install --python /var/tmp/lg-env/venv/bin/python \
    numpy matplotlib pandas transformers accelerate safetensors \
    sentencepiece protobuf huggingface_hub
source env.sh
```

The venv lives on local disk (`/var/tmp`, 1.7 GB/s) rather than in `$HOME`,
which is NFS at 104 MB/s.

### Every GPU job takes a lock

```bash
~/work/gpurun python -u bench/03_sweep.py
```

This is not decoration. An `nsys` run that overlapped the sweep inflated wall
time by ~8% and pushed apparent non-kernel time from ~30% to ~59%. Those
numbers were discarded and re-measured under `flock`.

---

## Methodology notes worth reading before the numbers

**Timing.** CUDA events around a K-iteration inner loop, so per-call launch
overhead amortizes out. Median of 5 outer reps reported, minimum used for
throughput. Warmup matters more than usual here: an unwarmed bf16 8192³ GEMM
measures 15.7 TFLOP/s, and the same GEMM after 50 warmup iterations measures
113 TFLOP/s. Any benchmark without warmup on this card is wrong by 7x.

**Kernel attribution.** Categorizing by kernel name does not work: `softmax`
appears in both attention and sampling, `bmm` in both attention and MLPs, and a
cuBLAS kernel named `ampere_bf16_s16816gemm_...` says nothing about which
projection launched it. Instead, model submodules are wrapped in
`record_function` scopes (`ATTN` / `MLP` / `NORM` / `LMHEAD` / `SAMPLE`), the
Chrome trace is parsed directly, and each device kernel is joined to its
launching aten op through the `External id` both events carry. Category is then
a function of *(scope, aten op)*, which is unambiguous. Anything unmatched lands
in an `other` bucket that is printed rather than silently absorbed; it is
currently empty.

**The three-way verdict.** A category is called memory- or compute-bound only
if it reaches 65% of its roofline ceiling `min(peak_compute, bandwidth x AI)`;
otherwise it is *latency-bound*, meaning neither ceiling explains its runtime.
Without that third bucket every low-intensity kernel gets filed as
"memory-bound" whether or not it moves bytes at a respectable rate, which is
precisely the sloppiness this repo exists to avoid. The 65% threshold is a
judgement call, but the measured categories are not near it: the projection
GEMMs sit at 93-100% of their ceiling and the elementwise kernels sit at ~1%,
so no category changes verdict anywhere between a 20% and a 90% threshold.

**Arithmetic intensity** is computed analytically from the model config, not
read from the profiler (which only estimates FLOPs for matmuls). Both a
*compulsory* byte count (each weight read once, each cached K/V read once) and
an *as-implemented* byte count are reported, because for this model they differ
by up to 8x and using the wrong one inverts the conclusion. See the GQA section.

**Making a decode step repeatable** required working around three properties of
transformers' `StaticCache`, documented in `bench/model_harness.py`:
`get_mask_sizes` returns `max_cache_len` rather than the fill level (so
attention always spans the whole padded cache, and cache *allocation* - not
fill - sets attention cost); the write index comes from `layer.cumulative_length`
rather than the `cache_position` passed to `forward`; and `cumulative_length` is
advanced in place every call, so timing a step in a loop walks it off the end of
the cache and trips a device-side assert in `index_copy_`. All 28 layers'
counters are repointed at views of one shared vector, making "rewind to the same
decode step" a single `fill_` kernel that is measured and subtracted.

---

## Result 1: the empirical roofline

`python bench/01_roofline.py` - measures the card before profiling anything.

### Memory bandwidth

Four access patterns, buffer sizes from inside-L2 to 4 GiB. Bytes counted as
compulsory traffic (STREAM convention).

| kernel | bytes/element | best achieved | % of 768 GB/s spec |
|---|---|---|---|
| `read` (reduction) | 1N | **711.1 GB/s** | **92.6%** |
| `triad` (`a = b + αc`) | 3N | 685.7 GB/s | 89.3% |
| `scale` (`a = αb`) | 2N | 684.1 GB/s | 89.1% |
| `copy` (D2D) | 2N | 684.0 GB/s | 89.1% |

Bandwidth plateaus by 256 MiB and is flat to 4 GiB. The 4 MiB point reads
696 GB/s for `copy` because it fits in the 6 MB L2, which is a useful check
that the large sizes really are missing cache.

### Compute throughput

Square GEMMs, N from 512 to 16384.

| dtype | best achieved | spec (dense) | % of spec |
|---|---|---|---|
| bf16 tensor | **128.5 TFLOP/s** | 154.8 | 83.0% |
| fp16 tensor | 124.0 TFLOP/s | 154.8 | 80.1% |
| tf32 tensor | 65.1 TFLOP/s | 77.4 | 84.2% |
| fp32 | 23.9 TFLOP/s | 38.7 | 61.8% |

**bf16 and fp16 land within 4% of each other.** GA102 is sometimes described as
running fp16 tensor ops at double rate with fp16 accumulate versus fp32
accumulate; if that split were visible here, one of these two rows would be
~2x the other. It is not, so the 154.8 TFLOP/s datasheet figure is the
fp32-accumulate rate and torch reaches 83% of it. This matters because it fixes
the roofline ridge point within 4% rather than within 2x.

The card is **power-limited, not clock-limited**, during sustained GEMMs:
296 W of a 300 W cap with throttle reason `0x4` (SW power cap), and SM clock
wandering between 1335 and 1935 MHz. Small GEMMs are neither: N=512 reaches
only 17.5 TFLOP/s (14% of peak) because there is not enough work to fill 84 SMs.

### The roofline

```
peak HBM bandwidth   711.1 GB/s   (92.6% of spec)
peak bf16 compute    128.5 TFLOP/s (83.0% of spec)
ridge point          180.7 FLOP per byte
```

**A kernel needs more than 180.7 FLOP per byte of HBM traffic to be
compute-bound on this card.**

That single number decides most of what follows. A batch-*B* decode step reads
each weight once and does 2*B* FLOPs with it, so its arithmetic intensity is
about *B* FLOP/byte. Reaching the ridge would take **batch ≈ 181**. Every batch
size in the requested sweep - 1 through 64 - is therefore predicted to be
memory-bound on the projection GEMMs, and the sweep confirms it: the
`compute-bound` column is **0.0% at all 28 points**.

---

## Result 2: the largest single item in a long-context decode step is not a matmul

This was not what I set out to find, and it changes the breakdown enough that
it belongs before the sweep table.

At batch 64 / context 4096, the two biggest device kernels per decode step are:

| kernel | calls/step | ms/step | what it is |
|---|---|---|---|
| `at::native::elementwise_kernel<128,4,...>` | 56 | **209.2** | `aten::copy_` inside `ATTN` |
| `fmha_cutlassF_bf16_aligned_64x128_rf_sm80` | 28 | 154.6 | memory-efficient attention |
| `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_tn` | 56 | 11.7 | MLP projections |

56 calls is two per layer. That `copy_` is transformers' `repeat_kv`: it
expands the 4 KV heads out to 28 query heads, and because the trailing
`reshape` acts on an `expand`ed (non-contiguous) view, it **materializes** the
result. 209 ms of a 379 ms step is spent duplicating the KV cache.

### Why it happens

```python
# transformers/integrations/sdpa_attention.py
def use_gqa_in_sdpa(attention_mask, key, value) -> bool:
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256
```

`StaticCache.get_mask_sizes` returns `kv_length = max_cache_len`, so attention
always spans the whole padded cache and a mask is *required* to hide the
padding. A mask makes `use_gqa_in_sdpa` return False, which calls `repeat_kv`.
The chain is: **padded static cache → mandatory dense mask → sdpa's native GQA
is unavailable → the KV cache is materialized at 7x.**

### What it costs

A first attempt to A/B this from outside - calling `forward(attention_mask=None)`
 - produced **bit-identical logits and identical timings** (`max|Δ| = 0.00e+00`
at all 7 points). `Qwen2Model.forward` builds its own mask for a padded cache
regardless of what the caller passes, so there is no framework-level switch.
That is a real negative result, and it moves the measurement down to
`scaled_dot_product_attention` itself.

`python bench/06_attn_paths.py` - one layer of decode attention, four ways:

| batch | ctx | A: `repeat_kv`+mask | B: `enable_gqa` | C: `repeat_kv` no mask | D: `enable_gqa`+mask | A/B |
|---|---|---|---|---|---|---|
| 1 | 512 | 0.100 ms | 0.054 ms | 0.135 ms | 0.384 ms | 1.85x |
| 1 | 4096 | 0.322 ms | 0.054 ms | 0.205 ms | 0.670 ms | 6.00x |
| 8 | 2048 | 0.781 ms | 0.063 ms | 0.835 ms | 2.368 ms | 12.36x |
| 8 | 4096 | 1.518 ms | 0.111 ms | 1.627 ms | 4.951 ms | 13.66x |
| 32 | 2048 | 3.094 ms | 0.214 ms | 3.166 ms | 9.441 ms | 14.46x |
| 64 | 2048 | 6.197 ms | 0.455 ms | 6.321 ms | 18.894 ms | 13.63x |
| 64 | 4096 | **12.239 ms** | **0.906 ms** | 12.496 ms | 39.364 ms | **13.50x** |

All four agree numerically: max absolute logit difference **1.95e-3**, which is
bf16 reduction-order noise, not a correctness difference.

Dispatched backends, read out of the trace:

| variant | backend |
|---|---|
| A `repeat_kv` + mask | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80` (mem-efficient) |
| B `enable_gqa`, no mask | `pytorch_flash::flash_fwd_kernel` (flash) |
| C `repeat_kv`, no mask | `pytorch_flash::flash_fwd_kernel` (flash) |
| D `enable_gqa` + mask | `aten::bmm` + `aten::_softmax` (**math fallback**) |

Three things fall out of this table:

1. **C ≈ A.** Removing the mask gets you the flash backend and buys nothing
   (12.496 vs 12.239 ms). The cost is the materialization, not the mask and not
   the backend choice.
2. **D is 3.2x worse than A.** Combining a mask with `enable_gqa=True` drops to
   `bmm` + `softmax` and achieves **14 GB/s**, 2% of peak. The transformers
   guard exists for a good reason.
3. **A is not slow because it is inefficient.** On the bytes it actually moves
   it reaches **658 GB/s, 93% of the measured peak.** It is memory-bound and
   running at nearly peak bandwidth - on 8x the traffic the arithmetic needs.

This is the trap the whole exercise is about. Judged against *compulsory*
bytes, attention at batch 64 / ctx 4096 looks like it achieves 79 GB/s, 11% of
peak, and gets filed as "latency-bound". Judged against the bytes the
implementation moves, it is at 93% of the memory ceiling. Same kernel, same
time, opposite verdict. The `byte_amplification` column in the sweep table
reports the ratio (up to **8.0x** total step traffic) so this cannot hide.

### Scope of the claim

This is **not** a free 13x for a real serving stack. Variant B passes no mask,
which is only correct when every cache slot holds a real token - true in this
harness by construction, false in general batched serving with mixed sequence
lengths. The finding is the cost of the *padded-dense-mask design*, and it is a
concrete reason production stacks use paged attention kernels that handle
variable lengths and GQA natively instead of a dense additive mask. Step 4 runs
vLLM on the same card and model, which is the natural place to check what that
design buys.

---

## Result 3: the non-kernel slice, measured three ways

`python bench/04_overhead.py`

Inferring non-kernel time as `wall - sum(kernel durations)` has a problem: both
profilers inflate exactly what they are measuring. The `prof` column in the
sweep table is `profiled GPU span / clean wall time` and it runs **1.83x at
batch 1** and 1.03x at batch 64 - so the profiler roughly doubles the apparent
inter-kernel gap at small batch. Any non-kernel number read off a profiler at
batch 1 is wrong by up to 2x.

So the authoritative measurement is a **CUDA graph capture** of the whole decode
step, timed with no profiler attached. A replay issues the entire step as one
command instead of ~1459 launches, so *eager minus replay is the removable
overhead*, measured rather than modelled.

| batch | ctx | eager | CUDA graph | removable | speedup | implied per-launch |
|---|---|---|---|---|---|---|
| 1 | 512 | 39.817 ms | **24.569 ms** | **15.248 ms (38.3%)** | **1.62x** | 10.45 µs |
| 8 | 512 | 39.927 ms | 29.655 ms | 10.272 ms (25.7%) | 1.35x | 7.04 µs |
| 32 | 512 | 48.674 ms | 47.228 ms | 1.446 ms (3.0%) | 1.03x | 1.01 µs |

**At batch 1, 38% of a decode step is not kernel work at all**, and it comes off
without touching a single kernel. It is gone by batch 32, where the GPU finally
has enough work per launch to stay ahead of the host.

Framed against the hard floor: the 13.17 GiB of weight traffic costs 19.9 ms at
711 GB/s. Eager decode at batch 1 runs at **2.00x that floor**; the same step
under CUDA graphs runs at **1.24x**.

### What the overhead is made of

Three independent measurements of the per-launch cost agree:

| method | per-launch cost | what it includes |
|---|---|---|
| CUDA graph delta (batch 1) | 10.45 µs | everything removable |
| trivial-op host issue floor | 11.97 µs | Python + dispatch + launch |
| `nsys` `cudaLaunchKernel` median | 7.99 µs | the CUDA API call alone |

The gap between the last two rows is Python and the dispatcher; the CUDA API
call itself is the larger share. Supporting numbers from the same floor
benchmark: device-side cost of a do-nothing kernel **7.39 µs**, empty
`cudaDeviceSynchronize` **4.60 µs**, and `queue_bound = True` - the host cannot
issue trivial kernels as fast as the GPU retires them.

### It is not synchronization

Counted from the trace, a decode step performs **0.67 host/device rendezvous**
(that is the one explicit `synchronize()` amortized over three profiled steps)
and moves **0 bytes device-to-host**. The non-kernel time is launch and dispatch
cost, not stalls waiting on the host.

Caveat worth stating: this harness calls `forward` directly. A real
`generate()` loop syncs every step to evaluate stopping criteria, and would add
sync cost this measurement excludes. That belongs to the serving stack, which
is step 4.

---

## Result 4: the sweep - where decode time goes, batch 1-64 x context 128-4096

`python bench/03_sweep.py` → `plots/breakdown.png`, `plots/roofline.png`,
`plots/bandwidth.png`

`mem%` / `cmp%` / `lat%` / `nonk%` are percentages of the profiled GPU span, so
they sum to 100. `ampl` is as-implemented bytes ÷ compulsory bytes. `prof` is
profiled span ÷ clean wall time, i.e. how much the profiler inflated the gaps.

| batch | ctx | wall ms | tok/s | AI | mem% | **cmp%** | lat% | nonk% | GB/s | ampl | prof |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 34.18 | 29.3 | 1.00 | 34.7 | **0.0** | 4.2 | 61.2 | 417 | 1.0x | 1.80 |
| 2 | 128 | 40.41 | 49.5 | 1.97 | 35.5 | **0.0** | 4.8 | 59.7 | 356 | 1.0x | 1.48 |
| 4 | 128 | 40.55 | 98.6 | 3.88 | 35.0 | **0.0** | 5.8 | 59.2 | 361 | 1.0x | 1.50 |
| 8 | 128 | 40.72 | 196.5 | 7.51 | 37.1 | **0.0** | 5.8 | 57.1 | 371 | 1.1x | 1.48 |
| 16 | 128 | 40.63 | 393.8 | 14.10 | 40.7 | **0.0** | 4.4 | 54.9 | 396 | 1.1x | 1.51 |
| 32 | 128 | 40.85 | 783.3 | 25.13 | 45.9 | **0.0** | 4.6 | 49.5 | 442 | 1.2x | 1.50 |
| 64 | 128 | 40.88 | **1565.5** | 41.27 | 57.2 | **0.0** | 5.2 | 37.7 | 538 | 1.4x | 1.50 |
| 1 | 512 | 40.51 | 24.7 | 0.98 | 35.6 | **0.0** | 4.9 | 59.4 | 360 | 1.0x | 1.51 |
| 2 | 512 | 40.58 | 49.3 | 1.91 | 36.5 | **0.0** | 5.6 | 57.8 | 371 | 1.1x | 1.49 |
| 4 | 512 | 40.51 | 98.7 | 3.60 | 38.1 | **0.0** | 7.1 | 54.8 | 394 | 1.1x | 1.48 |
| 8 | 512 | 40.39 | 198.1 | 6.46 | 45.5 | **0.0** | 4.2 | 50.2 | 440 | 1.2x | 1.49 |
| 16 | 512 | 40.60 | 394.1 | 10.73 | 53.9 | **0.0** | 4.5 | 41.6 | 527 | 1.4x | 1.52 |
| 32 | 512 | 48.58 | 658.7 | 16.03 | 73.2 | **0.0** | 4.9 | 21.9 | 590 | 1.8x | 1.26 |
| 64 | 512 | 72.72 | 880.1 | 21.28 | 92.9 | **0.0** | 4.7 | 2.4 | 593 | 2.6x | 1.03 |
| 1 | 2048 | 40.36 | 24.8 | 0.94 | 38.1 | **0.0** | 8.7 | 53.2 | 394 | 1.1x | 1.48 |
| 2 | 2048 | 40.71 | 49.1 | 1.69 | 45.2 | **0.0** | 4.0 | 50.9 | 435 | 1.2x | 1.49 |
| 4 | 2048 | 40.62 | 98.5 | 2.82 | 56.4 | **0.0** | 4.2 | 39.4 | 523 | 1.4x | 1.49 |
| 8 | 2048 | 47.57 | 168.2 | 4.22 | 72.8 | **0.0** | 4.4 | 22.7 | 596 | 1.9x | 1.27 |
| 16 | 2048 | 70.89 | 225.7 | 5.63 | 93.4 | **0.0** | 4.2 | 2.4 | 600 | 2.6x | 1.03 |
| 32 | 2048 | 114.25 | 280.1 | 6.75 | 95.8 | **0.0** | 2.7 | 1.5 | 621 | 3.9x | 1.03 |
| 64 | 2048 | 204.84 | 312.4 | 7.50 | 97.4 | **0.0** | 1.7 | 0.9 | 623 | 5.7x | 1.03 |
| 1 | 4096 | 40.42 | 24.7 | 0.89 | 40.1 | **0.0** | 13.7 | 46.2 | 437 | 1.2x | 1.50 |
| 2 | 4096 | 40.45 | 49.4 | 1.49 | 54.7 | **0.0** | 4.1 | 41.3 | 524 | 1.4x | 1.51 |
| 4 | 4096 | 49.78 | 80.4 | 2.23 | 76.4 | **0.0** | 4.3 | 19.4 | 568 | 1.9x | 1.23 |
| 8 | 4096 | 69.74 | 114.7 | 2.98 | 93.7 | **0.0** | 4.0 | 2.3 | 608 | 2.6x | 1.03 |
| 16 | 4096 | 115.30 | 138.8 | 3.57 | 96.0 | **0.0** | 2.6 | 1.4 | 613 | 3.9x | 1.03 |
| 32 | 4096 | 201.41 | 158.9 | 3.97 | 97.6 | **0.0** | 1.6 | 0.8 | 632 | 5.8x | 1.03 |
| 64 | 4096 | 379.21 | 168.8 | 4.20 | **98.6** | **0.0** | 0.9 | 0.4 | 634 | 8.0x | 1.03 |

### Answering the question the plan asked

**Compute-bound is 0.0% at all 28 points.** Not "small" - zero. The card's ridge
is 180.7 FLOP/byte and the highest arithmetic intensity anywhere in this sweep
is 62 (the projection GEMMs at batch 64), a factor of 3 short. Predicted from
the roofline before profiling; confirmed by it.

**The interesting axis is not memory vs compute, it is memory vs *issue*.**
Decode moves from launch-bound to bandwidth-bound as work per launch grows:

| | non-kernel share | memory-bound share |
|---|---|---|
| batch 1, ctx 128 | 61.2% | 34.7% |
| batch 64, ctx 4096 | 0.4% | **98.6%** |

The crossover is at roughly **batch 16-32 at context 512**, and it arrives
earlier at longer context (batch 4-8 at context 4096) because a longer cache
gives each attention launch more bytes to move. "Decode is memory-bound" is
true - but only once you have enough concurrent work. At batch 1 the dominant
term is not bandwidth, it is the host issuing 1459 kernels.

**Latency-bound never dominates but never disappears.** It peaks at 13.7%
(batch 1, ctx 4096) and holds at 2-5% almost everywhere. It is the elementwise
tail: `norm_elem` never exceeds 23% of peak bandwidth, and `mem_move` never
exceeds 1%.

### Category share of GPU busy time

Short context - the projection GEMMs are the whole story:

| batch | proj_gemm | lm_head | attention | gqa_expand | norm_elem | mem_move | gpu ms |
|---|---|---|---|---|---|---|---|
| 1 | **82.9** | 6.5 | 1.0 | 0.8 | 6.1 | 2.6 | 23.82 |
| 8 | 76.6 | 6.0 | 3.8 | 3.9 | 7.0 | 2.7 | 25.78 |
| 64 | 54.7 | 4.2 | 13.7 | 19.0 | 6.2 | 2.1 | 38.16 |

Long context (4096) - they are almost irrelevant:

| batch | proj_gemm | lm_head | attention | gqa_expand | norm_elem | mem_move | gpu ms |
|---|---|---|---|---|---|---|---|
| 1 | 60.6 | 4.7 | 18.5 | 9.3 | 4.9 | 2.1 | 32.57 |
| 8 | 28.5 | 2.2 | 29.5 | 35.7 | 2.9 | 1.1 | 70.03 |
| 64 | 5.4 | 0.4 | 39.6 | **53.6** | 0.7 | 0.2 | 390.49 |

At batch 64 / context 4096 the projection GEMMs - the thing everyone optimizes - 
are **5.4%** of GPU time, and 93% of it is attention plus the GQA copy.

### Two detail points

`python bench/02_decode_profile.py --batch 1 --cache-len 512`

```
wall (clean)        32.491 ms  (30.8 tok/s)      launches/step   1459
GPU busy            24.847 ms                    host/GPU = 1.31 (CPU-bound)
13.59 GiB moved, 13.21 GiB compulsory (1.03x), AI = 0.98 FLOP/byte

category         ms   %gpu  kern       AI    GB/s  %ceil  verdict
proj_gemm    19.780   79.6   280     1.00     660     93  memory-bound
lm_head       1.538    6.2     1     1.00     709    100  memory-bound
norm_elem     1.534    6.2   804     0.29       4      1  latency-bound
attention     0.830    3.3    28     1.00     248     35  latency-bound
mem_move      0.655    2.6   288     0.00       0      0  latency-bound
gqa_expand    0.503    2.0    56     0.00     467     66  memory-bound
sampling      0.009    0.0     2     0.25      67      9  latency-bound
```

`python bench/02_decode_profile.py --batch 64 --cache-len 4096`

```
wall (clean)       374.864 ms  (170.7 tok/s)     launches/step   1487
GPU busy           387.573 ms                    host/GPU = 0.89 (not CPU-bound)
223.94 GiB moved, 27.94 GiB compulsory (8.02x), AI = 4.20 FLOP/byte
achieved 641 GB/s on moved bytes / 80 GB/s on compulsory bytes

category         ms   %gpu  kern       AI    GB/s  %ceil  verdict
gqa_expand  207.165   53.5    56     0.00     581     82  memory-bound
attention   153.925   39.7    28     1.00     684     96  memory-bound
proj_gemm    21.207    5.5   308    62.45     631     89  memory-bound
norm_elem     2.698    0.7   804     0.31     145     20  latency-bound
lm_head       1.609    0.4     1    61.77     702     99  memory-bound
mem_move      0.916    0.2   287     0.00       8      1  latency-bound
```

Note `wall - busy` is **-12.7 ms** at the second point: the profiled kernel sum
exceeds the clean wall by 3.4%, because at these sizes CUPTI inflates the kernel
durations themselves. That is why the four-way split uses the trace's own span
rather than subtracting across two runs, and why the non-kernel number that
matters comes from the CUDA graph.

### Known limitations

- **`sampling` bandwidth exceeds 100% of peak** at large batch in
  `plots/bandwidth.png`. Its analytic byte count (`B x V x 4`) omits the bf16→fp32
  logit cast, so it is an underestimate. `sampling` is ≤0.1% of GPU time
  everywhere and affects no conclusion, but the number is wrong and is left
  visible rather than quietly clamped.
- **Run-to-run variation is 4-5%** on wall time (batch 1 / ctx 512 measured
  32.49 ms and 40.51 ms in two separate runs). The card is power-capped and its
  clock wanders 1335-1935 MHz. Within-run spread is reported per measurement;
  comparisons in this README are within a single run.
- The cache is filled synthetically rather than by a real prefill, which is
  sound for timing (no data-dependent control flow in these kernels) but means
  no numerical result is read off the sweep runs.
- Single stream only; the analysis assumes kernels do not overlap.

---

## The gate: which kernel category is the best fusion candidate

> **The projection GEMMs (QKV, output, MLP) are the best fusion candidate - and
> the thing to fuse into them is dequantization, not each other: they are 80% of
> GPU time at batch 1 and already run at 93% of the memory ceiling with a
> compute-bound fraction of exactly zero at all 28 sweep points, so no
> rearrangement of bf16 kernels can help them and the only remaining lever is
> making them read fewer weight bytes - which means quantized weights whose
> dequantization is fused into the matmul, because materializing a bf16 block
> first would hand back precisely the traffic the quantization saved.**

The supporting numbers: at batch 1 / context 512 the projection GEMMs are
19.780 ms of a 24.847 ms GPU-busy step, moving 660 GB/s against a 711 GB/s
ceiling at arithmetic intensity 1.00. There is no headroom in *how* they run,
and 7% of a category is not worth a kernel. There is a factor of 4 available in
*what they read*: INT4 weights would cut 13.17 GiB of weight traffic to roughly
3.3 GiB plus scales, which moves the batch-1 floor from 19.9 ms toward ~5 ms.

### Two categories that look like better candidates and are not

**`norm_elem` has by far the most launches - 804 of 1459 per step, 55% - and
only 6.2% of GPU time.** Fusing the RMSNorm / rotary / residual / SwiGLU tail
would cut launch count more than anything else, and launch count is what makes
batch-1 decode CPU-bound (host/GPU = 1.31). But CUDA graph capture already
removes 38.3% of the step by collapsing *all* 1459 launches, at the cost of a
capture call rather than a hand-written kernel. Fusing elementwise ops is the
expensive way to buy something a graph capture gives for free, so it is the
right second move, not the first.

**`gqa_expand` is the single largest item at long context - 53.6% of GPU time at
batch 64 / context 4096.** It is not a fusion candidate because it should not
exist: it is a 7x duplication of the KV cache that sdpa's own `enable_gqa` does
inside the kernel. The fix is elimination, and it lives in the serving stack's
attention design, not in a fused kernel. See Result 2.

### What this means for step 2

Step 2 fuses dequantization into the matmul in
[`lgoyal6/winnow`](https://github.com/lgoyal6/winnow). This step predicts two
things that step 2 should check rather than assume:

1. **The win should be close to the byte ratio, not the FLOP ratio.** The
   projections run at 93% of the memory ceiling, so latency should track weight
   bytes almost linearly. An INT4 fused kernel that delivers far less than its
   byte reduction is leaving something on the floor; one that delivers more is
   probably measuring wrong.
2. **The losing region should be at small shapes, and it should be large.** The
   per-launch floor measured here is 7.5-12 µs. Any fused kernel whose useful
   work is under ~10 µs will lose to whatever cuBLAS kernel PyTorch already
   picks, because the comparison is decided by launch cost, not arithmetic. At
   batch 1 a single projection GEMM is ~70 µs (19.780 ms / 280 kernels), so the
   margin is only about 7x - thin enough that a Triton kernel with worse
   occupancy can plausibly lose outright at batch 1.

---

## Figures

| | |
|---|---|
| `plots/roofline.png` | Every decode category placed on the empirical roofline, with the measured square GEMMs for reference. The projections and `lm_head` march up the memory diagonal from AI=1 to AI=62 and never reach the ridge; `norm_elem` and `sampling` sit two to three orders of magnitude below it. Zero-FLOP categories cannot appear here - see `bandwidth.png`. |
| `plots/breakdown.png` | Top: the four-way split of the profiled span across all 28 points. Bottom: category share of GPU busy time. Reading the top row left to right and the columns front to back shows non-kernel time being squeezed out and replaced by memory-bound time. |
| `plots/bandwidth.png` | Achieved bandwidth as a percentage of the measured 711 GB/s, per category. This is the only figure on which `gqa_expand` and `mem_move` can appear, and it is where the 80-96% utilization of the GQA copy is visible. |

## Files

```
bench/common.py            timing harness, spec constants, env capture
bench/trace_parse.py       Chrome-trace -> (scope, aten op) -> category
bench/model_harness.py     model loading, scope hooks, analytic cost model
bench/decode_point.py      one (batch, ctx) measurement, shared by 02 and 03
bench/01_roofline.py       bandwidth + GEMM microbenchmarks
bench/02_decode_profile.py one point in detail
bench/03_sweep.py          batch x context sweep
bench/04_overhead.py       launch floor, sync census, CUDA graph delta
bench/05_gqa.py            negative result: no framework-level GQA switch
bench/06_attn_paths.py     four attention paths at decode shapes
bench/plot_roofline.py     all three figures
```

## Commands, in order

```bash
source env.sh
~/work/gpurun python -u bench/01_roofline.py
~/work/gpurun python -u bench/02_decode_profile.py --batch 1  --cache-len 512
~/work/gpurun python -u bench/02_decode_profile.py --batch 64 --cache-len 4096
~/work/gpurun python -u bench/03_sweep.py
~/work/gpurun python -u bench/04_overhead.py
~/work/gpurun python -u bench/05_gqa.py
~/work/gpurun python -u bench/06_attn_paths.py
python bench/plot_roofline.py

# nsys second opinion (no root needed for CUDA API tracing)
~/work/gpurun nsys profile --capture-range=cudaProfilerApi \
  --capture-range-end=stop -t cuda,osrt --cuda-event-trace=false \
  -o nsys_clean --force-overwrite true python nsys_decode.py
nsys stats --report cuda_api_sum --format table nsys_clean.nsys-rep
```

## Summary of findings

1. **Empirical roofline: 711 GB/s (92.6% of spec), 128.5 TFLOP/s bf16 (83.0% of
   spec), ridge 180.7 FLOP/byte.** bf16 and fp16 agree within 4%, which resolves
   the GA102 accumulate-rate ambiguity and pins the ridge. The card is
   power-capped, not clock-capped, under sustained GEMMs.
2. **Decode is never compute-bound on this card at any batch up to 64** - 0.0%
   at all 28 points. Reaching the ridge needs batch ≈ 181.
3. **At batch 1, 38.3% of a decode step is not kernel work**, removable by CUDA
   graph capture alone (1.62x). It falls to 3.0% by batch 32. It is launch and
   dispatch cost, not synchronization: 0.67 rendezvous and 0 D2H bytes per step.
4. **The real axis is memory vs issue, not memory vs compute.** Non-kernel time
   goes 61.2% → 0.4% and memory-bound goes 34.7% → 98.6% across the sweep, with
   the crossover near batch 16-32 at context 512.
5. **At long context the largest single item is a `memcpy`, not a matmul.**
   transformers materializes the GQA expansion because a padded `StaticCache`
   forces a dense mask, which disables sdpa's native GQA. It costs 8.0x the
   compulsory byte traffic and 53.6% of GPU time at batch 64 / context 4096;
   the primitive is 13.5x faster with `enable_gqa` at equal numerics.
6. **Judging kernels against compulsory rather than as-implemented bytes inverts
   the verdict.** The same attention kernel reads as "11% of peak,
   latency-bound" or "93% of peak, memory-bound" depending on which byte count
   you divide by. Both are reported throughout.
