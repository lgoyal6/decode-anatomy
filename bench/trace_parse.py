"""Attribute every GPU kernel in a torch.profiler trace to a category.

Categorizing by kernel name alone does not work.  `softmax` appears in both
attention and sampling; `bmm` appears in attention and in MLPs; a cuBLAS kernel
called `ampere_bf16_...` tells you nothing about which projection launched it.

So we use the two pieces of structure the trace already carries:

  1. every device kernel carries an ``External id`` that matches the
     ``External id`` of the CPU-side aten op that launched it, and
  2. we wrap model submodules in ``record_function`` scopes, which land in the
     trace as ``user_annotation`` events, so each aten op sits inside a known
     scope (ATTN / MLP / NORM / LMHEAD / SAMPLE).

Category is then a function of (scope, aten op), which is unambiguous.  Any
kernel we fail to place lands in ``other`` and is printed, so an incomplete
mapping shows up as a visible bucket rather than a silent misattribution.
"""
import bisect
import json
from collections import defaultdict

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# aten ops that are matrix multiplies
GEMM_OPS = {
    "aten::linear", "aten::mm", "aten::addmm", "aten::matmul",
    "aten::bmm", "aten::baddbmm", "aten::_int_mm", "aten::einsum",
}
ATTN_OPS = {
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_flash_attention_forward",
    "aten::_efficient_attention_forward",
}
SAMPLING_OPS = {
    "aten::argmax", "aten::topk", "aten::multinomial", "aten::sort",
    "aten::cumsum", "aten::softmax", "aten::_softmax", "aten::log_softmax",
    "aten::div", "aten::gather", "aten::scatter",
}
MEM_OPS = {
    "aten::copy_", "aten::cat", "aten::index_put_", "aten::index_put",
    "aten::slice_scatter", "aten::index_copy_", "aten::clone",
    "aten::contiguous", "aten::to", "aten::_to_copy", "aten::empty_like",
    "aten::index_select", "aten::narrow_copy", "aten::stack",
    "aten::index_add_", "aten::masked_fill_", "aten::zero_", "aten::fill_",
}


def categorize(scope, op, kernel_name):
    """(module scope, aten op, kernel name) -> one of seven categories."""
    op = op or ""
    kn = (kernel_name or "").lower()

    # Sampling is decided by scope: the same aten::softmax is attention math
    # inside ATTN and sampling math inside SAMPLE.
    if scope == "SAMPLE":
        return "sampling"
    if scope == "LMHEAD":
        # Kept separate from the other projections on purpose: Qwen2.5-7B's
        # lm_head is 3584x152064, ~1.09 GiB in bf16, which is ~14% of all
        # weight bytes.  Folding it into "projection GEMMs" would hide that.
        return "lm_head" if op in GEMM_OPS else "norm_elem"

    if op in ATTN_OPS:
        return "attention"
    if op in GEMM_OPS:
        # A matmul inside the attention module is either a projection
        # (q/k/v/o, which are aten::linear) or the attention scores/values
        # product (bmm/matmul on 4-D tensors).
        if scope == "ATTN" and op in ("aten::bmm", "aten::baddbmm", "aten::matmul"):
            return "attention"
        return "proj_gemm"
    if op in MEM_OPS:
        # `aten::copy_` inside the attention module is transformers' repeat_kv:
        # it expands the 4 KV heads to 28 query heads and the trailing reshape
        # on an expanded (non-contiguous) view forces a materializing copy.
        # The genuine KV cache write is `aten::index_copy_`, and mask
        # construction is `aten::cat`/`aten::fill_`, so copy_-in-ATTN is
        # unambiguous for a GQA model.  It gets its own category because
        # burying 200 ms of avoidable duplication inside "memory movement"
        # is exactly the kind of thing this breakdown exists to surface.
        if scope == "ATTN" and op == "aten::copy_":
            return "gqa_expand"
        return "mem_move"
    if scope == "ATTN" and op in ("aten::softmax", "aten::_softmax"):
        return "attention"
    if op in SAMPLING_OPS:
        return "sampling"

    # Elementwise / normalization: RMSNorm decomposes into pow/mean/rsqrt/mul,
    # rotary into mul/neg/cat, SwiGLU into silu/mul, residuals into add.
    # `_native::bmm_triton_*` is torch 2.13's Triton bmm; here it is the
    # position x inv_freq outer product inside the rotary embedding, which is
    # overhead work, not a projection.
    if (op.startswith("aten::") or op.startswith("c10d::")
            or op.startswith("_native::")):
        return "norm_elem"

    # Fall back to the kernel name only when there is no aten parent at all.
    if "memcpy" in kn or "memset" in kn:
        return "mem_move"
    if "gemm" in kn or "cutlass" in kn or "gemv" in kn:
        return "proj_gemm"
    if "elementwise" in kn or "reduce" in kn or "norm" in kn:
        return "norm_elem"
    return "other"


class Trace:
    def __init__(self, path):
        with open(path) as f:
            self.raw = json.load(f)
        evs = self.raw["traceEvents"]

        self.cpu_by_extid = {}
        self.device = []
        anns = defaultdict(list)

        for e in evs:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat")
            args = e.get("args") or {}
            if cat in DEVICE_CATS:
                self.device.append(e)
            elif cat == "cpu_op":
                ext = args.get("External id")
                if ext is not None:
                    # Innermost op wins: aten::linear dispatches to aten::addmm,
                    # both share an External id lineage, and the kernel belongs
                    # to whichever op actually launched it.  Later/shorter
                    # events are deeper in the dispatch stack.
                    prev = self.cpu_by_extid.get(ext)
                    if prev is None or e.get("dur", 0) <= prev.get("dur", 0):
                        self.cpu_by_extid[ext] = e
            elif cat == "user_annotation":
                if not e["name"].startswith("ProfilerStep"):
                    anns[e["tid"]].append(e)

        # Per-thread sorted annotation starts, for innermost-scope lookup.
        self.ann = {}
        for tid, lst in anns.items():
            lst.sort(key=lambda x: x["ts"])
            self.ann[tid] = (lst, [x["ts"] for x in lst])

    def scope_of(self, cpu_ev):
        """Innermost enclosing record_function scope for a CPU op."""
        if cpu_ev is None:
            return None
        tid = cpu_ev["tid"]
        if tid not in self.ann:
            return None
        lst, starts = self.ann[tid]
        ts = cpu_ev["ts"]
        i = bisect.bisect_right(starts, ts) - 1
        # Walk back over candidates that start before ts; the innermost
        # containing scope is the one with the greatest start time.
        while i >= 0:
            a = lst[i]
            if a["ts"] + a.get("dur", 0) >= ts:
                return a["name"]
            i -= 1
        return None

    def kernels(self):
        """One record per device kernel, with category and attribution."""
        out = []
        for k in self.device:
            args = k.get("args") or {}
            ext = args.get("External id")
            cpu = self.cpu_by_extid.get(ext)
            scope = self.scope_of(cpu)
            op = cpu["name"] if cpu else None
            out.append({
                "kernel": k["name"],
                "cat_trace": k["cat"],
                "dur_us": k.get("dur", 0),
                "op": op,
                "scope": scope,
                "category": categorize(scope, op, k["name"]),
                "grid": args.get("grid"),
                "block": args.get("block"),
                "bytes": args.get("bytes"),
            })
        return out


    def gpu_span_us(self):
        """First kernel start to last kernel end, across all device streams.

        03_sweep originally derived non-kernel time as (clean wall) minus (sum
        of profiled kernel durations).  Mixing two different runs that way let
        the subtraction go negative once profiling overhead inflated the
        kernels.  Span minus busy comes from a single measurement and is
        therefore always a real, non-negative idle time -- inflated by the
        profiler, but self-consistent.
        """
        if not self.device:
            return 0.0
        lo = min(k["ts"] for k in self.device)
        hi = max(k["ts"] + k.get("dur", 0) for k in self.device)
        return hi - lo


def summarize(kernels, warn=True):
    """Aggregate kernel records by category."""
    agg = defaultdict(lambda: {"us": 0.0, "n": 0})
    for k in kernels:
        a = agg[k["category"]]
        a["us"] += k["dur_us"]
        a["n"] += 1
    total = sum(a["us"] for a in agg.values()) or 1.0
    rows = []
    for cat, a in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
        rows.append({"category": cat, "gpu_us": a["us"], "n_kernels": a["n"],
                     "pct_gpu": 100.0 * a["us"] / total})
    if warn:
        unk = [k for k in kernels if k["category"] == "other"]
        if unk:
            print(f"  WARNING: {len(unk)} kernels landed in 'other' "
                  f"({sum(k['dur_us'] for k in unk):.1f} us). Examples:")
            for k in sorted(unk, key=lambda x: -x["dur_us"])[:8]:
                print(f"    op={k['op']} scope={k['scope']} "
                      f"kernel={k['kernel'][:70]}")
    return rows, total
