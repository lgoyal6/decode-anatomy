"""Load a 7B model, put it in a decode state, and cost a decode step on paper.

Two things live here:

  * `annotate` -- wraps submodules in record_function scopes so the trace
    parser can tell an attention matmul from an MLP matmul.
  * `analytic_cost` -- FLOPs and compulsory HBM bytes per category for one
    decode step at (batch, cache_len), derived from the config.  Measured time
    alone cannot place a kernel on a roofline; we need bytes and FLOPs, and
    counting them from the architecture is more trustworthy than trusting a
    profiler's FLOP estimate (which only covers matmuls anyway).

The cache is filled synthetically rather than by a real prefill.  A real
B=64,S=4096 prefill is 262k tokens and OOMs the card in the MLP, and the
decode-step kernels we are timing depend on cache *shape*, not cache contents:
sdpa and cuBLAS have no data-dependent control flow. Numerical results are
never read off these runs, only timings.
"""
import torch
from torch.profiler import record_function
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load(model_id=MODEL, dtype=torch.bfloat16, attn="sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation=attn, device_map="cuda")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# record_function scopes
# ---------------------------------------------------------------------------
class _Scope:
    """Enter a record_function on module entry, exit it on module exit."""

    def __init__(self, name):
        self.name = name
        self.ctx = None

    def pre(self, mod, args, kwargs=None):
        self.ctx = record_function(self.name)
        self.ctx.__enter__()
        return None

    def post(self, mod, args, output):
        if self.ctx is not None:
            self.ctx.__exit__(None, None, None)
            self.ctx = None
        return output


def annotate(model):
    """Tag attention / MLP / norm / lm_head submodules. Returns handles."""
    handles = []

    def tag(mod, name):
        s = _Scope(name)
        handles.append(mod.register_forward_pre_hook(s.pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(s.post))

    for layer in model.model.layers:
        tag(layer.self_attn, "ATTN")
        tag(layer.mlp, "MLP")
        tag(layer.input_layernorm, "NORM")
        tag(layer.post_attention_layernorm, "NORM")
    tag(model.model.norm, "NORM")
    tag(model.lm_head, "LMHEAD")
    return handles


def remove(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Decode state
# ---------------------------------------------------------------------------
@torch.no_grad()
def make_decode_state(model, batch, cache_len, device="cuda"):
    """Return (ids, cache, pos, reset) for a repeatable decode step at `cache_len`.

    Three details of transformers' StaticCache drive this design:

    1. `StaticLayer.get_mask_sizes` returns `kv_length = max_cache_len`, not the
       number of valid entries.  Attention therefore always spans the whole
       padded cache, so `max_cache_len` -- not the fill level -- is what sets
       attention cost.  We allocate exactly `cache_len` so the measured cost is
       the cost of a `cache_len` cache.
    2. The write index comes from `layer.cumulative_length`, not from the
       `cache_position` handed to `forward` (that one only shapes the mask).
    3. `cumulative_length` is advanced in place on every call.  Timing a decode
       step in a loop therefore walks it off the end of the cache and trips a
       device-side assert in `index_copy_`.

    So we repoint all layers' `cumulative_length` at views of one shared vector,
    which makes "rewind to the same decode step" a single `fill_` kernel that
    can be measured and subtracted rather than 28 separate launches.
    """
    cfg = model.config
    cache = StaticCache(cfg, max_cache_len=cache_len)

    # One real token forward to trigger the cache's lazy allocation, so we get
    # correctly shaped and strided key/value tensors without hand-building them.
    warm = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
    model(input_ids=warm, past_key_values=cache, use_cache=True,
          cache_position=torch.tensor([0], device=device))

    # Plausible-magnitude garbage in the cache; only shapes affect timing.
    for layer in cache.layers:
        layer.keys.normal_(0, 1)
        layer.values.normal_(0, 1)

    # Write slot for the new token: the last valid slot, so the step attends
    # over exactly `cache_len` positions.
    start = cache_len - 1
    shared = torch.full((len(cache.layers),), start,
                        dtype=torch.int64, device=device)
    for i, layer in enumerate(cache.layers):
        layer.cumulative_length = shared[i]

    def reset():
        shared.fill_(start)

    ids = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
    pos = torch.tensor([start], device=device)
    return ids, cache, pos, reset


@torch.no_grad()
def decode_step(model, ids, cache, pos, sample=True):
    """One decode step, with the sampling tail in its own scope."""
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=pos)
    if sample:
        with record_function("SAMPLE"):
            nxt = out.logits[:, -1, :].argmax(dim=-1)
        return nxt
    return out.logits


# ---------------------------------------------------------------------------
# Analytic cost model
# ---------------------------------------------------------------------------
def analytic_cost(cfg, batch, cache_len, dtype_bytes=2, gqa_materialized=True):
    """FLOPs and HBM bytes for one decode step, by category.

    Two byte counts matter and they are not the same number:

    * *compulsory* traffic -- what the arithmetic requires: each weight read
      once, each cached K/V entry read once.
    * *as-implemented* traffic -- what the code actually moves.  For a GQA
      model under transformers' sdpa path, `repeat_kv` materializes the 4 KV
      heads out to 28 query heads before calling sdpa, so both the expansion
      write and the attention read are 7x the compulsory size.

    Dividing measured time by compulsory bytes makes attention look
    latency-bound when it is in fact running near peak bandwidth on 15x the
    traffic it needs.  `gqa_materialized` selects which world we are costing;
    both are reported so the amplification is explicit.
    """
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or H // nh
    I = cfg.intermediate_size
    V = cfg.vocab_size
    S = cache_len                  # StaticCache always attends the full pad
    B = batch
    e = dtype_bytes
    n_rep = nh // nkv

    # --- projection GEMMs (per layer): q, k, v, o, gate, up, down ----------
    proj_shapes = [
        ("q", H, nh * hd), ("k", H, nkv * hd), ("v", H, nkv * hd),
        ("o", nh * hd, H), ("gate", H, I), ("up", H, I), ("down", I, H),
    ]
    proj_w = sum(k * n for _, k, n in proj_shapes)
    proj_flops = 2 * B * proj_w * L
    proj_act = sum(B * (k + n) for _, k, n in proj_shapes) * e * L
    proj_bytes = proj_w * e * L + proj_act

    # --- attention math ---------------------------------------------------
    attn_flops = (2 * B * nh * hd * S + 2 * B * nh * S * hd) * L
    kv_compulsory = 2 * B * nkv * hd * S * e * L
    kv_expanded = 2 * B * nh * hd * S * e * L
    kv_read = kv_expanded if gqa_materialized else kv_compulsory
    attn_bytes = kv_read + B * nh * hd * e * 2 * L

    # --- GQA expansion (only exists because of the repeat_kv fallback) -----
    if gqa_materialized and n_rep > 1:
        gqa_bytes = kv_compulsory + kv_expanded    # read 4 heads, write 28
    else:
        gqa_bytes = 0.0

    # --- lm_head ----------------------------------------------------------
    lm_flops = 2 * B * H * V
    lm_bytes = H * V * e + B * H * e + B * V * 4   # logits come back fp32

    # --- norms / rotary / residual / SwiGLU -------------------------------
    ne_bytes = L * e * (
        2 * (2 * B * H + H)
        + 2 * (2 * B * nh * hd) + 2 * (2 * B * nkv * hd)
        + 2 * (3 * B * H)
        + 3 * B * I
    ) + e * (2 * B * H + H)
    ne_flops = L * B * (8 * H + 2 * I) + B * 4 * H

    # --- KV cache write ---------------------------------------------------
    mem_bytes = 2 * B * nkv * hd * e * L * 2
    # --- sampling ---------------------------------------------------------
    samp_bytes = B * V * 4 + B * 8

    cats = {
        "proj_gemm": (proj_flops, proj_bytes),
        "attention": (attn_flops, attn_bytes),
        "gqa_expand": (0, gqa_bytes),
        "lm_head": (lm_flops, lm_bytes),
        "norm_elem": (ne_flops, ne_bytes),
        "mem_move": (0, mem_bytes),
        "sampling": (B * V, samp_bytes),
    }
    out = {}
    for k, (f, b) in cats.items():
        out[k] = {"flops": float(f), "bytes": float(b),
                  "arith_intensity": (f / b) if b else 0.0}
    tf = sum(v["flops"] for v in out.values())
    tb = sum(v["bytes"] for v in out.values())
    # Compulsory total: same step with GQA handled inside the kernel.
    tb_compulsory = tb - gqa_bytes - (kv_expanded - kv_compulsory
                                      if gqa_materialized and n_rep > 1 else 0)
    out["TOTAL"] = {
        "flops": tf, "bytes": tb, "arith_intensity": tf / tb if tb else 0.0,
        "compulsory_bytes": tb_compulsory,
        "byte_amplification": tb / tb_compulsory if tb_compulsory else 1.0,
        "n_rep": n_rep,
    }
    return out


def config_summary(cfg):
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": hd,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "kv_bytes_per_token": 2 * cfg.num_key_value_heads * hd * 2
                              * cfg.num_hidden_layers,
    }
