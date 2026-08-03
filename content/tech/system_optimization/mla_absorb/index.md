---
title: "Absorbed MLA vs. Naive MLA: Same Linear FLOPs, 3.4x Core Attention"
slug: "absorbed-mla-vs-naive-mla"
date: 2026-08-02T09:30:00+08:00
draft: false
tags: ["MLA", "DeepSeek", "KV Cache", "FLOPs", "Roofline", "Sparse Attention", "Decode"]
categories: ["System Optimization"]
summary: "Matrix absorption rewrites MLA to attend directly on the compressed latent. A side-by-side PyTorch implementation plus a FLOPs breakdown shows the token-linear cost is bit-for-bit identical while core attention gets 3.4x more expensive — which is exactly why dense causal training keeps the naive form and sparse attention (DSA) training uses the absorbed one."
---

Multi-head Latent Attention (MLA) has two mathematically equivalent execution forms. The **naive** form up-projects the compressed latent back into per-head keys and values and then runs ordinary MHA. The **absorbed** form folds the up-projection matrices into the query and the output, so attention runs directly on the latent, which behaves like a single shared (MQA-style) KV head.

They produce identical outputs from identical weights. They do *not* have identical cost, and the difference is sharply localized:

* **Token-linear FLOPs are exactly the same.** Absorption moves work around; it does not remove it.
* **Core attention FLOPs differ by a fixed factor** — $3.4\times$ for the DeepSeek-V3 shape, independent of head count and sequence length.
* **Memory traffic differs by ~$71\times$** on the same shape, in the opposite direction.

That asymmetry is the entire decision rule. Dense causal attention is core-attention-dominated, so it keeps the naive form. Decoding and sparse attention are memory-dominated, so they take the absorbed form and eat the FLOPs.

## The Absorption Identity

MLA caches a compressed latent $\mathbf{c}^{KV}_t \in \mathbb{R}^{d_c}$ per token and reconstructs keys and values from it with per-head up-projections $W^{UK}_h$ and $W^{UV}_h$. The non-positional part of the score for head $h$ is

{{< math >}}
$$
\begin{aligned}
    s_{h,ij}^{\text{nope}}
      &= \left( W^{Q,\text{nope}}_h \mathbf{h}_i \right)^\top \left( W^{UK}_h \mathbf{c}^{KV}_j \right) \\
      &= \underbrace{\left( \left(W^{UK}_h\right)^\top W^{Q,\text{nope}}_h \mathbf{h}_i \right)}_{\text{absorbed query } \tilde{\mathbf{q}}_{h,i} \in \mathbb{R}^{d_c}}{}^{\top} \; \mathbf{c}^{KV}_j
\end{aligned}
$$
{{< /math >}}

Matrix multiplication is associative, so $W^{UK}_h$ can be applied to the query instead of the key. The key never has to exist. Symmetrically on the output side:

{{< math >}}
$$
\mathbf{o}_{h,i} = \sum_j p_{h,ij} \left( W^{UV}_h \mathbf{c}^{KV}_j \right) = W^{UV}_h \left( \sum_j p_{h,ij}\, \mathbf{c}^{KV}_j \right)
$$
{{< /math >}}

so $W^{UV}_h$ moves outside the attention sum and gets applied once per query token rather than once per cached token.

Two things do *not* absorb. The RoPE part of the key ($k_{pe}$, shared across heads) has a position-dependent rotation between $W^Q$ and $W^{UK}$, which breaks associativity — that is precisely why MLA carries a decoupled RoPE dimension at all, and why the absorbed score is computed as two separate matmuls that get summed. And the softmax sits between $W^{UK}$ and $W^{UV}$, so the two absorptions are independent of each other.

## Reference Implementation

Both paths below share `_project` (the query/latent/RoPE front end) and the same weights, so the outputs must match numerically. `forward_naive` materializes full-size K/V; `forward_absorbed` folds `W_UK` into the query and `W_UV` into the output. Every line carries its tensor shape.

```python
"""Minimal comparison: naive MLA vs. absorbed (matrix-absorption) MLA.

Shape names used in the trailing comments (default config in __main__):
    batch                      = 2
    seq       (seq_len)        = 512
    hidden    (hidden_size)    = 2048
    heads     (num_heads)      = 16
    q_lora    (q_lora_rank)    = 512
    kv_lora   (kv_lora_rank)   = 512     <- the compressed latent dim
    nope      (qk_nope_head_dim) = 128
    rope      (qk_rope_head_dim) = 64
    qk_dim    (nope + rope)    = 192     <- full per-head query/key dim
    v_dim     (v_head_dim)     = 128

Naive MLA:
    latent c_kv [batch, seq, kv_lora] --W_UK--> k_nope [batch, heads, seq, nope]
                                      --W_UV--> v      [batch, heads, seq, v_dim]
    then standard MHA over full-size k/v.  KV cache = k [heads, qk_dim] + v [heads, v_dim].

Absorbed MLA (what vLLM / SGLang do at decode):
    W_UK is folded into the query:  q' = q_nope @ W_UK   -> [batch, heads, seq, kv_lora]
    W_UV is folded into the output: o  = (attn @ c_kv) @ W_UV
    attention runs directly on the latent c_kv, which acts as a *single* shared
    (MQA-style) kv head.  KV cache = c_kv [kv_lora] + k_pe [rope] only.

Both paths use the exact same weights, so outputs must match numerically.
"""

import torch
from torch import nn


# --------------------------------------------------------------------------------------
# RoPE helpers (same as the HF DeepseekV3 implementation, non-interleaved variant)
# --------------------------------------------------------------------------------------
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # x [.., rope] -> x1 [.., rope/2], x2 [.., rope/2]
    return torch.cat((-x2, x1), dim=-1)  # -> [.., rope]


def apply_rope(q, k, cos, sin):
    # q [batch, heads, seq, rope] | k [batch, 1, seq, rope] | cos, sin [1, seq, rope]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [1, seq, rope] -> [1, 1, seq, rope]  (broadcasts over batch/heads)
    q = q * cos + rotate_half(q) * sin  # [batch,heads,seq,rope] * [1,1,seq,rope] -> [batch, heads, seq, rope]
    k = k * cos + rotate_half(k) * sin  # [batch,1,seq,rope]     * [1,1,seq,rope] -> [batch, 1, seq, rope]
    return q, k


def build_rope(seq_len, dim, device, dtype, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))  # [rope/2]
    t = torch.arange(seq_len, device=device).float()  # [seq]
    freqs = torch.outer(t, inv_freq)  # [seq] x [rope/2] -> [seq, rope/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # -> [seq, rope]
    return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # each [1, seq, rope]


# --------------------------------------------------------------------------------------
# MLA
# --------------------------------------------------------------------------------------
class MLA(nn.Module):
    def __init__(
        self,
        hidden_size=2048,  # hidden
        num_heads=16,  # heads
        q_lora_rank=512,  # q_lora
        kv_lora_rank=512,  # kv_lora
        qk_nope_head_dim=128,  # nope
        qk_rope_head_dim=64,  # rope
        v_head_dim=128,  # v_dim
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # qk_dim = nope + rope
        self.v_head_dim = v_head_dim
        self.scaling = self.qk_head_dim**-0.5  # scalar, 1/sqrt(qk_dim)

        # query: down-proj then up-proj (LoRA style)
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)  # weight [q_lora, hidden]
        self.q_a_layernorm = nn.RMSNorm(q_lora_rank)  # weight [q_lora]
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)  # weight [heads*qk_dim, q_lora]

        # kv: one down-proj producing [latent | k_pe], plus the up-proj
        self.kv_a_proj_with_mqa = nn.Linear(  # weight [kv_lora+rope, hidden]
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank)  # weight [kv_lora]
        self.kv_b_proj = nn.Linear(  # weight [heads*(nope+v_dim), kv_lora]
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)  # weight [hidden, heads*v_dim]

    # ---- shared front part: q, latent c_kv, k_pe ---------------------------------------
    def _project(self, hidden_states, cos, sin):
        b, s = hidden_states.shape[:2]  # hidden_states [batch, seq, hidden]

        # [batch,seq,hidden] @ [hidden,q_lora] -> [batch,seq,q_lora];  RMSNorm keeps shape;
        # then [batch,seq,q_lora] @ [q_lora,heads*qk_dim] -> [batch,seq,heads*qk_dim]
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))  # [batch, seq, heads*qk_dim]
        q = q.view(b, s, self.num_heads, self.qk_head_dim)  # -> [batch, seq, heads, qk_dim]
        q = q.transpose(1, 2)  # -> [batch, heads, seq, qk_dim]
        # split the last dim qk_dim into the non-positional part and the RoPE part
        q_nope, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope [batch, heads, seq, nope] | q_rot [batch, heads, seq, rope]

        # [batch,seq,hidden] @ [hidden,kv_lora+rope] -> [batch, seq, kv_lora+rope]
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        c_kv, k_rot = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # c_kv [batch, seq, kv_lora] | k_rot [batch, seq, rope]
        c_kv = self.kv_a_layernorm(c_kv)  # [batch, seq, kv_lora]  <- THIS is the whole KV cache in absorbed mode
        k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)  # -> [batch, 1, seq, rope]  (the 1 = shared across heads)

        q_rot, k_rot = apply_rope(q_rot, k_rot, cos, sin)  # shapes unchanged
        return q_nope, q_rot, c_kv, k_rot

    # ---- split kv_b_proj weight into the two absorbable matrices -----------------------
    @property
    def W_UK_UV(self):
        # kv_b_proj.weight [heads*(nope+v_dim), kv_lora] -> [heads, nope+v_dim, kv_lora]
        w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        W_UK = w[:, : self.qk_nope_head_dim, :]  # [heads, nope,  kv_lora]   latent -> key   up-projection
        W_UV = w[:, self.qk_nope_head_dim :, :]  # [heads, v_dim, kv_lora]   latent -> value up-projection
        return W_UK, W_UV

    # ---- path 1: naive (materialize full k/v) -----------------------------------------
    def forward_naive(self, hidden_states, cos, sin):
        b, s = hidden_states.shape[:2]
        q_nope, q_rot, c_kv, k_rot = self._project(hidden_states, cos, sin)
        # q_nope [batch,heads,seq,nope] | q_rot [batch,heads,seq,rope]
        # c_kv   [batch,seq,kv_lora]    | k_rot [batch,1,seq,rope]

        # up-project the latent into per-head keys and values
        kv = self.kv_b_proj(c_kv)  # [batch,seq,kv_lora] @ [kv_lora, heads*(nope+v_dim)] -> [batch, seq, heads*(nope+v_dim)]
        kv = kv.view(b, s, self.num_heads, -1)  # -> [batch, seq, heads, nope+v_dim]
        kv = kv.transpose(1, 2)  # -> [batch, heads, seq, nope+v_dim]
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k_nope [batch, heads, seq, nope] | value_states [batch, heads, seq, v_dim]

        k_rot_e = k_rot.expand(-1, self.num_heads, -1, -1)  # [batch,1,seq,rope] -> [batch, heads, seq, rope]  (no copy)
        key_states = torch.cat((k_nope, k_rot_e), dim=-1)  # [..,nope] + [..,rope] -> [batch, heads, seq, qk_dim]
        query_states = torch.cat((q_nope, q_rot), dim=-1)  # [..,nope] + [..,rope] -> [batch, heads, seq, qk_dim]

        # scores: contract over qk_dim = 192
        attn = query_states @ key_states.transpose(-1, -2)
        # [batch,heads,seq,qk_dim] @ [batch,heads,qk_dim,seq] -> [batch, heads, seq, seq]
        attn = attn * self.scaling  # [batch, heads, seq, seq]
        attn = attn + causal_mask(s, attn.dtype, attn.device)  # + [seq,seq] -> [batch, heads, seq, seq]
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(query_states.dtype)  # [batch, heads, seq, seq]

        out = attn @ value_states  # [batch,heads,seq,seq] @ [batch,heads,seq,v_dim] -> [batch, heads, seq, v_dim]
        out = out.transpose(1, 2)  # -> [batch, seq, heads, v_dim]
        out = out.reshape(b, s, -1)  # -> [batch, seq, heads*v_dim]
        return self.o_proj(out)  # [batch,seq,heads*v_dim] @ [heads*v_dim, hidden] -> [batch, seq, hidden]

    # ---- path 2: absorbed (attend directly on the latent) ------------------------------
    def forward_absorbed(self, hidden_states, cos, sin):
        b, s = hidden_states.shape[:2]
        q_nope, q_rot, c_kv, k_rot = self._project(hidden_states, cos, sin)
        # q_nope [batch,heads,seq,nope] | q_rot [batch,heads,seq,rope]
        # c_kv   [batch,seq,kv_lora]    | k_rot [batch,1,seq,rope]
        W_UK, W_UV = self.W_UK_UV  # W_UK [heads, nope, kv_lora] | W_UV [heads, v_dim, kv_lora]

        # absorb W_UK into the query: contract over nope, so the query now lives in latent space
        q_absorbed = torch.einsum("bhsd,hdr->bhsr", q_nope, W_UK)
        # [batch,heads,seq,nope] x [heads,nope,kv_lora] -> [batch, heads, seq, kv_lora]

        # c_kv / k_rot behave as ONE shared kv head -> scores are two MQA-style matmuls
        c_kv_h = c_kv.unsqueeze(1)  # [batch,seq,kv_lora] -> [batch, 1, seq, kv_lora]  (broadcasts over heads)
        attn = q_absorbed @ c_kv_h.transpose(-1, -2)
        # [batch,heads,seq,kv_lora] @ [batch,1,kv_lora,seq] -> [batch, heads, seq, seq]   (contract kv_lora = 512)
        attn = attn + q_rot @ k_rot.transpose(-1, -2)
        # [batch,heads,seq,rope] @ [batch,1,rope,seq] -> [batch, heads, seq, seq], added to the above
        attn = attn * self.scaling  # [batch, heads, seq, seq]
        attn = attn + causal_mask(s, attn.dtype, attn.device)  # + [seq,seq] -> [batch, heads, seq, seq]
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(q_nope.dtype)  # [batch, heads, seq, seq]

        # values live in the latent space too; absorb W_UV on the way out
        out = attn @ c_kv_h  # [batch,heads,seq,seq] @ [batch,1,seq,kv_lora] -> [batch, heads, seq, kv_lora]
        out = torch.einsum("bhsr,hdr->bhsd", out, W_UV)
        # [batch,heads,seq,kv_lora] x [heads,v_dim,kv_lora] -> [batch, heads, seq, v_dim]   (contract kv_lora)
        out = out.transpose(1, 2)  # -> [batch, seq, heads, v_dim]
        out = out.reshape(b, s, -1)  # -> [batch, seq, heads*v_dim]
        return self.o_proj(out)  # [batch,seq,heads*v_dim] @ [heads*v_dim, hidden] -> [batch, seq, hidden]


def causal_mask(s, dtype, device):
    # [seq, seq] additive mask: 0 on/below the diagonal, -inf above
    return torch.full((s, s), torch.finfo(dtype).min, device=device, dtype=dtype).triu(1)


# --------------------------------------------------------------------------------------
# FLOPs accounting, following Megatron's `num_floating_point_operations` convention
# --------------------------------------------------------------------------------------
FMA_EXPANSION = 2  # one multiply-accumulate = 2 FLOPs
FWD_BWD_EXPANSION = 3  # fwd (1x) + bwd dgrad (1x) + bwd wgrad (1x); use 1 for inference-only


def mla_flops(model, seq_len, absorbed=False, training=True):
    """FLOPs per token per layer.  Returns (token_linear, core_attention)."""
    h, hidden = model.num_heads, model.o_proj.weight.shape[0]
    nope, rope, v_dim = model.qk_nope_head_dim, model.qk_rope_head_dim, model.v_head_dim
    q_lora = model.q_a_proj.weight.shape[0]
    kv_lora = model.kv_lora_rank
    qk_dim = nope + rope

    expansion = FMA_EXPANSION * (FWD_BWD_EXPANSION if training else 1)

    # ---- token-linear part: identical for both paths (see note below) ------------------
    linear = (
        q_lora * (hidden + h * qk_dim + 1)  # q_a_proj + q_b_proj + q_a_layernorm
        + kv_lora * (hidden + h * (nope + v_dim) + 1)  # kv_a_proj(latent) + kv_b_proj/W_UK+W_UV + kv_a_layernorm
        + hidden * rope  # kv_a_proj, the k_pe slice
        + h * v_dim * hidden  # o_proj
    )

    # ---- core (L^2) part: THIS is where the two paths differ ---------------------------
    if absorbed:
        # QK contracts kv_lora (+ rope, computed separately); PV contracts kv_lora.
        # NOTE the `h`: c_kv has only ONE kv head, but the score matrix is still [heads, seq, seq]
        # and every head has its own q_absorbed -- broadcasting the single kv head saves BYTES
        # (that is the whole point of MQA at decode), not FLOPs.
        core = h * (kv_lora + rope) / 2 + h * kv_lora / 2
    else:
        # QK contracts qk_dim = nope + rope; PV contracts v_dim
        core = h * qk_dim / 2 + h * v_dim / 2  # /2 for the causal mask

    return expansion * linear, expansion * core * seq_len


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model = MLA().to(device=device, dtype=dtype).eval()
    b, s = 2, 512
    hidden_states = torch.randn(b, s, 2048, device=device, dtype=dtype)  # [batch, seq, hidden]
    cos, sin = build_rope(s, model.qk_rope_head_dim, device, dtype)  # each [1, seq, rope]

    with torch.no_grad():
        out_naive = model.forward_naive(hidden_states, cos, sin)  # [batch, seq, hidden]
        out_absorbed = model.forward_absorbed(hidden_states, cos, sin)  # [batch, seq, hidden]

    print(f"max abs diff : {(out_naive - out_absorbed).abs().max().item():.3e}")
    print(f"allclose     : {torch.allclose(out_naive, out_absorbed, atol=1e-3, rtol=1e-3)}")

    # ---- FLOPs per token per layer (Megatron convention: 3x fwd/bwd, 2x FMA) -----------
    lin_n, core_n = mla_flops(model, s, absorbed=False)
    lin_a, core_a = mla_flops(model, s, absorbed=True)
    print(f"\ntraining FLOPs / token / layer  (seq_len = {s})")
    print(f"  naive     linear {lin_n / 1e6:8.1f} M   core {core_n / 1e6:8.1f} M   total {(lin_n + core_n) / 1e6:8.1f} M")
    print(f"  absorbed  linear {lin_a / 1e6:8.1f} M   core {core_a / 1e6:8.1f} M   total {(lin_a + core_a) / 1e6:8.1f} M")
    print(f"  core ratio absorbed/naive : {core_a / core_n:.2f}x")
```

## FLOPs: Where the Two Paths Agree

The `linear` term in `mla_flops` has no `absorbed` branch, and that is not an oversight. Count the up-projection work on both sides, per token per layer:

| | naive | absorbed |
|---|---|---|
| $W^{UK}$ | `kv_b_proj` on the latent: $h \cdot d_{nope} \cdot d_c$ per **KV** token | einsum into `q_nope`: $h \cdot d_{nope} \cdot d_c$ per **query** token |
| $W^{UV}$ | inside `kv_b_proj`: $h \cdot d_v \cdot d_c$ per **KV** token | einsum on the output: $h \cdot d_v \cdot d_c$ per **query** token |

Same weights, same contraction dims, same count. Absorption relocates the multiplication from one operand to the other; it does not delete a GEMM. Everything else — `q_a_proj`, `q_b_proj`, the latent slice of `kv_a_proj_with_mqa`, the two RMSNorms, `o_proj` — is untouched by absorption and is shared by both paths verbatim.

The equality holds because $n_q = n_{kv}$ during training and prefill. **At decode it collapses**: $n_q = 1$ while $n_{kv} = S$. The naive path must up-project the entire cache each step ($S \cdot h \cdot d_c (d_{nope}+d_v)$), the absorbed path projects one query token. That is a different regime, discussed below.

## FLOPs: Where They Diverge

Core attention is the $O(S^2)$ part — the two matmuls per (query, key) pair. Their contraction dimensions are what absorption changes:

| | QK contracts | PV contracts | per pair per head |
|---|---|---|---|
| naive | $d_{nope} + d_{rope} = 192$ | $d_v = 128$ | $320$ |
| absorbed | $d_c + d_{rope} = 576$ | $d_c = 512$ | $1088$ |

$$R_{core} = \frac{2 d_c + d_{rope}}{d_{nope} + d_{rope} + d_v} = \frac{1024 + 64}{128 + 64 + 128} = 3.4$$

Two properties of this ratio are worth stating explicitly, because they are what make the decision rule simple:

1. **It is independent of head count.** The latent is a single shared KV head, but the score matrix is still $[h, S, S]$ and every head still has its own absorbed query. Broadcasting one KV head across $h$ query heads saves *bytes*, not *FLOPs* — the same reason MQA is a decode-time win and not a training-time one.
2. **It is independent of sequence length.** It is a pure ratio of contraction dims. What changes with $S$ is how much core attention *matters*.

For the DeepSeek-V3 shape ($h=128$, $d_{model}=7168$, $d_c=512$, $d_{q\_lora}=1536$), token-linear is $1122.6$ M FLOPs/token/layer (Megatron convention: $2\times$ FMA, $3\times$ fwd+bwd) and core is $6 \cdot h \cdot 320 \cdot S/2$:

| $S$ | naive total | absorbed total | absorbed/naive | core share (naive) |
|---:|---:|---:|---:|---:|
| 4 K | 1626 M | 2834 M | 1.74x | 31.0% |
| 32 K | 5149 M | 14813 M | 2.88x | 78.2% |
| 128 K | 17229 M | 55884 M | 3.24x | 93.5% |

The overhead starts modest and asymptotes to $3.4\times$: at short sequences the constant token-linear term dominates and hides the penalty, at long sequences core attention *is* the model and the full $3.4\times$ lands on the total. Breakeven — the point where the extra core cost equals the entire linear cost — is around $S \approx 1.6$ K for the toy config in the script. Beyond that, absorption is a straight tax on training throughput.

## Memory: The Other Direction

Per token per layer, in BF16, at $h = 128$:

* naive K/V: $h(d_{nope} + d_{rope} + d_v) = 40960$ elements
* absorbed latent: $d_c + d_{rope} = 576$ elements

A **71x** difference. In absolute terms, materializing K/V for one 128 K-token sequence costs **10.7 GB per layer**, against **151 MB** for the latent. This is the number that decides everything below.

## When to Use Which

### Dense causal attention, training and prefill → naive

Core attention dominates ($93\%$ of layer FLOPs at 128 K), so a $3.4\times$ core penalty is a $3.2\times$ penalty on the whole layer. Unaffordable.

The memory argument does not rescue absorption here, because a tiled kernel never pays it. FlashAttention loads a K/V tile into SRAM and *every query in the block reuses that tile*. So the naive path can up-project each latent tile inside SRAM (or recompute it in the backward pass) and the $71\times$ blow-up never reaches HBM. Dense attention amortizes the up-projection across the whole query block for free — which is exactly the condition under which absorption has nothing to offer and only costs.

### Decode → absorbed

Both terms flip. Core attention is $[h, 1, S]$ — a GEMV, negligible either way and hopelessly memory-bound. What dominates is streaming the KV cache: $71\times$ less traffic per cached token, at an arithmetic intensity where FLOPs are free. And the linear-term equality breaks in absorption's favor: the naive path would have to re-run `kv_b_proj` over all $S$ cached tokens every single step, while the absorbed path folds $W^{UK}$ into a single query vector. This is why vLLM and SGLang run absorbed MLA at decode and naive MLA at prefill, in the same server.

### Sparse attention (DSA) training → absorbed

This is the interesting case, and the one where the FLOPs table is the wrong table to read.

Under top-$k$ sparse attention, core attention stops being $O(S^2)$ and becomes $O(S \cdot k)$. At $S=128$ K and $k=2048$ that is a $32\times$ reduction — core attention falls back to $\approx 31\%$ of the layer, and the absorbed penalty on the *total* drops from $3.24\times$ to $1.74\times$. The $3.4\times$ is still there, but it now applies to a term that no longer dominates.

Meanwhile the property that made naive cheap for dense attention disappears. In sparse attention **every query selects its own key set**, so a tile of queries no longer shares a tile of keys. There is nothing to amortize the up-projection across. The naive path is left with two bad options: up-project all $S$ tokens once into HBM and gather from that (paying the full $71\times$ on materialization *and* on every gather — 10.7 GB per layer per sequence), or up-project per (query, selected-key) pair, which multiplies the linear term by $k$ and is far worse than $3.4\times$.

The absorbed path gathers a single 576-dim latent per selected token, shared across all heads, and that is the whole KV footprint. **GLM-5.2 trains its DSA with absorbed MLA for this reason** — the gather is $71\times$ cheaper and the score matmul's larger contraction dim is compensated by the sparsity. The same model's plain causal MLA layers do **not** use the absorbed form, because there the amortization argument holds and the $3.4\times$ is pure loss.

## Summary

| regime | core attn share | dominant cost | form |
|---|---|---|---|
| dense causal, train / prefill | high, $O(S^2)$ | core FLOPs | **naive** — $3.4\times$ core penalty unaffordable; tiling hides the $71\times$ memory cost |
| decode | negligible, $O(S)$ GEMV | KV cache bandwidth | **absorbed** — $71\times$ less traffic, FLOPs are free |
| sparse (DSA), train | reduced, $O(Sk)$ | gather bandwidth | **absorbed** — no cross-query amortization to lose; $3.4\times$ applies to a small term |

The one-line version: **absorption trades a fixed $3.4\times$ on core attention FLOPs for a fixed $71\times$ on KV memory traffic, at zero change to token-linear FLOPs.** Pick the side of that trade your bottleneck is on. For the roofline machinery behind "which side am I on", see [Roofline Analysis of LLMs on H200](/roofline-analysis-of-llms-on-h200-performance-modeling-and-recomputation-strategies/); for how tiled attention kernels avoid materializing anything, see [Demystifying FlashAttention](/demystifying-flashattention-forward-backward-and-triton-implementation/).