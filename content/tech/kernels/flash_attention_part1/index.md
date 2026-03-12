---
title: "Demystifying FlashAttention: Forward, Backward, and Triton Implementation"
date: 2026-03-11T23:30:00+08:00
draft: true
tags: ["Triton", "Flash Attention"]
categories: ["Kernels"]
summary: "A breakdown of FlashAttention's forward and backward passes, including Online Softmax, LogSumExp materialization, gradient recomputation, and core Triton implementations."
---

## Forward Pass

### Formulation

{{< math >}}
$$
\begin{align}
    \mathbf{S} &= \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N} \\
    \mathbf{P} &= \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N} \\
    \mathbf{O} &= \mathbf{P} \mathbf{V} \in \mathbb{R}^{N \times d}
\end{align}
$$
{{< /math >}}

where $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O} \in \mathbb{R}^{N \times d}$, $N$ is the sequence length, and $d$ is the head dimension.

### Trick: Online softmax

To avoid materializing the $N \times N$ matrix $\mathbf{S}$ in High Bandwidth Memory (HBM), the softmax operation is computed in blocks. 
Let vectors be processed in blocks $1$ and $2$. The standard softmax requires the global maximum and the sum of exponentials. The online update rules to transition from block 1 to block 2 are:

1. Update Local Maximum:

$$m^{(new)} = \max(m^{(1)}, m^{(2)})$$

where $m^{(1)} = \max(\mathbf{S}^{(1)})$ and $m^{(2)} = \max(\mathbf{S}^{(2)})$.

2. Update Sum of Exponentials ($l$) and Accumulator ($\mathbf{O}$):

To correct the previously computed values using the new global maximum, a decay factor (rescaling factor) is applied:

$$l^{(new)} = l^{(1)} e^{m^{(1)} - m^{(new)}} + \sum e^{\mathbf{S}^{(2)} - m^{(new)}}$$

$$\mathbf{O}^{(new)} = \mathbf{O}^{(1)} e^{m^{(1)} - m^{(new)}} + e^{\mathbf{S}^{(2)} - m^{(new)}} \mathbf{V}^{(2)}$$

Finally, the normalized output is $\mathbf{O} = \frac{\mathbf{O}^{(new)}}{l^{(new)}}$. For the backward pass, the LogSumExp (LSE) is saved to HBM:

$$LSE = m^{(new)} + \log(l^{(new)})$$

### Triton Implementation

The following snippet extracts the core inner loop of the forward pass from the official Triton tutorial. It iterates over blocks of $\mathbf{K}$ and $\mathbf{V}$ to update the state of a $\mathbf{Q}$ block in SRAM. The exponent base is optimized to 2 (`exp2`).

```python
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                    start_m, qk_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr):
    # Determine bounds based on causality stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
        
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    # Inner loop over blocks of K and V
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        qk = tl.dot(q, k)
        
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, 1))
            qk = qk * qk_scale - m_ij[:, None]
            
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        # Online Softmax update rules
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Update Output Accumulator
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij
        
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        
    return acc, l_i, m_i
```

## Backward Pass

### Formulation

{{< math >}}
$$
\begin{align}
  \mathbf{dV} &= \mathbf{P}^\top \mathbf{dO} \in \mathbb{R}^{N \times d} \\
  \mathbf{dP} &= \mathbf{dO} \mathbf{V}^\top \in \mathbb{R}^{N \times N} \\
  \mathbf{dS} &= \mathbf{dsoftmax (\mathbf{dP})} \in \mathbb{R}^{N \times N} \\
  &= \mathbf{P} \odot (\mathbf{dP} - \text{row\_sum}( \mathbf{dP} \odot \mathbf{P}))\\
  \mathbf{dQ} &= \mathbf{dS} \mathbf{K} \in \mathbb{R}^{N \times d} \\
  \mathbf{dK} &= \mathbf{dS}^\top \mathbf{Q} \in \mathbb{R}^{N \times d},
\end{align}
$$
{{< /math >}}


### preprocess

Important Trick

{{< math >}}
$$
\begin{aligned}
\mathbf{D}_i &= \sum_j \mathbf{dP}_{ij} \mathbf{P}_{ij} \\
&= \sum_j \left(\sum_k \mathbf{dO}_{ik} \mathbf{V}^\top_{kj}\right) \mathbf{P}_{ij} \\
&= \sum_k \mathbf{dO}_{ik} \sum_j (\mathbf{V}^\top_{kj} \mathbf{P}_{ij}) \\
&= \sum_k \mathbf{dO}_{ik} \sum_j (\mathbf{P}_{ij} \mathbf{V}_{jk}) \\
&= \sum_k \mathbf{dO}_{ik} \mathbf{O}_{ik}
\end{aligned}
$$
{{< /math >}}


### dq Part

### dkdv Part