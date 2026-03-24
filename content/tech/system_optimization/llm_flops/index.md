---
title: "Beyond Theoretical FLOPs: Analyzing MFU, HFU, and Attention Overhead in Transformers"
date: 2026-01-31T09:30:00+08:00
draft: false
tags: ["MFU", "HFU", "FLOPs", "GEMM", "Flash Attention"]
categories: ["System Optimization"]
summary: "A rigorous breakdown of FLOPs in Llama-style architectures, deriving the relationship between linear projections and quadratic attention overhead, with insights into sample packing efficiency."
---

Large Language Model (LLM) training efficiency is often distilled into the rule of thumb: $6CP$ for training FLOPs. However, as sequence lengths push into the hundreds of thousands, the "hidden" quadratic cost of attention begins to eclipse the linear projections of the model parameters. 

---

## MFU vs. HFU: Defining Compute Efficiency

Before calculating time distributions, we must strictly separate theoretical compute from actual hardware execution
* **MFU (Model FLOPs Utilization)**: Measures the absolute algorithmic throughput. Its numerator includes only the theoretical minimum FLOPs required for a forward and backward pass, strictly excluding any extra computation introduced to save memory (e.g., Activation Checkpointing).
* **HFU (Hardware FLOPs Utilization)**: Measures the actual computational density on the GPU. Its numerator includes all FLOPs executed by the hardware, including the recomputation of forward activations during the backward pass.

In standard Linear (Dense Matmul) operators, MFU and HFU are nearly identical (typically around **75%**). In FlashAttention (FA), however, their relationship depends strictly on the execution phase: during the forward pass, $\text{MFU} = \text{HFU}$ (typically around **65%**), but during the backward pass, they diverge to $\text{MFU} = \frac{4}{5} \text{HFU}$. This mathematical split occurs because FA recomputes the $O(N^2)$ attention weights ($QK^T$) during the backward pass to avoid massive HBM writes, resulting in $10\text{ssh}$ of actual executed work versus the $8\text{ssh}$ theoretical baseline. Consequently, if the backward pass achieves a physical HFU of **62.5%**, its effective algorithmic throughput (MFU) drops to **50%**.

## FLOPs Breakdown per Transformer Block

Taking a Llama-style architecture (using SwiGLU and standard MHA) as our baseline, we calculate the forward pass FLOPs for a single sequence of length $s$.

### Linear Operations (GEMM)

Note that the FLOPs for a matrix multiplication $(M, K) \times (K, N)$ is calculated as $2MKN$.

| Operation           | Input × Weight   | FLOPs (Forward) |
|---------------------|------------------|-----------------|
| Q, K, V Projections | $(s,h) \times (h,3h)$     | $6sh^2$            |
| Attention Output    | $(s,h) \times (h,h)$      | $2sh^2$            |
| MLP (SwiGLU)        | $(s,h) \times (h,\frac{8}{3}​h) \times 3$ | $16sh^2$           |
| Total Linear        | —                | $24sh^2$           |

From this table, we can see that for Linear operations:

$$\text{FLOPs}_{GEMM} = 2 \cdot s \cdot (12h^2) = 2 \cdot s \cdot P$$

Summing across all layers, where $C$ is the total tokens ($C = \sum s$) and $P$ is the total parameters, the total Linear GEMM FLOPs is $2CP$.


### Attention Mechanism

The Unlike linear layers, the self-attention core exhibits quadratic scaling with respect to sequence length.


| Operation           | Calculation        | FLOPs (Forward) |
|---------------------|--------------------|-----------------|
| $QK^T$ (Score)         | $(s,h) \times (h,s)$        | $2s^2h$            |
| Softmax             | $e^{x_i}​ / \sum{e^{x_j}​}$ | $\sim 5s^2$ [^1]      |
| $Score \cdot V$             | $(s,s)×(s,h)$        | $2s^2h$            |
| Total Attention | —                  | $4s^2h$            |


[^1]: Why 5 for Softmax? In practice, Softmax is computed as $\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum \exp(x_j - \max(x))}$ to ensure numerical stability. For each row of length $s$, this involves five $O(s)$ steps: 1) Finding the max ($1s$), 2) Subtraction ($1s$), 3) Exponential ($1s$), 4) Summation ($1s$), and 5) Division ($1s$), totaling approximately $5s^2$ FLOPs. Since $5 \ll h$, this term is negligible compared to the $2s^2h$ from attention GEMMs.

## The $s/6h$ Derivation

To find the relationship between these two parts, we calculate the ratio of Attention FLOPs to Linear FLOPs:

$$\text{Ratio} = \frac{\text{Attn FLOPs}}{\text{Linear FLOPs}} = \frac{4s^2h + 5s^2}{24sh^2} \approx \frac{4s^2h}{24sh^2} = \frac{s}{6h}$$

By applying this ratio, we arrive at the unified estimation for Total Forward FLOPs:

$$\text{FLOPs} \approx 2 s P \times \left(1 + \frac{s}{6h}\right)$$

### Theoretical Attention Overhead (Ignoring MFU)

The $\frac{s}{6h}$ term represents the Attention Overhead — the additional compute required beyond the standard parameter-driven projections.
Assuming all FLOPs execute at 100% efficiency, the table below illustrates how this quadratic overhead scales across mainstream architectures:

| Model                   | h (Hidden Size) | s (Sequence Length) | s/6h Ratio (Quadratic Overhead) |
|-------------------------|-----------------|---------------------|---------------------------------|
| Llama-3-8B              | 4,096           | 8,192               | ~33.3%                          |
| Llama-3-70B             | 8,192           | 8,192               | ~16.7%                          |
| Long-Context (Standard) | 4,096           | 32,768              | ~133.3%                         |
| Ultra-Long Context      | 4,096           | 131,072             | ~533.3% (Attention Dominates)   |

### Realistic Attention Overhead (Accounting for MFU)

Theoretical FLOPs fail to reflect actual wall-clock time due to the differing Hardware FLOPs Utilization (HFU) and Model FLOPs Utilization (MFU) of individual operators.

Execution time $T$ is inversely proportional to MFU: $T \propto \frac{\text{Theoretical FLOPs}}{\text{MFU}}$.
Using typical baseline efficiencies (GEMM $\approx 75\%$, FA Forward $\approx 65\%$, FA Backward MFU $\approx 50\%$ due to recomputation), we can calculate the realistic time distribution for a full training iteration (Forward + Backward).

**Linear (GEMM) Time Calculation:**
Total theoretical Linear FLOPs are $72sh^2$ (Forward $24sh^2$, Backward $48sh^2$).
$$T_{\text{linear}} \propto \frac{72sh^2}{0.75} = 96sh^2$$

**FlashAttention Time Calculation:**
Total theoretical FA FLOPs are $12s^2h$ (Forward $4s^2h$, Backward $8s^2h$).

$$T_{\text{attn}} \propto \frac{4s^2h}{0.65} + \frac{8s^2h}{0.50} \approx 6.15s^2h + 16.0s^2h = 22.15s^2h$$

**Realistic Time Ratio:**
$$R_{\text{real}} = \frac{T_{\text{attn}}}{T_{\text{linear}}} = \frac{22.15s^2h}{96sh^2} \approx 0.231 \frac{s}{h}$$

Comparing $R_{\text{real}}$ ($0.231 \frac{s}{h}$) to the theoretical ratio $R_{\text{theo}}$ ($0.167 \frac{s}{h}$), the realistic time overhead is approximately **1.38x** the theoretical compute overhead. The extreme MFU penalty in the FA backward pass disproportionately expands the actual time spent on Attention.

| Model                   | h (Hidden Size) | s (Sequence Length) | 0.231 s/h Ratio (Quadratic Overhead) |
|-------------------------|-----------------|---------------------|---------------------------------|
| Llama-3-8B              | 4,096           | 8,192               | ~46.2%                          |
| Llama-3-70B             | 8,192           | 8,192               | ~23.1%                          |
| Long-Context (Standard) | 4,096           | 32,768              | ~184.8%                         |
| Ultra-Long Context      | 4,096           | 131,072             | ~739.2% (Severe Bottleneck)   |


## Sample Isolation (Packing)

In modern training, multiple samples $s_i$ are often packed into one sequence. With sample isolation (e.g., via block-diagonal masking), the quadratic Attention cost only applies within each sample boundary $s_i$.
Given total tokens $C = \sum s_i$, the formula updates to:


$$\text{FLOPs} \approx \sum{ 2 s_i P \times \left(1 + \frac{s_i}{6h}\right) } = 2P \sum{\left(s_i + \frac{s_i^2}{6h}\right)}$$

Factoring out the total token count $C$, we get:

$$\text{FLOPs} \approx 2CP \left( 1 + \frac{1}{6h}\frac{\sum s_i^2}{\sum s_i} \right)$$

The expression $\frac{\sum s_i^2}{\sum s_i}$ represents a weighted mean where the $s_i$ values act as their own weights, **causing longer sequences to dominate the result**.

## Estimating the Overall MFU Upper Bound under Different Recomputation Strategies

The overall Model FLOPs Utilization (MFU) of a training system is the ratio of the theoretical minimum compute to the actual wall-clock time. When different Activation Checkpointing (recomputation) strategies are applied, the denominator (execution time) changes significantly depending on which forward operators are re-executed, resulting in distinct MFU upper bounds.

Let the ratio of sequence length to hidden dimension be $r = s/h$.

**1. Numerator: Total Theoretical FLOPs**

The strict theoretical FLOPs (excluding any recomputation) for a standard Transformer layer (forward + backward) is:

$$\text{FLOPs}_{\text{theo}} = 72sh^2 + 12s^2h = h^3 \cdot r \cdot (72 + 12r)$$

**2. Denominator: Base Hardware Time (No Recomputation)**

Using the previously established HFU baselines (GEMM at 75%, FA Forward at 65%, FA Backward at 50% with actual executed FLOPs of $10s^2h$):
* Base GEMM Forward Time: $\frac{24sh^2}{0.75} = 32sh^2$
* Base GEMM Backward Time: $\frac{48sh^2}{0.75} = 64sh^2$
* Base FA Forward Time: $\frac{4s^2h}{0.65} \approx 6.15s^2h$
* Base FA Backward Time: $\frac{8s^2h}{0.50} = 16s^2h$

### Analytical Solutions for Four Recomputation Strategies

By adding the specific time penalty of recomputing forward operators, we derive the MFU equations for four distinct memory-saving strategies:

**Case 1: No Recomputation**
Requires maximum HBM but executes zero redundant operations.

$$\text{MFU}_{\text{None}} = \frac{72 + 12r}{32 + 64 + 6.15r + 16r} = \frac{72 + 12r}{96 + 22.15r}$$

**Case 2: Recompute FA Only**
Requires executing FA forward passes twice.

$$\text{MFU}_{\text{FA\\_only}} = \frac{72 + 12r}{96 + 22.15r + 6.15r} = \frac{72 + 12r}{96 + 28.3r}$$

**Case 3: Recompute GEMM Only (Selective Checkpointing)**
Requires executing GEMM forward passes twice.

$$\text{MFU}_{\text{Gemm\\_only}} = \frac{72 + 12r}{96 + 22.15r + 32} = \frac{72 + 12r}{128 + 22.15r}$$

**Case 4: Full Activation Checkpointing**
Requires executing both GEMM and FA forward passes twice.

$$\text{MFU}_{\text{Full}} = \frac{72 + 12r}{96 + 22.15r + 32 + 6.15r} = \frac{72 + 12r}{128 + 28.3r}$$

**Practical MFU Upper Bounds at Scale ($h = 4096$)**
Substituting $h = 4096$ and varying the sequence length $s$, we calculate the respective $r$ ratios and map the theoretical MFU upper bounds across all four strategies.

| s (Seq Len) | r=s/h | Case 1: No Recomputation | Case 2: Recompute FA Only | Case 3: Recompute GEMM Only | Case 4: Full Checkpointing |
|-------------|-------|--------------------------|---------------------------|-----------------------------|----------------------------|
| 4,096       | 1     | 71.10                    | 67.58                     | 55.94                       | 53.74                      |
| 8,192       | 2     | 68.42                    | 62.91                     | 55.72                       | 52.00                      |
| 32,768      | 8     | 61.49                    | 52.11                     | 55.05                       | 47.40                      |
| 131,072     | 32    | 56.66                    | 45.53                     | 54.49                       | 44.12                      |

