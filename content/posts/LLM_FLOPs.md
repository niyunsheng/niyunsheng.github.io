---
title: "Transformer FLOPs and Attention Overhead"
date: 2026-01-31T09:30:00+08:00
draft: false
tags: ["LLM", "FLOPs", "AI Infra"]
categories: ["Engineering"]
summary: "A rigorous breakdown of FLOPs in Llama-style architectures, deriving the relationship between linear projections and quadratic attention overhead, with insights into sample packing efficiency."
---

Large Language Model (LLM) training efficiency is often distilled into the rule of thumb: $6CP$ for training FLOPs. However, as sequence lengths push into the hundreds of thousands, the "hidden" quadratic cost of attention begins to eclipse the linear projections of the model parameters. 

---


## FLOPs Breakdown per Transformer Block

Taking a Llama-style architecture (using SwiGLU and standard MHA) as our baseline, we calculate the forward pass FLOPs for a single sequence of length $s$.

### Linear Operations (GEMM)

Note that the FLOPs for a matrix multiplication $(M, K) \times (K, N)$ is calculated as $2MKN$.

| Operation           | Input × Weight   | FLOPs (Forward) |
|---------------------|------------------|-----------------|
| Q, K, V Projections | $(s,h) \times (h,3h)$     | $6sh^2$[^1]            |
| Attention Output    | $(s,h) \times (h,h)$      | $2sh^2$            |
| MLP (SwiGLU)        | $(s,h) \times (h,\frac{8}{3}​h) \times 3$ | $16sh^2$           |
| Total Linear        | —                | $24sh^2$           |

From this table, we can see that for Linear operations:

$$\text{FLOPs}_{GEMM} = 2 \cdot s \cdot (12h^2) = 2 \cdot s \cdot P$$

Summing across all layers, where $C$ is the total tokens ($C = \sum s$) and $P$ is the total parameters, the total Linear GEMM FLOPs is $2CP$.


### Attention MechanismThe

Unlike linear layers, the self-attention core exhibits quadratic scaling with respect to sequence length.


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

### Impact of Attention Overhead

The $\frac{s}{6h}$ term represents the Attention Overhead — the additional compute required beyond the standard parameter-driven projections. The table below illustrates how this overhead scales across mainstream architectures:

| Model                   | h (Hidden Size) | s (Sequence Length) | s/6h Ratio (Quadratic Overhead) |
|-------------------------|-----------------|---------------------|---------------------------------|
| Llama-3-8B              | 4,096           | 8,192               | ~33.3%                          |
| Llama-3-70B             | 8,192           | 8,192               | ~16.7%                          |
| Long-Context (Standard) | 4,096           | 32,768              | ~133.3%                         |
| Ultra-Long Context      | 4,096           | 131,072             | ~533.3% (Attention Dominates)   |


## Sample Isolation (Packing)

In modern training, multiple samples $s_i$ are often packed into one sequence. With sample isolation (e.g., via block-diagonal masking), the quadratic Attention cost only applies within each sample boundary $s_i$.
Given total tokens $C = \sum s_i$, the formula updates to:


$$\text{FLOPs} \approx \sum{ 2 s_i P \times \left(1 + \frac{s_i}{6h}\right) } = 2P \sum{\left(s_i + \frac{s_i^2}{6h}\right)}$$

Factoring out the total token count $C$, we get:

$$\text{FLOPs} \approx 2CP \left( 1 + \frac{1}{6h}\frac{\sum s_i^2}{\sum s_i} \right)$$

The expression $\frac{\sum s_i^2}{\sum s_i}$ represents a weighted mean where the $s_i$ values act as their own weights, **causing longer sequences to dominate the result**.

