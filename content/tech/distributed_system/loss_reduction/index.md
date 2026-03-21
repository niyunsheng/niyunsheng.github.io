---
title: "Loss Reduction in Distributed Training"
date: 2026-03-21T15:41:55+08:00
tags: ["LLM", "Video Generation", "CP", "DP"]
categories: ["Distributed System"]
summary: "An analysis of how Data Parallelism (DP) and Context Parallelism (CP) affect loss reduction, and how to maintain mathematical equivalence between distributed and single-device training for LLM and Video DiT models."
---

In modern distributed training, frameworks like PyTorch DDP or FSDP handle gradient synchronization automatically. However, as models move toward variable sequence lengths, Sequence Packing, and Context Parallelism (CP), the standard "All-Reduce Mean" approach often introduces subtle mathematical biases. The goal of any distributed strategy is to produce gradients identical to a single-device execution; achieving this requires shifting the reduction logic from the gradient synchronization phase to the forward loss computation.

## The Baseline: Why "Mean" Matters

The primary reason we use Mean reduction (rather than Sum) is to **decouple the gradient magnitude from the Batch Size (BS)**.

If a Sum reduction is used, doubling the batch size doubles the gradient scale, necessitating a retuning of the Learning Rate (LR). With a Mean reduction, the expected value of the gradient remains consistent regardless of the BS, allowing LR settings to remain stable during scaling.

The definition of "Mean" diverges depending on the architecture:

* **LLM (Global Token Mean)**: All tokens are treated as a single pool. The loss is the sum of all token cross-entropies divided by the total number of valid tokens.

$$L_{LLM} = \frac{1}{N_{global}} \sum_{i=1}^{N_{global}} l_{i}$$

* **Video Models (Hierarchical Mean)**: To ensure that a long video doesn't dominate the gradient of a short video in the same batch, we perform a two-step reduction: first, average tokens within each sample, then average those sample-level losses across the batch.

$$L_{Video} = \frac{1}{B} \sum_{b=1}^{B} \left( \frac{1}{T_b} \sum_{i=1}^{T_b} l_{b,i} \right)$$

**The Golden Rule**: Distributed training must produce gradients mathematically identical to these single-device baselines.

## Data Partitioning: DP vs. CP

In distributed training, we partition the data dimension to fit large models and long sequences into GPU memory:

* **Data Parallelism (DP)**: Partitions the Batch dimension. Each rank gets different samples.
* **Context Parallelism (CP)**: Partitions the Sequence (or Context) dimension. A single long sample is split across multiple ranks.

When DP and CP are combined, the data is no longer "identically distributed" across GPUs.
This non-uniformity is where standard framework-level reductions fail.

## Gradient `All-Reduce Mean` vs. `Loss / Total_Valid_Tokens`(LLM)

For LLMs, standard frameworks like PyTorch DDP default to **Grad-level Reduction** via `All-Reduce Mean`. This implicitly assumes that every rank contributes an equal denominator to the global average.

If the number of tokens $N_k$ varies per rank (common in sequence packing), the framework's default average skews the weights:

$$\nabla L_{DDP} = \frac{1}{W} \sum_{k=1}^{W} \left( \frac{1}{N_k} \sum_{i=1}^{N_k} \nabla l_{k,i} \right)$$

(Where $W$ is the DP World Size).

**The Bias**: Under this default reduction, the effective weight of a token becomes $\frac{1}{W \cdot N_k}$. Consequently, a token on a rank with fewer tokens ($N_{small}$) is disproportionately up-weighted compared to a token on a rank with more data ($N_{large}$).

**The Solution: Forward Loss Scaling**
To fix this, we move the scaling from the gradient-sync phase to the forward-loss phase:
1. **Sync Metadata**: Obtain $N_{global} = \sum N_k$ via All-Reduce Sum.
2. **Forward Scaling**: Locally compute $L_k = \frac{\sum l_{k,i}}{N_{global}}$.
3. **Grad Sync**: Use All-Reduce Sum on gradients. Now, every token globally has an identical weight of $1/N_{global}$.

## Implementing Pre-Backward Metadata Sync for Video DiT (CP)

Video DiT models introduce the "Hierarchical Mean" challenge. Under Context Parallelism, a single video sample is split across $W_{cp}$ ranks. No single rank knows the total length $T_s$ of that sample. Furthermore, with Sequence Packing, a rank might contain a fragment of a long video alongside several smaller, complete videos.

To keep distributed gradients consistent with the $L_{Video}$ baseline, metadata must be synchronized before the backward pass, and the gradient reduction must be controlled:
1. **CP-Group Sync (Intra-Sample)**: Within each CP group, use `All-Reduce Sum` to calculate the global length $T_b$ for every partitioned sample. Each token now knows its correct intra-sample weight ($1/T_s$).
2. **DP-Group Sync (Inter-Sample)**: Across the DP dimension, use `All-Reduce Sum` to find the true global sample count $B_{global}$.
3. **Local Loss Scaling**: Combine these into a local scalar without arbitrary compensation factors:
$$L_{loc} = \frac{1}{B_{global}} \sum_{b \in Local} \left( \frac{1}{T_b} \sum_{i \in Local} l_{b,i} \right)$$
4. **Gradient Synchronization**: Ensure the distributed optimizer or DDP/FSDP communication hook is configured for `All-Reduce Sum`.

By shifting from "Gradient Mean" to "Pre-Backward Loss Scaling + Gradient Sum," the complexities of DP and CP partitioning are resolved. This architecture guarantees that a multi-node cluster produces gradients identical to a single-device run, preserving the integrity of the learning rate and ensuring stable convergence for state-of-the-art models.

## Implementation Note: Gradient Norm and DP/CP

It is worth noting that this forward-scaling approach is entirely orthogonal to gradient norm calculations (e.g., for gradient clipping). Because the local gradients are already accurately scaled during the forward loss computation, the local gradients natively reflect their true global magnitudes.

If you are using DP/CP, the only required adjustment is changing the communication collective for gradient synchronization—specifically the `Reduce-Scatter` operation—from its default `Mean` to `Sum`. Once the `Reduce-Scatter` Sum is applied, the global gradients are correctly assembled, and the rest of your existing gradient norm calculation and clipping logic remains completely unchanged.