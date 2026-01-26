---
title: Visualizing 3D Attention: Bridging the Gap Between 1D Sequences and 3D Space"
date: 2026-01-26T23:30:00+08:00
draft: false
tags: ["Visualization", "DiT"]
categories: ["Tools"]
summary: "An interactive tool to visualize the mapping between 1D token sequences and 3D (T, H, W) sliding windows."
---

In the era of modern Video Generation, Diffusion Transformer (DiT) models have become the de facto standard. Unlike LLMs which deal with linear text, video models process tokens that carry 3D spatial-temporal information: **Frame (Time), Height, and Width**.

However, before these tokens are fed into the DiT blocks (or Attention kernels), they are typically flattened into a 1D sequence. The mapping from 3D coordinates $(t, h, w)$ to a 1D index $idx$ is theoretically simple:

$$idx_{1d} = t \times (H \times W) + h \times W + w$$

While the formula is straightforward, the **intuition** is not. When writing custom kernels (e.g., in Triton or CUDA), we often face a "locality mismatch":

1.  **1D Contiguity $\neq$ 3D Proximity:** Tokens that are adjacent in the flattened 1D memory layout might be spatially disjoint in the 3D structure (e.g., wrapping from the right edge of one row to the left edge of the next).
2.  **3D Proximity $\neq$ 1D Contiguity:** Tokens that are neighbors in 3D space (especially across different frames) can be extremely far apart in the 1D sequence.

This disconnect makes implementing **3D Sliding Window Attention** or **Causal Masking** notoriously error-prone. A single off-by-one error in your index calculation can break the causality.

To bridge this gap, I built an interactive visualization tool to help build intuition for 3D sliding windows.

{{< figure src="demo.gif" title="Fig. 1: Interactive Visualization Demo" caption="Click on any token to see its neighbors. The tool highlights how a 3D sliding window maps to the flattened 1D sequence in real-time." align="center" >}}

### Try it out
You can play with different window sizes and causal patterns in the live demo linked in the [Project Page](https://niyunsheng.github.io/sliding-window-viz-3D/).