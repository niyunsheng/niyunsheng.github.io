---
title: "Strategies for Handling Variable-Length Sequences (Varlen) in Triton"
date: 2026-03-10T22:00:00+08:00
draft: true
tags: ["Triton", "Varlen", "GPU Optimization"]
categories: ["Kernels"]
summary: "An analysis of different methods for mapping global threads or blocks to specific samples when processing variable-length sequences (varlen) in Triton"
---

Processing variable-length sequences (varlen) without padding is critical for training efficiency in models like Diffusion Transformers (DiT) and large language models. A primary challenge in writing varlen Triton kernels is efficiently mapping a flattened 1D thread or block ID to its corresponding `sample_id` using the cumulative sequence lengths (`cu_seqlens`) array.

Below is an analysis of four approaches to handle this mapping, evaluating their trade-offs in memory, compute, and kernel launch overhead.

## Approach 0: On-the-Fly SRAM Computation

If the batch size (`MAX_SAMPLES`) is small enough, the entire `cu_seqlens` array can be loaded into SRAM. The `sample_idx` is computed directly within the kernel.

```python
import triton
import triton.language as tl

@triton.jit
def _kernel(
    cu_seqlens_ptr,
    n_samples,
    MAX_SAMPLES: tl.constexpr,
    # ... other arguments ...
):
    pid = tl.program_id(0)
    
    # Load the entire cu_seqlens array into SRAM
    cu_seqlens = tl.load(
        cu_seqlens_ptr + tl.arange(0, MAX_SAMPLES), 
        mask=tl.arange(0, MAX_SAMPLES) < n_samples, 
        other=0x7FFFFFFF
    )
    
    # Calculate sample_idx by counting how many boundaries the current pid exceeds
    sample_idx = tl.sum((pid >= cu_seqlens), axis=0) - 1
    
    # ... kernel logic ...

# Launch
# _kernel[(total_rows,)](...)
```

**Pros**: 
* Requires no host-side preprocessing.
* Highly efficient for small `MAX_SAMPLES` (e.g., < 16): Branchless vectorized logic avoids warp divergence.
* Cache-friendly for small inputs: Redundant loads of small arrays are absorbed by L1/L2 cache, mitigating global memory bandwidth pressure.

**Cons**: 
* Poor scalability to large batch sizes: Constrained by block-level SRAM and register limitations.
* Bandwidth degradation at scale: The redundant loading of the entire `cu_seqlens` array by every thread block shifts from a cache advantage to a severe global memory bottleneck when the array size exceeds cache capacity.

## Approach 1: Precomputing Element-to-Sample Map via PyTorch

Generate a flat `row_to_sample` mapping tensor before launching the kernel. Each row (or token) maps directly to its sample ID.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    row_to_sample_ptr,
    # ... other arguments ...
):
    pid = tl.program_id(0)
    sample_idx = tl.load(row_to_sample_ptr + pid)
    # ... kernel logic ...

def get_row_to_sample(cu_seqlens: torch.Tensor) -> torch.Tensor:
    device = cu_seqlens.device
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_samples = cu_seqlens.shape[0] - 1
    
    # Generates [0, 0, 0, 1, 1, 2, 2, 2, 2...]
    row_to_sample = torch.repeat_interleave(
        torch.arange(n_samples, device=device), 
        seqlens, 
        dim=0
    )
    return row_to_sample

# Launch
# row_to_sample = get_row_to_sample(cu_seqlens)
# _kernel[(total_rows,)](...)
```

**Pros**
* Extremely simple kernel logic: Requires only a direct 1D global memory read, minimizing register pressure and compute instructions inside the main kernel.
* Constant time memory access per thread.

**Cons**
* High memory footprint: Materializing the dense `row_to_sample` tensor consumes $O(N)$ global memory ($N$ = total tokens), which can be prohibitive for very long sequences.
* Launch overhead: Host-side PyTorch dispatcher operations and memory allocation introduce latency before the Triton kernel executes.