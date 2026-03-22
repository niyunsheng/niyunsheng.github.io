---
title: "Progressive CUDA GEMM Optimization: From Memory-Bound to Swizzling"
date: 2026-03-22T08:00:00+08:00
tags: ["CUDA", "GEMM", "Shared Memory", "Bank Conflict", "Swizzling"]
categories: ["Kernels"]
summary: "A step-by-step guide to optimizing FP32 CUDA GEMM kernels. Learn how to overcome warp-level memory coalescing bottlenecks, eliminate 32-way shared memory bank conflicts using memory padding, and implement zero-waste XOR address swizzling."
---

This article explores the progressive optimization of FP32 General Matrix Multiplication (GEMM) on NVIDIA GPUs. By evolving from a naive memory-bound implementation to an architecturally aware kernel utilizing tiling and XOR swizzling, we achieve a 10.5x speedup on an RTX 3060 Ti.
The following results were measured for $M=N=K=4096$ with $TILE\\_SIZE=32$.

| Optimization Phase          | Execution Time (ms) | Key Improvement / Bottleneck Addressed |
|-----------------------------|---------------------|---------------------------|
| Naive (Global Memory)       | $1044.31$           | Uncoalesced HBM Access         |
| Shared (Conflict)           | $275.40$            | HBM Coalescing / 32-way Smem Read Conflict      |
| Transposed (Read Optimized) | $122.77$            | Eliminated 32-way Smem Read Conflict / 32-way Smem Write Conflict |
| Swizzled (Final)            | $99.31$             | Eliminated 32-way Smem Write Conflict |


## The Naive GEMM and Warp-Level Memory Strides

A baseline CUDA implementation assigns one thread to compute one element of $C(M, N)$. The matrices $A$ and $B$ are in row-major layout with shapes $(M, K)$ and $(N, K)$.

```cpp
// ---------------------------------------------------------
// Kernel 1: Naive Implementation (Global Memory Only)
// Input: A (M, K), B (N, K) - Row-Major
// ---------------------------------------------------------
__global__ void kernel_naive_nk(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float ans = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row][k] * B[col][k]
            // Causes severe uncoalesced memory access for B
            ans += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = ans;
    }
}
```

**Bottleneck Analysis: Uncoalesced Memory Access**
A common misconception is that because $(N, K)$ stores the $K$ dimension contiguously, reading along $K$ is optimal. While true for a single thread over time, GPU performance is dictated by warp-level memory coalescing—the access pattern of 32 threads executing a single load instruction simultaneously.

In the inner loop:
```cpp
ans += A[row * K + k] * B[col * K + k];
```

At any given iteration `k`, the instruction fetches data from HBM. Within a warp, the `row` index is constant, but the `col` index increments with `threadIdx.x`.
* Thread 0 requests memory at offset: `col * K + k`
* Thread 1 requests memory at offset: `(col + 1) * K + k`

The physical memory stride between adjacent threads is exactly $K$ elements. For $K=4096$, the warp's single load instruction shatters into 32 disjoint memory transactions, completely destroying memory coalescing and plummeting effective bandwidth. The arithmetic intensity remains strictly $1$ op/element, locking the kernel behind a severe memory wall.

## Shared Memory Tiling and the Bank Conflict Wall

To break the memory wall, we can load tiles of $A$ and $B$ into fast, on-chip shared memory (SRAM), allowing threads within a block to reuse data and massively increasing arithmetic intensity.

```cpp
// ---------------------------------------------------------
// Kernel 2: Shared Memory (Severe Bank Conflicts)
// ---------------------------------------------------------
__global__ void kernel_shared_conflict_nk(float* A, float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float Tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float Tile_B[TILE_SIZE][TILE_SIZE];

    float ans = 0.0f;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // Load A into Shared Memory (Coalesced)
        if (row < M && (ph * TILE_SIZE + tx) < K)
            Tile_A[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else 
            Tile_A[ty][tx] = 0.0f;

        // Load B into Shared Memory (Coalesced Global Load)
        // b_row maps to C's column (N dimension), b_col maps to K dimension
        int b_row = blockIdx.x * TILE_SIZE + ty;
        int b_col = ph * TILE_SIZE + tx;
        if (b_row < N && b_col < K)
            Tile_B[ty][tx] = B[b_row * K + b_col]; // Standard store
        else 
            Tile_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute Phase
        for (int k = 0; k < TILE_SIZE; k++) {
            // Reading Tile_B[tx][k] causes column-wise access.
            // In FP16 with TILE_SIZE=32, this results in a 16-way bank conflict.
            ans += Tile_A[ty][k] * Tile_B[tx][k];
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row * N + col] = ans;
}
```

**Bottleneck Analysis: Arithmetic Intensity vs. Bank Conflicts**

This implementation achieves excellent HBM coalescing and increases arithmetic intensity by a factor of $T$ (where $T$ is `TILE_SIZE`). However, it introduces a critical new constraint within the Streaming Multiprocessor (SM).

This implementation achieves two major architectural improvements over the naive approach, but introduces a critical new constraint within the SM:
1. **Coalesced HBM Access**: During the Global-to-Shared load phase, threads read `A` and `B` using `tx` as the contiguous inner dimension index. This triggers perfect memory coalescing across the warp.
2. **Increased Arithmetic Intensity**: Let $T$ represent `TILE_SIZE`. In each phase, a thread block loads a $T \times T$ tile of $A$ and a $T \times T$ tile of $B$, totaling $2T^2$ elements. Using these tiles, the block computes $T \times T \times T$ MACs ($2T^3$ operations). The compute-to-memory ratio is $\frac{2T^3}{2T^2} = T$. By setting $T=32$, the arithmetic intensity increases by a factor of $T$ ($32\times$), effectively shielding the ALUs from HBM latency.
3. **The Flaw: Shared Memory Bank Conflicts**:
While HBM traffic is optimized, the compute phase triggers a severe Shared Memory **Bank Conflict**. Shared memory is organized into 32 separate 4-byte banks.

In the inner loop:
```cpp
ans += Tile_A[ty][k] * Tile_B[tx][k];
```

For `Tile_B[tx][k]`, the column index `k` is constant across the warp, but the row index `tx` varies from $0$ to $31$. Assuming $T = 32$ and the `float` data type (4 bytes), the hardware bank index is calculated as:

{{<math>}}
$$
\begin{aligned}
\text{Bank} &= (\text{ByteAddress} / 4) \pmod{32} \\
&= \frac{(tx \times 32 + k) \times 4}{4} \pmod{32} \\
&= (tx \times 32 + k) \pmod{32} \\
&= k \pmod{32}
\end{aligned}
$$
{{</math>}}

Because $k$ is constant for the entire warp, all 32 threads attempt to access the exact same bank simultaneously. This forces the hardware to serialize the requests, resulting in a catastrophic **32-way bank conflict** that cripples compute performance.

## Shared Memory Transposition: Trading Read Conflicts for Write Conflicts

To mitigate the severe 32-way bank conflict observed in the compute phase of Example 2, we can physically reorient the memory layout within the shared memory tile. This is achieved by transposing the data during the Global-to-Shared load phase.

```cpp
// ---------------------------------------------------------
// Kernel 3: Shared Memory Optimized (Transposed Store)
// ---------------------------------------------------------
__global__ void kernel_shared_optimized_nk(float* A, float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float Tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float Tile_B[TILE_SIZE][TILE_SIZE];

    float ans = 0.0f;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // Load A (Coalesced)
        if (row < M && (ph * TILE_SIZE + tx) < K)
            Tile_A[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else 
            Tile_A[ty][tx] = 0.0f;

        // Load B with Transposed Store
        int b_row = blockIdx.x * TILE_SIZE + ty;
        int b_col = ph * TILE_SIZE + tx;
        if (b_row < N && b_col < K)
            Tile_B[tx][ty] = B[b_row * K + b_col]; // Swap tx and ty for transposed store
        else 
            Tile_B[tx][ty] = 0.0f;

        __syncthreads();

        // Compute Phase
        for (int k = 0; k < TILE_SIZE; k++) {
            // Reading Tile_B[k][tx] is now row-wise access.
            // Eliminates the severe 16-way conflict (reduces to FP16's inherent 2-way conflict).
            ans += Tile_A[ty][k] * Tile_B[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row * N + col] = ans;
}
```

**Bottleneck Analysis: Execution Frequency and Net Gains**
At first glance, this optimization appears contradictory. We have eliminated the 32-way read conflict during computation, but introduced a 32-way write conflict during the memory load phase. Why does this result in a net performance gain? The answer lies in the execution frequency of the inner vs. outer loops.

**1. The Compute Phase: 0-Way Read Conflict (High Frequency)**
In `ans += Tile_A[ty][k] * Tile_B[k][tx];`:
* **Tile_A**: `ty` and `k` are constant for the warp. All 32 threads read the exact same address. The hardware triggers a multicast (broadcast), resulting in a **0-way conflict**.
* **Tile_B**: `k` is constant, `tx` increments $0 \to 31$. Threads read 32 contiguous floats. Because `1 float = 1 bank`, they map perfectly across all 32 banks. This is a perfect **0-way conflict**.
* Crucially, this executes $32$ times per block phase.

**2. The Load Phase: 32-Way Write Conflict (Low Frequency)**
During the Global-to-Shared load phase, the assignment `Tile_B[tx][ty] = B[...]` uses `tx` as the row index. The stride between threads is $32$ floats (128 bytes).

$$\text{Bank} = (tx \times 32 + ty) \pmod{32} = ty \pmod{32}$$

Because `ty` is constant during the load warp, all 32 threads write to the same bank. This is a 32-way write conflict.

However, this executes only $1$ time per block phase. We traded a conflict in a high-frequency loop for a conflict in a low-frequency loop.

## Eliminating All Bank Conflicts: Padding and Swizzling

While Kernel 3 hides the latency of the write conflict, achieving 100% shared memory bandwidth requires eliminating it entirely. We have two primary methods: spatial modification (Padding) and logical permutation (Swizzling).

### Method 1: Memory Padding (The Spatial Trade-off)

The root cause of the write conflict is that the row stride ($32$ floats) perfectly wraps around the $32$ banks. Padding breaks this alignment by adding a dummy column, inflating the row stride to $33$ floats.

```cpp
#define TILE_SIZE 32
#define PADDING 1

// Padded Shared Memory Declaration
__shared__ float Tile_B[TILE_SIZE][TILE_SIZE + PADDING];
```

**Mathematical Resolution:**
With the new stride of $33$, when thread `tx` writes to `Tile_B[tx][ty]`, the bank index becomes:

$$\text{Bank} = (tx \times 33 + ty) \pmod{32} = (tx + ty) \pmod{32}$$

As `tx` goes from $0$ to $31$, `(tx + ty)` hits 32 unique values modulo 32. Perfect 0-way conflict!

**The Flaw: SRAM Waste** 
We waste 1 float per row ($\approx 3.1\%$ capacity waste). While small for FP32, SRAM is a strictly limited resource that dictates block occupancy.

### Method 2: Address Swizzling (The Zero-Waste Solution)

To achieve a 0-way conflict without inflating the memory footprint, we keep the array at $32 \times 32$ but scramble the logical mapping of coordinates to physical addresses using a bitwise XOR (`^`) bijection.

For a $32 \times 32$ tile of `float` types, applying a block-aware XOR swizzle logic operates as follows:

```cpp
// ---------------------------------------------------------
// Kernel 4: Shared Memory Swizzled (Zero Conflict, Zero Waste)
// ---------------------------------------------------------

// Helper macro for Swizzle address calculation
#define SWIZZLE(row, col) (row * TILE_SIZE + (col ^ row))

__global__ void kernel_shared_swizzled_nk(float* A, float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float Tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float Tile_B[TILE_SIZE * TILE_SIZE]; // 1D array for explicit address control

    float ans = 0.0f;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // Load A (Coalesced)
        if (row < M && (ph * TILE_SIZE + tx) < K)
            Tile_A[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else 
            Tile_A[ty][tx] = 0.0f;

        // Load B with Swizzled Write (Transposed logically)
        int b_row = blockIdx.x * TILE_SIZE + ty;
        int b_col = ph * TILE_SIZE + tx;
        if (b_row < N && b_col < K) {
            // Logical transpose: row=tx, col=ty. Apply XOR Swizzle.
            Tile_B[SWIZZLE(tx, ty)] = B[b_row * K + b_col]; 
        } else {
            Tile_B[SWIZZLE(tx, ty)] = 0.0f;
        }

        __syncthreads();

        // Compute Phase with Swizzled Read
        for (int k = 0; k < TILE_SIZE; k++) {
            // Read expects logical row=k, col=tx. Apply exact same XOR Swizzle.
            ans += Tile_A[ty][k] * Tile_B[SWIZZLE(k, tx)];
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row * N + col] = ans;
}
```

**Mechanism of XOR Swizzling:**
In the write phase, thread `tx` writes to logical coordinate `(tx, ty)`. The physical address offset is `tx * 32 + (ty ^ tx)`.

$$\text{Bank} = (tx \times 32 + (ty \oplus tx)) \pmod{32} = (ty \oplus tx) \pmod{32}$$

Because `ty` is constant for the warp, XORing it with `tx` ($0 \to 31$) deterministically produces 32 unique integers from $0 \to 31$. Every thread hits a different bank.

When the compute phase reads the data back using the exact same `SWIZZLE(k, tx)` macro, the bijection of XOR ensures data is fetched correctly. The underlying physical routing guarantees a 0-way bank conflict for both reads and writes, utilizing 100% of the SRAM bandwidth with zero wasted capacity.

