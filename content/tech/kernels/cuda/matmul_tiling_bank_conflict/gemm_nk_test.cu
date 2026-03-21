#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32
#define WARMUP_ITER 2
#define RUN_ITER 1

// ---------------------------------------------------------
// Kernel 1: Naive Implementation (Global Memory Only)
// Input: A (M, K), B (N, K) - Row-Major
// ---------------------------------------------------------
__global__ void kernel_naive_nk(half* A, half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float ans = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row][k] * B[col][k]
            // Causes severe uncoalesced memory access for B
            ans += __half2float(A[row * K + k] * B[col * K + k]);
        }
        C[row * N + col] = __float2half(ans);
    }
}

// ---------------------------------------------------------
// Kernel 2: Shared Memory (Severe Bank Conflicts)
// ---------------------------------------------------------
__global__ void kernel_shared_conflict_nk(half* A, half* B, half* C, int M, int N, int K) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ half Tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ half Tile_B[TILE_SIZE][TILE_SIZE];

    float ans = 0.0f;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // Load A into Shared Memory (Coalesced)
        if (row < M && (ph * TILE_SIZE + tx) < K)
            Tile_A[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else 
            Tile_A[ty][tx] = __float2half(0.0f);

        // Load B into Shared Memory (Coalesced Global Load)
        // b_row maps to C's column (N dimension), b_col maps to K dimension
        int b_row = blockIdx.x * TILE_SIZE + ty;
        int b_col = ph * TILE_SIZE + tx;
        if (b_row < N && b_col < K)
            Tile_B[ty][tx] = B[b_row * K + b_col]; // Standard store
        else 
            Tile_B[ty][tx] = __float2half(0.0f);

        __syncthreads();

        // Compute Phase
        for (int k = 0; k < TILE_SIZE; k++) {
            // Reading Tile_B[tx][k] causes column-wise access.
            // In FP16 with TILE_SIZE=32, this results in a 16-way bank conflict.
            ans += __half2float(Tile_A[ty][k] * Tile_B[tx][k]);
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row * N + col] = __float2half(ans);
}

// ---------------------------------------------------------
// Kernel 3: Shared Memory Optimized (Transposed Store)
// ---------------------------------------------------------
__global__ void kernel_shared_optimized_nk(half* A, half* B, half* C, int M, int N, int K) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ half Tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ half Tile_B[TILE_SIZE][TILE_SIZE];

    float ans = 0.0f;
    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ph++) {
        // Load A (Coalesced)
        if (row < M && (ph * TILE_SIZE + tx) < K)
            Tile_A[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else 
            Tile_A[ty][tx] = __float2half(0.0f);

        // Load B with Transposed Store
        int b_row = blockIdx.x * TILE_SIZE + ty;
        int b_col = ph * TILE_SIZE + tx;
        if (b_row < N && b_col < K)
            Tile_B[tx][ty] = B[b_row * K + b_col]; // Swap tx and ty for transposed store
        else 
            Tile_B[tx][ty] = __float2half(0.0f);

        __syncthreads();

        // Compute Phase
        for (int k = 0; k < TILE_SIZE; k++) {
            // Reading Tile_B[k][tx] is now row-wise access.
            // Eliminates the severe 16-way conflict (reduces to FP16's inherent 2-way conflict).
            ans += __half2float(Tile_A[ty][k] * Tile_B[k][tx]);
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row * N + col] = __float2half(ans);
}

// ---------------------------------------------------------
// Benchmark Sub-function
// ---------------------------------------------------------
// Function pointer type for the kernels
typedef void (*GemmKernel)(half*, half*, half*, int, int, int);

float benchmark_kernel(GemmKernel kernel, half* d_A, half* d_B, half* d_C, 
                       int M, int N, int K, dim3 dimGrid, dim3 dimBlock, const char* kernel_name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Measurement
    cudaEventRecord(start);
    for (int i = 0; i < RUN_ITER; ++i) {
        kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_ms = total_ms / RUN_ITER;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("[Benchmark] %-30s : %8.3f ms\n", kernel_name, avg_ms);
    return avg_ms;
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {
    // Matrix dimensions
    int M = 4096, N = 4096, K = 4096; 
    
    size_t size_A = M * K * sizeof(half);
    size_t size_B = N * K * sizeof(half);
    size_t size_C = M * N * sizeof(half);

    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    half *h_C = (half*)malloc(size_C);
    
    // Host Initialization
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(0.01f);
    for (int i = 0; i < N * K; ++i) h_B[i] = __float2half(0.01f);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Grid and Block configuration
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    // x dimension maps to N, y dimension maps to M
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: Warmup=%d, Run=%d\n\n", WARMUP_ITER, RUN_ITER);

    // Run Benchmarks
    benchmark_kernel(kernel_naive_nk, d_A, d_B, d_C, M, N, K, dimGrid, dimBlock, "Naive (Global Memory)");
    benchmark_kernel(kernel_shared_conflict_nk, d_A, d_B, d_C, M, N, K, dimGrid, dimBlock, "Shared Memory (Conflict)");
    benchmark_kernel(kernel_shared_optimized_nk, d_A, d_B, d_C, M, N, K, dimGrid, dimBlock, "Shared Memory (Transposed)");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}