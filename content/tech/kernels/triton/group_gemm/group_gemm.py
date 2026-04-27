# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K' : BK}, num_stages=s, num_warps=w) \
    for BM in [128]\
    for BN in [128, 256]\
    for BK in [64, 128]\
    for s in ([3, 4])\
    for w in [4, 8]\
]


@triton.autotune(
    tma_configs,
    key=['N', 'K', 'GROUP_SIZE'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    x_ptr,
    offsets,
    y_ptr,
    weight_ptr,
    N,
    K,
    GROUP_SIZE: tl.constexpr,
    NUM_SM: tl.constexpr, # number of virtual SM
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(GROUP_SIZE):
        # get the gemm size of the current problem
        start_m = tl.where(g == 0, 0, tl.load(offsets + g - 1))
        gm = tl.where(g == 0, tl.load(offsets + g), tl.load(offsets + g) - tl.load(offsets + g - 1))
        gn = N
        gk = K
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if gm > 0 and tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            a_ptr = x_ptr + start_m * K
            b_ptr = weight_ptr + g * N * K
            c_ptr = y_ptr + start_m * N

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[K, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[K, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[N, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(DTYPE)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def grouped_gemm_triton_tma(
    x: torch.Tensor, # (s, h_in)
    num_tokens_per_expert: torch.Tensor, # sum(num_tokens_per_expert) = s
    weight: torch.Tensor, # (group_num, h_out, h_in)
    bias: torch.Tensor=None, # (group_num, h_out,)
):
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    assert bias is None, "Bias is not supported yet"
    grid = lambda META: (META['NUM_SM'], )
    total_tokens, M = x.shape
    num_experts, N, K = weight.shape
    y = torch.empty((total_tokens, N), device=x.device, dtype=x.dtype)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    grouped_matmul_tma_kernel[grid](x, offsets, y, weight, N=N, K=K, GROUP_SIZE=num_experts, NUM_SM=num_sms(), DTYPE=tl.float16)
    return y


import torch.nn.functional as F
def grouped_gemm_torch_ref(
    x: torch.Tensor, # (s, h_in)
    num_tokens_per_expert: torch.Tensor, # sum(num_tokens_per_expert) = s
    weight: torch.Tensor, # (group_num, h_out, h_in)
    bias: torch.Tensor=None, # (group_num, h_out,)
) -> torch.Tensor:

    if bias is not None:
        raise NotImplementedError("Bias is not supported yet for torch 2.10")
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    return F.grouped_mm(x, weight.transpose(-2, -1), bias=bias, offs=offsets)

num_experts = 8
M, N, K = 1024, 8192, 8192
alpha = 32
total = num_experts * M
concentration = torch.full((num_experts,), alpha, dtype=torch.float32)
proportions = torch.distributions.Dirichlet(concentration).sample()
sizes = torch.round(proportions * total).to(torch.int32).to(DEVICE)
print(sizes)

total_tokens = sizes.sum()
x = torch.rand(total_tokens, K, device=DEVICE, dtype=torch.float16)
weight = torch.rand((num_experts, N, K), device=DEVICE, dtype=torch.float16)

y_torch = grouped_gemm_torch_ref(x, sizes, weight)
y_triton = grouped_gemm_triton_tma(x, sizes, weight)
print((y_torch - y_triton).abs().max())
assert torch.allclose(y_torch, y_triton, atol=1e-2, rtol=1e-2)
print("Test passed")

configs = []
for N, K, num_experts in [(7168, 2048, 8)]:
    for M in [1024]:
        configs.append(triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=["alpha"],
            x_vals=[1, 256, 1024],  # different possible values for `x_name`
            line_arg='provider',
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=['cublas'] + (['triton-tma'] if supports_tma() else []),
            # label name for the lines
            line_names=["cuBLAS"] + (['Triton + TMA'] if supports_tma() else []),
            # line styles
            styles=[('green', '-')] + ([('red', '-')] if supports_tma() else []),
            ylabel="runtime(ms)",  # label name for the y-axis
            plot_name="group-gemm-performance",
            # name for the plot. Used also as a file name for saving the plot.
            args={"M": M, "N": N, "K": K, "num_experts": num_experts},
        ))
@triton.testing.perf_report(configs)
def benchmark_batches(alpha, M, N, K, num_experts, provider):
    PEAK_TFLOPS = 989.0
    total = num_experts * M
    concentration = torch.full((num_experts,), alpha, dtype=torch.float32)
    proportions = torch.distributions.Dirichlet(concentration).sample()
    sizes = torch.round(proportions * total).to(torch.int32).to(DEVICE)
    group_size = num_experts
    
    total_tokens = sizes.sum().item()
    x = torch.rand(total_tokens, K, device=DEVICE, dtype=torch.float16)
    weight = torch.rand((num_experts, N, K), device=DEVICE, dtype=torch.float16)
    bias = None # torch.randn(num_experts, N, device=device, dtype=dtype)

    mfu = lambda ms: 2 * total_tokens * K * N / (ms * 1e-3) * 1e-12 / PEAK_TFLOPS
    

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_gemm_torch_ref(x, sizes, weight, bias), quantiles=quantiles)
    if provider == 'triton-tma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_triton_tma(x, sizes, weight, bias), quantiles=quantiles)
    return mfu(ms), mfu(max_ms), mfu(min_ms)
    # return ms, min_ms, max_ms

benchmark_batches.run(show_plots=False, print_data=True)