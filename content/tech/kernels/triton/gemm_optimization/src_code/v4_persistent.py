import triton
import triton.language as tl
import torch

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_persistent(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = num_pid_m * num_pid_n

    while pid < total_blocks:
        # -- Swizzling Logic (Inherited from Step 2) --
        num_pid_in_group = GROUP_SIZE * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # -- TMA Pointers Setup (Inherited from Step 3) --
        a_block_ptr = tl.make_block_ptr(
            a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0)
        )

        # -- Compute --
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            accumulator = tl.dot(a, b, accumulator)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -- Store --
        c_block_ptr = tl.make_block_ptr(
            c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0)
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))

        # Advance to the next logical block
        pid += num_programs

def matmul_persistent(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    
    # Query hardware SM count.
    # e.g., A100 has 108 SMs, H100 has 132 SMs.
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    # The grid size is bounded by the hardware SM count.
    grid = lambda Meta: (min(triton.cdiv(M, Meta['BLOCK_SIZE_M'])*triton.cdiv(N, Meta['BLOCK_SIZE_N']), num_sms), )

    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

if __name__ == "__main__":
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    c = matmul_persistent(a, b)
    c_ref = torch.matmul(a, b)
    print("Max error:", (c - c_ref).abs().max().item())