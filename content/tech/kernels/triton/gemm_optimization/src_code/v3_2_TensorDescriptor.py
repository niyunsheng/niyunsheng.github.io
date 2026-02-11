import triton
import triton.language as tl
import torch
from triton.tools.tensor_descriptor import TensorDescriptor

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_desc, b_desc, c_desc,
        # Matrix dimensions
        M, N, K,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE: tl.constexpr,
        IS_M_MAJOR: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ####### L2 Cache Optimizations #######
    if IS_M_MAJOR:
        # Number of programs in group
        num_pid_in_group = GROUP_SIZE * num_pid_n
        # Id of the group this program is in
        group_id = pid // num_pid_in_group
        # Row-id of the first program in the group
        first_pid_m = group_id * GROUP_SIZE
        # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
        group_size = min(num_pid_m - first_pid_m, GROUP_SIZE)
        # *Within groups*, programs are ordered in a column-major order
        # Row-id of the program in the *launch grid*
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
        pid_n = tl.where(group_id & 1, num_pid_n - 1 - (pid % num_pid_in_group) // group_size, (pid % num_pid_in_group) // group_size)
    else:
        # Number of programs in group
        num_pid_in_group = GROUP_SIZE * num_pid_m
        # Id of the group this program is in
        group_id = pid // num_pid_in_group
        # Row-id of the first program in the group
        first_pid_n = group_id * GROUP_SIZE
        # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
        group_size = min(num_pid_n - first_pid_n, GROUP_SIZE)
        # *Within groups*, programs are ordered in a column-major order
        # Row-id of the program in the *launch grid*
        pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size)
        pid_m = tl.where(group_id & 1, num_pid_m - 1 - (pid % num_pid_in_group) // group_size, (pid % num_pid_in_group) // group_size)
        
    ####### L2 Cache Optimizations #######

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    base_offset_m = pid_m * BLOCK_SIZE_M
    base_offset_n = pid_n * BLOCK_SIZE_N
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K
        a = a_desc.load([base_offset_m, offset_k]);
        b = b_desc.load([offset_k, base_offset_n]);
        accumulator = tl.dot(a, b, accumulator)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    c_desc.store([base_offset_m, base_offset_n], c)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 4
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N])
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    is_m_major = M >= N
    matmul_kernel[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE=GROUP_SIZE,  #
        IS_M_MAJOR=is_m_major,  #
        ACTIVATION=activation  #
    )
    return c
