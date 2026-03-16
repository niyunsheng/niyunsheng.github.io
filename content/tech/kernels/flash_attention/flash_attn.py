import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ],
    key=["HEAD_NUM", "HEAD_DIM"]
)
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, 
    sm_scale_log2, SEQ, 
    HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_head = tl.program_id(1)

    q_base = q_ptr + off_head * HEAD_DIM
    k_base = k_ptr + off_head * HEAD_DIM
    v_base = v_ptr + off_head * HEAD_DIM
    o_base = o_ptr + off_head * HEAD_DIM

    q_block_ptr = tl.make_block_ptr(
        base=q_base, shape=(SEQ, HEAD_DIM), strides=(HEAD_NUM * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_base, shape=(SEQ, HEAD_DIM), strides=(HEAD_NUM * HEAD_DIM, 1),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_base, shape=(SEQ, HEAD_DIM), strides=(HEAD_NUM * HEAD_DIM, 1),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    for start_n in range(0, SEQ, BLOCK_N):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2
        # it's necessary to mask scores for out-of-sequence positions to -inf before computing max for correctness
        if start_n + BLOCK_N >= SEQ:
            mask_n = start_n + tl.arange(0, BLOCK_N) < SEQ
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None]) 
        alpha = tl.math.exp2(m_i - m_ij)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    acc = acc.to(DTYPE)

    lse_base = lse_ptr + off_head * SEQ
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(lse_base + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < SEQ)

    o_block_ptr = tl.make_block_ptr(
        base=o_base, shape=(SEQ, HEAD_DIM), strides=(HEAD_NUM * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ],
    key=["HEAD_NUM", "HEAD_DIM"]
)
@triton.jit
def _bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr, lse_ptr, delta_ptr,
    sm_scale, sm_scale_log2e, SEQ, 
    HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)

    stride_seq = HEAD_NUM * HEAD_DIM
    q_base = q_ptr + off_h * HEAD_DIM
    k_base = k_ptr + off_h * HEAD_DIM
    v_base = v_ptr + off_h * HEAD_DIM
    do_base = do_ptr + off_h * HEAD_DIM
    dq_base = dq_ptr + off_h * HEAD_DIM 

    q_block_ptr = tl.make_block_ptr(
        base=q_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(0, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(0, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )

    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    do = tl.load(do_block_ptr, boundary_check=(0, 1))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse = tl.load(lse_ptr + off_h * SEQ + offs_m, mask=offs_m < SEQ)
    delta = tl.load(delta_ptr + off_h * SEQ + offs_m, mask=offs_m < SEQ)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, SEQ, BLOCK_N):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        # recompute qk
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2e
        # althought the lse is right in forward, but inf * 0 maybe NaN in backward, so we still need to mask qk for out-of-sequence positions to -inf to ensure p is 0 at those positions
        if start_n + BLOCK_N >= SEQ:
            mask_n = start_n + tl.arange(0, BLOCK_N) < SEQ
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
        
        p = tl.math.exp2(qk - lse[:, None])

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dq = tl.dot(ds.to(k.dtype), k, dq)

        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    dq_block_ptr = tl.make_block_ptr(
        base=dq_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_m * BLOCK_M, 0), 
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    tl.store(dq_block_ptr, dq.to(DTYPE), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ],
    key=["HEAD_NUM", "HEAD_DIM"]
)
@triton.jit
def _bwd_dk_dv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, dk_ptr, dv_ptr, lse_ptr, delta_ptr, 
    sm_scale, sm_scale_log2e, SEQ, 
    HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr
):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)

    stride_seq = HEAD_NUM * HEAD_DIM
    q_base = q_ptr + off_h * HEAD_DIM
    k_base = k_ptr + off_h * HEAD_DIM
    v_base = v_ptr + off_h * HEAD_DIM
    do_base = do_ptr + off_h * HEAD_DIM
    dk_base = dk_ptr + off_h * HEAD_DIM
    dv_base = dv_ptr + off_h * HEAD_DIM

    k_block_ptr = tl.make_block_ptr(
        base=k_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_n * BLOCK_N, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_n * BLOCK_N, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )

    q_block_ptr = tl.make_block_ptr(
        base=q_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(0, 0), 
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(0, 0), 
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0)
    )

    k = tl.load(k_block_ptr, boundary_check=(0, 1))
    v = tl.load(v_block_ptr, boundary_check=(0, 1))

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    for start_m_idx in range(0, SEQ, BLOCK_M):
        offs_m = start_m_idx + tl.arange(0, BLOCK_M)
        
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        do = tl.load(do_block_ptr, boundary_check=(0, 1))
        
        lse = tl.load(lse_ptr + off_h * SEQ + offs_m, mask=offs_m < SEQ)
        delta = tl.load(delta_ptr + off_h * SEQ + offs_m, mask=offs_m < SEQ)

        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2e

        if start_n + BLOCK_N >= SEQ:
            mask_n = start_n + tl.arange(0, BLOCK_N) < SEQ
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

        p = tl.math.exp2(qk - lse[:, None])

        dv = tl.dot(tl.trans(p).to(do.dtype), do, dv)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk)

        q_block_ptr = tl.advance(q_block_ptr, (BLOCK_M, 0))
        do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M, 0))

    dk_block_ptr = tl.make_block_ptr(
        base=dk_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_n * BLOCK_N, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    tl.store(dk_block_ptr, dk.to(DTYPE), boundary_check=(0, 1))
    dv_block_ptr = tl.make_block_ptr(
        base=dv_base, shape=(SEQ, HEAD_DIM), 
        strides=(stride_seq, 1), offsets=(start_n * BLOCK_N, 0), 
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0)
    )
    tl.store(dv_block_ptr, dv.to(DTYPE), boundary_check=(0, 1))


class FusedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Input tensors must be contiguous"
        seq, nh, hd = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / (hd ** 0.5)
        o = torch.empty_like(q)
        # save lse for backward
        lse = torch.empty((nh, seq), device=q.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), nh, 1)
        sm_scale_log2e = sm_scale  * math.log2(math.e)  # convert to log2 scale factor

        _fwd_kernel[grid](
            q, k, v, o, lse, sm_scale_log2e,
            seq, HEAD_NUM=nh, HEAD_DIM=hd,
            DTYPE=tl.float16
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.sm_scale_log2e = sm_scale_log2e
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        seq, nh, hd = do.shape

        do = do.contiguous()

        # pre process
        delta = (o * do).sum(dim=-1).transpose(0, 1).contiguous().to(torch.float32)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid_dq = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), H)
        _bwd_dq_kernel[grid_dq](
            q, k, v, do, dq, lse, delta, ctx.sm_scale, ctx.sm_scale_log2e,
            seq, HEAD_NUM=nh, HEAD_DIM=hd, DTYPE=tl.float16
        )
        
        grid_dk_dv = lambda meta: (triton.cdiv(seq, meta["BLOCK_N"]), nh, 1)
        _bwd_dk_dv_kernel[grid_dk_dv](
            q, k, v, do, dk, dv, lse, delta,  ctx.sm_scale, ctx.sm_scale_log2e,
            seq, HEAD_NUM=nh, HEAD_DIM=hd, DTYPE=tl.float16
        )

        return dq, dk, dv, None

def attention(q, k, v):
    return FusedAttention.apply(q, k, v)

def torch_attention(q, k, v):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    return o.transpose(0, 1)

if __name__ == "__main__":
    N, H, DIM = 300, 4, 64
    dtype = torch.float16
    device = "cuda"
    
    q = torch.randn(N, H, DIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(N, H, DIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(N, H, DIM, dtype=dtype, device=device, requires_grad=True)
    o_grad = torch.randn_like(q)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    o_ref = torch_attention(q_ref, k_ref, v_ref)
    o_ref.backward(o_grad)

    o = attention(q, k, v)
    o.backward(o_grad)

    print("Output max diff:", (o - o_ref).abs().max().item())
    print("DQ max diff:", (q.grad - q_ref.grad).abs().max().item())
    print("DK max diff:", (k.grad - k_ref.grad).abs().max().item())
    print("DV max diff:", (v.grad - v_ref.grad).abs().max().item())