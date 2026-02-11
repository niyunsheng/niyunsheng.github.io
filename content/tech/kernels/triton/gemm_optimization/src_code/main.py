import torch
import triton

def get_peak_tflops(dtype=torch.float16, device_index=0):
    device_name = torch.cuda.get_device_name(device_index)

    PEAK_TFLOPS_MAP = {
        "H100": {
            torch.float8_e4m3fn: 1979.0, # H100 SXM
            torch.float8_e5m2: 1979.0,
            torch.float16: 989.0,
            torch.bfloat16: 989.0,
        },
        "A100": {
            torch.float16: 312.0,        # A100 SXM/PCIe
            torch.bfloat16: 312.0,
        },
    }

    matched_device = None
    for key in PEAK_TFLOPS_MAP.keys():
        if key in device_name:
            matched_device = key
            break
    
    if not matched_device:
        raise NotImplementedError(f"Not implemented for device: {device_name}")
    
    if dtype not in PEAK_TFLOPS_MAP[matched_device]:
        raise NotImplementedError(f"Not implemented for dtype: {dtype}")
    
    return PEAK_TFLOPS_MAP[matched_device][dtype]

from v0_tiling import matmul as matmul_tiling
from v2_0_panel_swizzing import matmul as matmul_panel_swizzing
from v2_1_snake_panel_swizzing import matmul as matmul_snake_panel_swizzing
from v2_2_dynamic_swizzing import matmul as matmul_dynamic_swizzing
from v3_0_tma_make_block_ptr import matmul as matmul_tma_make_block_ptr
from v3_1_make_tensor_descriptor import matmul as matmul_make_tensor_descriptor
from v3_2_TensorDescriptor import matmul as matmul_TensorDescriptor
dtype = torch.float16
device = torch.device("cuda")
M, N, K = 8192, 8192, 4096
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)
c = matmul_tma_make_block_ptr(a, b)
print((c - torch.matmul(a, b)).abs().max())

def benchmark_matmul_mfu(func, a, b):
    ms,_,_ = triton.testing.do_bench(lambda: func(a, b), quantiles=[0.5, 0.2, 0.8], warmup=32)
    achieved_tflops = 2 * a.shape[0] * a.shape[1] * b.shape[1] / (ms * 1e-3) * 1e-12
    mfu = achieved_tflops / get_peak_tflops(a.dtype, a.device.index)
    return mfu

for name, func in [
        ("v0_tiling", matmul_tiling), 
        ("v2_0_panel_swizzing", matmul_panel_swizzing), 
        ("v2_1_snake_panel_swizzing", matmul_snake_panel_swizzing), 
        ("v2_2_dynamic_swizzing", matmul_dynamic_swizzing), 
        ("v3_0_tma_make_block_ptr", matmul_tma_make_block_ptr), 
        ("v3_1_make_tensor_descriptor", matmul_make_tensor_descriptor), 
        ("v3_2_TensorDescriptor", matmul_TensorDescriptor)
    ]:
    mfu = benchmark_matmul_mfu(func, a, b)
    print(f"{name}: {mfu:.2%} MFU")