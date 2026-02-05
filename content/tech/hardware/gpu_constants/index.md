---
title: "GPU & Network Constants"
date: 2026-01-25T23:21:35+08:00
draft: false
tags: ["NVIDIA", "A100", "H200", "B300", "Infiniband"]
categories: ["Hardware"]
summary: "A quick reference of Dense FLOPS and Unidirectional Bandwidth for A100, H100, H200, and Blackwell."
---

## 1. GPU Specs (A100 - B200)


| | BF16 Tensor Core (Dense) | FP8 Tensor Core (Dense) | GPU Memory | GPU Memory Bandwidth | NVLink (Unidirectional) | Host Interface |
| - | - | - | - | - | - | - |
| A100 80GB SXM | 312 TFLOPS | - | 80 GB HBM2e | 2,039 GB/s | 300 GB/s | PCIe Gen4 |
| H100 SXM | 989 TFLOPS | 1,979 TFLOPS | 80 GB HBM3 | 3.35 TB/s | 450 GB/s | PCIe Gen5 |
| H200 SXM |  989 TFLOPS | 1,979 TFLOPS | 141 GB HBM3e | 4.8 TB/s | 450 GB/s | PCIe Gen5 |
| GB200 NVL72 | 2.5 PFLOPS | 5 PFLOPS | 196 GB HBM3e | 8 TB/s | 900 GB/s | NVLink-C2C |
| HGX B200 | 2.25 PFLOPS | 4.5 PFLOPS | 180 GB HBM3e | 7.7 TB/s | 900 GB/s | PCIe Gen6 |
| HGX B300 | 2.25 PFLOPS | 4.5 PFLOPS | 288 GB HBM3e | 8 TB/s | 900 GB/s | PCIe Gen6 |


Note: 
* The **Host Interface** (PCIe) limits the GPU-to-NIC throughput. Ensure your simulation checks if `Port Speed > PCIe BW` to identify potential host-side bottlenecks (e.g., A100 on NDR).
* PCIe (Unidirectional) Bandwidth: Gen4 - 31.5 GB/s; Gen5 - 63 GB/s; Gen6 - 121 GB/s.


## 2. Switch Specs (InfiniBand)

|                       | Role          | Generation    | Port Speed| Port Count    |
| -                     | -             | -             | -         | -             |
| Quantum QM8700        | Leaf / Spine  | A100 (HDR)    | 25 GB/s   | 40            | 
| Quantum CS8500        | Director      | A100 (HDR)    | 25 GB/s   | Up to 800     |
| Quantum-2 QM9700      | Leaf / Spine  | H100 (NDR)    | 50 GB/s   | 64            |
| Quantum-2 CS9500      | Director      | H100 (NDR)    | 50 GB/s   | Up to 2,048   |
| Quantum-X800 Q3400    | Leaf / Spine  | B200 (XDR)    | 100 GB/s  | 144           |
| Quantum-X800 Director | Director      | B200 (XDR)    | 100 GB/s  | Scalable      |