---
title: "GPU & Network Constants"
date: 2026-01-25T23:21:35+08:00
draft: false
tags: ["GPU", "AI Infra", "NVIDIA"]
categories: ["Engineering"]
summary: "A quick reference of Dense FLOPS and Unidirectional Bandwidth for A100, H100, H200, and Blackwell."
---

## 1. GPU Specs (A100 - B200)

|                           | A100 80GB SXM | H100 SXM      | H200 SXM      | GB200 NVL72   | HGX B200      |
| -                         | -             | -             | -             | -             | -             |
| BF16 Tensor Core (Dense)  | 312 TFLOPS    | 989 TFLOPS    | 989 TFLOPS    | 2.5 PFLOPS    | 2.25 PFLOPS   |
| FP8 Tensor Core (Dense)   | -             | 1,979 TFLOPS  | 1,979 TFLOPS  | 5 PFLOPS      | 4.5 PFLOPS    |
| GPU Memory                | 80 GB HBM2e   | 80 GB HBM3    | 141 GB HBM3e  | 186 GB HBM3e  | 180 GB HBM3e  |
| GPU Memory Bandwidth      | 2,039 GB/s    | 3.35 TB/s     | 4.8 TB/s      | 8 TB/s        | 7.7 TB/s      |
| NVLink (Unidirectional)   | 300 GB/s      | 450 GB/s      | 450 GB/s      | 900 GB/s      | 900 GB/s      |
| Host Interface            | PCIe Gen4     | PCIe Gen5     | PCIe Gen5     | NVLink-C2C    | PCIe Gen5     |

> Note: The **Host Interface** (PCIe) limits the GPU-to-NIC throughput. Ensure your simulation checks if `Port Speed > PCIe BW` to identify potential host-side bottlenecks (e.g., A100 on NDR).
>> PCIe (Unidirectional) Bandwidth: Gen4 - 32 GB/s; Gen5 - 64 GB/s.

## 2. Switch Specs (InfiniBand)

|                       | Role          | Generation    | Port Speed| Port Count    |
| -                     | -             | -             | -         | -             |
| Quantum QM8700        | Leaf / Spine  | A100 (HDR)    | 25 GB/s   | 40            | 
| Quantum CS8500        | Director      | A100 (HDR)    | 25 GB/s   | Up to 800     |
| Quantum-2 QM9700      | Leaf / Spine  | H100 (NDR)    | 50 GB/s   | 64            |
| Quantum-2 CS9500      | Director      | H100 (NDR)    | 50 GB/s   | Up to 2,048   |
| Quantum-X800 Q3400    | Leaf / Spine  | B200 (XDR)    | 100 GB/s  | 144           |
| Quantum-X800 Director | Director      | B200 (XDR)    | 100 GB/s  | Scalable      |