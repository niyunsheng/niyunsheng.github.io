---
title: "Beyond Saved Tensors Hooks: Hiding PCIe Latency with Ping-Pong Prefetching"
date: 2026-02-10T23:20:00+08:00
draft: true
tags: ["PyTorch", "Offloading", "Memory Optimization", "Recomputation"]
categories: ["System Optimization"]
summary: "Standard PyTorch activation offloading relies on synchronous, on-demand data transfer, exposing the 'PCIe Wall' during backward passes. This post explores a proactive 'Ping-Pong' prefetching strategy—inspired by Megatron-Kwai—that utilizes double buffering and custom CUDA streams to fully overlap H2D transfer with GPU computation."
---

https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#saving-tensors-to-cpu

https://github.com/kwai/Megatron-Kwai/blob/sc25slimpipe/megatron/core/pipeline_parallel/offload.py

