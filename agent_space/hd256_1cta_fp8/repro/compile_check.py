"""Minimal FakeTensor compile check for SM100 path with forced arch.

Usage:
  FLASH_ATTENTION_ARCH=sm_103 CUTE_DSL_ARCH=sm_100 \
  FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/compile_check.py
"""
import os
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import flash_attn.cute.interface as I
from flash_attn.cute.interface import _get_device_arch

print("arch (kernel-select) =", _get_device_arch())

dtype = torch.bfloat16
b, s, h, d = 2, 512, 4, 128

with FakeTensorMode(allow_non_fake_inputs=True):
    q = torch.randn(b, s, h, d, dtype=dtype, device="cuda")
    k = torch.randn(b, s, h, d, dtype=dtype, device="cuda")
    v = torch.randn(b, s, h, d, dtype=dtype, device="cuda")
    out = I.flash_attn_func(q, k, v, causal=True)
    print("bf16 hd128 causal compiled, out shape:", tuple(out.shape))

print("OK")
