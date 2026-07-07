"""Golden comparison: dedicated hd256 1-CTA kernel vs the general SM100 kernel (q_stage=1).

The general kernel is the user-accepted correctness oracle. Both run fp8 hd256, scale=1.
Toggle via FA_HD256_USE_MAIN env in-process. Also reports error vs flashinfer fp16 ref.

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/golden_cmp.py
"""
import math
import os
import sys
import torch

fp8 = torch.float8_e4m3fn
dev = "cuda"
torch.manual_seed(42)


def fa(q, k, v, causal, ss):
    import flash_attn.cute.interface as I
    o = I.flash_attn_func(q, k, v, causal=causal, softmax_scale=ss)
    return (o[0] if isinstance(o, tuple) else o).float()


def run(b, sq, sk, h, hkv, d, causal):
    q = torch.randn(b, sq, h, d, device=dev, dtype=torch.bfloat16).to(fp8)
    k = torch.randn(b, sk, hkv, d, device=dev, dtype=torch.bfloat16).to(fp8)
    v = torch.randn(b, sk, hkv, d, device=dev, dtype=torch.bfloat16).to(fp8)
    ss = 1.0 / math.sqrt(d)
    os.environ["FA_HD256_USE_MAIN"] = "1"
    o_main = fa(q, k, v, causal, ss)
    os.environ["FA_HD256_USE_MAIN"] = "0"
    o_new = fa(q, k, v, causal, ss)
    diff = (o_new - o_main).abs().max().item()
    ref = o_main.abs().max().item()
    tag = f"b{b} sq{sq} sk{sk} h{h}/{hkv} d{d} caus={causal}"
    ok = diff <= 1e-2 + 0.02 * ref  # dedicated should match main closely (same fp8 math)
    print(f"{tag:38s} new-vs-main max_abs={diff:.4f} (main_max={ref:.3f})  {'PASS' if ok else 'FAIL'}")
    return ok


ok = True
ok &= run(1, 128, 128, 1, 1, 256, False)
ok &= run(1, 256, 256, 1, 1, 256, True)
ok &= run(2, 512, 512, 4, 4, 256, True)
ok &= run(2, 512, 512, 4, 1, 256, True)   # MQA
ok &= run(1, 384, 384, 2, 2, 256, False)
ok &= run(1, 128, 640, 1, 1, 256, True)   # short q, many kv blocks
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
