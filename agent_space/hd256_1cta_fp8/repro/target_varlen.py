"""Target shape: single seq (batch=1) len=8192 via VARLEN, Q heads=32, KV heads=2, hd=256, fp8, causal.

Golden = general SM100 kernel (FA_HD256_USE_MAIN=1). Also benchmarks the dedicated kernel.
Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/target_varlen.py
"""
import math
import os
import sys
import torch

fp8 = torch.float8_e4m3fn
dev = "cuda"
torch.manual_seed(0)

S = 8192
HQ, HKV, D = 32, 2, 256
CAUSAL = True
SS = 1.0 / math.sqrt(D)


def make_inputs():
    q = torch.randn(S, HQ, D, device=dev, dtype=torch.bfloat16).to(fp8)
    k = torch.randn(S, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
    v = torch.randn(S, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
    cu_q = torch.tensor([0, S], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, S], device=dev, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def call(q, k, v, cu_q, cu_k, pack_gqa, as_float=False):
    import flash_attn.cute.interface as I
    o = I.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                 max_seqlen_q=S, max_seqlen_k=S, causal=CAUSAL,
                                 softmax_scale=SS, pack_gqa=pack_gqa)
    out = o[0] if isinstance(o, tuple) else o
    return out.float() if as_float else out


def golden_check(pack_gqa):
    q, k, v, cu_q, cu_k = make_inputs()
    os.environ["FA_HD256_USE_MAIN"] = "1"
    o_main = call(q, k, v, cu_q, cu_k, pack_gqa, as_float=True)
    os.environ["FA_HD256_USE_MAIN"] = "0"
    o_new = call(q, k, v, cu_q, cu_k, pack_gqa, as_float=True)
    diff = (o_new - o_main).abs().max().item()
    ref = o_main.abs().max().item()
    print(f"[golden pack_gqa={pack_gqa}] new-vs-main max_abs={diff:.5f} (main_max={ref:.3f})  "
          f"{'PASS' if diff <= 1e-2 + 0.02*ref else 'FAIL'}")


def bench(pack_gqa, iters=50):
    q, k, v, cu_q, cu_k = make_inputs()
    f = lambda: call(q, k, v, cu_q, cu_k, pack_gqa, as_float=False)
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    fl = 2 * 2 * HQ * S * S * D * (0.5 if CAUSAL else 1.0)
    print(f"[bench pack_gqa={pack_gqa}] {ms*1000:.1f} us   {fl/(ms*1e-3)/1e12:.1f} TFLOP/s")


if __name__ == "__main__":
    pg = False
    golden_check(pg)
    bench(pg)
