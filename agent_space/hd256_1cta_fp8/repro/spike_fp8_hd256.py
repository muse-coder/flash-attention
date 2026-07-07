"""Spike: route fp8 hd256 through the general SM100 kernel (1-CTA, q_stage=1).

Validates the highest-risk items: tmem fit (S 2x128 + O 256 = 512), PV N=256, smem budget.
descale=None (=> scale 1). fp8 output is bf16. Reference computed in fp32 from fp8-casted inputs.

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/spike_fp8_hd256.py
"""
import sys
import torch
import flash_attn.cute.interface as I

torch.manual_seed(0)
fp8 = torch.float8_e4m3fn


def ref_attn(q, k, v, causal):
    # q,k,v are fp8; upcast to fp32 for reference
    qf, kf, vf = q.float(), k.float(), v.float()
    b, s, h, d = qf.shape
    sk = kf.shape[1]
    hkv = kf.shape[2]
    if h != hkv:  # GQA: repeat kv heads
        rep = h // hkv
        kf = kf.repeat_interleave(rep, dim=2)
        vf = vf.repeat_interleave(rep, dim=2)
    scale = 1.0 / (d ** 0.5)
    sc = torch.einsum("bshd,bthd->bhst", qf, kf) * scale
    if causal:
        mask = torch.triu(torch.ones(s, sk, device=q.device, dtype=torch.bool), 1 + sk - s)
        sc = sc.masked_fill(mask, float("-inf"))
    p = sc.softmax(-1)
    return torch.einsum("bhst,bthd->bshd", p, vf.float())


def run(b, s, h, hkv, d, causal):
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16).to(fp8)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16).to(fp8)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16).to(fp8)
    out = I.flash_attn_func(q, k, v, causal=causal)
    out = out[0] if isinstance(out, tuple) else out
    ref = ref_attn(q, k, v, causal)
    err = (out.float() - ref).abs().max().item()
    denom = ref.abs().max().item()
    tag = f"b{b} s{s} h{h}/{hkv} d{d} causal={causal}"
    print(f"{tag:40s} max_abs_err={err:.4f} (ref_max={denom:.3f})  {'PASS' if err < 0.15 else 'FAIL'}")
    return err < 0.15


ok = True
ok &= run(1, 128, 1, 1, 256, False)
ok &= run(1, 256, 1, 1, 256, True)
ok &= run(2, 512, 4, 4, 256, True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
