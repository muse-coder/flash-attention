"""Verify fp8 hd256 1-CTA using the official attention_ref + 2x tolerance methodology.

Mirrors tests/cute/test_flash_attn.py: out_ref = upcast fp32 ref, out_pt = fp8-intermediate ref,
pass iff (out-out_ref).amax <= 2*(out_pt-out_ref).amax + 2*rounding(out_ref).
descale=None (scale 1).
"""
import sys
import torch
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref

torch.manual_seed(0)
fp8 = torch.float8_e4m3fn
dtype_ref = torch.bfloat16


def run(b, sq, sk, h, hkv, d, causal):
    q_ref = torch.randn(b, sq, h, d, device="cuda", dtype=dtype_ref).to(fp8).to(dtype_ref).requires_grad_()
    k_ref = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype_ref).to(fp8).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype_ref).to(fp8).to(dtype_ref).requires_grad_()
    q, k, v = [x.detach().to(fp8) for x in (q_ref, k_ref, v_ref)]

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_pt, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=False,
                              reorder_ops=True, intermediate_dtype=fp8)
    out = flash_attn_func(q, k, v, causal=causal)
    out = out[0] if isinstance(out, tuple) else out

    err = (out.float() - out_ref.float()).abs().max().item()
    pt_err = (out_pt.float() - out_ref.float()).abs().max().item()
    atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    thresh = 2 * pt_err + atol
    ok = err <= thresh
    tag = f"b{b} sq{sq} sk{sk} h{h}/{hkv} d{d} causal={causal}"
    print(f"{tag:38s} err={err:.4f} pt_err={pt_err:.4f} thresh={thresh:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


ok = True
ok &= run(1, 128, 128, 1, 1, 256, False)
ok &= run(1, 256, 256, 1, 1, 256, True)
ok &= run(2, 512, 512, 4, 4, 256, True)
ok &= run(2, 512, 512, 4, 1, 256, True)   # MQA
ok &= run(1, 384, 384, 2, 2, 256, False)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
