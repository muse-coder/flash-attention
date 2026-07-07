"""Compare FA4 hd256 1-CTA fp8 vs flashinfer BatchPrefillWithPagedKVCacheWrapper fp8, scale=1.

Both consume the SAME fp8 values. flashinfer: q=fp8->fp16 (lossless), kv fp8 with k_scale=v_scale=1.
FA4: q/k/v fp8, descale=None. sm_scale = 1/sqrt(d) for both.

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/cmp_flashinfer_fp8.py
"""
import math
import sys
import torch
import flashinfer
from flash_attn.cute.interface import flash_attn_func

torch.manual_seed(42)
fp8 = torch.float8_e4m3fn
dev = "cuda"


def flashinfer_paged_fp8(q_f8, k_f8, v_f8, causal, sm_scale, backend="fa2"):
    # q_f8: (b, sq, h, d) fp8 ; k_f8/v_f8: (b, sk, hkv, d) fp8
    b, sq, h, d = q_f8.shape
    sk, hkv = k_f8.shape[1], k_f8.shape[2]
    page_size = sk  # one page per sequence
    q = q_f8.reshape(b * sq, h, d).to(torch.float16)
    # HND paged layout: (total_pages, 2, num_kv_heads, page_size, head_dim)
    kv_data = torch.empty(b, 2, hkv, page_size, d, dtype=fp8, device=dev)
    kv_data[:, 0] = k_f8.permute(0, 2, 1, 3)  # (b, hkv, sk, d)
    kv_data[:, 1] = v_f8.permute(0, 2, 1, 3)
    qo_indptr = (torch.arange(b + 1, device=dev).int()) * sq
    kv_indptr = (torch.arange(b + 1, device=dev).int()) * 1
    kv_indices = torch.arange(b, device=dev).int()
    kv_last_page_len = torch.full((b,), page_size, dtype=torch.int32, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=backend)
    w.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
           h, hkv, d, page_size, causal=causal,
           q_data_type=torch.float16, kv_data_type=fp8, sm_scale=sm_scale)
    o = w.run(q, kv_data, k_scale=1.0, v_scale=1.0)
    return o.reshape(b, sq, h, d).float()


def flashinfer_paged_fp16_ref(q_f8, k_f8, v_f8, causal, sm_scale, backend="fa2"):
    # High-precision ref: same fp8 values dequantized to fp16, bf16 compute.
    b, sq, h, d = q_f8.shape
    sk, hkv = k_f8.shape[1], k_f8.shape[2]
    page_size = sk
    q = q_f8.reshape(b * sq, h, d).to(torch.float16)
    kv_data = torch.empty(b, 2, hkv, page_size, d, dtype=torch.float16, device=dev)
    kv_data[:, 0] = k_f8.permute(0, 2, 1, 3).to(torch.float16)
    kv_data[:, 1] = v_f8.permute(0, 2, 1, 3).to(torch.float16)
    qo_indptr = (torch.arange(b + 1, device=dev).int()) * sq
    kv_indptr = (torch.arange(b + 1, device=dev).int()) * 1
    kv_indices = torch.arange(b, device=dev).int()
    kv_last_page_len = torch.full((b,), page_size, dtype=torch.int32, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend=backend)
    w.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
           h, hkv, d, page_size, causal=causal,
           q_data_type=torch.float16, kv_data_type=torch.float16, sm_scale=sm_scale)
    return w.run(q, kv_data).reshape(b, sq, h, d).float()


def run(b, sq, sk, h, hkv, d, causal, backend="fa2"):
    q = torch.randn(b, sq, h, d, device=dev, dtype=torch.bfloat16).to(fp8)
    k = torch.randn(b, sk, hkv, d, device=dev, dtype=torch.bfloat16).to(fp8)
    v = torch.randn(b, sk, hkv, d, device=dev, dtype=torch.bfloat16).to(fp8)
    sm_scale = 1.0 / math.sqrt(d)
    try:
        o_ref = flashinfer_paged_fp16_ref(q, k, v, causal, sm_scale, backend)
        o_fi = flashinfer_paged_fp8(q, k, v, causal, sm_scale, backend)
    except Exception as e:
        print(f"  flashinfer({backend}) ERR: {type(e).__name__}: {str(e)[:160]}")
        return None
    o_fa = flash_attn_func(q, k, v, causal=causal, softmax_scale=sm_scale)
    o_fa = (o_fa[0] if isinstance(o_fa, tuple) else o_fa).float()
    fi_err = (o_fi - o_ref).abs().max().item()
    fa_err = (o_fa - o_ref).abs().max().item()
    fa_fi = (o_fa - o_fi).abs().max().item()
    # flashinfer's own fp8 tolerance
    try:
        torch.testing.assert_close(o_fa, o_fi, atol=1e-2, rtol=2e-1)
        passed = "PASS"
    except AssertionError:
        passed = "FAIL"
    tag = f"b{b} sq{sq} sk{sk} h{h}/{hkv} d{d} caus={causal}"
    print(f"{tag:36s} FI8-vs-ref={fi_err:.3f} FA4-vs-ref={fa_err:.3f} FA4-vs-FI8={fa_fi:.3f}  [FI-tol]{passed}")
    return fa_err


for backend in ["fa2"]:
    print(f"=== backend={backend} ===")
    run(1, 128, 128, 1, 1, 256, False, backend)
    run(1, 256, 256, 1, 1, 256, True, backend)
    run(2, 512, 512, 4, 4, 256, True, backend)

print("=== hd128 fp8 (shipped path) multi-block sanity ===")
run(1, 128, 128, 1, 1, 128, False)   # 1 block
run(1, 128, 512, 1, 1, 128, False)   # q_stage=1, 4 kv blocks
run(2, 512, 512, 4, 4, 128, False)   # q_stage=2, multi block
run(2, 512, 512, 4, 4, 128, True)    # q_stage=2, causal
print("=== hd64 fp8 ===")
run(2, 512, 512, 4, 4, 64, True)
