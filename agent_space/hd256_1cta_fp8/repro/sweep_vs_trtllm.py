"""Sweep: FA4 1cta v9 (hd256 fp8 causal) vs FlashInfer trtllm-gen fp8, single & multi batch.

Both: hd256, HQ=32, HKV=2 (GQA x16), causal, fp8 (e4m3), sm_scale=1/sqrt(256), scale=1.
FA4: flash_attn_varlen_func (B seqs of length S). FI: BatchPrefillWithPagedKVCacheWrapper trtllm-gen.
Same event-timing method for both (relative comparison). FI o_dtype=fp8 (its fastest, strongest baseline).

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention:/home/mudi/flashinfer CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/sweep_vs_trtllm.py
"""
import math, torch
import flash_attn.cute.interface as I
import flashinfer

fp8 = torch.float8_e4m3fn
dev = "cuda"
HQ, HKV, D = 32, 2, 256
PAGE = 64
torch.manual_seed(0)


def timeit(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000  # us


def make_fa4(B, S):
    q = torch.randn(B * S, HQ, D, device=dev, dtype=torch.bfloat16).to(fp8)
    k = torch.randn(B * S, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
    v = torch.randn(B * S, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32) * S
    ss = 1.0 / math.sqrt(D)
    def run():
        return I.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                        max_seqlen_q=S, max_seqlen_k=S, causal=True,
                                        softmax_scale=ss, pack_gqa=False)
    return run


def make_fi(B, S):
    npages_seq = S // PAGE
    total_pages = npages_seq * B
    q = torch.randn(B * S, HQ, D, device=dev, dtype=torch.bfloat16).to(fp8)
    kv = torch.randn(total_pages, 2, HKV, PAGE, D, device=dev, dtype=torch.bfloat16).to(fp8)
    qo_indptr = torch.arange(B + 1, device=dev, dtype=torch.int32) * S
    kv_indptr = torch.arange(B + 1, device=dev, dtype=torch.int32) * npages_seq
    kv_indices = torch.arange(total_pages, device=dev, dtype=torch.int32)
    last = torch.full((B,), PAGE, dtype=torch.int32, device=dev)
    out = torch.empty(B * S, HQ, D, device=dev, dtype=fp8)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="trtllm-gen")
    w.plan(qo_indptr, kv_indptr, kv_indices, last, HQ, HKV, D, PAGE, causal=True,
           pos_encoding_mode="NONE", sm_scale=1.0 / math.sqrt(D),
           q_data_type=fp8, kv_data_type=fp8, o_data_type=fp8)
    def run():
        return w.run(q, kv, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out)
    return run


print(f"{'batch':>5} {'seqlen':>7} {'FA4 us':>9} {'trtllm us':>10} {'speedup':>8}")
for B in [1, 4, 8]:
    for S in [1024, 2048, 4096, 8192]:
        try:
            t_fa = timeit(make_fa4(B, S))
            t_fi = timeit(make_fi(B, S))
            sp = t_fi / t_fa
            print(f"{B:>5} {S:>7} {t_fa:>9.1f} {t_fi:>10.1f} {sp:>7.2f}x")
        except Exception as e:
            print(f"{B:>5} {S:>7}  ERR {type(e).__name__}: {str(e)[:70]}")
