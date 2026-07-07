"""Ragged varlen multi-batch: 3 seqs [4096,5120,6144] together. FA4 vs trtllm-gen fp8.
Accuracy (vs fp32 ref from same fp8 inputs; + FA4 vs main-kernel oracle) and perf (event).
hd256, HQ=32, HKV=2, causal, fp8, scale=1/sqrt(256).

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flash-attention:/home/mudi/flashinfer CUTE_DSL_ARCH=sm_103a \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/bench_scripts/varlen_multibatch.py
"""
import math, os, torch
fp8 = torch.float8_e4m3fn; dev = "cuda"; torch.manual_seed(0)
HQ, HKV, D, PAGE = 32, 2, 256, 64
SEQS = [4096, 5120, 6144]
SS = 1.0 / math.sqrt(D)
total = sum(SEQS)
B = len(SEQS)

# shared fp8 inputs (contiguous ragged)
q = torch.randn(total, HQ, D, device=dev, dtype=torch.bfloat16).to(fp8)
k = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
v = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16).to(fp8)
cu = torch.tensor([0] + list(torch.tensor(SEQS).cumsum(0).tolist()), device=dev, dtype=torch.int32)


def fp32_ref():
    out = torch.empty(total, HQ, D, device=dev, dtype=torch.float32)
    rep = HQ // HKV
    for b in range(B):
        s, e = cu[b].item(), cu[b + 1].item()
        L = e - s
        qf = q[s:e].float()                       # (L,HQ,D)
        kf = k[s:e].float().repeat_interleave(rep, dim=1)  # (L,HQ,D)
        vf = v[s:e].float().repeat_interleave(rep, dim=1)
        sc = torch.einsum("ihd,jhd->hij", qf, kf) * SS      # (HQ,L,L)
        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
        sc = sc.masked_fill(mask, float("-inf"))
        p = sc.softmax(-1)
        out[s:e] = torch.einsum("hij,jhd->ihd", p, vf)
    return out


def run_fa4(use_main=False):
    os.environ["FA_HD256_USE_MAIN"] = "1" if use_main else "0"; os.environ["FA_HD256_2CTA"] = "0"
    import flash_attn.cute.interface as I
    o = I.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=max(SEQS), max_seqlen_k=max(SEQS),
                                 causal=True, softmax_scale=SS, pack_gqa=False)
    return (o[0] if isinstance(o, tuple) else o)


def build_fi():
    import flashinfer
    npages = [(s + PAGE - 1) // PAGE for s in SEQS]
    tot_pages = sum(npages)
    kv = torch.zeros(tot_pages, 2, HKV, PAGE, D, device=dev, dtype=fp8)
    # pack contiguous k/v into paged layout
    pg = 0
    for b in range(B):
        s = cu[b].item()
        for p in range(npages[b]):
            n = min(PAGE, SEQS[b] - p * PAGE)
            kv[pg, 0, :, :n] = k[s + p * PAGE: s + p * PAGE + n].transpose(0, 1)
            kv[pg, 1, :, :n] = v[s + p * PAGE: s + p * PAGE + n].transpose(0, 1)
            pg += 1
    qo_indptr = cu.clone()
    kv_indptr = torch.tensor([0] + list(torch.tensor(npages).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    kv_indices = torch.arange(tot_pages, device=dev, dtype=torch.int32)
    last = torch.tensor([(s - 1) % PAGE + 1 for s in SEQS], device=dev, dtype=torch.int32)
    out = torch.empty(total, HQ, D, device=dev, dtype=fp8)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="trtllm-gen")
    w.plan(qo_indptr, kv_indptr, kv_indices, last, HQ, HKV, D, PAGE, causal=True,
           pos_encoding_mode="NONE", sm_scale=SS, q_data_type=fp8, kv_data_type=fp8, o_data_type=fp8)
    return w, kv, out


def timeit(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


print(f"ragged varlen: seqs={SEQS} total={total} HQ={HQ} HKV={HKV} d={D} causal fp8")
ref = fp32_ref()
o_fa = run_fa4(False).float()
o_main = run_fa4(True).float()
w, kv, out = build_fi()
o_fi = w.run(q.reshape(total, HQ, D), kv, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out).float()

def err(a): return (a - ref).abs().max().item()
print(f"[accuracy] ref_max={ref.abs().max().item():.3f}")
print(f"  FA4  vs fp32-ref : {err(o_fa):.4f}")
print(f"  trtllm vs fp32-ref: {err(o_fi):.4f}")
print(f"  FA4 vs main-oracle: {(o_fa-o_main).abs().max().item():.5f}  (should be ~0)")
print(f"  FA4 vs trtllm     : {(o_fa-o_fi).abs().max().item():.4f}")

t_fa = timeit(lambda: run_fa4(False))
t_fi = timeit(lambda: w.run(q.reshape(total, HQ, D), kv, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out))
print(f"[perf event] FA4={t_fa:.1f}us  trtllm={t_fi:.1f}us  FI/FA4={t_fi/t_fa:.2f}x")
