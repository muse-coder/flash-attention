"""FlashInfer trtllm-gen paged prefill benchmark for the target hd256 fp8 shape.

Target shape mirrors PLAN.md:
  batch=1, seqlen=8192, Q heads=32, KV heads=2, head_dim=256, causal, fp8.

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flashinfer CUDA_VISIBLE_DEVICES=0 \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/bench_flashinfer_trtllm_prefill.py
"""

import math
import os
import sys

import numpy as np
import torch


FLASHINFER_ROOT = "/home/mudi/flashinfer"
if FLASHINFER_ROOT not in sys.path:
    sys.path.insert(0, FLASHINFER_ROOT)

import flashinfer  # noqa: E402


def bench_gpu_time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.array(times)


def make_inputs(S=8192, HQ=32, HKV=2, D=256, page_size=64, o_dtype=torch.float8_e4m3fn):
    assert S % page_size == 0
    fp8 = torch.float8_e4m3fn
    q = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16).to(fp8)
    num_pages = S // page_size
    # HND layout for trtllm-gen: [num_pages, 2, num_kv_heads, page_size, head_dim].
    kv_cache = torch.randn(
        num_pages, 2, HKV, page_size, D, device="cuda", dtype=torch.bfloat16
    ).to(fp8)
    qo_indptr = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, num_pages], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)
    out = torch.empty(S, HQ, D, device="cuda", dtype=o_dtype)
    return q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, out


def run_one(o_dtype, page_size=64):
    torch.manual_seed(0)
    S, HQ, HKV, D = 8192, 32, 2, 256
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, out = make_inputs(
        S, HQ, HKV, D, page_size, o_dtype
    )
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="trtllm-gen"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        HQ,
        HKV,
        D,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        sm_scale=1.0 / math.sqrt(D),
        q_data_type=q.dtype,
        kv_data_type=kv_cache.dtype,
        o_data_type=o_dtype,
    )

    def fn():
        return wrapper.run(q, kv_cache, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out)

    fn()
    torch.cuda.synchronize()
    times = bench_gpu_time(fn)
    med = float(np.median(times))
    p20 = float(np.percentile(times, 20))
    p80 = float(np.percentile(times, 80))
    flops = 2 * 2 * HQ * S * S * D * 0.5
    tflops = flops / (med * 1e-3) / 1e12
    print(
        f"[flashinfer trtllm-gen o_dtype={str(o_dtype).split('.')[-1]} page={page_size}] "
        f"median={med * 1000:.1f} us p20={p20 * 1000:.1f} p80={p80 * 1000:.1f} "
        f"{tflops:.1f} TFLOP/s out={tuple(out.shape)} {out.dtype}"
    )
    return med


if __name__ == "__main__":
    os.environ.setdefault("FLASHINFER_LOGGING_LEVEL", "warning")
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        run_one(dtype)
