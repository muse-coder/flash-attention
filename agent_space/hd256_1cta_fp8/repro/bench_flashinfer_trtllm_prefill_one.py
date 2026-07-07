"""Single-dtype FlashInfer trtllm-gen paged prefill benchmark/profiling target.

Use this for NCU so launch filtering does not have to skip through both output
dtypes from bench_flashinfer_trtllm_prefill.py.

Run:
  cd /home/mudi/flash-attention
  PYTHONPATH=/home/mudi/flashinfer CUDA_VISIBLE_DEVICES=0 \
  FLASHINFER_OUT_DTYPE=fp8 /home/mudi/mudi_env/bin/python \
    agent_space/hd256_1cta_fp8/repro/bench_flashinfer_trtllm_prefill_one.py
"""

import math
import os
import sys

import torch


FLASHINFER_ROOT = "/home/mudi/flashinfer"
if FLASHINFER_ROOT not in sys.path:
    sys.path.insert(0, FLASHINFER_ROOT)

import flashinfer  # noqa: E402


def make_inputs(S=8192, HQ=32, HKV=2, D=256, page_size=64, o_dtype=torch.float8_e4m3fn):
    fp8 = torch.float8_e4m3fn
    q = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16).to(fp8)
    num_pages = S // page_size
    kv_cache = torch.randn(
        num_pages, 2, HKV, page_size, D, device="cuda", dtype=torch.bfloat16
    ).to(fp8)
    qo_indptr = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, num_pages], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    last_page_len = torch.tensor([page_size], device="cuda", dtype=torch.int32)
    out = torch.empty(S, HQ, D, device="cuda", dtype=o_dtype)
    return q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, out


def run_one(o_dtype):
    torch.manual_seed(0)
    S, HQ, HKV, D, page_size = 8192, 32, 2, 256, 64
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
    warmup = int(os.environ.get("FLASHINFER_WARMUP", "1"))
    for _ in range(warmup):
        wrapper.run(q, kv_cache, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out)
    torch.cuda.synchronize()
    wrapper.run(q, kv_cache, q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out)
    torch.cuda.synchronize()
    print(f"done out={out.dtype} shape={tuple(out.shape)}")


if __name__ == "__main__":
    os.environ.setdefault("FLASHINFER_LOGGING_LEVEL", "warning")
    dtype_name = os.environ.get("FLASHINFER_OUT_DTYPE", "fp8")
    out_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float8_e4m3fn
    run_one(out_dtype)
