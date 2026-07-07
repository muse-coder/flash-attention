# Benchmark 脚本(FA4 1cta v9 vs FlashInfer trtllm-gen fp8)

统一环境:
```
PYTHONPATH=/home/mudi/flash-attention[:/home/mudi/flashinfer] CUTE_DSL_ARCH=sm_103a \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=<gpu> B=.. S=.. HQ=.. HKV=.. python <script>
```

- `fa4_one_param.py` / `fi_one_param.py`:参数化(B,S,HQ,HKV)单发,给 ncu 抓 kernel(FA4 名 `hd256`,FI 名 `fmha`)。
  FI 支持非 64 对齐 seqlen(向上取整分页 + last_page_len)。
- `ncu_sweep.sh`:batch{1,4,8}×seqlen{1k,2k,4k,8k} ncu kernel-only 对比 → `../bench_data/ncu_kernel_sweep.txt`
- `ncu_long.sh`:b1 × {8k,10k,12k,14k,16k} → `ncu_long_sweep.txt`
- `ncu_irr.sh`:b1 × 不规则 {8500,9700,11300,12900,14500,15900} → `ncu_irr_sweep.txt`
- `ncu_heads.sh`:HQ/HKV∈{(16,1),(8,1),(16,2)} × {8k,12k,16k} → `ncu_heads_sweep.txt`
- `golden_irr.py` / `golden_heads.py`:对应正确性(vs FA_HD256_USE_MAIN=1 主 kernel oracle)。
- 上层 event 计时 sweep:`../sweep_vs_trtllm.py`(→ `sweep_out.txt`)。
口径:ncu `gpu__time_duration.sum` 中位/3 次(kernel-only,排除 host wrapper 开销)。结论见 `../BENCHMARK_vs_trtllm.md`。
