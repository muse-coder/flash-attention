# 版本迭代留痕（versions/）

约定：每次有意义的迭代存一个 `versions/vN_<desc>/`，含：
- `flash_fwd_hd256_1cta_sm100.py`：该版 kernel 快照
- `NOTE.md`：改了什么 + golden 结果 + 性能(目标 shape TFLOP/s) + 结论/下一步
每版单独 git commit（message 前缀 `vN:`）。回退直接 `cp versions/vN/....py flash_attn/cute/`。

| 版本 | 描述 | golden | 目标shape TFLOP/s | 备注 |
|---|---|---|---|---|
| v0 | 忠实副本(阶段1) | 逐位一致 | 1468.6 | 基线/回退点 |
| v1 | K-ping-pong plumbing only | 逐位一致 | 1463.6 | 新增 s_pp 与拆分 pipeline，未接线 |
| v2 | First K-ping-pong schedule | 逐位一致 | 1461.5 | 可运行但未提速，split-P 暂未接入 |
| v3 | True dedicated baseline + compile-key split | 逐位一致 | 1889.6 | 修复 main/dedicated cache alias；NCU 确认真实 `FlashAttentionForwardHd256_1CTA_Sm100`，v1/v2 需重验 |
| v4 | KPP fixed + 10-warp compact + split-P + regs | 逐位一致 | 2514.0 | ncu 重测：FA4 ~430.9us vs FI trtllm-gen fp8 ~445.4us → **快~3.3%**（旧 492us event 计时作废） |
| v5 | kv_stage 4→5 (overlap_sO_sQ) | 逐位一致 | ~430.1us(ncu) | **负结果**：内存非瓶颈，已回退 |
| v6 | split_P sweep {96,64,32} | 逐位一致 | ~431us(ncu) | **负结果**：无提升，已回退 |
| v7 | ex2 emulation sweep {0,8,16} | (计算实验) | ~432us(ncu) | **负结果**:exp2 非瓶颈,已回退 |
| v8 | softmax 关键路径探针(FA_SM_DOUBLE) | (诊断) | softmax×5→时间不变 | **决定性**:softmax 已被掩盖,双 softmax 无用,已回退 |
| v9 | 非causal 紧凑 bonus | causal目标不变(431.9us) | 非causal 973→936.5us | 源自2cta判据实验;causal target 0.00000 |
| r2 | correction+pack_gqa(Round2) | correction: 逐位一致;pack_gqa golden PASS | correction null;pack_gqa 562(慢30