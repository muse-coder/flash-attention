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
