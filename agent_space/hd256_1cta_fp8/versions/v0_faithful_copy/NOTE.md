# v0 — 忠实副本（阶段1）
- 内容：主 kernel `FlashAttentionForwardSm100` 的忠实副本，改类名为 `FlashAttentionForwardHd256_1CTA_Sm100`，
  interface 路由 fp8+hd256 → 本 kernel。q_stage=1，S/P 已双缓冲(tmem)但调度只用单缓冲（无 K-ping-pong）。
- golden（new vs 主 kernel）：逐位一致 max_abs=0.00000（全 shape PASS）。
- 性能（目标 shape varlen b1 s8192 hq32 hkv2 d256 fp8 causal）：748.7us / 1468.6 TFLOP/s。
- 用途：K-ping-pong 的正确性/性能基线与回退点。
