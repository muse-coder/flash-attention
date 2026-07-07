# v2 (S1+tune) — 非causal 2cta 紧凑 + 判据结论
- 改动:非 causal TMA-O 路径也做 16→11 warp 紧凑(去掉 5 个 parked empty/softmax1 warp)。golden 0.00000 PASS。
- 机制确认(profile):multicast 生效,DRAM 降到 2.05%(vs 1cta ~4.8%),每 CTA 只存半份 K/V。
- **判据(apples-to-apples,都紧凑 11 warp,非 causal b1 s8192 hq32 hkv2)**:
  - **1cta 936.5us vs 2cta 962.5us → 1cta 快 ~2.8%**。
- **调度指标**:紧凑后 2cta 仍 eligible 0.26 / no-eligible 74.5% / compute 54% —— cross-CTA 同步饥饿未解决。
- **结论**:即使都调优,**2cta 输给 1cta**。multicast 省的 DRAM 用不上(latency-bound),pair-UMMA cross-CTA 同步是净负担。→ **causal 2cta 大工程不值得投,2cta 方向关闭。**
