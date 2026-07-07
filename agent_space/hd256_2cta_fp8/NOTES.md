# hd256 2-CTA fp8 kernel — 开发笔记

最小同步 pair-UMMA 方案。详见 `PLAN.md`。1cta 方案在 `../hd256_1cta_fp8/`(最佳 v4 ≈430us,作对照/回退基准)。

## 环境(同 1cta)
```
cd /home/mudi/flash-attention
PYTHONPATH=/home/mudi/flash-attention CUTE_DSL_ARCH=sm_103a \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 CUDA_VISIBLE_DEVICES=<free> \
/home/mudi/mudi_env/bin/python <script>
```

## 迭代记录
### [init] 2026-07-07 — 建目录 + 写 PLAN
- 建 `agent_space/hd256_2cta_fp8/`(独立嵌套 git)。写 PLAN.md。待 review 后按 S0→S6 实施。

### [v0/S0] 2026-07-07 — 忠实副本 + 路由骨架
- 拷贝 v4→`flash_fwd_hd256_2cta_sm100.py`,改类名;interface 加 `FA_HD256_2CTA` 开关 + 独立 compile-key。
- golden vs 主 kernel:逐位一致(0.00000)全 PASS。S0 完成。

### [v1/S1] 2026-07-07 — 启用继承 2-SM 机制;非causal 通过但不提速,causal 崩溃
- 捷径:2cta 文件继承 flash_fwd_sm100.py 的完整 use_2cta_instrs 机制。interface 置 use_2cta_instrs=True。
- 非 causal:golden 0.00000 PASS;性能 997.8us vs 1cta 973.0us(慢 2.5%)。ncu 确认 cluster_dim_x=2 真 2-CTA。
- causal:CUDA Error 912(需 union-trip-range,S5)。
- **重要判据:M=256 pair-UMMA 未转化为 wall-time 收益 → 再次确认 latency-bound,2-SM 不划算。**

### [S5 设计] 2026-07-07 — causal+LPT 的 2-CTA 适配(待斟酌)
- causal 2cta 死锁根因:① block_info tile_m=128 vs pair 256 → causal 范围错;② LPT 调度器 cluster 对齐需确认。
- 适配:范围用 pair-M(union,两 CTA 同范围)+ LPT cluster 粒度负载均衡 + 本地 mask。mask 已确认对(tScS 坐标+m_block*2 base)。仍是最小同步。
- 诚实斟酌点:非 causal 2cta 已证明不快过 1cta(973 vs 997),causal 适配大工程且收益存疑。详见 PLAN.md S5。

### [v2/判据] 2026-07-07 — 非causal 2cta 紧凑后仍不敌 1cta(方向关闭)
- 非causal 2cta 紧凑 16→11 warp:golden PASS;997.8→962.5us。
- 公平对比(1cta 非causal 也紧凑):**1cta 936.5 vs 2cta 962.5 → 1cta 快 2.8%**。
- profile:2cta eligible 仅 0.26(cross-CTA 同步饥饿未解),multicast 省 DRAM(2.05%)但 latency-bound 用不上。
- **判据结论:2cta 调优后仍输 1cta → causal 2cta 不值得做,方向关闭。接受 1cta v4 定版。**
