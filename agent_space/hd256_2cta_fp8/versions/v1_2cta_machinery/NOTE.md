# v1 (S1) — 启用继承的 2-SM(pair-UMMA)机制
- 关键发现:2cta 文件继承自 flash_fwd_sm100.py,**本身就含完整 use_2cta_instrs 机制**(cluster、pair-UMMA、multicast load、CTA-local softmax、对称 tmem)。主 kernel 的 2cta 本就是"CTA-local softmax + pair-UMMA"(即用户要的最小同步设计),开源独立文件才是多余同步那个。
- 改动:interface 对 `use_fp8_hd256_2cta` 置 `use_2cta_instrs=True`(kernel 文件未改,仍是 v0 拷贝)。
- 验证(ncu 确认真跑 2-CTA:cluster_dim_x=2, block 512, 持久 grid=148):
  - **非 causal golden 逐位一致 0.00000 PASS**(b2 s512 h4)。
  - **非 causal 性能(b1 s8192 hq32 hkv2 d256):2cta 997.8us vs 1cta(v4)973.0us → 2cta 慢 ~2.5%**。
  - **causal:CUDA Error 912 崩溃**(继承的 2cta 不支持 causal,需 union-trip-range = S5)。
- 结论:M=256 pair-UMMA 的张量效率**没有转化为 wall-time**(若真 tensor-bound,997→应≈500us)→ 再次印证 kernel 是 **latency-bound 非 tensor-bound**,2-SM 收益被 cluster 同步开销吃掉。与文档"attention tile fit 512 + latency-bound 优先 1-SM"一致。
