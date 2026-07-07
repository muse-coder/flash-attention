# 版本迭代留痕(versions/)

约定:每个跑通/有结论的节点存 `versions/vN_<desc>/{flash_fwd_hd256_2cta_sm100.py, NOTE.md}`,单独 commit(`vN:` 前缀)。

| 版本 | 描述 | golden | 目标shape TFLOP/s / us | 备注 |
|---|---|---|---|---|
| (待 review 后开始 S0) | | | | 对照:1cta v4 ≈430us |
| v0 | S0 忠实副本+路由骨架 | 逐位一致 | (=v4,未提速) | 1cta 行为,验证路由/compile-key |
| v1 | S1 启用继承2-SM机制 | 非causal逐位一致;causal崩溃 | 非causal 997.8us(vs 1cta 973.0) | M=256 不敌 1cta,latency-bound 确认 |
| v2 | 非causal紧凑+判据 | 非causal逐位一致 | 2cta 962.5 vs 1cta 936.5 | **判据:2cta输给1cta**,方向关闭 |
