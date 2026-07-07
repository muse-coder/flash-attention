# 1-CTA causal 优化 Round 2:correction 关键路径 + pack_gqa

对象:1cta hd256 fp8 causal 目标 shape(varlen b1 s8192 hq32 hkv2 d256),当前 v4/v9 ≈ 431.9us。
背景:v5-v8 单点调优(kv_stage/split_P/ex2/softmax探针)全 null,softmax 已证明不在关键路径;
profile 最大 stall = barrier 29% / long_scoreboard 22% / wait 16%,占用率 15%(1 CTA/SM,tmem 锁死)。

## 方向 1:correction 关键路径探针 + 针对性优化

**目的**:确定 causal 1cta 的真正关键路径(已排除 softmax)。若 correction 在关键路径 → 针对性优化。

**依据**:
- 最大 stall 是 **barrier(29%,named barrier / CTA sync)**。
- softmax 探针(v8)证明 softmax 有大量余量(5× softmax→时间不变)。
- correction warp 做 **O 的在线 rescale**,并通过 `O_rescaled`(单 O 版)/`pipeline_s_p_o` 门控 mma 的**下一路 PV**——是 barrier stall 的头号嫌疑:MMA 每块要等 correction 把 O rescale 完才能累加下一个 PV。

**方法**:与 softmax 探针同法。在 `correction_loop` 注入 N 遍额外工作(env `FA_CORR_DOUBLE`),做有副作用不被 DCE 的冗余 rescale/搬运,看 kernel 时间涨幅:
- 涨 ≈ 线性 → correction 在关键路径,值得优化。
- 不涨 → 排除,关键路径在别处(load/mma wait)。

**若确认在关键路径,候选优化**:
- 放宽/优化 O_rescale 门控:`rescale_threshold`(fp8=4.0)调大 → 跳过更多不必要的 O rescale(row_max 增长小的块不 rescale)。
- O_rescale 与 LSE 计算 / 最终 store 解耦,缩短门控 MMA 的关键段。
- 减少 correction↔mma 的 barrier 粒度(如 split-O rescale 早放行)。

## 方向 2:pack_gqa

**目的**:目标 GQA 16:1(hq32/hkv2)。pack_gqa 把 `qhead_per_kvhead=16` 个 q-head 打进 M 维,一次 KV tile 服务多个 q-head,提升 K/V 复用、减少冗余 KV load 与 tile 调度开销。

**依据**:
- 现 `pack_gqa=False`(interface 对 hd256 强制关闭,line ~907/924)→ 32 个 q-head 各自独立走 tile,KV 靠 L2 复用但调度/load 有冗余。
- pack_gqa 后 m_block 内含多个 q-head 的行,KV load 一次多用 → 潜在减少 long_scoreboard(等 KV)与 launch 数。

**方法**:对 fp8 hd256 1cta 放开 pack_gqa;golden(vs 主 kernel,注意 pack_gqa 的输出布局)+ 目标 shape ncu。
**风险/约束**:hd256 历史禁 pack_gqa;`m_block % qhead_per_kvhead==0` 约束(128%16=0 ✓);tmem/正确性需验证;varlen + pack_gqa 组合。

## 验证 & 版本管理
- 每步:`golden_cmp.py` / `target_varlen.py`(vs `FA_HD256_USE_MAIN=1` 主 kernel)+ 干净 ncu 中位。
- 留痕:`versions/vN_<desc>/{kernel 快照, NOTE.md}` + git commit(`vN:`)+ 更新 VERSIONS.md。回退点:v4/v9。
- 环境:`PYTHONPATH=仓库 CUTE_DSL_ARCH=sm_103a FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 /home/mudi/mudi_env/bin/python`。

## 诚实预期
方向 1 若命中(correction 关键路径)有真实收益;方向 2 pack_gqa 对 latency/占用-bound 帮助不确定(不改 1 CTA/SM 上限),但减少冗余 KV load 可能压 long_scoreboard。均以实测为准,null 也留痕。
