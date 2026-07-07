# FA4 1cta v9 vs FlashInfer trtllm-gen fp8 — 完整 benchmark

**日期**:2026-07-07  **GPU**:B300(sm_103a)

## 配置
- 两者均:hd256,HQ=32 / HKV=2(GQA×16),causal,**fp8(e4m3)**,sm_scale=1/√256,scale=1。
- FA4:1cta v9(`flash_fwd_hd256_1cta_sm100.py`),`flash_attn_varlen_func`,B 个长度 S 的序列。
- FI:`BatchPrefillWithPagedKVCacheWrapper(backend="trtllm-gen")`,q/kv 均 fp8,page=64,o_dtype=fp8(其最快档)。
- **口径:ncu kernel-only(`gpu__time_duration.sum` 中位/3 次)**——排除 host wrapper 开销,公平对比 kernel 本身。

## 完整对比表(ncu kernel-only)

| batch | seqlen | FA4 kernel (us) | trtllm kernel (us) | FI/FA4 (>1=FA4更快) |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 28.1 | 23.5 | 0.84x |
| 1 | 2048 | 56.9 | 50.7 | 0.89x |
| 1 | 4096 | 142.7 | 138.9 | 0.97x |
| **1** | **8192** | **434.7** | **452.0** | **1.04x** ✅ |
| 4 | 1024 | 79.7 | 55.9 | 0.70x |
| 4 | 2048 | 193.7 | 145.0 | 0.75x |
| 4 | 4096 | 536.6 | 459.9 | 0.86x |
| 4 | 8192 | 1717.6 | 1676.3 | 0.98x |
| 8 | 1024 | 152.0 | 99.3 | 0.65x |
| 8 | 2048 | 376.7 | 268.5 | 0.71x |
| 8 | 4096 | 1060.9 | 903.1 | 0.85x |
| 8 | 8192 | 3437.2 | 3338.0 | 0.97x |

## 结论

1. **FA4 只在目标 shape(b1 s8192)领先:1.04x**(434.7 vs 452.0us)。这与之前记录的 ~432us 一致。
2. **其余所有配置 FA4 都慢于 trtllm-gen**(kernel-only,已排除 wrapper 开销 → 是真实 kernel 差距):
   - **小 seqlen 最差**:s1024 各 batch 0.65–0.84x。
   - **多 batch 拉大差距**:同 seqlen 下 batch 越大 FA4 越吃亏(如 s2048:b1 0.89x → b8 0.71x)。
   - **seqlen 越大越接近持平**:s8192 各 batch ≈0.97–1.04x。
3. **趋势**:FA4 1cta 是**为"大 seqlen + 低 batch"调优的**;GPU 填满时(tile 多)K-ping-pong overlap 发挥,反超 trtllm。小 shape / 多 batch 时:
   - kernel 欠载(小 seqlen 的 CTA 数少,~1-2 waves);
   - 每-CTA(1 CTA/SM,占用率 15%)固定开销占比高;
   - trtllm-gen 的持久 + paged 调度对多 batch/小 shape 扩展更好。

## 与 event 计时的差异(重要)
- 本表是 **ncu kernel-only**。之前 `sweep_vs_trtllm.py` 的 **event 端到端**数字更高(FA4 含 ~25–58us host wrapper 开销:张量校验/layout 转置/cache 查找)。
- 例:b1 s8192 → ncu kernel 434.7us,event ~490us。**目标 shape 两口径下 FA4 均胜**。
- 小 shape 的 event 差距(如 b1 s1024 event 0.39x vs kernel 0.84x)主要来自 FA4 的 host 开销,而非 kernel。

## 定位:FA4 落后的两个可优化方向(未做)
- **A. host wrapper 开销**:小 shape 端到端受其拖累(~25-58us)。可精简 `flash_attn_varlen_func` 的 host 路径。
- **B. 小 seqlen kernel 欠载 / 多 batch 扩展**:1cta 占用率 15% 上限(tmem 锁死)是根因;多 batch 下更明显。持久调度/更好的 tile 负载均衡可能改善,但受结构约束。

repro:`repro/sweep_vs_trtllm.py`(event)、`repro/ncu_kernel_sweep.txt`(本表原始数据)。

## 长 seqlen 单并发(b1)扩展(ncu kernel-only,2026-07-07)

| batch | seqlen | FA4 kernel (us) | trtllm kernel (us) | FI/FA4 |
|---:|---:|---:|---:|---:|
| 1 | 8192 | 434.7 | 444.6 | 1.02x |
| 1 | 10240 | 641.6 | 671.3 | 1.05x |
| 1 | 12288 | 891.5 | 950.1 | 1.07x |
| 1 | 14336 | 1174.2 | 1274.4 | **1.09x** |
| 1 | 16384 | 1508.4 | 1628.6 | 1.08x |

**结论**:单并发下 **seqlen 越长,FA4 领先越大**(8k 1.02x → 14k 1.09x → 16k 1.08x)。
长上下文单并发 prefill 是 FA4 1cta v9 的甜点区(K 块多 → K-ping-pong overlap + 深 KV 流水收益大,固定 per-CTA 开销被摊薄)。
repro:`repro/ncu_long_sweep.txt`。

## 长 seqlen 瓶颈分析(b1 s16384,ncu)
- **Compute(SM) 72%**(8k 时 61%)、Memory 47%、DRAM 3.2%、Occupancy 15%(1 CTA/SM)、eligible 0.45 / No-Eligible 59%。
- stall(cyc/issued,总~4.6,比 8k 6.54 少):**barrier 1.74(最大)/ long_scoreboard 1.23 / wait 1.00 / short 0.47**。
- **为何长序列 FA4 更快**:K 块多 → 更好摊薄 per-CTA 固定开销、张量核更忙(compute 61%→72%)、eligible warps 0.41→0.45。
- **能否再优化(实测)**:kv_stage 4→5(overlap_sO_sQ)在 16k **仍 null**(1513 vs 1515us)→ 深化 KV 流水无效,long_scoreboard 非绑定约束。
- **结论**:长序列瓶颈 = barrier 同步(warp-role 握手,softmax/correction 已证非关键)+ 残余内存/MMA 延迟,在 1 CTA/SM 占用下。compute 已 72%,便宜旋钮(kv_stage)用尽。再压需攻 per-block barrier 同步(深、风险高,预期 ~10-15%)。FA4 长序列已领先 trtllm 1.05-1.09x。

## 不规则 seqlen 单并发(b1,ncu kernel-only,2026-07-07)
非 64/128 对齐的 seqlen(测 varlen 部分 tile / paged 部分页路径):

| batch | seqlen | FA4 kernel (us) | trtllm kernel (us) | FI/FA4 |
|---:|---:|---:|---:|---:|
| 1 | 8500 | 470.0 | 486.5 | 1.04x |
| 1 | 9700 | 582.2 | 613.5 | 1.05x |
| 1 | 11300 | 772.9 | 820.2 | 1.06x |
| 1 | 12900 | 966.0 | 1041.2 | 1.08x |
| 1 | 14500 | 1212.6 | 1304.4 | 1.08x |
| 1 | 15900 | 1448.2 | 1559.2 | 1.08x |

- **平滑无断崖**:不规则 seqlen 下 FA4 依然稳定领先 1.04-1.08x,随 seqlen 增长扩大,与规则 seqlen 完全一致。
- **正确性**:S=9700/12900/15900 golden vs 主 kernel **逐位一致 max_abs=0.00000 PASS**(varlen 部分 tile 处理正确)。
- repro:`repro/ncu_irr_sweep.txt`。

## 不同 attention head 配置(b1,ncu kernel-only,2026-07-07)
hd256 fp8 causal,单并发,varlen。测不同 (HQ, HKV):

| HQ | HKV | (GQA比) | seqlen | FA4 (us) | trtllm (us) | FI/FA4 |
|---:|---:|:---:|---:|---:|---:|---:|
| 16 | 1 | 16:1 | 8192 | 223.1 | 252.4 | 1.13x |
| 16 | 1 | 16:1 | 12288 | 447.0 | 504.9 | 1.13x |
| 16 | 1 | 16:1 | 16384 | 749.9 | 852.8 | 1.14x |
| 8 | 1 | 8:1 | 8192 | 119.2 | 155.4 | **1.30x** |
| 8 | 1 | 8:1 | 12288 | 229.6 | 298.1 | **1.30x** |
| 8 | 1 | 8:1 | 16384 | 377.0 | 471.4 | **1.25x** |
| 16 | 2 | 8:1 | 8192 | 223.2 | 253.9 | 1.14x |
| 16 | 2 | 8:1 | 12288 | 451.1 | 510.0 | 1.13x |
| 16 | 2 | 8:1 | 16384 | 751.9 | 855.1 | 1.14x |

**关键发现:head 数越少,FA4 领先越大**:
- HQ=32(之前):~1.02–1.09x
- HQ=16(nkv=1 或 2):~1.13–1.14x
- **HQ=8:1.25–1.30x**(最大)
- nkv(1 vs 2)对 FA4 几乎无影响(16/1 与 16/2 时间基本一致)——KV 都小,瓶颈在 q 侧处理。
- 推测:head 少 → CTA 少 → trtllm-gen 的持久调度尾效应/负载不均更明显,而 FA4 的 per-CTA 结构相对更稳。
- **正确性**:HQ/HKV=(16,1)/(8,1)/(16,2) golden vs 主 kernel **逐位一致 0.00000 PASS**。
- repro:`repro/ncu_heads_sweep.txt`。

## 总体结论(截至 2026-07-07)
FA4 1cta v9 在 **hd256 fp8 causal 单并发长上下文(≥8k)** 全面稳定领先 FlashInfer trtllm-gen fp8:
- HQ=32:1.02–1.09x;HQ=16:~1.14x;HQ=8:~1.25–1.30x(head 越少领先越大)。
- 规则/不规则 seqlen 均平滑,golden 全部逐位一致。
FA4 落后的场景:小 seqlen(<4k)、多 batch(kernel 欠载 + 1cta 占用率 15% 上限 + host wrapper 开销)。

---

## 追加:ragged 多batch + 单并发长序列(2026-07-07,含 fp8 P-saturation 修复)

> 本节数据在 `max_offset=8→4` 修复之后测(见 `FP8_RESCALE_SATURATION_FIX.md`)。修复前 fp8
> causal 精度 rel_L1≈15.6%;修复后 ~4.5%,追平 trtllm-gen 且对 fp32 golden 比其更准。
> GPU:B300(sm_103a),空闲卡,ncu kernel-only(`gpu__time_duration.sum` 中位);精度 vs
> trtllm-gen 输出,quant scale=1,sm_scale=1/√256,HQ=32/HKV=2。

### 精度(全场景稳定 ~4.5% rel_L1)

| 场景 | total | rel_L1 | cos | max | vs fp32: FA4 / trtllm(MAE) |
|---|---:|---:|---:|---:|---|
| ragged [4096,5120,6144] | 15360 | 4.36% | 0.99917 | 0.156 | — |
| ragged [1200,2800,1600,2000,1400] | 9000 | 4.21% | 0.99921 | 0.188 | 0.00138 / 0.00194 |
| ragged [7800,9100,10400,8600] | 35900 | 4.51% | 0.99915 | 0.172 | 0.00068 / 0.00100 |
| ragged [8000,9500,10200,8300,9800] | 45800 | 4.51% | 0.99915 | 0.188 | 0.00068 / 0.00100 |
| 单并发 8000 | 8000 | 4.50% | 0.99915 | 0.148 | — |
| 单并发 9000 | 9000 | 4.49% | 0.99916 | 0.188 | — |
| 单并发 10000 | 10000 | 4.55% | 0.99914 | 0.125 | — |

对 fp32 golden,**FA4 一致比 trtllm-gen 更准**(FA4 MAE < trtllm MAE)。

### 性能(ncu kernel-only)

单并发长序列 —— FA4 领先,且序列越长优势越大:

| batch | seqlen | FA4 (us) | trtllm (us) | FI/FA4 |
|---:|---:|---:|---:|---:|
| 1 | 8000 | 421.7 | 434.7 | **1.03x** ✅ |
| 1 | 9000 | 516.2 | 539.9 | **1.05x** ✅ |
| 1 | 10000 | 621.6 | 680.6 | **1.09x** ✅ |

ragged 多batch长序列 —— 打平:

| seqs | total | FA4 (us) | trtllm (us) | FI/FA4 |
|---|---:|---:|---:|---:|
| [7800,9100,10400,8600] | 35900 | 2060.4 | 2045.4 | 0.99x |

ragged 多batch短序列 —— 欠载落后(已知,暂缓优化):

| seqs | total | FA4 (us) | trtllm (us) | FI/FA4 |
|---|---:|---:|---:|---:|
| [1200,2800,1600,2000,1400] | 9000 | 211.1 | 162.7 | 0.77x |

### 小结
- 修复后精度全场景 ~4.5% rel_L1,不再受 P 饱和影响;对 fp32 比 trtllm 更准。
- 单并发长序列(≥8k)FA4 领先 1.03–1.09x;多batch长序列打平;多batch短序列因 kernel 欠载落后(0.77x)。
- repro:`repro/bench_scripts/varlen_multibatch.py`(+ ad-hoc SEQS 参数化脚本)。
