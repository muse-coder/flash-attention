# hd256 1-CTA fp8 kernel — 开发笔记

独立开发日志。每次迭代追加一节：方案要点 + 改动 + 验证结果。代码快照放 `snapshots/`，repro 脚本放 `repro/`。

## 目标
新建 1-CTA hd256 causal varlen fp8(e4m3) GQA 前向 kernel（B300/sm103）。fp8 外部 descale=1。
详见 `PLAN.md`。

## 环境
- Python: `/home/mudi/mudi_env/bin/python`
- GPU: B300（nvidia-smi 报 "NVIDIA L20D / cc 8.9" 是代号，实际 sm_103）
- arch 强制：`FLASH_ATTENTION_ARCH`（kernel 选择）+ `CUTE_DSL_ARCH`（编译目标），见 `interface.py:76-92`。

## 关键设计（速记）
- tmem 512 列：S0=[0,128) S1=[128,256) O=[256,512)；P(fp8 32列)覆写 S 上半。
- QK K=256 CuTe 自动分 8 条 tcgen05.mma；PV N=256 先试单发，不行回退 2×128。
- 单 128 行 Q，S 沿 K 双缓冲 ping-pong，QK(i+1)‖softmax(i)，两路 PV 累加进单一 O，单一 running max/sum。
- 复用 softmax/block_info/seqlen_info/varlen scheduler/gemm helper/epilogue。

---

## 迭代记录

### [init] 2026-07-07 — 建立版本管理
- 建 `agent_space/hd256_1cta_fp8/`（主仓库 .gitignore 忽略 agent_space，故此处为独立嵌套 git 仓库）。
- 拷贝 PLAN.md，建 NOTES.md。下一步：阶段1 骨架。

### [env] 2026-07-07 — 打通工具链（重要）
**运行方式（务必带这些）**：
```
cd /home/mudi/flash-attention
PYTHONPATH=/home/mudi/flash-attention \
CUTE_DSL_ARCH=sm_103a \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
CUDA_VISIBLE_DEVICES=<free gpu> \
/home/mudi/mudi_env/bin/python <script>
```
- **PYTHONPATH 必须指向仓库**：mudi_env 里另装了旧的 site-package `flash_attn`（且 import 就报错），
  必须让仓库版本优先。
- **CUTE_DSL_ARCH=sm_103a**：DSL 的 `is_family_of(sm_100f/sm_103f)` 需要 **带 `a` 后缀**；`sm_100`/`sm_103`(无a) 都 False。
  torch 看到的真实 cap 是 (10,3)=B300，所以 `FLASH_ATTENTION_ARCH` 可不设（自动=103）；nvidia-smi 的 "L20D/8.9" 是代号掩码。
- **物理 GPU 是 B300 sm_103，可真机执行**（不只是 fake 编译）。
- **依赖升级**（用户授权）：仓库 HEAD 需 CuTe DSL 4.6.0+（commit 82d6441）。原 mudi_env 是 cutlass-dsl 4.5.2 + quack 0.5.0，
  两者都太旧（`nvvm.fmax` 签名、`cute.core.ThrMma`）。已升级到 **nvidia-cutlass-dsl 4.6.0 + quack-kernels 0.6.1**。
- 验证：bf16 hd128 causal 真机跑通，max_abs_err=0.008 PASS（`repro/real_run_check.py`）。

### [spike] 2026-07-07 — 用主 kernel 验证 hd256 1-CTA fp8 可行性
- interface 加路由：fp8+hd256 → 通用 `FlashAttentionForwardSm100`（强制 q_stage=1, use_2cta=False），
  绕过不支持 fp8 的 2cta 专用 kernel。改动见 interface.py（`use_fp8_hd256_1cta`）。
- **三大风险清除**：hd256 1-CTA fp8 能**编译+真机运行**（tmem 512 放得下、PV N=256 可行、smem 够）。
- 精度：单 K 块(s128) PASS；多 K 块用 attention_ref+2x 容差判 FAIL。
- **但**：诊断发现连**已发布的 hd128 q_stage=2 fp8**（`diag_qstage1.py`）也过不了 attention_ref 判据
  （err 0.227）→ 说明是**我的参考/容差不对**，不是 kernel bug。

### [baseline 调整] 2026-07-07 — 改用 flashinfer fp8 做 oracle（用户指定）
- baseline = flashinfer `BatchPrefillWithPagedKVCacheWrapper`（`/home/mudi/flashinfer`），fp8 + scale=1。
- flashinfer fp8 prefill：**Q=fp16、KV=fp8**（fa2 后端反量化 KV→bf16 计算），传 k_scale/v_scale。
- 对齐方式：两边吃相同 fp8 数值、scale=1。flashinfer 喂 q=(fp8→fp16)、kv fp8 + k_scale=v_scale=1；FA4 喂 fp8 q/k/v, descale=None。
- 目标：FA4 hd256 1-CTA fp8 输出与 flashinfer fp8 输出在 fp8 容差内对上。

### [关键结论] 2026-07-07 — hd256 1-CTA fp8 已验证正确；fa2 fp8≠真fp8
用 `cmp_flashinfer_fp8.py` 对比（FA4 vs flashinfer paged, scale=1，相同 fp8 数值）：
- **flashinfer fa2 后端的 "fp8" 实为 fp16 计算**：`FI8-vs-ref=0.000`（把 fp8 KV 反量化成 fp16 算），是高精度 oracle。
- FA4 vs 高精度参考的偏差（max_abs）：
  - hd256 1块非causal 0.015；hd256 多块 causal 0.49–0.57
  - **hd128（已发布）1块 0.014；多块非causal 0.40；多块 causal 0.51**
  - **hd64（已发布）causal 0.66**
- **判定**：hd256 1-CTA fp8 与已发布 hd128/hd64 fp8 偏差**同量级** → hd256 1-CTA 路径**和现有 FA4 fp8 一样正确，无 hd256 特有 bug**。
  ~0.4-0.6 是 FA4 **真 fp8 计算**（fp8 张量核 QK + P 重量化 fp8）的固有代价，随 K 块数增长。
- **后续**：要做真 fp8 vs 真 fp8 公平对比，需用 flashinfer **trtllm-gen / fa3** 后端（q_dtype=fp8, kv_dtype=fp8），
  而非 fa2。待与用户确认基准定义。

### [baseline] 2026-07-07 — spike 性能基线（主 kernel q_stage=1 hd256 fp8）
用户确认：正确性以"与现有 FA4 fp8 一致"为准（spike 已满足）。性能基线（B300, `bench_spike.py`）：
- b2 s2048 h8 causal: 49us 699 TFLOP/s（小 shape 欠载）
- b2 s4096 h8 causal: 85us 1624 TFLOP/s
- b4 s4096 h16 causal: 323us 1700 TFLOP/s
- b2 s2048 h8 非causal: 47us 1454 TFLOP/s
B300 fp8 峰值约 4500-5000 TFLOP/s → 现~35% MFU，有 overlap 空间。专用 K-ping-pong kernel 目标：超过此基线。
主 kernel q_stage=1 分析：S 只用单缓冲（s_stage=2 的 S1 槽位闲置），QK(i+1) 要等 softmax(i) 读完 S0 → 串行，overlap 弱。

### [阶段1] 2026-07-07 — 专用文件建立 + 忠实副本
- 新建 `flash_attn/cute/flash_fwd_hd256_1cta_sm100.py`（拷贝主 kernel，类名 `FlashAttentionForwardHd256_1CTA_Sm100`）。
- interface：import 新类；fp8+hd256 路由到新类（`use_fp8_hd256_1cta`）。加 env `FA_HD256_USE_MAIN=1` 可切回主 kernel 生成黄金参考。
- 验证 `golden_cmp.py`：新 kernel vs 主 kernel **逐位一致 max_abs=0.0000**（MHA/MQA/causal/非causal/短q多kv 全 PASS）。
- 快照：`snapshots/phase1/`。下一步：阶段2/3 —— 把 mma()/softmax 改成 S 双缓冲 K-ping-pong overlap。

### [profile] 2026-07-07 — 确认 overlap 受限（ncu）
b2 s4096 h8 d256 causal，SpeedOfLight+WarpStall：
- Compute(SM) 25.8%，Memory 34.2% → 都低，非算力/带宽瓶颈。
- **71.6% warp cycle 是 scoreboard stall（等 smem 数据）**，平均每 warp stall 17.8/24.9 cycle。
- 判定：latency/overlap 受限 → **K-ping-pong overlap 有价值**（让 QK 张量核在 softmax 期间保持忙碌）。
- 存 `repro/ncu_phase1.txt`。

### [目标shape] 2026-07-07 — 用户指定主目标 shape（写入 PLAN.md）
- 单并发 batch=1、len=8192、**varlen 表示**、Q head=32 / KV head=2（GQA x16）、hd256 fp8 causal, scale=1/sqrt(256)。
- `repro/target_varlen.py`。phase1 副本：new-vs-main **0.00000 PASS**；**748.7us / 1468.6 TFLOP/s**（目标 shape 基线）。
- varlen+GQA(非pack) 路径正常。K-ping-pong 目标：在此 shape 上超过 1468 TFLOP/s。

### [K-ping-pong 实现配方] 2026-07-07 — 待实施（阶段3核心）
**本质**：把 S/P 双缓冲数从 q_stage 解耦。q_stage 仍=1（单 O、单 running stat、单 softmax 流），
但 S/P 用 2 个 buffer 按 **K-block parity** 轮换，让 `QK(块 i+1)→S[(i+1)%2]` 与 `softmax(块 i, 读 S[i%2])` overlap。

**当前 q_stage=1 串行原因**（`flash_fwd_hd256_1cta_sm100.py` mma()）：只用 S0；`gemm_Si(QK i+1)` 覆写 S0 前，
`gemm_Pi(PV i)` 的 `pipeline_s_p_o.producer_acquire` 已等 softmax(i) 读完 S0 → QK(i+1) 被串行到 softmax(i) 之后。

**改造点（函数级）**：
1. 引入 `self.s_pp = 2`（S/P ping-pong 深度），与 `self.q_stage=1` 解耦。tmem：S0=[0,128) S1=[128,256)，
   P0/P1 覆写各自 S 上半（fp8 P 32 列），O=[256,512) 单份。`tmem_s_offset/tmem_p_offset` 已是 2 元素，复用。
2. barriers `mbar_S_full_*`/`mbar_P_full_*`/`pipeline_p_lastsplit`：stage 数从 q_stage(1) 改为 s_pp(2)。
   O 相关（`pipeline_o_acc`/`O_rescaled`/`sm_stats`/`O_epi`）保持单份(q_stage=1)。
3. `mma()`：prologue 发 QK(block0)→S[0]。主循环让 QK 领先 PV 一个块：
   `QK(i+1)→S[(i+1)%2]`（提交 S_full[(i+1)%2]）在 `PV(i)`（等 P_full[i%2]，累加单 O）之前发射。
   V 的 pipeline release 时机按块。边界：最后一块只 PV 不再 QK。
4. `softmax_loop()`：单 warp group(w0-3)，单 running max/sum；按 K-parity 读 S[i%2]、写 P[i%2]、
   提交 P_full[i%2]；stats 仍单份传 correction。
5. `correction_loop()`：单 O 的在线 rescale 不变（q_stage=1 语义），只是被 s_pp 的 P/S 轮换驱动。
6. O_rescaled↔PV 依赖：单 O 被两路 PV 交替累加，rescale 必须在下一路 PV 前完成——用单份 O_rescaled barrier 串起。

**验证**：每改一步 `golden_cmp.py`（新 vs 主，逐位/fp8容差）+ `target_varlen.py`。目标：正确 + >1468 TFLOP/s。
**回退**：phase1 快照 `snapshots/phase1/` 是已知可用副本。

### [v1] 2026-07-07 — K-ping-pong plumbing（S1/S2）
- 改动：在 `FlashAttentionForwardHd256_1CTA_Sm100` 中新增 `self.s_pp = 2`，并添加独立
  `mbar_S_full_kpp` / `mbar_P_full_kpp` / `mbar_P_full_lastsplit_kpp` / `mbar_O_rescaled_kpp`
  以及对应 pipeline 对象。旧 `pipeline_s_p_o` 仍保持现有行为，`mma()` / `softmax_loop()` /
  `correction_loop()` 尚未接入新 pipeline。
- 验证：
  - `py_compile flash_attn/cute/flash_fwd_hd256_1cta_sm100.py` PASS。
  - `golden_cmp.py` PASS：全 shape new-vs-main max_abs=0.0000。
  - `target_varlen.py` PASS：new-vs-main max_abs=0.00000；751.3us / 1463.6 TFLOP/s。
- 结论：S1/S2 plumbing 可用，行为保持 phase1/v0。下一步迁移 S/P/O_rescaled 的实际 wait/release，
  然后重写 `mma()` 为 QK 领先 PV 一个 K-block 的 parity 调度。

### [v2] 2026-07-07 — 首个 K-ping-pong 调度版本
- 改动：
  - `mma()` dense `q_stage=1` 路径改为 K-ping-pong：prologue 发 `QK0->S0`；循环内先发
    `QK(i+1)->S[(i+1)%2]`，再等 `P(i)` + `O_rescaled` 做 `PV(i)`；PV0/PV1 均累加到单一 O。
  - `softmax_loop()` 对 dense K blocks 维护 `k_iter`，按 parity 读写 `S0/P0` 或 `S1/P1`。
  - `correction_loop()` 的 KPP 路径显式等待 `O_full`，rescale 单一 O 后用 `O_rescaled_kpp` 放行 MMA。
- 验证：
  - `py_compile flash_attn/cute/flash_fwd_hd256_1cta_sm100.py` PASS。
  - `timeout 180s ... golden_cmp.py` PASS：全 shape new-vs-main max_abs=0.0000。
  - `timeout 240s ... target_varlen.py` PASS：new-vs-main max_abs=0.00000；752.3us / 1461.5 TFLOP/s。
- 结论：KPP 同步协议闭合且正确，但目标 shape 性能未改善（v0 1468.6，v1 1463.6，v2 1461.5 TFLOP/s）。
  首要怀疑：首版 KPP 等完整 P（未用 split-P partial arrive）、每轮新增 `O_rescaled` full wait/release、
  或 `V(i)`/`K(i+1)` 等待顺序没有实际隐藏 load latency。下一步：建立 flashinfer trtllm-gen 基线，
  同时 profile v2 并恢复 split-P 或减少 O_rescaled 同步开销。

### [v3] 2026-07-07 — 修复 compile-key alias，建立真实 dedicated baseline
- 发现：`FA_HD256_USE_MAIN=1` 只改变 forward class 选择，但旧 compile key 没有记录 class/variant。
  因此同一进程先编译 main kernel 做 golden 后，后续 `FA_HD256_USE_MAIN=0` 可能直接复用 main CUBIN。
- 改动：`interface.py` 新增 `fwd_kernel_variant`：
  - `hd256_fp8_main`：`FA_HD256_USE_MAIN=1` 的通用 `FlashAttentionForwardSm100` oracle。
  - `hd256_fp8_1cta`：默认专用 `FlashAttentionForwardHd256_1CTA_Sm100`。
  - 该字段加入 `_flash_attn_fwd` compile key。
- 为排除未落版 KPP/10-warp 实验干扰，工作 kernel 回退到 v0 忠实副本；诊断快照保存在
  `repro/diagnostic_snapshots/flash_fwd_hd256_1cta_sm100_before_v0_restore.py`。
- 验证：
  - `golden_cmp.py` PASS：全部 shape new-vs-main max_abs=0.0000。
  - `target_varlen.py` PASS：`581.9us / 1889.6 TFLOP/s`。
  - NCU launch-only：kernel 名含
    `flash_fwd_hd256_1cta_sm100FlashAttentionForwardHd256_1CTA_Sm100`，`582304 ns`，
    block size `512`，grid size `2048`，register/thread `128`，shared memory/block `232448` bytes。
- FlashInfer trtllm-gen baseline（已有脚本 `bench_flashinfer_trtllm_prefill.py`）：
  - fp8 output：约 `481.9us / 2281.8 TFLOP/s`。
  - bf16 output：约 `524.8us / 2095.0 TFLOP/s`。
  - FA4 当前输出 bf16，因此按 bf16-output baseline 的 10% 优势目标约为 `<=472us`；当前 `~582us` 还需约 19% 降耗时。
- 重要更正：v1/v2 的 KPP PASS/性能结论是在 compile-key alias 修复前获得的，不能再作为真实 KPP 证据。
  修复后重跑未落版 KPP/compaction 状态，在 `golden_cmp.py` 首个非 varlen/noncausal shape 上出现 launch failure。
- 下一步：从 v0 真实 baseline 重新引入优化，优先把 KPP 限定到目标 causal-varlen 路径，其它 shape 继续 legacy schedule；
  每次性能结论必须附 kernel-name/launch 证据。

### [v4] 2026-07-07 — KPP 修正 + 10-warp compact + split-P + register tuning
- 改动：
  - KPP 只在目标类路径启用：`q_stage==1`、causal、varlen-q、dense、non-local、non-split-KV、non-pack-GQA；
    其它 golden shape 继续走 v0/legacy schedule。
  - 修正 KPP 同步状态：P/O-rescaled consumer state 不随 work tile 重置；softmax 的 `kpp_iter_global`
    跨 work tile 递增；KPP `P_full` commit 用 `sync_warp()` + `elect_one()` 保护。
  - 目标路径从 16 warp compact 到 10 warp：softmax `0..3`、correction/store `4..7`、MMA `8`、load `9`。
  - KPP 恢复 split-P partial arrive，`split_P_arrive=96`。
  - fp8 tuning 支持显式 `num_regs_other`，最终配置为 softmax/correction/other = `160/120/96`。
- 验证：
  - `golden_cmp.py` 全 PASS，全部 listed shapes new-vs-main max_abs=0.0000。
  - `target_varlen.py` PASS，new-vs-main max_abs=0.00000。
  - `pytest -q tests/cute/test_flash_attn_varlen.py -k hd256_fp8_1cta -x` PASS：
    `1 passed, 5832 deselected`。
  - 目标 shape 独立进程 5 次：`437.8, 437.6, 437.3, 436.8, 437.0 us`。
  - NCU launch-only：真实 kernel 名含 `FlashAttentionForwardHd256_1CTA_Sm100`，`436544 ns`，
    block size `320`，grid size `2048`，register/thread `160`，shared memory/block `232448` bytes。
- FlashInfer trtllm-gen 当前基线：
  - fp8 output median `492.3 us`，bf16 output median `540.1 us`。
  - 严格按 fp8-output baseline 的 10% 目标为 `<=443.1 us`；v4 约 `437.3 us`，已满足。
- 快照：`versions/v4_kpp_compact_splitp_regs/`。下一步若继续优化，应从 v4 出发做 profile-driven 小改，
  并保留 v5+ 版本。

### [baseline 修正] 2026-07-07 — 用 ncu 重测（event 计时的 492us 作废）
用户指出 492us baseline 不准（event 计时含 wrapper/launch 开销）。同一 ncu 方法（`-k regex:` 名字过滤，
`--metrics gpu__time_duration.sum`，各 4 次）在目标 shape（b1 s8192 hq32 hkv2 d256 fp8 causal, scale=1）实测：
- **FA4 dedicated (v4)**：430496/430528/431296/434496 ns → 中位 **~430.9us**
- **FlashInfer trtllm-gen fp8-out**：439360/444736/446080/448928 ns → 中位 **~445.4us**
- FlashInfer trtllm-gen bf16-out（已有 csv）：459936/470784 ns
**修正结论**：v4 比 FI trtllm-gen fp8-out 快 **~3.3%（14.5us）**，非 11.2%。
注意 FA4 输出 bf16（写 2× 字节）而此 FI 输出 fp8；对 FI bf16-out（~460-471us）FA4 快 ~6-8%。
kernel 名：FA4 `flash_fwd_hd256_1cta...`；FI `fmhaSm103aKernel_QkvE4m3OE4m3H256PagedKvCausalP64VarSeqQ128Kv128PersistentContext`。

### [v5-neg] 2026-07-07 — 深化 KV 流水无效（barrier-bound 确认）
- v5 试 overlap_sO_sQ→kv_stage 5：golden PASS，但干净 ncu 430.1us ≈ v4 430.9us，无实质变化 → 已回退。
- stall 分解（cyc/issued，总 6.54）：barrier 1.88 / long_scoreboard 1.44 / wait 1.04 / short_scoreboard 0.50。
- 占用率：2.41 active、0.41 eligible warps/sched，63% No-Eligible。tmem 满 → 1 CTA/SM 硬顶。
- 结论：瓶颈是 barrier 同步 + 低占用，非内存带宽/深度。下一杠杆需攻 softmax 速度/同步（高风险）。

### [v6-neg] 2026-07-07 — split_P_arrive sweep 无效
- sweep split_P_arrive ∈ {96(v4),64,32}，干净 ncu 中位：431.5 / 431.0 / 430.9us —— 全在噪声内，无提升。
- 结论：参数旋钮（kv_stage、split_P）对 barrier-bound 瓶颈均无效。已回退，文件 == v4。
- 结构性天花板确认：tmem 满 → 1 CTA/SM + S ping-pong 深度=2，softmax 延迟经 barrier 直接卡 MMA。
  进一步提速只剩高风险项：(a) 加 softmax warp（更多 warp/sched + 更快 softmax 生产 P）；(b) 减少 warp-role 同步。

### [softmax 计算调查] 2026-07-07 — 已高度优化,exp2 非瓶颈
应用户"softmax 计算还能提速吗"深入排查 softmax_step + softmax.py:
- **已有优化(确认)**:① P 生产不依赖 row_sum,`update_row_sum` 在 commit P **之后**(line 2769-2770)→ row_sum 不在 MMA-解锁关键路径;
  ② causal masked/full 块分离(内部满块不套 mask,line 2461);③ 硬件 exp2(sm103, ex2_emu_freq=0);
  ④ packed FMA(fma_packed_f32x2);⑤ split-P 早放行(96);⑥ correction 经 sm_stats_barrier 早通知 row_max。
- **实验 ex2 emulation**(把部分 exp2 从 SFU 挪到 FMA,FA4 对 hd256-2cta 用 freq=14):
  sweep FA_EX2_FREQ {0,8,16} → 431.7/431.7/433.0us **无提升**。印证 profile:math_pipe/mio throttle≈0,exp2 非瓶颈。
- **结论**:softmax 计算层面基本到顶。其对性能的影响是经 barrier 门控 MMA(结构性 ping-pong 深度=2),非计算低效。
  唯一残留理论杠杆:4线程 warp-reduction(绑定 MMA tmem 布局,深且不确定)。文件保持 == v4。

### [v8-probe] 2026-07-07 — softmax 不在关键路径(决定性,否定双 softmax 方向)
用户提议:hd256 1-CTA 开双 softmax(softmax0→tmem, softmax1→smem)。
- 硬件约束:tcgen05 的 **O(PV 累加器)必须在 tmem,不能放 smem**;两份独立 O(2×256 列)装不下 512 tmem → 双独立 tile 本就不可行。P 可放 smem 但 P 不占瓶颈。
- **关键路径探针**:在 softmax_step 注入 N 遍额外完整 softmax(t2r+reduce+exp2+r2t,写同一 tmem P,有副作用不被 DCE,随后被真值覆盖),env `FA_SM_DOUBLE=N`。干净 ncu:
  - softmax ×1(基线):434.9us
  - softmax ×3:432.3us
  - softmax ×5:433.4us
- **结论:softmax 工作量翻到 5 倍,kernel 时间几乎不变 → softmax 已被 MMA 彻底掩盖,完全不在关键路径。**
  → **双 softmax / 任何 softmax 加速收益 ≈ 0**,不值得做。瓶颈在 MMA/张量管线侧(ping-pong 深度=2 限制 MMA 领先、O rescale 耦合),非 softmax。
- 探针已回退,文件 == v4。

### [profile] 2026-07-07 — best v4 当前瓶颈定位(NCU full sections)
按 `/home/mudi/atrex-kernel-agent/gpu-wiki/docs/nvidia/common/profiling` 的流程，对 best v4 跑完整 NCU：
`SpeedOfLight/ComputeWorkload/Scheduler/WarpState/Occupancy/MemoryWorkload/Instruction/PmSampling`。
artifact 见 `repro/ncu_best_v4_bottleneck.*` 和 `repro/ncu_best_v4_bottleneck_summary.md`。

关键指标：
- Compute(SM) 61.01%，Memory 36.92%，DRAM 4.74%，L2 hit 90.97% → 不是 DRAM 带宽瓶颈。
- Issue slots busy 35.55%，No Eligible 63.20%，eligible warps/sched 0.40，active warps/sched 2.42。
- achieved occupancy 15.03%，register/shared-mem 都限制为 1 block/SM。
- stall: barrier 1.90、long_scoreboard 1.45、wait 1.04、short_scoreboard 0.50 cyc/issued。

结论：当前瓶颈是 **1 CTA/SM 下的 warp-specialized pipeline 同步/低 eligible-warp**，具体在单 O accumulator 周围的
`P_full` / `O_full` / `O_rescaled` barrier 链；不是 softmax math、exp2、split-P 参数、KV stage 或 DRAM 带宽。
下一步若继续优化，应优先尝试减少 O-rescale 同步边，而不是继续加速 softmax。

### [v9] 2026-07-07 — 非causal 紧凑 bonus
- 2cta 判据实验副产品:给 1cta 非causal 也加 16→11 紧凑 → 973→936.5us。causal 目标不受影响(431.9us,golden PASS)。
- 2cta 方向结论:调优后 2cta(962.5)仍输 1cta(936.5),cross-CTA 同步净负担,方向关闭。详见 ../hd256_2cta_fp8/。

### [Round2] 2026-07-07 — correction 关键路径 + pack_gqa,均负结果
- **方向1 correction**:rescale_threshold sweep(FA_RESCALE_THR 4/8/16/100,干净测)→ 432.9/431.8/431.8/432.3us **全无变化**。
  thr=100 几乎不 rescale 仍不快 → **correction rescale 不在关键路径**。(correction 探针 FA_CORR_DOUBLE 因编辑本身污染基线 702us 作废,改用此干净判据。)
- **方向2 pack_gqa**:pack_gqa=True golden PASS 但 **562.8us,比 pack_gqa=False 432us 慢 30%** → packing 打碎并行/加索引开销,**对该 shape 有害**。
- 澄清:干净 v9 causal 目标 = 432.4us(ncu),与 v4 一致,无回归。
- **合并结论**:softmax(v8)、correction(round2)均证明不在关键路径;kv_stage/split_P/ex2/pack_gqa 全 null/负。
  关键路径在 **MMA+load 流水**本身,而占用率被 tmem 锁死在 15%(1 CTA/SM)。causal 1cta v4 ≈432us 是**实际天花板**。
