# K-ping-pong 实现计划（阶段3）

目标：在 `flash_fwd_hd256_1cta_sm100.py`（当前=主 kernel 忠实副本，q_stage=1）上实现 K 方向 S 双缓冲，
使 `QK(块 i+1)` 与 `softmax(块 i)` overlap，单 O、单 running stat、单 softmax warp group。
正确性 oracle：`FA_HD256_USE_MAIN=1`（主 kernel）。主目标 shape：varlen b1 s8192 hq32 hkv2 d256 fp8 causal。

## 现状分析（已确认）
- **tmem 已双缓冲**：`s_stage=2`（:356），`tmem_s_offset=[0,128]`、`tmem_p_offset=[64,192]`、
  `tOtO/tP` 用 `s_stage` 建 fragment。即 S0/S1、P0/P1 的 tmem 空间**已存在**，O=[256,512) 单份。
- **缺的是调度**：barriers 按 `q_stage*2`（=2，实用 1）建；mma/softmax 循环按 `stage in range(q_stage=1)` 只用槽 0。
- **难点**：`pipeline_s_p_o`（`mbar_S_full_P_full_O_rescaled`，:688,960）**一个 barrier 三义**：
  - MMA `producer_commit_w_index(stage)`（:1709,1779）= "S[stage] ready" → softmax。
  - MMA `producer_acquire_w_index_phase(stage)`（:1733,1799）= 等 "P[stage] ready + O rescaled" ← softmax+correction。
  - 即 S_full 与 P_full/O_rescaled 共用槽位。

## 设计决策
**方案 A（推荐，清晰）**：拆分 conflated barrier，S/P 深度=2（K-parity），O 相关深度=1。
- `pipeline_S_full`：MMA→softmax，**2 槽**（K-parity）。QK(i) 后 commit S_full[i%2]；softmax 等 S_full[i%2]。
- `pipeline_P_full`：softmax→MMA，**2 槽**。softmax 写完 P(i) commit P_full[i%2]；PV(i) 前 MMA 等 P_full[i%2]。
  （保留 `pipeline_p_lastsplit` 的 split-P 优化，同样 2 槽；首版可先禁用 split_P 简化。）
- `pipeline_O_rescaled`：correction→MMA，**1 槽**。correction 对单 O rescale 完后 commit；PV(i) 前 MMA 等它。
- `pipeline_o_acc`：MMA→correction，**1 槽**（不变）。PV 累加 O 后 commit。
- `pipeline_sm_stats`：softmax→correction，**1 槽**（单 stat 流，不变）。

**方案 B（最小改动）**：保留 conflated `s_p_o` 但改 2 槽、按 K-parity 索引。风险：O_rescaled 语义跨 parity 变 2 块陈旧，
易死锁/竞争。**首版不用，作为备选。**

采用 **方案 A**。

## 分步实施（每步：编译 → `golden_cmp.py` 逐位/容差 → `target_varlen.py`）

**S0. 基线固化**：确认从 phase1 快照可回退；当前 HEAD 已是 phase1。建分支思想：改坏就 `cp snapshots/phase1/... 回去`。

**S1. 引入 `self.s_pp = 2` 常量**（`__init__`，与 q_stage 解耦），仅作为后续 barrier/循环深度来源。暂不改行为。编译过。

**S2. barrier 扩容**：`mbar_S_full`（新，s_pp*2）、`mbar_P_full`（新，s_pp*2）、`mbar_P_full_lastsplit`（s_pp*2）；
`mbar_O_rescaled`（新，1*2）、`mbar_O_full`/`mbar_softmax_stats`/`mbar_O_epi` 保持 q_stage。
构造对应 pipeline 对象。**先建但不接线**，保证编译过、行为不变（旧 s_p_o 仍在用）。

**S3. 改 `mma()`**：把"QK 领先 PV 一块"的循环写出来：
- prologue：QK(block0)→S[0]，commit S_full[0]。
- 主循环 i=0..N-1：先发 `QK(块 i+1)→S[(i+1)%2]`（若 i+1<N），等 K(i+1)；再等 P_full[i%2] + O_rescaled，
  发 `PV(块 i)`→单 O（accumulate），commit o_acc；commit S_full[(i+1)%2]。
- 末块只 PV。V 的 release 按块。
- **注意**：QK 领先要求 K(i+1) 已 load（kv_stage=4 足够）。

**S4. 改 `softmax_loop()`/`softmax_step()`**：单 warp group(w0-3)，单 running max/sum；
块 i：等 S_full[i%2] → t2r 读 S[i%2] → 在线更新 max/sum → 写 P[i%2] → commit P_full[i%2] → 传 stats(单份)。

**S5. 改 `correction_loop()`**：单 O 在线 rescale（逻辑不变），驱动来源改为 s_pp 的 P/stats 轮换；
rescale 完 commit O_rescaled（单份）。

**S6. 删除旧 conflated `s_p_o` 引用**，全部切到 S_full/P_full/O_rescaled。编译 + golden。

**S7. 正确性回归**：`golden_cmp.py` 全部 PASS（逐位或 fp8 容差）；`target_varlen.py` golden PASS。

**S8. 性能**：`target_varlen.py` bench，对比 1468 TFLOP/s；`bench_spike.py` 多 shape。ncu 复测 stall。

**S9. 调优**（阶段4-7 合并）：split_P_arrive、kv_stage、寄存器数、ex2；PV N=256 vs 2×128。

## 验证门 & 回退
- 每步编译用 `FLASH_ATTENTION_FAKE_TENSOR=1` 快速过；正确性用真机 `golden_cmp.py`。
- 任一步 golden FAIL 且短时查不出 → `cp snapshots/phase1/flash_fwd_hd256_1cta_sm100.py flash_attn/cute/` 回退，缩小改动重来。
- 每个**有意义的迭代**存 `versions/vN_<desc>/`（kernel 快照 + NOTE.md：改动/golden/性能），单独 git commit（`vN:` 前缀），
  并更新 `VERSIONS.md` 表格。v0=忠实副本基线。回退：`cp versions/vN/....py flash_attn/cute/`。
  中间小改动可只 commit 不建 version；跑通/有结论的节点必须建 vN。

## 风险
1. 单 O 的 O_rescaled↔两路 PV 依赖（S5/S3 交界）—— 最易死锁，重点 printf 验证。
2. 各 pipeline 初始 phase（producer/consumer phase 初值）——参考主 kernel `mma_*_phase` 初值。
3. QK 领先一块的边界（首块 prologue、末块无 QK、causal 下块数少）。
4. `split_P_arrive` 与 ping-pong 交互——首版禁用（设 `self.split_P_arrive=0` 或绕过），跑通后再加。
