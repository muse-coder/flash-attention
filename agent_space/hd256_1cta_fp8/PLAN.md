# Plan: hd256 1-CTA causal varlen fp8 forward kernel (SM100/B300)

## Context

FA4 (CuTeDSL, Blackwell) 目前在 SM100 上所有 `head_dim==256` 都被强制路由到独立的 **2-CTA** kernel
`sm100_hd256_2cta_fmha_forward.py`（`interface.py:602-603` 的 `use_dedicated_hd256_kernel`
把 `use_2cta_instrs` 置真）。该 2-CTA kernel：

- **不支持 fp8**：`sm100_hd256_2cta_fmha_forward.py:208` 直接 `assert descale_tensors is None`。
- causal/varlen 虽支持，但依赖 2-SM UMMA 锁步 + `use_semantic_trip_range` /
  `get_trip_start_count_via_block_info`（`:112,:958-963,:1345-1385`）这套"并集 trip range + mask"
  硬凑，同步复杂、易错。

**目标**：新建一个**独立文件**的 **1-CTA** hd256 前向 kernel，支持 **causal + varlen + fp8(e4m3) + GQA/MQA**，
fp8 外部 descale 恒为 1（省掉 descale 路径，但保留 fp8 内部 P 放大精度技巧）。首版**不做** non-causal-only
优化、local/sliding window、paged KV、split-KV、pack_gqa、score_mod/mask_mod、softcap、learnable_sink。

**为什么 1-CTA 可行且更优**：单 CTA 只处理 128 行 Q → tmem 恰好放得下 `S 双缓冲(2×128) + O(256) = 512 列`
（2-CTA 之所以存在只是为了把 tile_m 拉到 256 换吞吐）。1-CTA 没有跨 CTA 锁步：causal 直接用
`BlockInfo` trip-range 免费拿到、varlen 用现成 `SingleTileVarlenScheduler` 无 cluster 对齐问题、
fp8 全套机制（max_offset 256×、窄 P 列、V 在 host 侧转置无需 kernel 内转置）都现成。fp8 每元素 1 字节
使 K/V smem 压力减半，是 hd256 单 CTA 能双缓冲的关键（bf16 版会退化到 kv_stage=1）。

## 核心 tensor 调度方案（本 kernel 的新颖点）

与现有方案的区别（已确认）：
- hd128 1-CTA（主 kernel `q_stage=2`）：overlap 来自**两个独立 128 行 Q 子块**（各自独立 running max/sum、
  独立 O），mma 循环 `flash_fwd_sm100.py:1685-1826` 是"2 Q 子块共进同一个 K"。
- 主 kernel `q_stage=1`：只有单 S 缓冲，**不做** K 方向的 QK/softmax overlap。
- **本方案**：单个 128 行 Q 子块，沿 **K 方向对 S 双缓冲 ping-pong**，`QK(块 i+1)` 与 `softmax(块 i)` overlap，
  两路 PV 累加进**同一个** O(128×256)，维护**单一** running max/sum。

流水（S buffer parity = n_block 序号 %2）：
```
prologue: QK(块0) -> S0
loop i = 0..N-1:
    QK(块 i+1) -> S[(i+1)%2]      # 异步 tcgen05 MMA，与下面 softmax(块 i) overlap
    wait P[i%2] ready
    PV(块 i): P[i%2] · V[i] -> O   # 累加进单一 O
softmax warp: 顺序处理 S0,S1,S0,... -> P0,P1,...（单一 running max/sum，跨块在线更新）
correction warp: 当某块把 row_max 抬高时，对单一 O 乘 correction factor 再让 PV 累加
```
overlap 本质 = 异步 tensor core 发 QK 与 SIMT softmax(exp2) 并行；S 双缓冲让 QK(i+1) 写 S1 时 softmax 仍在读 S0。

## tmem 布局（512 列恰好占满，FP32 计列）
```
S0 = [0, 128)      S1 = [128, 256)     # QK 累加器双缓冲 (s_stage=2)
O  = [256, 512)    # 单一 PV 累加器 128×256 (fp32)
P  : fp8 仅 32 列，覆写各自 S buffer 上半（softmax 先 t2r 读完整 S 再 r2t 写 P）
row_max/row_sum 向量：复用 S 区（tmem_vec_offset）
```
参照 `flash_fwd_sm100.py:294-307` 的偏移公式（本 kernel 需重算：S 双缓冲对应 K-parity 而非 q_stage）。

## MMA / tiling（已确认）
- `mma_tiler_qk = (128, n_block=128, 256)`：**K=256 无需手动分段**，`make_trivial_tiled_mma` 只吃 `[:2]`，
  K 由 fragment K-mode 承载，CuTe 自动发 `256/32=8` 条 `tcgen05.mma` 累加进同一 S（机制同主 kernel K=192→6 条）。
  复用 `blackwell_helpers.py` 的 `gemm_ptx_partial`/`gemm_ptx_precomputed_varname`（`:97,:396,:617` 附近）。
- `mma_tiler_pv = (128, 256, n_block=128)`：N=256 硬件合法（单条 atom N≤256，`blackwell-gemm-tensor-memory.md:58-65`）。
  **首版先试 N=256 单发**（O 为一个 128×256 tmem 累加器，`correction_epilogue` 已按 `corr_tile_size` 循环泛化到 256）；
  若 tmem/correction 层出现布局问题，**回退到 `iterations_pv=2`（N 256→2×128，两半分别累加进 O 的两半）**，参照
  `sm100_hd256_2cta_fmha_forward.py:1162-1187`。
- QK/K major = K，V major = MN（V 在 host 侧已转置，`flash_fwd_sm100.py:440-441,479`）。P source = TMEM。

## warp specialization（仿主 kernel 16 warp / 512 线程）
- softmax = w0-3（单一 running stat，顺序处理 S0/S1 ping-pong）；**w4-7 空闲或改作辅助 KV load**。
- correction = w8-11（对单一 O 做在线 rescale + 最终归一化 + 写 LSE）。
- mma = w12（唯一发 tcgen05）；epilogue = w13；load = w14；empty = w15。
- 寄存器分配走 fp8 tuning（`flash_fwd_sm100.py:448-465`，`_FP8_TUNING_CONFIG`）。

## pipeline barriers（mbarrier）
复用主 kernel 的 barrier 集合（`flash_fwd_sm100.py:686-695,919-1017`），按"单 O + K-ping-pong"调整：
- `mbar_load_Q`（load→mma，1 个 Q）、`mbar_load_KV`（load→mma，kv_stage=4）。
- `S_full`（mma→softmax，2 个 S buffer，按 parity 索引）。
- `P_full`（softmax→mma，2 个 P buffer）+ `split_P_arrive` 分半提交优化（`:163-166`，可后加）。
- `O_rescaled`（correction→mma：告知"O 已被上一块 rescale 完，可累加下一路 PV"）——**单 O 关键同步点**，
  取代主 kernel `pipeline_s_p_o` 里 O 那一义；串起 correction rescale 与 mma 的下一路 PV，防 RAW/WAR。
- `softmax_stats`（softmax→correction，传 row_max/row_sum，`sm_stats_barrier` 细粒度 arrive）。
- `O_epi`（correction→epilogue，仅非 varlen TMA-O 路径）。
- **删除**：2-CTA 的 cluster barrier、`tmem_dealloc_mbar` 跨 CTA 逻辑退化为单 CTA、CLC 字段（用静态/varlen 调度）。
- **注意 varlen ⇒ `use_tma_O=False`**（`flash_fwd_sm100.py:188-192`），O 走 correction warp 的
  `_store_O_to_gmem`（`:2814-2861`）逐行 cp.async，而非 TMA store。

## 直接复用（import，不复制）
- `softmax.py`：`SoftmaxSm100`、`scale_subtract_rowmax`（`:330-345`）。fp8 常量保留：
  `max_offset=8`、`max_offset_scale=256`（`flash_fwd_sm100.py:2014,2430-2432`）、`rescale_threshold=4.0`（`:2022-2026`）。
- `block_info.py`：`BlockInfo.get_n_block_min_max`（`:24-55`）、`get_n_block_min_causal_local_mask`（`:105-121`）——causal+varlen 与 CTA 数无关。
- `seqlen_info.py`：`SeqlenInfoQK.create`（`:83-146`）。
- `tile_scheduler.py`：`SingleTileVarlenScheduler`（`:788`，`cluster_shape_mn=(1,1)`）；非 varlen 用 `SingleTileLPTScheduler`。
- `blackwell_helpers.py`：`gemm_ptx*` 系列。
- `mask.py` / `named_barrier.py` / `pipeline.py` / `copy_utils.py` / `mma_sm100_desc.py` / `fast_math.py`。
- GQA：`_kv_head_idx`（`flash_fwd_sm100.py:1839-1843`）取 KV head 索引。
- descale 恒 1：**不传** `DescaleTensors`，`_load_effective_descales` 返回 `(1.0,1.0)`（`:1846-1862`）。

## 文件改动
1. **新建** `flash_attn/cute/flash_fwd_hd256_1cta_sm100.py`：新类
   `FlashAttentionForwardHd256_1CTA_Sm100`。骨架仿 `flash_fwd_sm100.py` 的 1-CTA 结构（**不是** 2-CTA 文件的
   `q_stage=iterations` 语义），重写 `mma()`/`softmax_loop()`/`correction_loop()` 的 K-ping-pong + 单 O 逻辑，
   其余方法（`load`、`epilogue_s2g`/`_store_O_to_gmem`、smem/tmem 布局构造、tiled MMA 构造）尽量照搬并裁剪掉
   2CTA/paged/split/local/score_mod/pack_gqa 分支。
2. **修改** `flash_attn/cute/interface.py`：在 `_flash_attn_fwd`（约 `:601-603,:885-939`）加分支——
   当 `arch in [10,11] and head_dim==head_dim_v==256 and is_fp8` 时选新 kernel（`use_2cta_instrs=False`），
   否则 hd256 仍走原 2-CTA kernel。保持非 fp8 hd256 行为不变。
3. **测试**：在 `tests/cute/test_flash_attn.py` 放开/新增 `d==256 & fp8 & descale=None` 用例（现有 `:112,:170-180`
   对 hd256 有 skip；对 fp8 现有用例用随机 descale，需加 descale=None 变体）。

## 实现阶段（分步，每步可编译+验证）
1. **骨架 + 编译通过**：新文件类定义、tmem/smem/tiled-MMA 布局、SharedStorage、warp 分发框架；先跑
   FakeTensor 编译（`FLASH_ATTENTION_FAKE_TENSOR=1`）确认 tmem_total==512 断言过、smem 在预算内。
2. **正确性优先（可暂不 overlap）**：先实现最简正确版——单 S 缓冲、标准 flash K-loop、单 O、causal 非 varlen fp8，
   小 shape 数值对齐 bf16 参考。
3. **加 K-ping-pong overlap**：切到 S 双缓冲 + `QK(i+1)‖softmax(i)`，验证数值不变、观察是否提速。
4. **varlen**：接 `SingleTileVarlenScheduler` + `use_tma_O=False` 的 `_store_O_to_gmem`，验证 varlen 用例。
5. **GQA/MQA**：`qhead_per_kvhead>1`，验证。
6. **PV N 决策**：确认 N=256 单发是否正确/更快；否则切 `iterations_pv=2`。
7. **调优**：kv_stage、寄存器数、`split_P_arrive`、`enable_ex2_emu`（B300=sm103 用硬件 ex2）。

## 版本管理（agent_space + git）
- 新建 `agent_space/hd256_1cta_fp8/`，用 git 跟踪。目录内容：
  - `PLAN.md`（本计划的拷贝）、`NOTES.md`（每次迭代的方案与实验记录，追加式）。
  - `kernel/`：kernel 源码的工作副本 / 软链，或每次迭代的快照。
  - `repro/`：临时 repro 脚本、profiling 输出、SASS/PTX dump。
- 每完成一个实现阶段或一次有意义的迭代就 `git add -A && git commit`，commit message 记录该次改动与验证结果，
  使方案与代码的每次迭代都可回溯。（现处于 plan mode 无法建目录/提交，批准退出后立即执行。）

## 验证（环境：Python `/home/mudi/mudi_env`，GPU = B300）
- **arch 探测坑**：`nvidia-smi` 报 `NVIDIA L20D / cc 8.9`（B300 代号），CuTeDSL 自动探测可能误判为 SM89 而拒编 SM100。
  测试时大概率需用 `FLASH_ATTENTION_ARCH`（`interface.py:76-92`）或 `CUTE_DSL_ARCH` 强制到 `sm_103`。上手第一步先确认强制方式。
- 编译期（无 GPU）：`FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x` 只验证编译。
- 运行期数值：fp8 用 bf16 参考（`test_flash_attn.py:196` `dtype_ref`，`upcast=False, intermediate_dtype=fp8` 见 `:296-298`），
  descale=None，容差按现有 fp8 分支（`:312-314`）。命令：
  `/home/mudi/mudi_env/bin/python -m pytest tests/cute/test_flash_attn.py -k "256 and ..." -x`（视 GPU 空闲用 `CUDA_VISIBLE_DEVICES`）。
- 端到端：先小 shape（seqlen 128/256，causal，MHA），再 GQA、varlen、大 seqlen。对比 out 与 bf16 参考、检查 LSE。

## 主要风险（相对主 kernel q_stage=2 最易出问题处）
1. **单 O 的 rescale 与两路 PV 的 RAW/WAR**：`O_rescaled` barrier 时序错误会死锁或算错——最需要仔细设计与 printf 验证。
2. **barrier phase 初始化**：S/P 双缓冲的 parity 与各 pipeline 的初始 phase（参照 `:1945-1947` 的 phase 初值）。
3. **PV N=256 单发未经验证**：主 kernel 从未用 N>128 的 PV，tmem O 布局/`tmem_warp_shape` 可能需按 128 拆。
4. **causal 下 n_block 很少时 ping-pong 退化**（如只有 1 个 K 块），prologue/边界要处理干净。
5. **smem 贴预算**：Q(32KB)+O(64KB)+4×32KB(KV)=224KB 紧贴 ~227KB 硬顶，加 mbar/sScale 后需实测不溢出（可退 kv_stage=3 或开 `overlap_sO_sQ`）。

## 参考资料
- 架构/CuTeDSL：`/home/mudi/atrex-kernel-agent/gpu-wiki/docs/nvidia/blackwell/`
  （`hands-on/tcgen05-mma-tmem.md`、`three-role-warp-specialization.md`、`two-cta-cooperation.md`、
  `cutedsl/blackwell-gemm-tensor-memory.md`、`blackwell-gemm-low-precision.md`、`articles/flash-attention-4-kernel-architecture.md`）。
- PTX：`/home/mudi/ptx-isa-markdown/cuda_skill/references/`（`ptx-isa.md`、`ptx-docs/`）——tcgen05.mma / mbarrier / cp.async.bulk。
- 仓库调试文档：`AI/DEBUG_2CTA.md`（hang/deadlock、printf bisection）、`AI/RACECHECK_TMA_HAZARD.md`。






---

## 主目标 shape（用户指定，2026-07-07）—— 优化与验证的首要对象

- **单并发（batch=1）**，序列长度 **8192**，但**用 varlen 的 tensor 布局表示**（`cu_seqlens_q=cu_seqlens_k=[0,8192]`，
  q shape `(8192, 32, 256)`，k/v shape `(8192, 2, 256)`）。走 `flash_attn_varlen_func`。
- **Q head = 32，KV head = 2**（GQA，qhead_per_kvhead = 16）。
- head_dim = head_dim_v = 256，fp8(e4m3)，causal，softmax scale = 1/sqrt(256)（外部 descale=1）。
- 正确性 oracle：通用 SM100 kernel（`FA_HD256_USE_MAIN=1`，用户认可"与现有 FA4 fp8 一致"）。
- 性能：此 shape 为主要 benchmark 对象；K-ping-pong 目标是在此 shape 上超过 phase1 副本基线。
- repro：`repro/target_varlen.py`。
- 注意：varlen ⇒ `use_tma_O=False`（走 `_store_O_to_gmem`）；GQA 首版用非 pack_gqa（`pack_gqa=False`）。
