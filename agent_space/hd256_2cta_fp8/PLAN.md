# Plan: hd256 2-CTA fp8 forward — 最小同步 pair-UMMA

## Context / 动机

1-CTA hd256 fp8 kernel(`hd256_1cta_fp8/`,当前最佳 **v4 ≈ 430us**,比 FlashInfer trtllm-gen fp8 ~445us 快 ~3.3%)
已到**结构性天花板**,证据(详见 `../hd256_1cta_fp8/NOTES.md`):
- 占用率 15%(1 CTA/SM,2.5 warp/sched,63% no-eligible)——被 **tmem(512 满)+reg(160)+smem(231KB)** 三重锁死。
- 多瓶颈 latency 区(barrier 29% / long_scoreboard 22% / wait 16% CPI),单点调优 v5/v6/v7 三连 null(三振=实际天花板)。
- softmax 探针:5× softmax 工作量→时间不变,softmax 完全被掩盖(不是瓶颈)。
- **未被利用的杠杆**:v4 是 **M=128 单 SM MMA,张量核只 ~50% 峰值**(文档 `blackwell/hardware/2sm-cooperative.md`)。tensor pipe SOL 58.7% 与此吻合。

**本方案目标**:用 **2-CTA pair-UMMA(M=256)把张量核效率推向 ~100%**,同时**只保留最小必需的跨 CTA 同步**,
避开开源 hd256 2cta(`flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`)那套"大量不必要同步"(实测 bf16 1158us / SOL 42.9% / 78% no-eligible,比 v4 还慢)。

**核心 trick(与开源版的区别)**:pair-UMMA 的**累加器天然按 M 切分到两个 CTA**(文档 §5.2),所以
**S/softmax/P/O 全部是 CTA-local 的,不需要任何跨 CTA 同步**;开源版在这里加了本不需要的 cross-CTA sync。
真正必需的同步只有 3 处:① K/V 的 TMA multicast mbarrier;② pair-UMMA 到达(leader 发);③ tmem 对称 alloc/dealloc。

**目标 shape**(沿用):varlen b1 s8192,Q head=32 / KV head=2(GQA×16),hd256 fp8 causal,scale=1/√256。
**成功标准**:golden 与主 kernel(`FA_HD256_USE_MAIN=1`)在 fp8 容差内一致,且目标 shape ncu kernel 时长 **< v4 的 ~430us**。

## 设计

### Cluster / tile 映射
- `cluster_shape=(2,1)`,一对 CTA(even=leader / odd=peer)组成一个 pair,处理 **M=256 = 两个 128 行 Q 子块**(同一 head、相邻行段)。CTA0→rows[0:128],CTA1→rows[128:256]。
- QK、PV 都用 **pair-UMMA(`tcgen05.CtaGroup.TWO`)**:mma_tiler M=256,N/K 同 1cta。
- 两 CTA 共享**同一份 K/V**(同 head、同 n_block 范围);causal 下取两者 n_block 的**并集范围**,各 CTA 本地 mask 掉多余块(CTA0 早行会 mask 尾部块)——这是**本地 mask,不是跨 CTA 同步**。

### tmem 布局(每个 CTA 独立 512 列,与 1cta v4 相同)
pair-UMMA 累加器沿 M 切分(文档 §5.2:each CTA holds 128×N accumulator):
```
CTA0 tmem: S0[0,128) S1[128,256)（自己 128 行的 QK,双缓冲) | O[256,512)（自己 128 行 ×256)
CTA1 tmem: 同结构,存后 128 行
```
→ 每 CTA 仍 512 列(和 v4 一样),**S/P/O 全是自己的**。P(fp8 32 列)覆写 S 上半,同 v4。

### K/V TMA multicast（operand 流量减半)
- 参照文档 §5.3 + 开源 2cta:K/V 用 **pair-aware TMA atom(`make_tma_atom_[A|B]_sm100` / `CopyBulkTensorTileG2SOp(cta_group=TWO)`)**,bitmask 只 multicast 给**同 parity**;两半在 MMA 时由 pair 机制凑齐。
- 每 CTA 只从 GMEM load 自己那半 K/V(operand DRAM/smem 流量减半),但 pair-UMMA 读取整块。
- tx_count 需 ×cta_group_size(文档/CLAUDE.md 提示:2CTA 时 tx bytes 翻倍)。

### softmax / correction / epilogue —— **保持 CTA-local,不加任何跨 CTA 同步**(核心)
- 直接复用 v4 的 `softmax_loop`/`softmax_step`/`correction_loop`/epilogue:每个 CTA 对**自己的** S→P→O 做全套,running max/sum 本地。
- fp8 全套保留(max_offset=8 / ×256 / rescale_threshold=4 / 窄 P / descale=1)。

### 必需的 3 处同步(且仅此)
1. **load→mma**:K/V multicast 的 mbarrier(`shared::cluster`,set_barrier_transaction_bytes,tx×2)。
2. **pair-UMMA 到达**:只有 leader(even)CTA 发 MMA;用 `umma_arrive_multicast_2x1SM`(cta_group::2 专用,与 cta_group::1 的 commit 走不同 pipeline)。peer mbarrier 地址由 CUTLASS/CuTeDSL helper 处理,**不手写**。
3. **tmem 对称 alloc/dealloc**:两 CTA 必须对称分配/释放(不对称会死锁,文档明确)。dealloc 前 cluster-wide sync。
- **不加**:S 的跨 CTA 收集、softmax 前后的 cross-CTA barrier、correction 的跨 CTA 协调——这些是开源版多余的部分,全部省掉。

## 实现路径

**基线选择**:以**干净 v4**(`flash_fwd_hd256_1cta_sm100.py`)为主体(保留 fp8 + CTA-local softmax/correction/epilogue),
**把 pair-UMMA/cluster/multicast 机制从开源 `sm100_hd256_2cta_fmha_forward.py` 移植进来**(那里有已验证的
descriptor/bitmask/mbarrier/对称 tmem 细节),但**只取机制、丢弃其多余同步**。

- **新文件**:`flash_attn/cute/flash_fwd_hd256_2cta_sm100.py`,类 `FlashAttentionForwardHd256_2CTA_Sm100`。
- **interface 路由**:fp8+hd256 增加二选一开关(env `FA_HD256_2CTA=1` 或参数),默认仍走 v4(1cta),
  开关走新 2cta —— 便于 A/B 对比,不破坏 v4。

### 函数级改动清单(相对 v4)
1. `__init__`/`_setup_attributes`:`use_2cta_instrs=True`,`cta_group_size=2`,`cluster_shape_mn=(2,1)`;
   `mma_tiler_qk=(2*128,n_block,256)`、`mma_tiler_pv=(2*128,256,n_block)`;tmem 每 CTA 仍 512(pair 沿 M 切)。
2. tiled MMA:`make_trivial_tiled_mma(..., cta_group=TWO)`(QK、PV 都是)。
3. `load`:K/V 换 pair-aware multicast TMA atom + 同 parity bitmask;Q 各 CTA load 自己的 128 行(不 multicast)。tx_count×2。
4. `kernel` 分发:`is_leader_cta = cta_rank_in_cluster % 2 == 0`;只有 leader 发 pair-UMMA;`block_idx_in_cluster` 定位。
5. `mma()`:QK/PV 用 cta_group::2 发射(参照开源 `sm100_hd256_2cta_fmha_forward.py:1116-1294` 的 pair-UMMA 循环),
   但 **S/P/O 的 producer/consumer 仍走 v4 的 CTA-local pipeline**(不引入 cross-CTA S/softmax 同步);
   K/V consumer_release 对齐 multicast。
6. tmem:`TmemAllocator` 用 `two_cta_tmem_dealloc_mbar`(对称 dealloc),cluster_sync 在末尾(参照开源 :703-711,:1547)。
7. `softmax_loop`/`correction_loop`/`epilogue`:**基本不动**(CTA-local)。仅确认 tmem 指针/warp 数与 2cta 线程布局一致。
8. grid:`cute.round_up(grid, cluster)`;varlen 用 `SingleTileVarlenScheduler` + cluster(检查 cluster_shape_mn[1]==1 约束)。
9. causal 并集 trip range:CTA0/CTA1 取 n_block 并集,各自本地 mask(可先用开源的 `get_n_block_min_max` 取 pair 的 max 范围,leader 决定范围广播——注意这只是范围数值,非重同步)。

### 复用(import,不复制)
softmax.py / mask.py / block_info.py / seqlen_info.py / tile_scheduler.py / blackwell_helpers(pair-UMMA gemm)/
pipeline.py / copy_utils.py / mma_sm100_desc.py。descale=1 省掉。

## 分阶段实施(每步:编译 → golden → 目标 shape ncu,建 vN 留痕)

- **S0 基线固化**:新文件 = v4 拷贝 + 改类名 + interface 加 `FA_HD256_2CTA` 开关(此时仍是 1cta 行为)。golden 应逐位一致。→ v0。
- **S1 cluster 骨架**:cluster_shape=(2,1)、grid round_up、leader/peer 判定、tmem 对称 alloc/dealloc + cluster_sync。**先不改 MMA**(仍 1cta MMA,每 CTA 独立跑自己的 128 行,不共享 K/V)——验证 cluster 启动/tmem 对称不死锁,golden 仍对。→ v1。
- **S2 pair-UMMA QK**:QK 换 cta_group::2(M=256),S 沿 M 切;K 仍每 CTA 各自 load 全量(先不 multicast)。leader 发 MMA + `umma_arrive_multicast_2x1SM`。golden 对(重点验 S 切分 + softmax CTA-local 正确)。→ v2。
- **S3 pair-UMMA PV**:PV 换 cta_group::2;O 沿 M 切。golden 对。→ v3。
- **S4 K/V multicast**:K/V 换 pair-aware multicast TMA(同 parity bitmask,tx×2),每 CTA 只 load 半。golden 对 + ncu 看 DRAM/operand 流量降。→ v4。
- **S5 causal 并集 trip range + varlen**:接目标 shape(varlen b1 s8192 GQA)。golden 对。→ v5。
- **S6 调优 + 对比**:ncu vs 1cta-v4(430us);split_P、kv_stage、寄存器、bitmask 子簇同步(只等必需的 CTA)。→ v6+。

**关键验证点**:S2 之后每步都要确认 **softmax/correction 没有引入跨 CTA 同步**(这是本方案价值所在);
用 ncu 的 barrier stall + `smsp__warp_issue_stalled_barrier` 对比 v4,确认没变差。

## S5 详细设计（causal + LPT 的 2-CTA 适配 —— 需斟酌）

**现状(v1 实测)**:非 causal 2cta 走持久调度器→能跑但不提速(997 vs 973us);causal 2cta **死锁**(小 shape hang / 大 shape CUDA Err 912)。

**根因(两个)**:
1. **调度器 cluster 分配**:causal 用 `SingleTileLPTScheduler`。它有 `cluster_shape_m`/`use_cluster_idx`:
   - `use_cluster_idx=True`(causal 非持久时=True)→ `block_idx//cluster_m`,cluster 内两 CTA 拿同一 pair-tile(pair-UMMA 需要)。
   - 但要确认 STATIC 路径 `get_current_work`(非 CLC)也应用了这个除法 + `get_grid_shape` 按 cluster round_up、grid_x = num_block*cluster_m。
2. **causal n_block 范围**:`block_info` 用 `tile_m=cta_tiler[0]=128`,但 2cta 下 m_block 索引 **256 行 pair-tile**;`get_n_block_min_max` 的 `m_idx_max=(m_block+1)*128` 算成半个 pair → 范围错 → 两 CTA 的 K-load / pair-UMMA mbarrier 计数与实际不符 → 死锁。

**适配方案**:
- **A. 范围用 pair-M(union)**:causal 2cta 时,`get_n_block_min_max` 用 `tile_m = cta_tiler[0]*cta_group_size = 256`(pair 的最后一行 = CTA1 边界),两 CTA 取**相同 union 范围**。做法:2cta 时给 `block_info` 传 `tile_m=256`(或专用 BlockInfo)。**验证 BlockInfo.tile_m 只影响范围计算,不影响 mask**(mask 用 tScS 坐标 + `m_block*2*128` base,已确认对)。
- **B. LPT cluster 对齐**:确认 `SingleTileLPTScheduler` STATIC 路径 grid = `num_block*cluster_m`(每 cluster 一个 pair-tile)、`get_current_work` 按 `cluster_m` 除、两 CTA 同 tile。LPT 的负载均衡(causal 尾块多)在 **cluster 粒度**做。若现成 STATIC 路径未对齐,需给 2cta 造一个 cluster-aware 变体(参考开源 `sm100_hd256_2cta_fmha_forward.py` 的 `SingleTileLPTScheduler` 用法 + `round_up(grid, cluster)`)。
- **C. masked-block 循环**:`get_n_block_min_causal_local_mask` 同样用 pair-M(256),两 CTA 迭代相同 masked 区,各自本地 mask(CTA0 早行 mask 更多)。
- **D. 边界**:causal 下 pair 尾部 CTA0 多算的 masked 块 = 浪费但不影响正确性;varlen 的 cluster round_up 对齐要处理。

**仍是最小同步**:以上都是**范围/调度/本地 mask** 的调整,**不引入 softmax/S 的跨 CTA 同步**(pair-UMMA 到达 + K/V multicast mbarrier 之外无新同步)。

**⚠️ 斟酌点(诚实)**:非 causal 2cta(已能跑)**不快过 1cta**(M=256 的张量效率未转化为 wall-time,因 latency-bound)。causal 适配是大工程(scheduler + range + 死锁调试),而预期收益按非 causal 证据看**存疑**。三个选项:
- (i) 做完 causal 适配再最终判定(尊重"实测为准",但投入大);
- (ii) 先给非 causal 2cta 做 v4 同款调优(10-warp/split_P),若仍不反超 973us,则强判 2cta 无用、停;
- (iii) 接受 v4 定版。

## 验证 & 环境
- 环境同 1cta:`PYTHONPATH=仓库 CUTE_DSL_ARCH=sm_103a FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 /home/mudi/mudi_env/bin/python`。
- golden:`repro/golden_cmp.py`(复用 1cta 的,新 2cta vs `FA_HD256_USE_MAIN=1` 主 kernel)。
- 性能:`repro/target_varlen.py` + 干净 ncu(`-k regex:hd256 -c 4 --metrics gpu__time_duration.sum` 取中位)。
- 对照基线:1cta v4 ≈ 430us(ncu);开源 2cta bf16 ≈ 1158us(仅作"多同步反例")。

## 版本化留痕(versions/vN_<desc>/)
每个跑通/有结论的节点:存 `versions/vN_<desc>/{flash_fwd_hd256_2cta_sm100.py, NOTE.md}` + git commit(`vN:` 前缀)+ 更新 `VERSIONS.md`。
回退:`cp versions/vN/....py flash_attn/cute/`。中间小改只 commit。

## 主要风险(2-SM 最易踩)
1. **死锁**:pair-UMMA leader/peer mbarrier、tmem 对称 dealloc 顺序、cluster_sync 时机。参考 `AI/DEBUG_2CTA.md`(printf 二分)。**逐步引入(S1 先只上 cluster 不上 pair-UMMA)**。
2. **bitmask 错误**:multicast/MMA bitmask 用 CUTLASS helper(`create_tma_multicast_mask`),不手写。
3. **fp8 + pair-UMMA**:开源 2cta 无 fp8;fp8 的窄 P(A 操作数)在 cta_group::2 下的 tmem 布局需核对(P 沿 M 切后每 CTA 128×n_block)。
4. **causal 并集范围**:CTA0 多算的被 mask 的块 = 浪费,但不影响正确性;注意 varlen 边界。
5. **文档警示**:该 tile 本就 fit 512 tmem + latency-bound,1-SM 可能本就更优;**若 S4/S5 后仍不快过 v4,即验证了文档结论,接受 v4 定版**(这本身是有价值的实证)。
