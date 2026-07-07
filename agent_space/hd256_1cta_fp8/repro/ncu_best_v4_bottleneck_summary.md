# v4 best-kernel bottleneck summary

Date: 2026-07-07

Scope: current best hd256 1CTA fp8 causal varlen target kernel, v4 code
(`versions/v4_kpp_compact_splitp_regs`), using the profiling workflow from
`/home/mudi/atrex-kernel-agent/gpu-wiki/docs/nvidia/common/profiling`.

## Measurement

Target harness:

```bash
PYTHONPATH=/home/mudi/flash-attention \
CUTE_DSL_ARCH=sm_103a \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
CUDA_VISIBLE_DEVICES=0 \
/home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/target_varlen.py
```

Full NCU capture:

```bash
ncu --target-processes all \
  --kernel-name 'regex:.*FlashAttentionForwardHd256.*' \
  --launch-count 1 \
  --import-source on \
  --section SpeedOfLight \
  --section ComputeWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section Occupancy \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --section SourceCounters \
  --section InstructionStats \
  --section PmSampling \
  --section PmSampling_WarpStates \
  -o agent_space/hd256_1cta_fp8/repro/ncu_best_v4_bottleneck \
  /home/mudi/mudi_env/bin/python agent_space/hd256_1cta_fp8/repro/target_varlen.py
```

Artifacts:

- `repro/ncu_best_v4_bottleneck.ncu-rep`
- `repro/ncu_best_v4_bottleneck.raw.csv`
- `repro/ncu_best_v4_bottleneck.details.txt`

Note: per the profiling docs, NCU `Duration` is used only as a launched-kernel
sanity check. Bottleneck diagnosis below uses ratios, stalls, occupancy, and
pipeline evidence. The separate launch-only NCU baseline correction remains:
FA4 v4 median ~430.9 us vs FlashInfer trtllm-gen fp8-out median ~445.4 us,
so current best is ~3.3% faster, not the earlier event-timing based ~10%.

## Captured kernel identity

- Kernel: `FlashAttentionForwardHd256_1CTA_Sm100`
- Grid: 2048 CTAs
- Block: 320 threads = 10 warps
- Registers/thread: 160
- Dynamic shared memory/block: 231.424 KiB
- Total shared memory/block: 232.448 KiB
- Theoretical occupancy: 15.62%
- Achieved occupancy: 15.03%
- Achieved active warps/SM: 9.62
- Occupancy limit: registers = 1 block/SM, shared memory = 1 block/SM

Interpretation: this kernel is effectively capped at one CTA per SM. There is
no extra CTA-level latency hiding available for the inner pipeline.

## Main counter signals

Speed of Light:

- Compute(SM) throughput: 61.01%
- Memory throughput: 36.92%
- DRAM throughput: 4.74%
- L1/TEX throughput: 45.46%
- L2 throughput: 36.82%
- L2 hit rate: 90.97%

Scheduler:

- Issue slots busy: 35.55%
- One or more eligible warp: 36.80%
- No eligible warp: 63.20%
- Issued warp/scheduler: 0.37
- Active warps/scheduler: 2.42
- Eligible warps/scheduler: 0.40
- Warp cycles per issued instruction: 6.58

Pipeline:

- Tensor pipe active: 58.82% elapsed / 60.62% active
- ALU pipe active: 26.05% elapsed
- FMA pipe active: 17.21% elapsed
- TMA pipe active: 0.47% elapsed
- Local/shared spilling: 0

Warp stall ratios, cycles per issued instruction:

- barrier: 1.90
- long_scoreboard: 1.45
- wait: 1.04
- short_scoreboard: 0.50
- no_instruction: 0.36
- mio_throttle: 0.03
- math_pipe_throttle: 0.02

PM sampling distribution:

- barrier: 7313 samples, 26.7%
- long_scoreboard: 6568 samples, 24.0%
- wait: 4425 samples, 16.2%
- selected: 4096 samples, 15.0%
- short_scoreboard: 2118 samples, 7.7%
- no_instruction: 1542 samples, 5.6%
- other throttle/front-end classes are each <=1.4%

## Diagnosis

The current best version is not DRAM-bandwidth-bound. DRAM is only 4.74% of
peak, L2 hit rate is high, and increasing KV pipeline depth in v5 did not move
runtime. L2/L1 traffic exists, but it is not the limiting resource.

It is not softmax math-bound. v7 ex2-emulation sweeps did not improve runtime,
math-pipe and MIO throttle are tiny, and the v8 probe made softmax work 3x/5x
heavier with essentially no runtime increase. Softmax work is already hidden by
the surrounding pipeline.

The dominant bottleneck is the warp-specialized dependency chain around the
single O accumulator:

1. MMA produces `S(i)` into one of two tmem S slots.
2. Softmax waits for `S_full`, writes `P(i)`, and signals `P_full`.
3. MMA waits for both `P_full(i)` and `O_rescaled` before launching `PV(i)`.
4. Correction waits for `O_full`, rescales the single O accumulator when
   needed, then signals `O_rescaled`.
5. The next `PV` cannot consume the single O accumulator until that signal.

The code-level anchors are:

- S/P ping-pong depth is only 2 while O is single-buffered:
  `flash_fwd_hd256_1cta_sm100.py:159-162`, `327-333`.
- KPP split pipelines are separate but still form a hard producer/consumer
  chain: `1013-1044`.
- MMA waits on `P_full` and `O_rescaled` before each `PV`:
  `1832-1835`, `1870-1873`.
- Correction waits on `O_full`, performs optional O rescale, then releases
  `O_rescaled`: `2904-2917`, final path `2946-2968`.
- Softmax writes P and does split-P early arrive, but row_sum update is after
  P release, so row_sum itself is not on the MMA-unlock path:
  `2737-2769`.

Because tmem is full (`S0/S1` 2x128 columns + `O` 256 columns = 512 columns),
the kernel cannot add a third S/P buffer or a second independent O accumulator.
This makes the barrier chain visible directly in scheduler metrics: only 0.40
eligible warps/scheduler and 63.20% no-eligible cycles.

## Excluded hypotheses

- More KV staging: v5 `kv_stage 4->5` was flat. Memory depth is not the next
  lever.
- Earlier split-P threshold: v6 `{96,64,32}` was flat. The current `96` is not
  the active limiter.
- Faster exp2/softmax math: v7 was flat and v8 5x softmax work was flat. Dual
  softmax should not help.
- Register spill reduction: NCU reports zero local/shared spills. Registers are
  an occupancy limiter, but simply lowering registers risks starving the
  already tight warp-specialized roles without freeing another CTA unless both
  register and shared-memory/tmem constraints are solved.
- DRAM bandwidth: DRAM throughput is too low to explain the ceiling.

## Current bottleneck statement

Best-v4 is latency/synchronization bound inside a one-CTA-per-SM,
warp-specialized pipeline. The immediate limiter is not arithmetic throughput
or global memory bandwidth; it is low eligible-warp supply caused by the
`P_full` / `O_rescaled` / `O_full` barrier chain around a single tmem O
accumulator. Tensor pipe reaches about 59% elapsed utilization, but issue slots
are only about 36% busy because the MMA/correction/softmax roles frequently
wait on each other.

## Next direction to test

Only one plausible next experiment family remains before a larger redesign:

- Try to reduce or bypass one synchronization edge in the O-rescale chain,
  not to speed up softmax. Example: first probe whether `scale_prev == 1` is
  frequent enough to matter; if it is, specialize that fast path so correction
  warps can release `O_rescaled` with less work and fewer waits. If it is not,
  the next edit should instead target merging/reordering the `O_full` ->
  correction -> `O_rescaled` handshake.

Verification target:

- Same harness and same NCU sections.
- Runtime must be judged with launch-only NCU or a same-process timing A/B.
- Expected metric movement if successful:
  - `smsp__average_warps_issue_stalled_barrier_per_issue_active` down from
    ~1.90.
  - `No Eligible` down from ~63%.
  - `Issue Slots Busy` up from ~35.5%.
  - Tensor pipe elapsed utilization up without increasing DRAM throughput.
