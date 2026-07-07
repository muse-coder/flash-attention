# v4 - Target KPP + compact warps + split-P + register tuning

- Content:
  - Re-introduced K-direction ping-pong only for the target-like path:
    `q_stage == 1`, causal, varlen-q, dense, non-local, non-split-KV, non-pack-GQA.
    Other shapes keep the faithful-copy schedule and remain bitwise equal to the main oracle.
  - Fixed KPP pipeline ordering:
    - P/O-rescaled consumer states persist across work tiles.
    - `kpp_iter_global` persists across work tiles in the softmax loop.
    - KPP `P_full` producer commits are guarded by `sync_warp()` + `elect_one()`.
  - Compacted the target path from 16 warps to 10 warps:
    softmax `0..3`, correction/varlen-store `4..7`, MMA `8`, load `9`.
  - Restored split-P partial arrive at 96 columns for KPP.
  - Added explicit `num_regs_other` override for fp8 tuning and selected
    `num_regs_softmax=160`, `num_regs_correction=120`, `num_regs_other=96`.
  - Added a focused pytest regression for SM100 hd256 fp8 varlen causal GQA,
    comparing the dedicated kernel against the `FA_HD256_USE_MAIN=1` oracle.
- Golden:
  - `golden_cmp.py` PASS for all listed shapes, new-vs-main max_abs=0.0000.
  - Target `target_varlen.py` PASS, new-vs-main max_abs=0.00000.
  - `pytest -q tests/cute/test_flash_attn_varlen.py -k hd256_fp8_1cta -x` PASS:
    `1 passed, 5832 deselected`.
- Target shape:
  - Independent-process target runs after this tuning:
    `437.8, 437.6, 437.3, 436.8, 437.0 us`.
  - NCU launch-only capture: kernel name contains
    `flash_fwd_hd256_1cta_sm100FlashAttentionForwardHd256_1CTA_Sm100`, duration
    `436544 ns`, block size `320`, grid size `2048`, registers/thread `160`,
    shared memory/block `232448` bytes.
- FlashInfer trtllm-gen baseline:
  - Current run: fp8 output median `492.3 us`, bf16 output median `540.1 us`.
  - Strict fp8-output 10% target is `<=443.1 us`; v4 median target timing is about
    `437.3 us`, roughly 11.2% lower than the fp8-output FlashInfer median.
- Conclusion:
  - v4 satisfies the requested 10% latency reduction against the current
    FlashInfer trtllm-gen fp8-output baseline on the target shape.
  - Keep this as the current rollback point before attempting further profile-driven
    work.
