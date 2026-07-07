# FP8 P-saturation bug in hd256 1-CTA (max_offset + rescale_threshold overflow)

## Symptom

hd256 1-CTA causal varlen **fp8** had large accuracy error vs a trusted reference
(FlashInfer `BatchPrefillWithPagedKVCacheWrapper(backend="trtllm-gen")`, fp8, quant scale=1).

Multi-batch ragged varlen `[4096, 5120, 6144]`, HQ=32/HKV=2, hd256, causal, fp8:

| metric | before fix | after fix | trtllm-gen |
|---|---|---|---|
| rel_L1 (Σ\|err\|/Σ\|ref\|) | **15.6%** | **4.36%** | 3.5% |
| cosine | 0.9863 | 0.9992 | 0.9994 |
| max abs | 1.08 | 0.156 | 0.155 |

bf16 was unaffected (only fp8 hit).

## How it was localized

1. **Per-batch == single-batch, bit-identical** → varlen multi-batch logic is correct;
   the error is per-batch, not a batching/cross-batch-leak bug.
2. **LSE matched fp32 exactly** (octile diff `-0.0000`) → softmax normalization / row_sum /
   row_max are all correct. The error is only in `O = P·V`.
3. **Relative error is flat across query positions** (~18%), not concentrated → not a
   causal-diagonal / boundary bug. The apparent "early-query spike" in *absolute* error was
   just the output-magnitude effect (early causal queries have fewer keys → larger \|O\|).
4. **Constant inputs (q=k=v=1.0/1.5/2.0) → exact (error 0)**; error only appears with a
   *spread* of scores, and grows with `softmax_scale` (peakier P). A plain "P→fp8" numeric
   model gives only ~2.5% — i.e. FA4 was ~7× worse than an ideal fp8 P-cast.
5. Setting `rescale_threshold=0` for fp8 collapsed the error to ~2.5% → confirmed the bug is
   the interaction of `rescale_threshold` with the fp8 P scaling.

## Root cause

fp8 flash-attention scales the softmax probabilities up by `2^max_offset` before the e4m3
cast so small probs don't underflow (`scale_subtract_rowmax` adds `max_offset` to the exp2
bias in `softmax.py`). Separately, the `rescale_threshold` optimization avoids rescaling the
O accumulator every K-block: when a new block's max exceeds the running max by `≤
rescale_threshold` (log2 units), the kernel **keeps the old row_max and skips the O rescale**
(`update_row_max` in `softmax.py`).

Consequence: within such a block an entry can have `(S − old_max)·scale_log2 > 0`, up to
`+rescale_threshold`. The exp2 argument therefore reaches `max_offset + rescale_threshold`, so

```
P_max = 2^(max_offset + rescale_threshold) = 2^(8 + 4) = 2^12 = 4096
```

which **exceeds the e4m3 max of 448 (≈ 2^8.8) and saturates**. The clipped entries are the
*dominant* probabilities, so peaky distributions (large softmax_scale) are hit hardest — and
constant inputs (max never grows → excess = 0 → P_max = 2^max_offset = 256 < 448) are exact.
This matched every observation.

Constraint that must hold for e4m3:

```
max_offset + rescale_threshold ≤ log2(448) ≈ 8.8
```

The old values `max_offset=8`, `rescale_threshold=4` violate it (sum = 12).

## Fix

Lower `max_offset` from 8 → 4, keep `rescale_threshold=4` (sum = 8 ≤ 8.8, so `P_max = 2^8 =
256 < 448`). Three sites in `flash_fwd_hd256_1cta_sm100.py` must stay consistent:

- P-path `max_offset` (softmax loop)
- correction/stats-path `max_offset` (`Float32`)
- `max_offset_scale = 2^max_offset` (256.0 → 16.0)

**Why this and not `rescale_threshold=0`:** `max_offset` barely affects accuracy (small probs
contribute negligibly to O; a max=1 P-cast already gives ~2.5%), whereas `rescale_threshold`
is a real perf knob. Reducing the threshold instead costs performance:

| config | kernel (ncu, locked clock, GPU idle) | rel_L1 |
|---|---|---|
| max_offset=8, thr=4 (orig) | 588.9us | 15.6% ❌ |
| thr=0 | ~753us ❌ | 4.3% |
| thr=0.5 | ~684us ❌ | 4.3% |
| **max_offset=4, thr=4 (fix)** | **589.5us** | **4.36%** ✅ |

`max_offset` is a compile-time constant in the exp2 FMA bias — zero runtime cost. ncu A/B
(same idle GPU, locked clock) shows 589.5us vs 588.9us = within noise. **Accuracy fixed,
performance unchanged.**

## Note: other fp8 sm100 kernels share this latent bug

`flash_fwd_sm100.py` (hd128/192) and `flash_fwd_hd256_2cta_sm100.py` use the same shared
`SoftmaxSm100` with the same `max_offset=8` / `rescale_threshold=4`. hd128 1-CTA fp8 was
measured with the identical failure (relerr 0.17–0.32, worse at larger scale). bf16 is
immune (no 448 ceiling). This fix is scoped to hd256 1-CTA only, per request; the others
would need the same `max_offset + rescale_threshold ≤ 8.8` correction if fp8 accuracy there
matters.

## Repro

`agent_space/hd256_1cta_fp8/repro/bench_scripts/varlen_multibatch.py` (accuracy + perf,
FA4 vs trtllm-gen). Scale-sweep / constant-input / LSE diagnostics were run ad-hoc.
