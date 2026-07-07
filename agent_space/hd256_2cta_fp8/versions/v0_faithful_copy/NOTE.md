# v0 (S0) — 忠实副本 + 路由骨架
- 新文件 `flash_fwd_hd256_2cta_sm100.py` = 干净 1cta v4 的逐字节拷贝,仅改类名 `FlashAttentionForwardHd256_2CTA_Sm100`。此时仍是 1-CTA 行为。
- interface:import 新类;加 `FA_HD256_2CTA=1` 开关 + 独立 compile-key `hd256_fp8_2cta`(防别名复用 CUBIN)。
- golden(`repro/golden_2cta.py`,FA_HD256_2CTA=1 vs FA_HD256_USE_MAIN=1):全 shape **max_abs=0.00000 PASS**。
- 下一步:S1 cluster 骨架(cluster=(2,1)+对称 tmem,先不上 pair-UMMA)。
