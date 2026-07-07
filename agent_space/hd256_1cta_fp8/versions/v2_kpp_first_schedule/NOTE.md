# v2 — First K-ping-pong schedule

- 内容：实现 dense `q_stage=1` 的 K-direction ping-pong 调度：
  `QK(i+1)` 写 `S[(i+1)%2]`，再等待 `P(i)` + `O_rescaled` 做 `PV(i)`，两路 P 累加到单一 O。
- softmax：按 `k_iter % 2` 读 `S0/S1`、写 `P0/P1`，使用独立 `S_full_kpp` / `P_full_kpp`。
- correction：KPP 路径显式等待 `O_full`，rescale 单一 O 后通过 `O_rescaled_kpp` 放行下一轮 PV。
- 限制：首版 KPP 禁用了 split-P partial arrive（PV 等完整 P），block-sparse/split-KV 仍走 legacy 路径。
- golden（new vs 主 kernel）：`golden_cmp.py` 全 PASS，max_abs=0.0000。
- 目标 shape（varlen b1 s8192 hq32 hkv2 d256 fp8 causal）：golden PASS，752.3us / 1461.5 TFLOP/s。
- 结论：同步协议可运行，但当前性能未超过 v0/v1。可能被完整-P 等待、额外 O_rescaled 同步、
  或 KV wait 顺序抵消。下一步需要恢复/重设 split-P、profile KPP stall，并建立 flashinfer trtllm-gen 目标基线。
