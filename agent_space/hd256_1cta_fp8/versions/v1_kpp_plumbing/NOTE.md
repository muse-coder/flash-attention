# v1 — K-ping-pong plumbing only

- 内容：引入 `self.s_pp = 2`，为后续 K 方向 S/P ping-pong 拆出独立 barrier/pipeline：
  `S_full_kpp`、`P_full_kpp`、`P_full_lastsplit_kpp`、`O_rescaled_kpp`。
- 行为：旧的 conflated `pipeline_s_p_o` 仍然驱动 `mma()` / `softmax_loop()` / `correction_loop()`；
  新 pipeline 只初始化、不接线，用于降低后续迁移风险。
- golden（new vs 主 kernel）：`golden_cmp.py` 全 PASS，max_abs=0.0000。
- 目标 shape（varlen b1 s8192 hq32 hkv2 d256 fp8 causal）：golden PASS，751.3us / 1463.6 TFLOP/s。
- 结论：S1/S2 plumbing 编译和真机验证通过；性能与 v0 基本持平。下一步迁移 `softmax_step`
  的 S_full/P_full wait/release 到独立 kpp pipeline，再改 `mma()` 的 K-block parity 调度。
