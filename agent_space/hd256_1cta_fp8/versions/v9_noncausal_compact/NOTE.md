# v9 — 非causal 紧凑 bonus(源于 2cta 判据实验)
- 给 1cta 非 causal(TMA-O)路径也加 16→11 warp 紧凑(此前只 varlen/causal 路径紧凑)。
- 结果:非 causal b1 s8192 hq32 hkv2 从 973→936.5us(-3.8%)。causal 目标不受影响(仍 431.9us / golden 0.00000 PASS)。
- 来源:2cta 判据实验中为公平对比给 1cta 补的同款紧凑,顺带成为 1cta 非causal 的改进。
