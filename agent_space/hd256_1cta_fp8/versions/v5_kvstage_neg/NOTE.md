# v5 — kv_stage 4→5 (overlap_sO_sQ) — 负结果，已回退

- 假设：ncu 显示 long_scoreboard(等内存) 1.44 cyc/instr，deeper KV 流水应能掩盖。
- 改动：目标路径开 `overlap_sO_sQ`（sO 复用 sQ 省 32KB smem）→ kv_stage 4→5。
- 结果：golden 逐位一致 PASS；干净 ncu 中位 **430.1us** vs v4 430.9us —— **无实质变化（噪声内）**。
- 结论：**内存流水不是瓶颈，barrier 才是**。已回退（本快照 == v4 代码）。
- 根因洞察：S ping-pong 深度被 tmem 死锁在 2（S 2×128 + O 256 = 512 满），MMA 最多领先 softmax 1 块，
  softmax 延迟直接以 barrier stall 卡住 MMA。真正杠杆=加速 softmax（更多 warp）或减少 warp-role 同步。
