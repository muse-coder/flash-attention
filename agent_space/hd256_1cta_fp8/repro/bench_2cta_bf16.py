"""Feasibility probe: existing 2-CTA hd256 (bf16) kernel at the target shape, for NCU.
2cta kernel is bf16-only (no fp8). Target dims: b1 s8192 hq32 hkv2 d256 causal, dense.
"""
import math, os, torch
import flash_attn.cute.interface as I
S,HQ,HKV,D=8192,32,2,256
dev="cuda"; torch.manual_seed(0)
q=torch.randn(1,S,HQ,D,device=dev,dtype=torch.bfloat16)
k=torch.randn(1,S,HKV,D,device=dev,dtype=torch.bfloat16)
v=torch.randn(1,S,HKV,D,device=dev,dtype=torch.bfloat16)
ss=1.0/math.sqrt(D)
run=lambda: I.flash_attn_func(q,k,v,causal=True,softmax_scale=ss,pack_gqa=False)
for _ in range(int(os.environ.get("FA_WARMUP","3"))): run()
torch.cuda.synchronize()
run(); torch.cuda.synchronize()
print("done 2cta bf16")
