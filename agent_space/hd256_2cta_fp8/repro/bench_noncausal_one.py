import math, os, torch
import flash_attn.cute.interface as I
S,HQ,HKV,D=8192,32,2,256
fp8=torch.float8_e4m3fn; dev="cuda"; torch.manual_seed(0)
q=torch.randn(1,S,HQ,D,device=dev,dtype=torch.bfloat16).to(fp8)
k=torch.randn(1,S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
v=torch.randn(1,S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
ss=1.0/math.sqrt(D)
run=lambda: I.flash_attn_func(q,k,v,causal=False,softmax_scale=ss,pack_gqa=False)
for _ in range(int(os.environ.get("FA_WARMUP","3"))): run()
torch.cuda.synchronize(); run(); torch.cuda.synchronize(); print("done")
