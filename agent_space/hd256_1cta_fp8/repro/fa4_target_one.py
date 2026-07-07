"""FA4 dedicated hd256 1-CTA fp8 target-shape one-shot for NCU (warmup then 1 timed launch)."""
import math, os, torch
import flash_attn.cute.interface as I
S,HQ,HKV,D=8192,32,2,256
fp8=torch.float8_e4m3fn; dev="cuda"
torch.manual_seed(0)
q=torch.randn(S,HQ,D,device=dev,dtype=torch.bfloat16).to(fp8)
k=torch.randn(S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
v=torch.randn(S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
cu=torch.tensor([0,S],device=dev,dtype=torch.int32)
ss=1.0/math.sqrt(D)
run=lambda: I.flash_attn_varlen_func(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=S,max_seqlen_k=S,causal=True,softmax_scale=ss,pack_gqa=False)
for _ in range(int(os.environ.get("FA_WARMUP","3"))): run()
torch.cuda.synchronize()
run(); torch.cuda.synchronize()
print("done")
