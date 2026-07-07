import math, os, torch
import flash_attn.cute.interface as I
B=int(os.environ["B"]); S=int(os.environ["S"]); HQ,HKV,D=int(os.environ.get("HQ","32")),int(os.environ.get("HKV","2")),256
fp8=torch.float8_e4m3fn; dev="cuda"; torch.manual_seed(0)
q=torch.randn(B*S,HQ,D,device=dev,dtype=torch.bfloat16).to(fp8)
k=torch.randn(B*S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
v=torch.randn(B*S,HKV,D,device=dev,dtype=torch.bfloat16).to(fp8)
cu=torch.arange(B+1,device=dev,dtype=torch.int32)*S; ss=1/math.sqrt(D)
run=lambda: I.flash_attn_varlen_func(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=S,max_seqlen_k=S,causal=True,softmax_scale=ss,pack_gqa=False)
for _ in range(5): run()
torch.cuda.synchronize(); run(); torch.cuda.synchronize(); print("done")
