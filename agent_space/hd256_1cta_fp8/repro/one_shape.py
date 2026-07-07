import math, torch, sys
from flash_attn.cute.interface import flash_attn_func
fp8=torch.float8_e4m3fn; dev="cuda"
b,s,h,d=2,4096,8,256; causal=True
q=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
k=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
v=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
ss=1.0/math.sqrt(d)
for _ in range(int(sys.argv[1]) if len(sys.argv)>1 else 5):
    o=flash_attn_func(q,k,v,causal=causal,softmax_scale=ss)
torch.cuda.synchronize()
