import math, torch
from flash_attn.cute.interface import flash_attn_func
fp8=torch.float8_e4m3fn; dev="cuda"
def bench(b,s,h,d,causal,iters=50):
    q=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
    k=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
    v=torch.randn(b,s,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
    ss=1.0/math.sqrt(d)
    f=lambda: flash_attn_func(q,k,v,causal=causal,softmax_scale=ss)
    for _ in range(10): f()
    torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): f()
    e1.record(); torch.cuda.synchronize()
    ms=e0.elapsed_time(e1)/iters
    # flops: causal ~ 0.5, 2 matmuls * 2
    fl=2*2*b*h*s*s*d*(0.5 if causal else 1.0)
    tflops=fl/(ms*1e-3)/1e12
    print(f"b{b} s{s} h{h} d{d} caus={causal}: {ms*1000:.1f} us  {tflops:.1f} TFLOP/s")
bench(2,2048,8,256,True)
bench(2,4096,8,256,True)
bench(4,4096,16,256,True)
bench(2,2048,8,256,False)
