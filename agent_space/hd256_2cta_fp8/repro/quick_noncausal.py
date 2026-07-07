import math, os, torch
fp8=torch.float8_e4m3fn; dev="cuda"; torch.manual_seed(0)
def fa(q,k,v,ss):
    import flash_attn.cute.interface as I
    o=I.flash_attn_func(q,k,v,causal=False,softmax_scale=ss)
    return (o[0] if isinstance(o,tuple) else o).float()
b,sq,h,hkv,d=2,512,4,4,256
q=torch.randn(b,sq,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
k=torch.randn(b,sq,hkv,d,device=dev,dtype=torch.bfloat16).to(fp8)
v=torch.randn(b,sq,hkv,d,device=dev,dtype=torch.bfloat16).to(fp8)
ss=1.0/math.sqrt(d)
os.environ["FA_HD256_USE_MAIN"]="1"; os.environ["FA_HD256_2CTA"]="0"
om=fa(q,k,v,ss)
os.environ["FA_HD256_USE_MAIN"]="0"; os.environ["FA_HD256_2CTA"]="1"
on=fa(q,k,v,ss)
diff=(on-om).abs().max().item(); ref=om.abs().max().item()
print(f"noncausal 2cta-vs-main max_abs={diff:.5f} ref={ref:.3f} {'PASS' if diff<=1e-2+0.02*ref else 'FAIL'}")
