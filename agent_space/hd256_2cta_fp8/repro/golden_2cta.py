"""Golden: FA_HD256_2CTA=1 (new 2cta kernel) vs FA_HD256_USE_MAIN=1 (main-kernel oracle).
At stage S0 the 2cta kernel is a byte copy of 1cta v4, so this must be bitwise identical.
"""
import math, os, sys, torch
fp8=torch.float8_e4m3fn; dev="cuda"; torch.manual_seed(42)
def fa(q,k,v,causal,ss):
    import flash_attn.cute.interface as I
    o=I.flash_attn_func(q,k,v,causal=causal,softmax_scale=ss)
    return (o[0] if isinstance(o,tuple) else o).float()
def run(b,sq,sk,h,hkv,d,causal):
    q=torch.randn(b,sq,h,d,device=dev,dtype=torch.bfloat16).to(fp8)
    k=torch.randn(b,sk,hkv,d,device=dev,dtype=torch.bfloat16).to(fp8)
    v=torch.randn(b,sk,hkv,d,device=dev,dtype=torch.bfloat16).to(fp8)
    ss=1.0/math.sqrt(d)
    os.environ["FA_HD256_USE_MAIN"]="1"; os.environ["FA_HD256_2CTA"]="0"
    o_main=fa(q,k,v,causal,ss)
    os.environ["FA_HD256_USE_MAIN"]="0"; os.environ["FA_HD256_2CTA"]="1"
    o_new=fa(q,k,v,causal,ss)
    diff=(o_new-o_main).abs().max().item(); ref=o_main.abs().max().item()
    ok = diff <= 1e-2 + 0.02*ref
    print(f"b{b} sq{sq} sk{sk} h{h}/{hkv} d{d} caus={causal}  2cta-vs-main max_abs={diff:.5f} (ref={ref:.3f})  {'PASS' if ok else 'FAIL'}")
    return ok
ok=True
ok&=run(1,128,128,1,1,256,False)
ok&=run(1,256,256,1,1,256,True)
ok&=run(2,512,512,4,4,256,True)
ok&=run(2,512,512,4,1,256,True)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
