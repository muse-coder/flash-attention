import math, os, torch
fp8=torch.float8_e4m3fn; dev="cuda"
def call(HQ,HKV,S,use_main):
    os.environ["FA_HD256_USE_MAIN"]="1" if use_main else "0"; os.environ["FA_HD256_2CTA"]="0"
    import flash_attn.cute.interface as I
    torch.manual_seed(0)
    q=torch.randn(S,HQ,256,device=dev,dtype=torch.bfloat16).to(fp8)
    k=torch.randn(S,HKV,256,device=dev,dtype=torch.bfloat16).to(fp8)
    v=torch.randn(S,HKV,256,device=dev,dtype=torch.bfloat16).to(fp8)
    cu=torch.tensor([0,S],device=dev,dtype=torch.int32)
    o=I.flash_attn_varlen_func(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=S,max_seqlen_k=S,causal=True,softmax_scale=1/math.sqrt(256),pack_gqa=False)
    return (o[0] if isinstance(o,tuple) else o).float()
for HQ,HKV in [(16,1),(8,1),(16,2)]:
    om=call(HQ,HKV,8192,True); on=call(HQ,HKV,8192,False)
    d=(on-om).abs().max().item(); r=om.abs().max().item()
    print(f"HQ={HQ} HKV={HKV}: max_abs={d:.5f} {'PASS' if d<=1e-2+0.02*r else 'FAIL'}")
