import torch
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref
torch.manual_seed(0)
fp8=torch.float8_e4m3fn; dr=torch.bfloat16
def run(sq,sk,d,causal):
    q=torch.randn(1,sq,1,d,device="cuda",dtype=dr).to(fp8).to(dr)
    k=torch.randn(1,sk,1,d,device="cuda",dtype=dr).to(fp8).to(dr)
    v=torch.randn(1,sk,1,d,device="cuda",dtype=dr).to(fp8).to(dr)
    qq,kk,vv=[x.to(fp8) for x in (q,k,v)]
    oref,_=attention_ref(q,k,v,causal=causal)
    opt,_=attention_ref(q,k,v,causal=causal,upcast=False,reorder_ops=True,intermediate_dtype=fp8)
    o=flash_attn_func(qq,kk,vv,causal=causal); o=o[0] if isinstance(o,tuple) else o
    err=(o.float()-oref.float()).abs().max().item(); pe=(opt.float()-oref.float()).abs().max().item()
    th=2*pe+2*(oref+0.3-0.3-oref).abs().max().item()
    print(f"d{d} sq{sq} sk{sk} causal={causal}: err={err:.4f} th={th:.4f} {'PASS' if err<=th else 'FAIL'} (qstage={'1' if sq<=128 else '2'})")
# hd128 fp8: q_stage=1 (sq<=128) with 1 vs many K-blocks
run(64,128,128,False)
run(64,512,128,False)
run(128,512,128,False)
run(128,512,128,True)
# hd128 fp8: q_stage=2 (sq>128) many K-blocks (known-good path)
run(256,512,128,False)
