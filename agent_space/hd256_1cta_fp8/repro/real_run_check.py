import torch
import flash_attn.cute.interface as I
from flash_attn.cute.interface import _get_device_arch
print("arch =", _get_device_arch(), "| torch cap =", torch.cuda.get_device_capability(0))
b,s,h,d = 2,512,4,128
q,k,v = [torch.randn(b,s,h,d,dtype=torch.bfloat16,device="cuda") for _ in range(3)]
out = I.flash_attn_func(q,k,v,causal=True)
out = out[0] if isinstance(out, tuple) else out
# reference
qf,kf,vf = [x.float() for x in (q,k,v)]
scale = 1.0/(d**0.5)
sc = torch.einsum("bshd,bthd->bhst",qf,kf)*scale
mask = torch.triu(torch.ones(s,s,device="cuda",dtype=torch.bool),1)
sc = sc.masked_fill(mask,float("-inf"))
p = sc.softmax(-1)
ref = torch.einsum("bhst,bthd->bshd",p,vf)
err = (out.float()-ref).abs().max().item()
print("bf16 hd128 causal max_abs_err =", err)
print("PASS" if err < 1e-2 else "FAIL")
