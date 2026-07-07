import math, os, torch, sys
sys.path.insert(0,"/home/mudi/flashinfer")
import flashinfer
B=int(os.environ["B"]); S=int(os.environ["S"]); HQ,HKV,D,PAGE=int(os.environ.get("HQ","32")),int(os.environ.get("HKV","2")),256,64
fp8=torch.float8_e4m3fn; dev="cuda"; torch.manual_seed(0)
npages=(S+PAGE-1)//PAGE; total=npages*B
q=torch.randn(B*S,HQ,D,device=dev,dtype=torch.bfloat16).to(fp8)
kv=torch.randn(total,2,HKV,PAGE,D,device=dev,dtype=torch.bfloat16).to(fp8)
qo=torch.arange(B+1,device=dev,dtype=torch.int32)*S
kvi=torch.arange(B+1,device=dev,dtype=torch.int32)*npages
kvidx=torch.arange(total,device=dev,dtype=torch.int32)
last=torch.full((B,),(S-1)%PAGE+1,dtype=torch.int32,device=dev)
out=torch.empty(B*S,HQ,D,device=dev,dtype=fp8)
ws=torch.empty(256*1024*1024,dtype=torch.uint8,device=dev)
w=flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(ws,"HND",backend="trtllm-gen")
w.plan(qo,kvi,kvidx,last,HQ,HKV,D,PAGE,causal=True,pos_encoding_mode="NONE",sm_scale=1/math.sqrt(D),q_data_type=fp8,kv_data_type=fp8,o_data_type=fp8)
run=lambda: w.run(q,kv,q_scale=1.0,k_scale=1.0,v_scale=1.0,out=out)
for _ in range(5): run()
torch.cuda.synchronize(); run(); torch.cuda.synchronize(); print("done")
