import numpy as np, os, subprocess, sys
HERE='/workspace/ascend_mega_kernel/.tower/worktrees/wt-117/m25_attn_fa_core'
rng=np.random.default_rng(7)
def to_bf16(a):
    u=(a.astype(np.float32).view(np.uint32)+0x7FFF+((a.astype(np.float32).view(np.uint32)>>16)&1))&0xFFFF0000
    return (u>>16).astype('<u2')
def from_bf16(u):
    return (u.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
m,ctx,NH,NKV,HD=37,64,24,2,256
scale=HD**-0.5
q=from_bf16(to_bf16(rng.uniform(-1,1,(m,NH,HD)))).reshape(m,NH,HD)
k=from_bf16(to_bf16(rng.uniform(-1,1,(ctx,NKV,HD)))).reshape(ctx,NKV,HD)
v=from_bf16(to_bf16(rng.uniform(-1,1,(ctx,NKV,HD)))).reshape(ctx,NKV,HD)
pos=np.arange(m)[:,None]; idx=np.arange(ctx)[None,:]; mask=idx<=pos
ref=np.zeros((m,NH,HD))
for h in range(NH):
    n2=h//12
    s=(q[:,h,:]@k[:,n2,:].T)*scale
    s=np.where(mask,s,-np.inf); w=np.exp(s-s.max(1,keepdims=True)); p=w/w.sum(1,keepdims=True)
    ref[:,h,:]=p@v[:,n2,:]
d='/tmp/selfcheck1'; os.makedirs(d,exist_ok=True)
for name,arr in (('q.bin',q),('k.bin',k),('v.bin',v)):
    to_bf16(arr).tofile(os.path.join(d,name))
open(os.path.join(d,'params.txt'),'w').write(f'm={m}\nposBase=0\nctx={ctx}\nnBlk=1\nnh={NH}\nhd={HD}\n')
to_bf16(ref).tofile(os.path.join(d,'out.bin'))
subprocess.run([sys.executable,HERE+'/check_ref.py',d],check=False)
print('--- 现在把 out 弄坏（乘 1.05）---')
to_bf16(ref*1.05).tofile(os.path.join(d,'out.bin'))
subprocess.run([sys.executable,HERE+'/check_ref.py',d],check=False)
