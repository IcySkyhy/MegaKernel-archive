import re, subprocess, os, sys
root='/workspace/ascend_mega_kernel/.tower/worktrees/wt-117'
p=os.path.join(root,'m25_attn_fa_core/evidence/exclusions.md')
lines=open(p).read().splitlines()
rows=[l for l in lines if re.match(r'^\| \d+ \|', l)]
ev=os.path.join(root,'m25_attn_fa_core/evidence')
tracked=set(subprocess.run(['git','ls-tree','-r','--name-only','HEAD'],cwd=root,capture_output=True,text=True).stdout.split())
ok_all=True
print(f"共 {len(rows)} 行；逐行核对（日志文件存在 / commit 存在 / 读数串出现在该日志）")
for l in rows:
    n=l.split('|')[1].strip()
    names=re.findall(r'`([A-Za-z0-9_./]+\.log)`', l)
    hashes=re.findall(r'`([0-9a-f]{7})`', l)
    reads=re.findall(r'`?(EXIT[A-Z_0-9]*=\d+|rc=\d+|hang=\d+|若[^`]*)?`?', l)
    reads=[r for r in re.findall(r'(EXIT[A-Z_0-9]*=\d+)', l)]
    prob=[]
    for nm in names:
        cand=nm if '/' in nm else nm
        full=os.path.join('m25_attn_fa_core/evidence',cand)
        if not os.path.exists(os.path.join(root,full)):
            # 白名单：`run_m64b.log` 只出现在"已不在库中"的免责句里（本条读数以别的日志为准）
            if nm == 'run_m64b.log' and '已不在库中' in l:
                continue
            prob.append(f"日志缺:{nm}")
        elif reads:
            txt=open(os.path.join(root,full),errors='replace').read()
            miss=[r for r in reads if r not in txt]
            if miss: prob.append(f"读数不在日志:{miss}")
    for h in hashes:
        r=subprocess.run(['git','cat-file','-e',h+'^{commit}'],cwd=root,capture_output=True)
        if r.returncode!=0: prob.append(f"commit缺:{h}")
    print(f"  行{n:>2}: 日志={names or '-'} commit={hashes or '-'} 读数={reads or '-'}  => {'OK' if not prob else ' !! '+'; '.join(prob)}")
    if prob: ok_all=False
print("ALL_OK" if ok_all else "HAS_PROBLEM")
