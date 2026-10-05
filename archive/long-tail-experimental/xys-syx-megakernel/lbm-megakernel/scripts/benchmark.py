#!/usr/bin/env python3
"""Fresh matched event trials; refuses a binary/source snapshot without gates."""
import argparse
import datetime
import json
from pathlib import Path
import random
import re
import statistics
from common import BIN, ROOT, digest, environment, run, save, source_hashes, variants

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, default=ROOT/'results/local')
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--steps',type=int,nargs='+',default=[100,1000])
p.add_argument('--seed',type=int,default=20260924)
a=p.parse_args()
a.out=a.out.resolve()
if a.rounds<2 or any(n<=0 or n%2 for n in a.steps): p.error('Use >=2 rounds and positive even step counts')
gate=json.loads((a.out/'validation.json').read_text())
if digest(BIN)!=gate['binary_sha256'] or source_hashes()!=gate['sources']:
    raise SystemExit('The build differs from the validation gate; rerun validation first.')
rng=random.Random(a.seed)
rows=[]
path=a.out/'events.json'
if path.exists(): raise SystemExit(f'Refusing to overwrite existing measurements: {path}')
metadata={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'binary_sha256':digest(BIN),'sources':source_hashes(),'seed':a.seed,
          'rounds':a.rounds,'warmups':5,'trials_per_process':20,'environment':environment()}
for steps in a.steps:
    for r in range(a.rounds):
        order=variants();rng.shuffle(order)
        for name in order:
            telemetry=environment()['gpu']
            text=run([BIN,name,'--bench','run',steps],a.out/f'bench-{steps}-{r}-{name}.log')
            trials=[float(s) for s in re.findall(r'TRIAL \d+ us_per_step=([\d.]+)',text)]
            if len(trials)!=20:raise RuntimeError(f'Expected 20 trials: {name}')
            resource=dict(re.findall(r'(\w+)=(\S+)',next(s for s in text.splitlines() if s.startswith('RESOURCE'))))
            rows.append({'variant':name,'steps':steps,'round':r,'order':order,
                         'trials_us_per_step':trials,'resource':resource,'gpu_before':telemetry})
            save(path,{'metadata':metadata,'complete':False,'runs':rows})
            print(f'{steps} steps round {r+1}: {name} {statistics.median(trials):.3f} us/step',flush=True)
metadata['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(path,{'metadata':metadata,'complete':True,'runs':rows})
print(path,flush=True)
