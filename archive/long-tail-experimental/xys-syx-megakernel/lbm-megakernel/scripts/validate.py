#!/usr/bin/env python3
"""Bitwise gates for the extracted code; optional long histories and sanitizers."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from common import BIN, ROOT, digest, environment, run, save, source_hashes, variants

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, default=ROOT / 'results/local')
p.add_argument('--full', action='store_true')
p.add_argument('--sanitizers', action='store_true')
p.add_argument('--reference', type=Path, help='Optional pre-existing upstream B0 executable')
a = p.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
# A failed rerun must never leave a stale PASS marker.
gate = a.out / 'validation.json'
gate.unlink(missing_ok=True)
run([sys.executable, ROOT / 'scripts/check_source.py'], a.out / 'source-check.log')
obstacle = a.out / 'obstacles.txt'
# Obstacles on and across all three experimental tile interfaces, including tails.
with obstacle.open('w') as f:
    for z in range(150):
        for y in range(120):
            f.write(''.join('#' if ((x in (7,8,15,16,29,30,59,60,89,90,111,112) or
                         y in (3,4,7,8,115,116) or z in (3,4,7,8,143,144,147,148))
                         and (x*3+y*5+z*7)%11==0) else '.' for x in range(120)) + '\n')
        f.write('\n')
rows = []
for name in variants():
    cases = [('sentinels', 6, ['--patterned','--sentinels','-i',obstacle])]
    if a.full:
        cases += [('default',1000,[]), ('parity',1002,[]),
                  ('patterned',1000,['--patterned']), ('obstacles',102,['-i',obstacle])]
    for label, steps, extra in cases:
        text = run([BIN,name,'--check',steps,*extra], a.out / f'check-{name}-{label}.log')
        if f'PASS physical_steps={steps}' not in text or 'full_allocation_equal=0' in text:
            raise RuntimeError(f'Missing numerical PASS: {name} {label}')
        rows.append({'variant':name,'case':label,'steps':steps,'pass':True})
    # Exercise output buffer selection after an odd number of fused launches.
    for steps in (2,6):
        target = a.out / f'velocity-{name}-{steps}.dat'
        run([BIN,name,steps,'-o',target], a.out / f'velocity-{name}-{steps}.log')
        reference = a.out / f'velocity-B0-{steps}.dat'
        if digest(target) != digest(reference):
            raise RuntimeError(f'Wrong velocity output buffer: {name} at {steps}')
    for steps in ('0','-2','3','nonnumeric'):
        r = subprocess.run([str(BIN),name,steps], capture_output=True, text=True)
        if r.returncode != 1: raise RuntimeError(f'Accepted invalid steps: {name} {steps}')
    print(f'{name}: bitwise allocations, current/inactive parity, output and CLI PASS', flush=True)
external = None
if a.reference:
    a.reference = a.reference.resolve()
    external = {'binary_sha256':digest(a.reference), 'velocity_checks':[]}
    for steps in (2,6,100):
        target=a.out/f'upstream-B0-{steps}.dat'
        run([a.reference,steps,'-o',target],a.out/f'upstream-B0-{steps}.log')
        own=a.out/f'velocity-B0-{steps}.dat'
        if not own.exists(): run([BIN,'B0',steps,'-o',own],a.out/f'velocity-B0-{steps}.log')
        if digest(target)!=digest(own): raise RuntimeError(f'Upstream B0 differs at {steps}')
        external['velocity_checks'].append({'steps':steps,'sha256':digest(own),'pass':True})
    print('Independent upstream B0 velocity comparisons PASS',flush=True)
sanitizers=[]
if a.sanitizers:
    tool=shutil.which('compute-sanitizer') or '/usr/local/cuda-12.9/bin/compute-sanitizer'
    for name in variants():
        for check in ('memcheck','racecheck','synccheck','initcheck'):
            text=run([tool,'--tool',check,'--error-exitcode','3',BIN,name,'6',
                      '--patterned','--sentinels','-i',obstacle],a.out/f'{check}-{name}.log')
            if 'ERROR SUMMARY: 0 errors' not in text and 'RACECHECK SUMMARY: 0 hazards' not in text:
                raise RuntimeError(f'Missing clean sanitizer summary: {check} {name}')
            sanitizers.append({'variant':name,'tool':check,'steps':6,'pass':True})
        print(f'{name}: all four sanitizers PASS',flush=True)
# Binary dumps and generated obstacle data are reproducible, not shipped evidence.
for f in a.out.glob('*.dat'): f.unlink()
obstacle.unlink()
save(gate,{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'binary_sha256':digest(BIN),'sources':source_hashes(),'environment':environment(),
           'full':a.full,'checks':rows,'sanitizers':sanitizers,'upstream_reference':external,
           'velocity_steps':[2,6],'invalid_counts':['0','-2','3','nonnumeric']})
print(f'All gates passed: {gate}',flush=True)
