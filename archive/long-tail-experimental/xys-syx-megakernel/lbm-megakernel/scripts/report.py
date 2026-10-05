#!/usr/bin/env python3
"""Render a compact summary from saved events; never launches GPU workloads."""
import argparse
import csv
import io
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from common import ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,default=ROOT/'results/rtx5090')
p.add_argument('--output',type=Path,help='Write a separate compact report instead of updating README')
a=p.parse_args()
a.results=a.results.resolve()
data=json.loads((a.results/'events.json').read_text())
gate=json.loads((a.results/'validation.json').read_text())
if not data['complete']:raise SystemExit('Campaign is incomplete; refusing a partial summary')
if data['metadata']['binary_sha256']!=gate['binary_sha256'] or data['metadata']['sources']!=gate['sources']:
    raise SystemExit('Event and correctness source/binary snapshots differ')
runs=data['runs']; names=runs[0]['order']
# Stable order matches the source registry, independent of shuffled execution order.
order=['B0','C1','C2','C3','S2-C128','S2-D128','S2-C256','S2-D256',
       'C-V1-128','C-V1-256','C-V2-128','C-V2-256','D-V1-128','D-V1-256',
       'D-V2-128','D-V2-256','T30-S1-128','T30-S1-256','T30-V2-128','T30-V2-256']
assert set(names)==set(order)
steps_all=sorted({r['steps'] for r in runs})
summary={}
for steps in steps_all:
    summary[steps]={}
    for name in order:
        selected=sorted([r for r in runs if r['variant']==name and r['steps']==steps],key=lambda r:r['round'])
        if len(selected)!=data['metadata']['rounds']:raise SystemExit('Incomplete round matrix')
        if [r['round'] for r in selected]!=list(range(data['metadata']['rounds'])):
            raise SystemExit('Duplicate or missing rounds')
        xs=sorted(v for r in selected for v in r['trials_us_per_step'])
        if len(xs)!=20*len(selected) or any(not math.isfinite(x) or x<=0 for x in xs):
            raise SystemExit('Invalid event trials')
        summary[steps][name]={'median_us':statistics.median(xs),'minimum_us':xs[0],
            'p90_us':xs[math.ceil(.9*len(xs))-1],'trials':len(xs),
            'round_medians_us':[statistics.median(r['trials_us_per_step']) for r in selected],
            'resource':selected[0]['resource']}
steps=max(steps_all); current=summary[steps]; baseline=current['B0']['median_us']
us=lambda name:current[name]['median_us']
best=min(order,key=us)
cube=min(['C-V2-128','C-V2-256'],key=us)
t30=min(['T30-V2-128','T30-V2-256'],key=us)
headline=(f'**Best measured: {best}, {us(best):.2f} µs/physical step '
          f'({baseline/us(best):.2f}× B0), versus B0 at {baseline:.2f} µs.** '
          f'That is {us(best)*steps/1000:.2f} ms versus {baseline*steps/1000:.2f} ms '
          f'for {steps} timesteps. The large gain comes from two-step fusion; '
          'it establishes that hand optimization helps, not that the optimum has been found.')
def table(names):
    return '\n'.join(['| Kernel | µs / physical step ↓ | Speedup over B0 ↑ |',
                      '| --- | ---: | ---: |']+
                     [f'| {n} | {us(n):.2f} | {baseline/us(n):.2f}× |' for n in names])
env=data['metadata']['environment']
gpus=list(csv.DictReader(io.StringIO(env['gpu']),skipinitialspace=True))
visible=env['CUDA_VISIBLE_DEVICES'].split(',')[0]
gpu=next((g for g in gpus if g.get('index')==visible or g.get('uuid')==visible),gpus[0])
compiler=env['build'].splitlines()[0].split(' (')[0]
cuda_match=re.search(r'release ([\d.]+)',env['nvcc'])
cuda_version=cuda_match.group(1) if cuda_match else 'see build metadata'
arch=re.search(r'--cuda-gpu-arch=(\S+)',env['build']).group(1)
protocol=(f"{gpu['name']}, driver {gpu['driver_version']}, CUDA {cuda_version}, "
          f"{compiler} (`{arch}`, `-O3 -ffast-math`), measured "
          f"{data['metadata']['started_utc'][:10]}. Default cavity, no obstacle file. "
          f"Pooled medians of {current['B0']['trials']} measured trials per variant "
          f"at **{steps} physical steps/trial**, across {data['metadata']['rounds']} "
          "seeded shuffled process rounds; each process has five warm-ups and "
          "twenty trials. Both device buffers are restored outside each timed "
          "interval. Allocation, validation, and output are excluded.")
quick=protocol+'\n\n**One-step spatial controls**\n\n'+table(order[:6])
mapping_best=min(['C1','C2','C3'],key=us)
quick+=(f'\n\nThe best C1–C3 mapping is {mapping_best}, with '
        f'{(us(mapping_best)/baseline-1)*100:+.2f}% time versus B0. '
        f'S2-C128 changes time by {(us("S2-C128")/baseline-1)*100:+.2f}%; '
        f'S2-D128 by {(us("S2-D128")/baseline-1)*100:+.2f}%.')
quick+='\n\n**Two-step fusion** (C = cube, D = elongated; suffix = workers)\n\n'
quick+=table(['D-V1-128','D-V1-256','D-V2-128','D-V2-256',cube,t30])
quick+='\n\nSpeedup = B0 time / candidate time. Values below 1 mean slower than B0. Treat sub-percent differences cautiously.\n'
lessons=(f"In this campaign, D-V2-256 takes {us('D-V2-256'):.2f} µs/step: "
         f"{us('S2-D256')/us('D-V2-256'):.2f}× faster than its same-worker "
         f"one-step control, S2-D256 ({us('S2-D256'):.2f} µs). "
         f"V2 changes time by {(us('D-V2-256')/us('D-V1-256')-1)*100:+.2f}% "
         f"versus D-V1-256; shared capacity alone does not predict the time change.\n\n"
         f"The fastest cube V2 ({cube}) takes {us(cube):.2f} µs; the fastest "
         f"T30 V2 ({t30}) takes {us(t30):.2f} µs. At 128 workers, T30 V2 "
         f"changes time by {(us('T30-V2-128')/us('D-V2-128')-1)*100:+.2f}% versus D V2; "
         f"at 256 workers, by {(us('T30-V2-256')/us('D-V2-256')-1)*100:+.2f}%. "
         "A shape that helps one worker count need not improve the best overall configuration. "
         "Small percentage differences are descriptive; two process rounds do not establish "
         "statistical significance.")
validation=(f"The shipped standalone campaign passed all **{len(order)} variants**: "
            "full-allocation bitwise comparisons, velocity output parity, and invalid-count checks. ")
if gate['full']:
    validation+='It includes default histories at 1000/1002 steps, patterned input at 1000, and external obstacles at 102. '
if len(gate['sanitizers'])==4*len(order):
    validation+='All variants also passed memcheck, racecheck, synccheck, and global-memory initcheck on six-step sentinel/obstacle input. '
if gate['upstream_reference']:
    validation+='B0 velocity output at 2/6/100 steps additionally matches the independently built original executable. '
validation+='See [the validation contract](docs/VALIDATION.md) and [saved gates](results/rtx5090/validation.json).'
full=['# Complete standalone event summary','',protocol,'',
      'These tables retain both worker counts. They use CUDA events, not profiler replay times.',
      'Round medians describe process/order variation; pooled trials within one process are not independent process replicates.','']
for nsteps in steps_all:
    full += [f'## {nsteps} physical steps per trial','',
             '| Kernel | Median µs/step | Min | P90 | Round medians | Speedup vs B0 |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    b=summary[nsteps]['B0']['median_us']
    for name in order:
        r=summary[nsteps][name]
        rounds=' / '.join(f'{x:.3f}' for x in r['round_medians_us'])
        full.append(f"| {name} | {r['median_us']:.3f} | {r['minimum_us']:.3f} | {r['p90_us']:.3f} | {rounds} | {b/r['median_us']:.3f}× |")
    full.append('')
full += ['## Runtime resource queries','',
         '| Kernel | Steps/launch | Registers/thread | Dynamic shared B | Active blocks/SM |',
         '| --- | ---: | ---: | ---: | ---: |']
for name in order:
    r=current[name]['resource']
    full.append(f"| {name} | {r['steps_per_launch']} | {r['registers']} | {r['dynamic_shared']} | {r['active_blocks_per_sm']} |")
full += ['', 'These are runtime attributes and theoretical residency, not achieved occupancy or measured spill traffic.',
         'Raw events preserve launch geometry, resources, execution order and pre-process telemetry.',
         'No profiler work ran concurrently with this serial event campaign.',
         '',f"Binary SHA-256: `{data['metadata']['binary_sha256']}`.", '']
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('# Standalone LBM results\n\n'+headline+'\n\n'+quick+'\n\n'+lessons+'\n\n'+ '\n'.join(full))
else:
    readme=(ROOT/'README.md').read_text()
    for tag,value in [('RESULT_HEADLINE',headline),('RESULTS',quick),('LESSONS',lessons),('VALIDATION',validation)]:
        pattern=f'<!-- {tag}_BEGIN -->.*?<!-- {tag}_END -->'
        readme,count=re.subn(pattern,lambda _:f'<!-- {tag}_BEGIN -->\n{value}\n<!-- {tag}_END -->',readme,flags=re.S)
        if count!=1:raise SystemExit(f'Expected one README marker: {tag}')
    (ROOT/'README.md').write_text(readme)
    (ROOT/'docs/BENCHMARKS.md').write_text('\n'.join(full))
    (a.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f'Rendered complete saved campaign: {best} {us(best):.3f} us/step, {baseline/us(best):.3f}x B0')
