#!/usr/bin/env python3
"""Run the guide in measured stages; no edits to the original Z2 runner/data."""
import argparse
import csv
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics as st
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(os.environ.get('LBM_SWEEP_RESULTS',ROOT/'results/cluster-sweep')).resolve()
BASE.mkdir(parents=True,exist_ok=True)
STAGES={
 'a':['D-V2-256','D-C0-Z2-Spread','D-C0-Z2-LoadBalancing'],
 'a2':['D-V2-256','D-C0-Z2-Spread','D-C0-Z2-LoadBalancing',
       'D-C1-Z2-Spread','D-C1-Z2-LoadBalancing','D-V3-Z2-Spread','D-V3-Z2-LoadBalancing'],
 'd':['D-V2-256','D-V3-Z2-LoadBalancing','D-C0-XZ4-LoadBalancing','D-C1-XZ4-LoadBalancing','D-V3-XZ4-LoadBalancing'],
 'b':['D-V2-256']+[f'D-C0-{shape}-{policy}' for shape in ('Z2','XZ4','YZ6') for policy in ('Spread','LoadBalancing')],
}
SHAPES={'Z2':(1,1,2),'XZ4':(2,1,2),'YZ6':(1,3,2)}
# Reuse the already validated event protocol and profiler machinery.
spec=importlib.util.spec_from_file_location('z2_runner',ROOT/'experiments/z2/run.py')
z2=importlib.util.module_from_spec(spec);spec.loader.exec_module(z2)
sha=z2.sha

def configure(stage):
    global OUT,NAMES,BIN
    OUT=BASE/stage;OUT.mkdir(exist_ok=True)
    NAMES=list(STAGES[stage]);BIN=ROOT/'build/sweep/lbm'
    if stage in ('d','e'):BIN=ROOT/'build/sweep/lbm-xz4'
    z2.OUT=OUT;z2.BIN=BIN;z2.NAMES=NAMES
    z2.ENV=dict(os.environ);z2.ENV.setdefault('CUDA_VISIBLE_DEVICES','0')
    (OUT/'tmp').mkdir(exist_ok=True);z2.ENV['TMPDIR']=str(OUT/'tmp')
    z2.PATTERNS={n:'z2_sync' if n.startswith('D-C1-Z2') else 'z2_dedup' if n.startswith('D-V3-Z2') else
                 'xz4_sync' if n.startswith('D-C1-XZ4') else 'xz4_dedup' if n.startswith('D-V3-XZ4') else 'v2<16, 8, 4>' for n in NAMES}
    z2.require_gates=require_gates
    z2.sources=sources
    z2.binaries=binaries

def sources():
    paths=[p for d in ('include','kernels') for p in (ROOT/d).iterdir() if p.is_file()]
    paths += [ROOT/p for p in ('src/grid.cuh','src/driver.cuh','Makefile','experiments/z2/run.py')]
    paths += [ROOT/'experiments/cluster_sweep'/p for p in ('main.cu','variants.cuh','launch.cuh','prepare_driver.py','model.cpp','Makefile')]
    paths += [ROOT/'build/sweep/driver.cuh']
    if BIN.name=='lbm-xz4':
        paths += [ROOT/'experiments/cluster_sweep'/p for p in ('xz4.cuh','xz4_debug.cuh','xz4_main.cu','xz4_ownership.cu','xz4_model.cpp','Makefile.xz4')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def binaries():
    paths=[BIN,ROOT/'build/sweep/model']
    if BIN.name=='lbm-xz4':paths += [ROOT/'build/sweep/xz4-model',ROOT/'build/sweep/xz4-ownership']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def save(name,obj):z2.save(name,obj)
def run(cmd,log,expected=0):return z2.run(cmd,log,expected)
def require_gates():
    g=json.loads((OUT/'gates.json').read_text())
    assert g['passed'] and g['sources']==sources() and g['binaries']==binaries(),'Changed validated code'
    assert g['variants']==NAMES,'Stage variants differ'
    return g

def generated_driver_gate():
    original=(ROOT/'src/driver.cuh').read_text()
    a=original.index('static const Variant *selected = nullptr;');b=original.index('static void pair(')
    assert (ROOT/'build/sweep/driver.cuh').read_text()==original[:a]+'#include "launch.cuh"\n'+original[b:]
    z2.source_gate()
    if BIN.name=='lbm-xz4':
        v2=(ROOT/'kernels/temporal_v2.cuh').read_text()
        xz=(ROOT/'experiments/cluster_sweep/xz4.cuh').read_text()
        phase='    for (int i = int(threadIdx.x); i < TX * TY * TZ;'
        final=v2[v2.index(phase):].replace('} // namespace temporal','').strip()
        for name,next_marker in [('xz4_sync','// Rank linearization'),('xz4_dedup','} // namespace temporal')]:
            body=xz[xz.index('__global__ void '+name):]
            body=body[:body.index(next_marker)]
            assert body.count('cluster.sync();')==2
            assert body[body.index(phase):].replace('        XZ_CORE(x,y,z);\n','').strip()==final
        first='    const int ox = int(blockIdx.x) * TX'
        assert xz[xz.index(first):xz.index('    cluster.sync();',xz.index(first))]==v2[v2.index(first):v2.index('    __syncthreads();')]
    return {'validation_and_event_body_unchanged':True,'collision_and_temporal_body_unchanged':True}


def obstacle_file():
    p=OUT/'interfaces.obstacles'
    with p.open('w') as f:
        for z in range(150):
            for y in range(120):
                f.write(''.join('#' if (((z%4 in (0,3) or x%16 in (0,15) or y%8 in (0,7) or
                    x in (0,119) or y in (0,119) or z in (0,149)) and (x+3*y+7*z)%11<3)
                    or (x*73+y*131+z*197)%997==0) else '.' for x in range(120))+'\n')
            f.write('\n')
    return p

def xz4_ownership(obstacle,prospective):
    model=json.loads(run([ROOT/'build/sweep/xz4-model'],'ownership-cpu.json'))
    text=run([ROOT/'build/sweep/xz4-ownership',obstacle],'ownership-gpu.log')
    gpu=json.loads(next(s.split(' ',1)[1] for s in text.splitlines() if s.startswith('OWNERSHIP_JSON ')))
    assert gpu['passed'] and gpu['counts']==model['counts']
    assert sum(model['counts'][:4])==prospective['shapes']['XZ4']['candidates']
    assert sum(model['counts'][4:8])==prospective['shapes']['XZ4']['valid']
    names=[f'producer_rank{r}' for r in range(4)]+[f'valid_rank{r}' for r in range(4)]+[
        'local','remote','fallback_local','fallback_remote','remote_x','remote_z','remote_xz']
    save('ownership.json',{'cpu':model,'gpu':gpu,'counts':dict(zip(names,model['counts']))})
    for flag in ('--bad-threads','--bad-cluster'):
        text=run([ROOT/'build/sweep/xz4-ownership',flag],flag[2:]+'.log',1)
        assert 'Assertion' in text and 'EXPECTED_ASSERT' in text
    print('XZ4 CPU/GPU producers, all writers/core cells, X/Z/XZ DSM paths and assertions: PASS',flush=True)

def xz4_sass():
    text=(OUT/'production.sass').read_text();result={};codegen={}
    for part in text.split('Function : ')[1:]:
        symbol=part.splitlines()[0].strip()
        if any(key in symbol for key in ('z2_dedup','xz4_sync','xz4_dedup')):
            assert not re.search(r'\b(?:ATOM|RED)[A-Z_.]*\b|__assertfail',part)
            opcodes=collections.Counter()
            for line in part.splitlines():
                match=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z0-9_.]+)\s',line)
                if match:opcodes[match[1].split('.')[0]]+=1
            codegen[symbol]={'static_opcode_counts':dict(opcodes),'no_atomic_family_or_assertion':True}
        if not ('xz4_sync' in symbol or 'xz4_dedup' in symbol):continue
        assert part.count('UCGABAR_ARV ;')==2 and part.count('UCGABAR_WAIT ;')==2
        instructions=[s.strip() for s in part.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s)]
        result[symbol]={'sha256':hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),'cluster_arrive_wait_pairs':2,'no_atomics_or_asserts':True}
    assert len(result)==2
    save('xz4-sass.json',result);save('codegen.json',codegen);return result

def gates():
    if (OUT/'gates.json').exists():raise RuntimeError('Use a new LBM_SWEEP_RESULTS directory; gates already saved')
    source_checks=generated_driver_gate()
    model=json.loads(run([ROOT/'build/sweep/model'],'producer-model.json'))
    obstacle=obstacle_file()
    if BIN.name=='lbm-xz4':xz4_ownership(obstacle,model)
    reference={}
    for steps in (2,6,100):
        p=OUT/f'B0-{steps}.dat';run([BIN,'B0',steps,'-o',p],f'B0-output-{steps}.log')
        reference[steps]=sha(p);p.unlink()
    cases=[('two',2,[]),('sentinels',6,['--patterned','--sentinels','-i',obstacle]),
           ('default1000',1000,[]),('default1002',1002,[]),('patterned1000',1000,['--patterned']),
           ('obstacle100',100,['-i',obstacle]),('obstacle102',102,['-i',obstacle])]
    resources={};rows=[];unsupported=[];reused={}
    prior=[]
    for path in sorted(BASE.glob('*/gates.json')):
        if path.parent==OUT:continue
        candidate=json.loads(path.read_text())
        if (candidate.get('passed') and candidate['sources']==sources() and
            candidate['binaries']==binaries() and candidate['obstacle_sha256']==sha(obstacle)):
            prior.append((path,candidate))
    for name in list(NAMES):
        match=next(((path,g) for path,g in prior if name in g['variants']),None)
        if match:
            path,g=match
            original_rows=[r for r in g['cases'] if r['variant']==name]
            assert len(original_rows)==len(cases)
            resources[name]=json.loads((path.parent/'resources.json').read_text())[name]
            rows.extend(original_rows)
            reused[name]={'gate':str(path.relative_to(BASE)),'gate_sha256':sha(path)}
            print(name+': reusing full numerical/sanitizer gates for identical binary, sources and input',flush=True)
            continue
        log=OUT/f'probe-{name}.log'
        with log.open('w') as f:p=subprocess.run([str(BIN),name,'--check','2'],cwd=ROOT,env=z2.ENV,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode==77:
            unsupported.append({'name':name,'reason':log.read_text()});print('UNSUPPORTED',name,flush=True);continue
        assert p.returncode==0,log
        for label,steps,extra in cases:
            text=run([BIN,name,'--check',steps,*extra],f'check-{name}-{label}.log')
            count=2 if steps==2 else 6 if steps==6 else 8
            assert f'PASS physical_steps={steps}' in text
            assert text.count('bit_mismatches=0 flag_byte_mismatches=0 nonfinite=0 full_allocation_equal=1')==count
            resources[name]={prefix:dict(re.findall(r'(\w+)=(\S+)',next(s for s in text.splitlines() if s.startswith(prefix+' ')))) for prefix in ('RESOURCE','CLUSTER')}
            rows.append({'variant':name,'case':label,'steps':steps,'pass':True})
        for steps in (2,6,100):
            p=OUT/f'{name}-{steps}.dat';run([BIN,name,steps,'-o',p],f'output-{name}-{steps}.log')
            assert sha(p)==reference[steps];p.unlink()
        for count in ('0','-2','3','abc'):run([BIN,name,count],f'invalid-{name}-{count}.log',1)
        for tool in ('memcheck','racecheck','synccheck','initcheck'):
            text=run([z2.SAN,'--tool',tool,'--error-exitcode','3',BIN,name,'6','--patterned','--sentinels','-i',obstacle],f'{tool}-{name}.log')
            assert 'ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text
        print(name+': all bitwise histories, output and four sanitizers PASS',flush=True)
    if unsupported:
        save('unsupported.json',unsupported)
        # Unsupported shapes remain explicit in the campaign record and are excluded
        # from timing, never represented by synthetic measurements.
        NAMES[:]=[n for n in NAMES if n not in {x['name'] for x in unsupported}]
    sass=z2.inspect_sass()
    if BIN.name=='lbm-xz4':sass.update(xz4_sass())
    old=json.loads((ROOT/'results/z2/sass.json').read_text())
    assert sass['V2_C0']['sha256']==old['V2_C0']['sha256'],'C0 device code changed'
    save('resources.json',resources)
    save('gates.json',{'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'variants':NAMES,'unsupported':unsupported,'validation_reused':reused,'sources':sources(),'binaries':binaries(),
        'source_gate':source_checks,'cases':rows,'reference_outputs':reference,'obstacle_sha256':sha(obstacle),
        'sanitizers':['memcheck','racecheck','synccheck','initcheck'],'sass_checks':sass})
    print('STAGE GATES PASS',flush=True)

def profile():
    z2.profile()
    # The reused Z2 collector's static cluster_config field is only explanatory.
    # Replace it with this campaign's requested shape and verify actual NCU metadata.
    traces=json.loads((OUT/'nsys.json').read_text());resources=json.loads((OUT/'resources.json').read_text())
    for name in NAMES:
        dims=[int(x) for x in resources[name]['CLUSTER']['dim'].split('x')]
        traces[name]['cluster_config']=dims
        traces[name]['requested_policy']=resources[name]['CLUSTER']['policy']
        for cache in ('all','none'):
            with (OUT/f'{name}-ncu-{cache}.csv').open() as f:units,row=list(csv.DictReader(f))
            assert [int(row['launch__cluster_dim_'+a]) for a in 'xyz']==dims
            if name!='D-V2-256':
                requested='Policy'+resources[name]['CLUSTER']['policy']
                assert row['launch__cluster_scheduling_policy']==requested,(name,requested,row['launch__cluster_scheduling_policy'])
            else:assert int(row['launch__cluster_size'])==0
    save('nsys.json',traces)
    print('NCU actual cluster dimensions and policies match every requested launch',flush=True)

def summary():
    j=json.loads((OUT/'events.json').read_text());assert j['complete']
    med={}
    for n in NAMES:
        med[n]={}
        for steps in (100,1000):
            rs=[r for r in j['runs'] if r['name']==n and r['steps']==steps]
            med[n][steps]={'median':st.median([x for r in rs for x in r['trials']]),
                'round_medians':[st.median(r['trials']) for r in rs]}
        print(n,med[n][1000],flush=True)
    save('summary.json',med)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES)
    p.add_argument('action',choices=('gates','bench','profile','campaign','summary'))
    args=p.parse_args();configure(args.stage)
    if args.action!='gates' and (OUT/'gates.json').exists():NAMES[:]=json.loads((OUT/'gates.json').read_text())['variants']
    with (BASE/'.campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='campaign':gates();z2.bench();summary();profile()
        else:{'gates':gates,'bench':z2.bench,'profile':profile,'summary':summary}[args.action]()

if __name__=='__main__':main()
