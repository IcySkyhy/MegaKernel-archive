#!/usr/bin/env python3
"""Isolated Z2 cluster gates, serial event trials, and matched profiler evidence."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import statistics as st
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('LBM_Z2_RESULTS',ROOT/'results/z2')).resolve()
OUT.mkdir(parents=True,exist_ok=True)
ENV=dict(os.environ); ENV.setdefault('CUDA_VISIBLE_DEVICES','0')
(OUT/'tmp').mkdir(exist_ok=True);ENV['TMPDIR']=str(OUT/'tmp')
BIN=ROOT/'build/lbm-z2'
DEBUG=ROOT/'build/z2-ownership'
NAMES=['D-V2-256','D-C0-Z2-256','D-C1-Z2-256','D-V3-Z2-256']
PATTERNS={NAMES[0]:'v2<16, 8, 4>',NAMES[1]:'v2<16, 8, 4>',NAMES[2]:'z2_sync',NAMES[3]:'z2_dedup'}
COUNTER_NAMES=['producer_rank0','producer_rank1','valid_rank0','valid_rank1','local',
               'remote','fallback_local','fallback_remote','remote_rank0','remote_rank1',
               'interior_remote_rank0','interior_remote_rank1']
NSYS=os.environ.get('NSYS','/usr/local/cuda-12.9/bin/nsys')
NCU=os.environ.get('NCU','/opt/nvidia/nsight-compute/2025.2.1/ncu')
SAN=os.environ.get('SANITIZER','/usr/local/cuda-12.9/bin/compute-sanitizer')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def run(cmd,log,expected=0):
    with (OUT/log).open('w') as f:
        p=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    text=(OUT/log).read_text()
    if p.returncode!=expected or (not expected and '==ERROR==' in text):
        raise RuntimeError(f'Exit {p.returncode}: {cmd}; see {OUT/log}')
    return text

def sources():
    paths=[p for d in ('src','include','kernels') for p in sorted((ROOT/d).iterdir()) if p.is_file()]
    paths += [ROOT/'Makefile',ROOT/'experiments/z2/model.cpp']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def binaries():return {p.name:sha(p) for p in (BIN,DEBUG,ROOT/'build/z2-model')}
def require_gates():
    g=json.loads((OUT/'gates.json').read_text())
    assert g['passed'] and g['sources']==sources() and g['binaries']==binaries(),'Code differs from completed gates'
    return g

def source_gate():
    run([sys.executable,ROOT/'scripts/check_source.py'],'collision-source.log')
    original=json.loads((ROOT/'docs/provenance.json').read_text())['files']
    for key in ('kernels/cell.cuh','kernels/collision.cuh','kernels/temporal_v2.cuh',
                'include/layout_config.h','include/lbm_macros.h','include/lbm.h'):
        assert sha(ROOT/key)==original[key]['extracted_sha256'],key
    v2=(ROOT/'kernels/temporal_v2.cuh').read_text()
    control=(ROOT/'kernels/cluster_sync.cuh').read_text()
    start='    const int ox = int(blockIdx.x) * TX'
    first=v2[v2.index(start):v2.index('    __syncthreads();')]
    same=control[control.index(start):control.rindex('    cluster.sync();')]
    assert first==same,'C1 changes more than synchronization'
    phase='    for (int i = int(threadIdx.x); i < TX * TY * TZ;'
    for file in ('cluster_sync.cuh','cluster_dedup.cuh'):
        s=(ROOT/'kernels'/file).read_text().replace('        Z2_CORE(x,y,z);\n','')
        assert s[s.index(phase):].strip()==v2[v2.index(phase):].strip(),file
    for file in ('cluster_sync.cuh','cluster_dedup.cuh'):
        s=(ROOT/'kernels'/file).read_text()
        assert s.count('cluster.sync();')==2 and '__syncthreads()' not in s
    return {'collision_unchanged':True,'c1_phase1_unchanged':True,'phase2_unchanged':True,'cluster_syncs':2}

def inspect_sass():
    text=run(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',BIN],'production.sass')
    chosen={}
    for part in text.split('Function : ')[1:]:
        name=part.splitlines()[0].strip()
        key='C2' if 'z2_dedup' in name else 'C1' if 'z2_sync' in name else 'V2_C0' if '2v2ILi16ELi8ELi4E' in name else None
        if not key:continue
        assert not re.search(r'\b(?:ATOM|RED)(?:\.|\s)|__assertfail',part),name
        instructions=[s.strip() for s in part.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s)]
        barriers=[s for s in instructions if re.search(r'\b(?:BAR\.|MEMBAR|SYNCS|FENCE|CGAERRBAR|UCGABAR_)',s)]
        # This campaign targets sm_120: each cooperative cluster sync lowers to
        # one uniform arrive/wait pair plus its block-local synchronization.
        expected=0 if key=='V2_C0' else 2
        assert part.count('UCGABAR_ARV ;')==expected and part.count('UCGABAR_WAIT ;')==expected
        assert part.count('BAR.SYNC.DEFER_BLOCKING ')==(1 if key=='V2_C0' else 2)
        chosen[key]={'symbol':name,'sha256':hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),
                     'barrier_instructions':barriers,'no_atomics_or_asserts':True}
    assert set(chosen)=={'V2_C0','C1','C2'}
    save('sass.json',chosen)
    return chosen

def obstacles():
    p=OUT/'cluster-interfaces.obstacles'
    with p.open('w') as f:
        for z in range(150):
            for y in range(120):
                f.write(''.join('#' if (((z%8 in (3,4) or z in (0,149) or
                    x in (0,15,16,111,112,119) or y in (0,7,8,111,112,119)) and
                    (x+3*y+7*z)%11<3) or (x*73+y*131+z*197)%997==0) else '.' for x in range(120))+'\n')
            f.write('\n')
    return p

def gates():
    (OUT/'gates.json').unlink(missing_ok=True)
    checks=source_gate()
    model=json.loads(run([ROOT/'build/z2-model'],'ownership-cpu.json'))
    obstacle=obstacles()
    text=run([DEBUG,obstacle],'ownership-gpu.log')
    gpu=json.loads(next(s.split(' ',1)[1] for s in text.splitlines() if s.startswith('OWNERSHIP_JSON ')))
    assert gpu['passed'] and gpu['counts']==model['counts']
    ownership={'cpu':model,'gpu':gpu,'counts':dict(zip(COUNTER_NAMES,gpu['counts']))}
    save('ownership.json',ownership)
    for flag in ('--bad-threads','--bad-cluster'):
        text=run([DEBUG,flag],flag[2:]+'.log',expected=1)
        assert 'Assertion' in text and 'EXPECTED_ASSERT' in text
    print('CPU/GPU producer ownership, all shared writers, DSM counts, negative launch assertions: PASS',flush=True)
    cases=[('two',2,[]),('sentinels',6,['--patterned','--sentinels','-i',obstacle]),
           ('default1000',1000,[]),('default1002',1002,[]),('patterned1000',1000,['--patterned']),
           ('obstacle100',100,['-i',obstacle]),('obstacle102',102,['-i',obstacle])]
    reference={}
    for n in (2,6,100):
        p=OUT/f'B0-{n}.dat';run([BIN,'B0',n,'-o',p],f'B0-output-{n}.log');reference[n]=sha(p)
    rows=[];resource={}
    for name in NAMES:
        for label,n,extra in cases:
            text=run([BIN,name,'--check',n,*extra],f'check-{name}-{label}.log')
            assert f'PASS physical_steps={n}' in text
            count=2 if n==2 else 6 if n==6 else 8
            assert text.count('bit_mismatches=0 flag_byte_mismatches=0 nonfinite=0 full_allocation_equal=1')==count
            rows.append({'variant':name,'case':label,'steps':n,'pass':True})
            resource[name]={prefix:dict(re.findall(r'(\w+)=(\S+)',next(s for s in text.splitlines() if s.startswith(prefix+' ')))) for prefix in ('RESOURCE','CLUSTER')}
        for n in (2,6,100):
            p=OUT/f'{name}-{n}.dat';run([BIN,name,n,'-o',p],f'output-{name}-{n}.log')
            assert sha(p)==reference[n];p.unlink()
        for count in ('0','-2','3','abc'):run([BIN,name,count],f'invalid-{name}-{count}.log',expected=1)
        for tool in ('memcheck','racecheck','synccheck','initcheck'):
            text=run([SAN,'--tool',tool,'--error-exitcode','3',BIN,name,'6','--patterned','--sentinels','-i',obstacle],f'{tool}-{name}.log')
            assert 'ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text
            print(f'{name} {tool}: PASS',flush=True)
        print(f'{name}: all histories, buffers, velocity files and sanitizers PASS',flush=True)
    sass=inspect_sass()
    save('resources.json',resource)
    save('gates.json',{'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources':sources(),'binaries':binaries(),'source_gate':checks,'cases':rows,
        'sanitizers':['memcheck','racecheck','synccheck','initcheck'],'reference_outputs':reference,
        'obstacle_sha256':sha(obstacle),'ownership_match':True,'sass_checks':sass})
    print('ALL Z2 GATES PASS',flush=True)

def bench():
    g=require_gates()
    if (OUT/'events.json').exists():raise RuntimeError('Refusing to overwrite saved event measurements')
    rng=random.Random(20260924);rows=[]
    metadata={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':20260924,
              'warmups':5,'trials_per_process':20,'rounds':{'100':4,'1000':2},
              'sources':g['sources'],'binaries':g['binaries'],
              'build':(ROOT/'build/config.txt').read_text()}
    for steps,rounds in ((100,4),(1000,2)):
        for rnd in range(rounds):
            order=list(NAMES);rng.shuffle(order)
            for name in order:
                tag=f'bench-{steps}-{rnd}-{name}'
                telemetry=run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv'],tag+'-gpu.csv')
                text=run([BIN,name,'--bench','run',steps],tag+'.log')
                values=[float(x) for x in re.findall(r'^TRIAL \d+ us_per_step=([\d.]+)$',text,re.M)]
                assert len(values)==20
                rows.append({'name':name,'steps':steps,'round':rnd,'order':order,'trials':values})
                save('events.json',{'complete':False,'metadata':metadata,'runs':rows})
                print(f'BENCH {steps} round={rnd} {name}: {st.median(values):.3f} us/step',flush=True)
    save('events.json',{'complete':True,'metadata':metadata,'runs':rows})

def profile():
    g=require_gates()
    if (OUT/'profiling.json').exists():
        raise RuntimeError('Refusing to overwrite completed profiler measurements')
    # Retain available names and selected descriptions, not a hard-coded assumption
    # that every architecture/tool release exposes the same DSM counters.
    query=OUT/'ncu-available-metrics.txt'
    if not query.exists():
        run([NCU,'--query-metrics','--query-metrics-mode','all','--devices','0'],query.name)
    available={}
    for line in query.read_text().splitlines():
        match=re.match(r'^(\w+__[\w.]+)\s+',line)
        if match:available[match[1]]=line
    candidates={k:v for k,v in available.items() if re.search(r'cluster|distributed|dsm|remote|shared',k,re.I)}
    (OUT/'ncu-dsm-metric-candidates.txt').write_text('\n'.join(candidates.values())+'\n')
    requested=[f'l1tex__t_{kind}_pipe_lsu_mem_dshared_op_{op}.sum'
               for kind in ('requests','sectors') for op in ('ld','st')]
    requested += ['l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum',
                  'l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum']
    metrics=[key for key in requested if key in available]
    assert metrics,'Installed NCU exposes no requested DSM metrics; inspect query output'
    save('dsm-metrics.json',{'query':[NCU,'--query-metrics','--query-metrics-mode','all','--devices','0'],
         'selected':metrics,'unavailable':[k for k in requested if k not in available],
         'descriptions':{k:available[k] for k in metrics}})
    versions={}
    for name,cmd in [('nsys',[NSYS,'--version']),('ncu',[NCU,'--version']),
                     ('sanitizer',[SAN,'--version']),('gpu',['nvidia-smi','-q'])]:
        versions[name]=run(cmd,'environment-'+name+'.txt')
    traces={}
    for name in NAMES:
        tag=name+'-nsys';prefix=OUT/tag;output=OUT/(tag+'.dat')
        run([NSYS,'profile','--trace=cuda','--cuda-event-trace=false','--sample=none',
             '--cpuctxsw=none','--force-overwrite=true','-o',prefix,BIN,name,'100','-o',output],tag+'.log')
        run([NSYS,'export','--type=sqlite','--force-overwrite=true','--output',prefix.with_suffix('.sqlite'),
             prefix.with_suffix('.nsys-rep')],tag+'-export.log')
        with sqlite3.connect(prefix.with_suffix('.sqlite')) as db:
            db.row_factory=sqlite3.Row
            rows=db.execute('SELECT k.*,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
            apis=db.execute('SELECT r.*,s.value name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id').fetchall()
        assert len(rows)==50
        for r in rows:
            assert PATTERNS[name].replace(' ','') in r['name'].replace('(int)','').replace(' ',''),r['name']
            assert tuple(r['grid'+a] for a in 'XYZ')==(8,15,38)
            assert tuple(r['block'+a] for a in 'XYZ')==(256,1,1)
            assert r['staticSharedMemory']==0 and r['dynamicSharedMemory']==38912
        duration=[(r['end']-r['start'])/1000 for r in rows]
        gap=[(b['start']-a['end'])/1000 for a,b in zip(rows,rows[1:])]
        launch=[(r['end']-r['start'])/1000 for r in apis if r['name'].startswith('cudaLaunchKernel')]
        assert len(launch)==50
        assert sha(output)==g['reference_outputs']['100'];output.unlink()
        traces[name]={'kernel_count':len(rows),'kernel_name':rows[0]['name'],
            'grid':[8,15,38],'block':[256,1,1],'cluster_config':[1,1,1 if name==NAMES[0] else 2],
            'cluster_from_sqlite':{k:rows[0][k] for k in rows[0].keys() if 'cluster' in k.lower()},
            'dynamic_shared':38912,'registers':sorted({r['registersPerThread'] for r in rows}),
            'local_bytes':sorted({r['localMemoryPerThread'] for r in rows}),
            'duration_us':duration,'gap_us':gap,'launch_api_us':launch,
            'total_kernel_ms':sum(duration)/1000,'gpu_span_ms':(rows[-1]['end']-rows[0]['start'])/1e6,
            'steady_kernel_us':st.median(duration[1:]),'steady_us_per_step':st.median(duration[1:])/2,
            'output_sha256':g['reference_outputs']['100']}
        save('nsys.json',traces)
        print(f'NSYS {name}: 50 launches, geometry/resources/output PASS',flush=True)
    ncu_rows={}
    for cache in ('all','none'):
        for name in NAMES:
            tag=f'{name}-ncu-{cache}';output=OUT/(tag+'.dat')
            cmd=[NCU,'--launch-skip','10','--launch-count','1','--cache-control',cache,
                 '--clock-control','base','--replay-mode','kernel']
            for section in ('SpeedOfLight','MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables',
                            'Occupancy','LaunchStats','SchedulerStats','WarpStateStats'):
                cmd+=['--section',section]
            cmd+=['--metrics',','.join(metrics),'-f','-o',OUT/tag,BIN,name,'100','-o',output]
            run(cmd,tag+'.log')
            run([NCU,'--import',OUT/(tag+'.ncu-rep'),'--csv','--page','raw','--print-units','base'],tag+'.csv')
            run([NCU,'--import',OUT/(tag+'.ncu-rep'),'--page','details'],tag+'-details.txt')
            assert sha(output)==g['reference_outputs']['100'];output.unlink()
            with (OUT/(tag+'.csv')).open() as f:units,*values=list(csv.DictReader(f))
            assert len(values)==1
            assert PATTERNS[name].replace(' ','') in values[0]['Kernel Name'].replace('(int)','').replace(' ',''),values[0]['Kernel Name']
            for key in metrics:
                assert key in values[0] and values[0][key] not in ('','n/a'),key
            ncu_rows[tag]={'output_sha256':g['reference_outputs']['100'],'csv_sha256':sha(OUT/(tag+'.csv')),
                           'launch_metadata':{k:v for k,v in values[0].items() if k.startswith('launch__')},
                           'dsm':{k:{'unit':units[k],'value':values[0][k]} for k in metrics}}
            print(f'NCU {name} cache={cache}: counters collected, output PASS',flush=True)
    save('profiling.json',{'complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'sources':g['sources'],'binaries':g['binaries'],'versions':versions,'ncu':ncu_rows})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['gates','bench','profile','inspect'])
    a=p.parse_args()
    with (OUT/'.campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'gates':gates,'bench':bench,'profile':profile,'inspect':inspect_sass}[a.action]()
