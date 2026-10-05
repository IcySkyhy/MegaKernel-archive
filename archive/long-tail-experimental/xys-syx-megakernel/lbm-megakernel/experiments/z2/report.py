#!/usr/bin/env python3
"""Generate the Z2 report and its isolated README summary from saved evidence."""
import csv
import json
import math
import statistics as st
from run import ROOT, OUT, NAMES, require_gates, inspect_sass, sha


def p90(xs):return sorted(xs)[math.ceil(.9*len(xs))-1]
def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    lines.append('')

def main():
    gates=require_gates()
    events=json.loads((OUT/'events.json').read_text())
    profile=json.loads((OUT/'profiling.json').read_text())
    traces=json.loads((OUT/'nsys.json').read_text())
    resources=json.loads((OUT/'resources.json').read_text())
    ownership=json.loads((OUT/'ownership.json').read_text())
    assert events['complete'] and profile['complete']
    for data in (events['metadata'],profile):
        assert data['sources']==gates['sources'] and data['binaries']==gates['binaries']
    sass=inspect_sass()  # CPU disassembly only; no GPU workloads.
    runs={};median={}
    for steps,expected in ((100,4),(1000,2)):
        for name in NAMES:
            rs=[r for r in events['runs'] if r['steps']==steps and r['name']==name]
            assert len(rs)==expected and all(len(r['trials'])==20 for r in rs)
            runs[name,steps]=sorted(rs,key=lambda r:r['round'])
            median[name,steps]=st.median([x for r in rs for x in r['trials']])
    metrics={}
    for cache in ('all','none'):
        for name in NAMES:
            with (OUT/f'{name}-ncu-{cache}.csv').open() as f:units,*rows=list(csv.DictReader(f))
            assert len(rows)==1
            record=profile['ncu'][f'{name}-ncu-{cache}']
            assert sha(OUT/f'{name}-ncu-{cache}.csv')==record['csv_sha256']
            expected=(0,0,0) if name==NAMES[0] else (1,1,2)
            assert tuple(int(rows[0]['launch__cluster_dim_'+axis]) for axis in 'xyz')==expected
            assert int(rows[0]['launch__cluster_size'])==(0 if name==NAMES[0] else 2)
            for key in json.loads((OUT/'dsm-metrics.json').read_text())['selected']:
                value=float(rows[0][key].replace(',',''))
                if name!=NAMES[-1] or '_op_ld.' in key:assert value==0
                else:assert value>0
            metrics[name,cache]=(units,rows[0])
    def m(name,key,cache='all'):
        return float(metrics[name,cache][1][key].replace(',',''))
    def amount(name,key,kind,cache='all'):
        factors={'seconds':{'ns':1e-9,'us':1e-6,'ms':1e-3,'second':1,'s':1},
                 'bytes':{'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9},
                 'hz':{'cycle/second':1,'Hz':1,'hz':1,'KHz':1e3,'MHz':1e6,'GHz':1e9}}
        return m(name,key,cache)*factors[kind][metrics[name,cache][0][key]]
    def us(name,cache='all'):return amount(name,'gpu__time_duration.sum','seconds',cache)*1e6
    def dram(name,op,cache='all'):return amount(name,f'dram__bytes_op_{op}.sum','bytes',cache)/2e6
    def total(name,cache='all'):return dram(name,'read',cache)+dram(name,'write',cache)
    base,c0,c1,c2=NAMES
    ratio=median[c2,1000]/median[base,1000]
    changes=[(a,b,100*(median[b,1000]/median[a,1000]-1)) for a,b in zip(NAMES,NAMES[1:])]
    lines=['# D-V3-Z2-256: cluster/DSMEM experiment','',
        f"Measured {events['metadata']['utc'][:10]} (UTC) on RTX 5090 GPU 0, driver 575.57.08, CUDA 12.9.",
        'Standalone Clang 23.0.0git (`c813428fb1b5678a6e4c541f4aee82a7977e95ea`),',
        '`sm_120`, original `-O3 -ffast-math` numerical flags. All four variants were',
        'freshly measured together; older README measurements are not pooled here.','',
        f'**C2 / D-V2 time = {ratio:.5f}× ({100*(ratio-1):+.2f}%).**',
        f'D-V2 takes {median[base,1000]:.3f} µs/physical step; C2 takes {median[c2,1000]:.3f}.',
        f'C2 changes time by {100*(median[c2,1000]/median[c1,1000]-1):+.2f}% versus C1.',
        'The deduplication helps within the clustered design, but does not beat the',
        'ordinary D-V2 launch in this campaign.' if ratio>1 else
        'The complete clustered design beats the ordinary D-V2 launch in this campaign.',
        '', 'See the [implementation and ownership contract](Z2_DESIGN.md) for kernel',
        'links, the boundary argument, build requirements, and reproduction commands.',
        '', '## Implementations and runtime resources', '',
        'All variants own 16×8×4 per block, use 256 threads, grid 8×15×38,',
        '38,912 B dynamic shared/block, and two physical timesteps/launch.',
        'Static application shared and runtime local bytes/thread are zero. C0 uses',
        'the exact D-V2 device function; C1 changes synchronization; C2 adds deduplication',
        'and DSM stores. C0/C1/C2 here are unrelated to the older spatial C1/C2/C3 names.']
    table(lines,['Variant','Cluster','Reg/thread','Ordinary blocks/SM','Max active clusters/device','Potential max cluster size'],[
        [name,'ordinary (query 1×1×1)' if name==base else '1×1×2',resources[name]['RESOURCE']['registers'],
         resources[name]['RESOURCE']['active_blocks_per_sm'],resources[name]['CLUSTER']['active_clusters_device'],
         resources[name]['CLUSTER']['potential_cluster_size']] for name in NAMES])
    lines+=['Cluster occupancy is queried through `cudaOccupancyMaxActiveClusters`;',
        'the result is device-wide capacity, not measured residency or clusters/SM.',
        'The ordinary reference is queried with explicit singleton dimensions, but',
        'its actual launch has no cluster attribute. All rows use `cudaLaunchKernelExC`',
        'in this experiment; the default executable remains unchanged.','',
        '## Correctness gates completed before timing','',
        '- One fused launch equals two B0 launches across all initialized bytes.',
        '- Default histories 1000/1002, patterned 1000, external obstacles 100/102 pass for all four variants.',
        '- Current N and inactive N−2 are compared to the corresponding B0 states; short checkpoints 2/4/6 are also checked.',
        '- Nonzero sentinels, never-produced population perturbations, flags, unused flag-slot bytes, padding and margins pass bitwise.',
        '- Physical boundaries and internal interfaces z=3/4, 67/68, 147/148 are exercised by external obstacles.',
        '- Independent B0 velocity output is byte-identical at 2/6/100 steps; malformed/odd/nonpositive step counts are rejected.',
        '- Memcheck, racecheck, synccheck and global initcheck pass six-step patterned/sentinel/obstacle runs for all four variants.',
        '- Debug assertions reject a 192-thread block and a singleton cluster.',
        '- CPU/GPU ownership counts match exactly, with one visit per producer candidate/core and one writer per consumed shared slot.',
        '', 'No numerical tolerance was introduced. Every compared physical population',
        'is finite; bit mismatches, maximum absolute/RMS/relative errors, flag mismatches',
        'and full-allocation differences are zero. Global initcheck is not a shared',
        'initialization proof; writer counts, racecheck and full-state comparisons',
        'provide the additional evidence.','',
        'Production SASS has no debug ATOM/RED/assertion instructions. D-V2/C0',
        'contains one block barrier. C1 and C2 each contain two `UCGABAR_ARV` /',
        '`UCGABAR_WAIT` pairs, with their block barriers and memory ordering.',
        'Source checks preserve the collision bodies and exact V2 Phase 2. Binary',
        'and source hashes bind all timed/profiled workloads to completed gates.','',
        '## Ownership and source-level work']
    counts=ownership['counts'];cpu=ownership['cpu']
    table(lines,['Quantity per fused launch','Count'],[
        ['Original / C2 producer candidates',f"{cpu['old_candidates']:,} / {counts['producer_rank0']+counts['producer_rank1']:,}"],
        ['Original / C2 valid Phase 1 collisions',f"{cpu['old_valid']:,} / {counts['valid_rank0']+counts['valid_rank1']:,}"],
        ['Phase 2 collisions / shared writers','2,160,000 / 41,040,000'],
        *[[key,f'{counts[key]:,}'] for key in counts]])
    lines+=['Local/remote rows count valid producers; fallback rows are separate and',
        'disjoint. Rank remote totals include fallbacks. Remote stores total 2,736,000',
        'floats = 10.944 MB/fused launch = 5.472 MB/physical step. A representative',
        'full interior interface has 640 stores each way. These are logical FP32',
        'payload and source visits, not hardware sectors or executed instruction counts.','',
        '## Unprofiled CUDA-event timing','',
        'Seed 20260924. Four shuffled 100-step rounds and two shuffled 1000-step',
        'rounds; each process has five warm-ups and twenty trials. Both buffers are',
        'restored from identical snapshots outside every timed interval. Allocation,',
        'validation, restoration and output are excluded; GPU launch gaps are included.',
        'All times below are µs/physical timestep, with nearest-rank P90.',
        'MLUPS = 2,160,000 / µs_per_step. Default LDC, no external obstacles.']
    for steps in (100,1000):
        lines+=['',f'### {steps} physical steps/trial']
        rows=[]
        for name in NAMES:
            trials=[x for r in runs[name,steps] for x in r['trials']]
            rows.append([name,len(trials),f'{st.median(trials):.3f}',f'{min(trials):.3f}',f'{p90(trials):.3f}',
                         f'{2160000/st.median(trials):.2f}',' / '.join(f"{st.median(r['trials']):.3f}" for r in runs[name,steps])])
        table(lines,['Variant','Trials','Median','Minimum','P90','MLUPS','Round medians'],rows)
    lines+=['', '### Complete-kernel time ratios — below one favors the numerator']
    rows=[]
    pairs=[(c2,base),(c0,base),(c1,c0),(c2,c1)]
    for steps in (100,1000):
        for num,den in pairs:
            ratios=[st.median(a['trials'])/st.median(b['trials']) for a,b in zip(runs[num,steps],runs[den,steps])]
            rows.append([steps,f'{num} / {den}',f'{median[num,steps]/median[den,steps]:.5f}',
                         ' / '.join(f'{x:.5f}' for x in ratios)])
    table(lines,['Steps','Comparison','Pooled median ratio','Same-round median ratios'],rows)
    lines+=['Twenty trials in one process are not twenty independent process replicates.',
        'Same-round ratios compare separately shuffled processes, not simultaneous',
        'paired trials. Round ranges are descriptive, not confidence intervals.','',
        '## NSYS: actual launches, runtime geometry and gaps','',
        'NSYS 2025.1.3 CUDA tracing; no CPU sampling/context-switch capture. Every',
        'launch in each 100-physical-step run passes identity/grid/block/shared checks.',
        'All profiled velocity outputs match B0. Steady medians exclude launch one.']
    table(lines,['Variant','Launches','Reg/thread','Kernel sum ms','GPU span ms','Steady µs/kernel','Steady µs/step','Median gap µs','Median launch API µs'],[
        [n,traces[n]['kernel_count'],traces[n]['registers'][0],f"{traces[n]['total_kernel_ms']:.3f}",
         f"{traces[n]['gpu_span_ms']:.3f}",f"{traces[n]['steady_kernel_us']:.3f}",f"{traces[n]['steady_us_per_step']:.3f}",
         f"{st.median(traces[n]['gap_us']):.3f}",f"{st.median(traces[n]['launch_api_us']):.3f}"] for n in NAMES])
    lines+=['Grid/block/shared are the common configuration above. This NSYS SQLite',
        'schema does not expose cluster dimensions; those come from the runtime',
        'launch configuration and debug assertions, with NCU launch metadata retained.',
        'Host API time overlaps GPU execution and is not added to kernel duration.',
        'Trace timing is auxiliary; event medians determine speed comparisons.','',
        '## NCU: matched t=20→22, two replay cache policies','',
        'NCU 2025.2.1, kernel replay, `--clock-control base`. Each variant samples',
        'launch 11. Additive counters and duration are divided by two for physical-step',
        'reporting; resource counts, percentages, hit rates and ratios are not.',
        'CSV base units are converted explicitly. No-flush is replay sensitivity,',
        'not an unprofiled steady-state timing estimate. All outputs match B0.']
    for cache,label in [('all','Flush before replay'),('none','No flush before replay')]:
        lines+=['',f'### {label}']
        table(lines,['Variant','Read MB/step','Write MB/step','Total MB/step','µs/kernel','µs/step','Actual DRAM TB/s'],[
            [n,f'{dram(n,"read",cache):.3f}',f'{dram(n,"write",cache):.3f}',f'{total(n,cache):.3f}',
             f'{us(n,cache):.3f}',f'{us(n,cache)/2:.3f}',f'{2*total(n,cache)/us(n,cache):.3f}'] for n in NAMES])
    lines+=['', '### NCU launch metadata — verified for both replay policies']
    table(lines,['Variant','Cluster dimensions','Cluster size','Max active clusters/device','Scheduling policy'],[
        [n,'×'.join(metrics[n,'all'][1]['launch__cluster_dim_'+a] for a in 'xyz'),
         metrics[n,'all'][1]['launch__cluster_size'],metrics[n,'all'][1]['launch__cluster_max_active'],
         metrics[n,'all'][1]['launch__cluster_scheduling_policy']] for n in NAMES])
    lines+=['NCU reports zero cluster dimensions/size/capacity for the ordinary launch.',
        'The runtime occupancy diagnostic above explicitly queries singleton clusters,',
        'so its 340-cluster capacity has a different configuration. The cluster variants',
        'use the default scheduling policy; no scheduling preference was set.', '',
        'Actual bandwidth uses measured DRAM bytes and NCU duration, not the',
        'equivalent 152-B useful population convention. Event timing remains primary.','',
        '### Flush replay: memory, resources and scheduling']
    rows=[]
    fields=[('Registers/thread','launch__registers_per_thread'),
        ('Allocated registers/thread','launch__registers_per_thread_allocated'),
        ('Block limit: shared memory','launch__occupancy_limit_shared_mem'),
        ('Block limit: registers','launch__occupancy_limit_registers'),
        ('Theoretical occupancy %','sm__maximum_warps_per_active_cycle_pct'),
        ('Achieved occupancy %','sm__warps_active.avg.pct_of_peak_sustained_active'),
        ('Active warps/SM','sm__warps_active.avg.per_cycle_active'),
        ('Eligible warps/scheduler/active cycle','smsp__warps_eligible.avg.per_cycle_active'),
        ('Issue active %','smsp__issue_active.avg.pct_of_peak_sustained_active'),
        ('Global load useful B/sector','smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio'),
        ('Global store useful B/sector','smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio'),
        ('L1/TEX hit %','l1tex__t_sector_hit_rate.pct'),('L2 hit %','lts__t_sector_hit_rate.pct'),
        ('DRAM throughput %','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),
        ('L1/TEX throughput %','l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),
        ('L2 throughput %','lts__throughput.avg.pct_of_peak_sustained_elapsed'),
        ('SM throughput %','sm__throughput.avg.pct_of_peak_sustained_elapsed')]
    for label,key in fields:rows.append([label,*[f'{m(n,key):.3f}' for n in NAMES]])
    for op,word in [('ld','load'),('st','store')]:
        for interface in ('global','local'):
            for count in ('requests','sectors'):
                key=f'l1tex__t_{count}_pipe_lsu_mem_{interface}_op_{op}.sum'
                rows.append([f'{interface.capitalize()} {word} {count}/step',*[f'{m(n,key)/2:.1f}' for n in NAMES]])
        for kind,key in [('bank conflicts',f'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum'),
                         ('wavefronts',f'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum')]:
            rows.append([f'Shared {word} {kind}/step',*[f'{m(n,key)/2:.1f}' for n in NAMES]])
    for stall in ('long_scoreboard','short_scoreboard','barrier','lg_throttle','mio_throttle','wait'):
        rows.append([f'Stall {stall} per issue-active ratio',*[f'{m(n,"smsp__average_warps_issue_stalled_"+stall+"_per_issue_active.ratio"):.3f}' for n in NAMES]])
    for label,key in [('SM clock GHz','gpc__cycles_elapsed.avg.per_second'),('DRAM clock GHz','dram__cycles_elapsed.avg.per_second')]:
        rows.append([label,*[f'{amount(n,key,"hz")/1e9:.3f}' for n in NAMES]])
    table(lines,['Metric',*NAMES],rows)
    lines+=['Shared conflicts measure access serialization, not races. Local traffic is',
        'measured, not inferred from register count. Achieved occupancy uses active',
        'cycles; stall ratios are not elapsed-time percentages or an additive time budget.','',
        '### Installed DSM-specific counters — per physical step, flush replay']
    dsm=json.loads((OUT/'dsm-metrics.json').read_text())
    table(lines,['Metric (installed name)',*NAMES],[[f'`{key}`',*[f'{m(n,key)/2:.1f}' for n in NAMES]] for key in dsm['selected']])
    lines+=['D-V2, C0 and C1 have zero DSM counters. C2 has nonzero remote stores',
        'and zero remote loads in both replay policies, consistent with Phase 1-only',
        'remote writes and local Phase 2 reads. Its hardware interface reports',
        f'{m(c2,"l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum")/1e6:.6f} MB/fused launch,',
        'versus 10.944 MB of logical remote float payload; these are different quantities.', '',
        'Selected metrics were found in the installed query output; names and descriptions',
        'are retained in `dsm-metrics.json`. DSM request/sector/interface-byte counters',
        'describe hardware interfaces and must not be identified with the source-level',
        'remote float-store count. The full base-unit CSV includes both cache policies.',
        '', '## What the experiment supports','']
    for a,b,change in changes:lines.append(f'- {a} → {b}: {change:+.2f}% event time at 1000 steps.')
    lines+=[f'- C2 changes flush-replay DRAM bytes/step by {100*(total(c2)/total(base)-1):+.2f}% versus D-V2.',
        '- Fewer duplicated producer collisions can help the clustered kernel, while cluster placement and synchronization still affect the complete result.',
        '- Identical shared capacity and ordinary block occupancy do not establish identical cluster placement, scheduler behavior or communication cost.',
        '- C1→C2 changes computation, DSM addressing/communication and generated resources together; these controls are not additive phase timings.',
        '', 'The current fastest hand-written configuration remains D-V2-256 in this',
        'comparison. The result argues for a compiler profitability decision, rather',
        'than unconditional replacement of redundant halo work with cluster communication.' if ratio>1 else
        'The result establishes profitability for this specific synchronous Z2 design.',
        '', 'Scope remains one GPU/toolchain, fixed 120×120×150, even-step FP32 evolution,',
        'immutable flags and matching invariant initial slots. External obstacles extend',
        'correctness coverage; performance uses default LDC. Two long-run process rounds',
        'do not establish a broad statistical claim. No larger cluster, TMA, rolling',
        'storage, asynchronous barrier, precision or worker-count change was included.','',
        '## Reproduce and inspect','',
        'See [Z2_DESIGN.md](Z2_DESIGN.md#build-validate-measure). Run from the repo folder:',
        '', '```sh','make cluster-report  # saved evidence only',
        'LBM_Z2_RESULTS="$PWD/results/z2-local" CUDA_VISIBLE_DEVICES=0 make cluster-all','```','',
        'Evidence is in `results/z2/`: source/binary/SASS hashes, independent model and',
        'GPU ownership counts, numerical/sanitizer logs, every event trial and execution',
        'order, pre-process telemetry, NSYS timeline JSON, NCU base-unit CSV and details.',
        'Large `.nsys-rep`, `.sqlite` and `.ncu-rep` files are retained locally and ignored',
        'by Git. The report generator performs no GPU workloads.']
    target=ROOT/'docs/Z2_RESULTS.md';target.write_text('\n'.join(lines)+'\n')
    summary=['## 6. Z2 clusters: replace duplicate halo work with DSM stores','',
        'This follow-up keeps D-V2’s 16×8×4 tile, 256 workers, FP32 arithmetic and',
        '38,912 B shared/block. Two neighboring Z blocks form a 16×8×8 cluster tile.',
        'C2 assigns each first-step producer to one block and streams crossing populations',
        'through DSMEM. Two cluster barriers protect and order those writes; Phase 2 stays local.','',
        '**Fresh matched campaign; 1000-step event medians:**']
    table(summary,['Variant','Change','µs/physical step','Time / D-V2'],[
        [n,desc,f'{median[n,1000]:.3f}',f'{median[n,1000]/median[base,1000]:.3f}×']
        for n,desc in zip(NAMES,['Ordinary D-V2','C0: cluster launch only','C1: two cluster syncs','C2: deduplication + DSMEM'])])
    summary += [f'C2 removes 16.7% of candidate producer visits and improves on C1 by',
        f'{100*(1-median[c2,1000]/median[c1,1000]):.2f}%, but takes **{100*(ratio-1):.2f}% longer than D-V2**.' if ratio>1 else
        f'{100*(1-median[c2,1000]/median[c1,1000]):.2f}% and beats D-V2 by {100*(1-ratio):.2f}%.',
        'All four variants pass full-allocation bitwise, long-history and sanitizer gates.',
        'The counts prove DSM communication replaces duplicated work; the timing shows',
        'why a compiler still needs a profitability model. These are complete-kernel',
        'controls, not additive phase timings. Cluster C1/C2 names are separate from',
        'the older one-step C1/C2 variants.','',
        '[Implementation and reproduction](docs/Z2_DESIGN.md) · [Full evidence](docs/Z2_RESULTS.md)']
    begin='<!-- Z2_SUMMARY_BEGIN -->';end='<!-- Z2_SUMMARY_END -->'
    readme=ROOT/'README.md';text=readme.read_text();block=begin+'\n'+'\n'.join(summary)+'\n'+end
    if begin in text:
        a=text.index(begin);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
    else:
        anchor='## Correctness is part of the transformation';assert anchor in text
        text=text.replace(anchor,block+'\n\n'+anchor,1)
    old='Longer temporal tiles, rolling storage, clusters/DSMEM, input staging, and submission controls remain separate experiments. Reduced precision changes the numerical contract. None is implemented or credited with the reported speedup.'
    new='The Z2 cluster/DSMEM experiment above tests replacing duplicated halo work with communication. Longer temporal tiles, rolling storage, input staging, and submission controls remain separate experiments. Reduced precision changes the numerical contract. None of those follow-ups is credited with the original two-step speedup.'
    text=text.replace(old,new);readme.write_text(text)
    print(target)

if __name__=='__main__':main()
