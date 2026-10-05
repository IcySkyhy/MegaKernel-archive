#!/usr/bin/env python3
"""Summarize only completed, hash-checked cluster-sweep stages."""
import csv
import json
import math
from pathlib import Path
import statistics as st
import run as sweep

ROOT=sweep.ROOT;BASE=sweep.BASE

def table(lines,headers,rows):
    lines+=['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(map(str,r))+' |' for r in rows];lines.append('')
def p90(x):return sorted(x)[math.ceil(.9*len(x))-1]
def main():
    lines=['# Cluster profitability sweep: scheduling policy, shape, then DSMEM','',
        'Fresh stages on RTX 5090; FP32 D3Q19, fixed 120×120×150, sm_120.',
        'Each row uses 256 threads, a 16×8×4 per-block core, 38,912 B shared/block,',
        'grid 8×15×38 and two physical timesteps/launch. Explicit policy preferences',
        'are launch attributes. Ordinary D-V2 has no cluster attribute.', '',
        'The [decision sequence](../experiments/cluster_sweep/DECISIONS.md) follows',
        'the supplied guide. The [implementation notes](CLUSTER_SWEEP_DESIGN.md)',
        'describe code reuse, correctness gates and reproduction. Older Z2 timings',
        'are not pooled into these measurements.']
    if (BASE/'decision-d.json').exists():
        decision=json.loads((BASE/'decision-d.json').read_text())
        lines+=['', '**Outcome: stop this Tt=2 cluster tuning branch under the guide’s criterion.**', '',
            f"The best clustered finalist is {decision['best_cluster']}, still {100*(decision['best_cluster_ratio']-1):.2f}% slower than ordinary D-V2 in the fresh Stage D campaign.",
            'LoadBalancing materially improves C0, so the required Z2 C1/V3 rerun was',
            'completed before the shape sweep. XZ4 had acceptable C0 cost and was then',
            'implemented and fully validated, but its DSM variant is slower. Stage E',
            'pointer hoisting was not entered; deeper temporal blocking remains future work.']
        table(lines,['Stage D finalist','µs/physical step','Time / ordinary'],[
            ['D-V2-256',f"{decision['ordinary_us']:.3f}",'1.00000'],
            *[[n,f"{r['us']:.3f}",f"{r['ratio_to_ordinary']:.5f}"] for n,r in decision['finalists'].items()]])
    model=json.loads((BASE/'a/producer-model.json').read_text())
    lines+=['', '## Prospective producer work — independent flat-address model', '',
        'C0 still computes independent halos. The combined-box counts below describe',
        'a possible deduplicated implementation, not work saved by C0 itself.']
    table(lines,['Shape','Launch sweep?','Combined candidates','Candidate reduction','Valid producers','Valid reduction'],[
        [name,'yes' if name in ('Z2','XZ4','YZ6') else 'model only',f"{row['candidates']:,}",
         f"{100*(1-row['candidates']/model['old_candidates']):.2f}%",f"{row['valid']:,}",
         f"{100*(1-row['valid']/model['old_valid']):.2f}%"] for name,row in model['shapes'].items()])
    lines+=['Independent halos have 4,924,800 candidates and 4,536,000 valid first',
        'collisions per fused launch. Phase 2 remains 2,160,000 collision sites.']
    summaries={}
    for stage,label in [('a','A — Z2 scheduling policy'),('a2','A2 — existing Z2 synchronization and DSM under both policies'),
                        ('b','B — C0 shape sweep'),('d','D — selected larger DSM design'),('e','E — separate code-generation experiment')]:
        directory=BASE/stage
        if not (directory/'events.json').exists():continue
        ev=json.loads((directory/'events.json').read_text())
        if not ev['complete']:continue
        gates=json.loads((directory/'gates.json').read_text())
        assert gates['passed'] and ev['metadata']['sources']==gates['sources'] and ev['metadata']['binaries']==gates['binaries']
        for p,h in gates['sources'].items():assert sweep.sha(ROOT/p)==h,p
        for p,h in gates['binaries'].items():assert sweep.sha(ROOT/p)==h,p
        for provenance in gates.get('validation_reused',{}).values():
            assert sweep.sha(BASE/provenance['gate'])==provenance['gate_sha256']
        names=gates['variants'];resource=json.loads((directory/'resources.json').read_text())
        lines+=['',f'## Stage {label}','',f"Event campaign started {ev['metadata']['utc']}.",
            'Four seeded shuffled 100-step rounds (80 trials/variant), two 1000-step',
            'rounds (40 trials/variant), five warm-ups and twenty trials/process.',
            'Both buffers restore outside event timing. Times include launch gaps;',
            'allocation, validation, output and restores are excluded. Default LDC input.']
        med={};rounds={}
        for n in names:
            for steps in (100,1000):
                rs=[r for r in ev['runs'] if r['name']==n and r['steps']==steps]
                assert len(rs)==(4 if steps==100 else 2)
                rounds[n,steps]=[st.median(r['trials']) for r in rs]
                med[n,steps]=st.median([x for r in rs for x in r['trials']])
        summaries[stage]={n:{'us':med[n,1000],'ratio':med[n,1000]/med['D-V2-256',1000]} for n in names}
        table(lines,['Variant','Cluster','Requested policy','Registers/thread','Shared B/block','Ordinary blocks/SM','Ordinary theoretical occupancy %','Max active clusters/device'],[
            [n,resource[n]['CLUSTER']['dim'],resource[n]['CLUSTER']['policy'],resource[n]['RESOURCE']['registers'],
             resource[n]['RESOURCE']['dynamic_shared'],resource[n]['RESOURCE']['active_blocks_per_sm'],
             resource[n]['RESOURCE']['theoretical_occupancy_pct'],resource[n]['CLUSTER']['active_clusters_device']] for n in names])
        lines+=['The occupancy API capacity is device-wide. The ordinary row queries singleton',
            'clusters only for that diagnostic; its actual launch remains unclustered.']
        for steps in (100,1000):
            lines+=['',f'### {steps} steps/trial — µs per physical timestep']
            rows=[]
            for n in names:
                values=[x for r in ev['runs'] if r['name']==n and r['steps']==steps for x in r['trials']]
                rows.append([n,f'{med[n,steps]:.3f}',f'{min(values):.3f}',f'{p90(values):.3f}',
                    f'{2160000/med[n,steps]:.2f}',f'{med[n,steps]/med["D-V2-256",steps]:.5f}',
                    ' / '.join(f'{x:.3f}' for x in rounds[n,steps])])
            table(lines,['Variant','Median','Min','P90','MLUPS','Time / ordinary','Round medians'],rows)
        comparisons=[(n,'D-V2-256') for n in names if n!='D-V2-256']
        comparisons += [(n,n.replace('LoadBalancing','Spread')) for n in names if 'LoadBalancing' in n and n.replace('LoadBalancing','Spread') in names]
        for n in names:
            if '-C1-' in n and n.replace('-C1-','-C0-') in names:comparisons.append((n,n.replace('-C1-','-C0-')))
            if '-V3-' in n and n.replace('-V3-','-C1-') in names:comparisons.append((n,n.replace('-V3-','-C1-')))
        table(lines,['1000-step comparison (numerator / denominator)','Pooled ratio','Same-round ratios'],[
            [f'{a} / {b}',f'{med[a,1000]/med[b,1000]:.5f}',
             ' / '.join(f'{x/y:.5f}' for x,y in zip(rounds[a,1000],rounds[b,1000]))] for a,b in comparisons])
        lines+=['These are complete-kernel ablations, not additive phase timings. Process',
            'rounds expose drift; twenty trials in one process are not independent',
            'process replicates. Ratios and ranges are descriptive, not confidence intervals.']
        if (directory/'ownership.json').exists():
            ownership=json.loads((directory/'ownership.json').read_text())
            assert ownership['gpu']['passed'] and ownership['cpu']['counts']==ownership['gpu']['counts']
            lines+=['', '### Four-rank ownership gate', '',
                'Every physical shared (r,q) slot, Phase 2 core site and producer candidate',
                'has exactly one visit/writer. CPU Cartesian and GPU production-loop counts',
                'match; the debug launch rejects wrong worker count and cluster shape.']
            table(lines,['Quantity per fused launch','CPU = GPU count'],[
                [k,f'{v:,}'] for k,v in ownership['counts'].items()])
            lines+=['Local/remote count valid producers; fallback categories are separate.',
                'X/Z/XZ remote categories include fallbacks and sum to all remote stores.',
                'Source-level float payload and producer visits are not hardware transactions.',
                'Both XZ4 Phase 2 bodies match V2; C1 Phase 1 is unchanged. Production SASS',
                'has two cluster arrive/wait pairs and no debug ATOM/RED/assertions.']
        if gates['unsupported']:
            lines+=['','Unsupported configurations (not timed):']+[f"- {x['name']}: see its probe log." for x in gates['unsupported']]
        if not (directory/'profiling.json').exists():continue
        prof=json.loads((directory/'profiling.json').read_text());assert prof['complete']
        assert prof['binaries']==gates['binaries'] and prof['sources']==gates['sources']
        traces=json.loads((directory/'nsys.json').read_text())
        lines+=['','### NSYS and NCU diagnostics','',
            'NSYS checks 50 launches over 100 physical steps, grid/block/shared and output',
            'against B0. Cluster dimensions/policy are verified independently through NCU',
            'launch metadata. NSYS 2025.1.3 SQLite has no cluster-dimension columns.']
        table(lines,['Variant','Kernels','Kernel sum ms','GPU span ms','Steady µs/step','Median gap µs','Median launch API µs'],[
            [n,traces[n]['kernel_count'],f"{traces[n]['total_kernel_ms']:.3f}",f"{traces[n]['gpu_span_ms']:.3f}",
             f"{traces[n]['steady_us_per_step']:.3f}",f"{st.median(traces[n]['gap_us']):.3f}",f"{st.median(traces[n]['launch_api_us']):.3f}"] for n in names])
        lines+=['NCU 2025.2.1 kernel replay, base clocks, launch 11 (t=20→22), both cache',
            'policies. Additive counters/duration divide by two; ratios, resource counts',
            'and percentages do not. Event timing remains the performance authority.',
            'No-flush is a replay diagnostic. All profiler outputs match B0.']
        metrics={}
        for cache in ('all','none'):
            for n in names:
                path=directory/f'{n}-ncu-{cache}.csv'
                assert sweep.sha(path)==prof['ncu'][f'{n}-ncu-{cache}']['csv_sha256']
                with path.open() as f:units,row=list(csv.DictReader(f))
                metrics[n,cache]=row
                assert units['dram__bytes_op_read.sum']=='byte' and units['gpu__time_duration.sum']=='ns'
                expected=tuple(int(x) for x in resource[n]['CLUSTER']['dim'].split('x'))
                assert tuple(int(row['launch__cluster_dim_'+a]) for a in 'xyz')==expected
                if n!='D-V2-256':assert row['launch__cluster_scheduling_policy']=='Policy'+resource[n]['CLUSTER']['policy']
        def m(n,key,cache='all'):return float(metrics[n,cache][key].replace(',',''))
        table(lines,['Variant','NCU policy field','Flush read MB/step','Flush write MB/step','Flush total MB/step','No-flush total MB/step','Flush µs/step','No-flush µs/step'],[
            [n,metrics[n,'all']['launch__cluster_scheduling_policy'],f'{m(n,"dram__bytes_op_read.sum")/2e6:.3f}',
             f'{m(n,"dram__bytes_op_write.sum")/2e6:.3f}',f'{(m(n,"dram__bytes_op_read.sum")+m(n,"dram__bytes_op_write.sum"))/2e6:.3f}',
             f'{(m(n,"dram__bytes_op_read.sum","none")+m(n,"dram__bytes_op_write.sum","none"))/2e6:.3f}',
             f'{m(n,"gpu__time_duration.sum")/2000:.3f}',f'{m(n,"gpu__time_duration.sum","none")/2000:.3f}'] for n in names])
        fields=[('Registers/thread','launch__registers_per_thread',1),
          ('Achieved occupancy %','sm__warps_active.avg.pct_of_peak_sustained_active',1),
          ('Eligible warps/scheduler/active cycle','smsp__warps_eligible.avg.per_cycle_active',1),
          ('Issue active %','smsp__issue_active.avg.pct_of_peak_sustained_active',1),
          ('L2 hit %','lts__t_sector_hit_rate.pct',1),('L2 throughput %','lts__throughput.avg.pct_of_peak_sustained_elapsed',1),
          ('SM throughput %','sm__throughput.avg.pct_of_peak_sustained_elapsed',1)]
        for op in ('ld','st'):
            for kind in ('requests','sectors'):
                fields += [(f'Global {op} {kind}/step',f'l1tex__t_{kind}_pipe_lsu_mem_global_op_{op}.sum',2),
                           (f'Local {op} {kind}/step',f'l1tex__t_{kind}_pipe_lsu_mem_local_op_{op}.sum',2)]
            fields += [(f'Shared {op} wavefronts/step',f'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum',2),
                       (f'Shared {op} conflicts/step',f'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum',2)]
        for stall in ('long_scoreboard','short_scoreboard','barrier','lg_throttle','mio_throttle','wait'):
            fields += [(f'Stall {stall} per issue-active ratio',f'smsp__average_warps_issue_stalled_{stall}_per_issue_active.ratio',1)]
        for key in json.loads((directory/'dsm-metrics.json').read_text())['selected']:fields += [(key+' / step',key,2)]
        table(lines,['Flush-replay metric',*names],[[label,*[f'{m(n,key)/div:.3f}' for n in names]] for label,key,div in fields])
        if stage=='d':
            z='D-V3-Z2-LoadBalancing';x='D-V3-XZ4-LoadBalancing'
            lines+=['', '### What the final controlled experiment supports', '',
                f"XZ4 V3 changes time by {100*(med[x,1000]/med['D-C1-XZ4-LoadBalancing',1000]-1):+.2f}% versus its matched C1. It saves producer work but loses even within the same cluster topology and policy.",
                'Both temporal variants use 64 registers/thread, 38,912 shared bytes/block',
                'and the same ordinary two-block/SM resource bound. Local-memory traffic',
                'is zero; the results do not support a spill explanation.']
            table(lines,['Quantity per fused launch (unless noted)','Z2 V3 LB','XZ4 V3 LB'],[
                ['Valid Phase 1 collisions','3,766,500','3,543,300'],
                ['Logical remote FP32 payload MB','10.944000','13.678080'],
                ['DSM store requests',f'{m(z,"l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum"):.0f}',f'{m(x,"l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum"):.0f}'],
                ['DSM store sectors',f'{m(z,"l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum"):.0f}',f'{m(x,"l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum"):.0f}'],
                ['DSM interface write MB',f'{m(z,"l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum")/1e6:.6f}',f'{m(x,"l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum")/1e6:.6f}'],
                ['Global-load sectors/physical step',f'{m(z,"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")/2:.1f}',f'{m(x,"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")/2:.1f}'],
                ['DRAM MB/physical step',f'{(m(z,"dram__bytes_op_read.sum")+m(z,"dram__bytes_op_write.sum"))/2e6:.3f}',f'{(m(x,"dram__bytes_op_read.sum")+m(x,"dram__bytes_op_write.sum"))/2e6:.3f}']])
            lines+=['The logical remote payload grows only 25%, but measured DSM interface',
                'bytes grow about 2.6× and store requests about 4.4×. Global-load sectors',
                'fall while DRAM bytes stay nearly unchanged. This is consistent with a',
                'more expensive communication realization, not evidence that source-level',
                'halo savings were absent. Aggregate counters do not isolate X versus Z',
                'transaction costs or prove a single timing cause.', '',
                'Static SASS inspection retains 19 generic Phase 1 stores, 19 local shared',
                'loads and 19 final global stores in each V3. XZ4 has more static bit/shift',
                'instructions (LOP3 44→69, SHF 23→37), but these are static code counts,',
                'not dynamic execution or an instruction-level timing attribution. The',
                'full disassembly and `codegen.json` are retained. No pointer-hoisting',
                'variant was implemented after the measured stop condition was reached.']
        lines+=['Shared bank conflicts are access serialization, not races. Stall ratios are',
            'not time percentages. DSM interface bytes are not logical float payload.',
            'The SASS useful-bytes/sector heuristic is deliberately omitted: its mixed',
            'global/DSM kernel needs numerator/denominator verification (the earlier Z2 C2',
            'report could exceed 32 B/sector). Raw metric values remain in the CSV.']
    lines+=['','## Validation and evidence','',
        'Every timed configuration passes full-allocation bitwise comparisons against B0',
        'at 2 steps, sentinel/pattern/obstacle at 6, default 1000/1002, patterned 1000',
        'and obstacles 100/102. Current N and inactive N−2, byte flags, unused flag-slot',
        'bytes, padding/margins and invariant/Y-alias cases are preserved. Independent',
        'velocity outputs match at 2/6/100; invalid counts are rejected. Four sanitizers',
        'run six patterned/sentinel/obstacle steps. No tolerances are introduced.',
        'A previously checked configuration may reuse its complete gate evidence only',
        'when binary, compiled sources and obstacle bytes match; manifests name and hash',
        'the reused gate. Event and profiler measurements are fresh in every stage.',
        'C0 and ordinary D-V2 share the exact device function; its SASS hash matches',
        'the original Z2 campaign. Source gates prove the timing and comparison driver',
        'bodies are reused verbatim. Stage manifests bind source/binary hashes to results.','',
        'Exact prospective producer counts are saved in each stage\'s `producer-model.json`.',
        'These are scalar flat-address model counts; C0 does not perform deduplication.',
        'Raw event trials/order, pre-process telemetry, NSYS JSON and NCU CSV/details are',
        'under `results/cluster-sweep/`. Large profiler binaries are retained locally',
        'and ignored by Git. Report generation performs no GPU workloads.']
    decisions=[]
    for path in sorted(BASE.glob('decision-*.json')):
        decisions.append((path.name,json.loads(path.read_text())))
    if decisions:
        lines+=['', '## Recorded stage decisions', '']
        for filename,decision in decisions:
            lines += [f'`{filename}`:', '', '```json', json.dumps(decision,indent=2), '```', '']
    (ROOT/'docs/CLUSTER_SWEEP_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (BASE/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if 'd' in summaries:
        summary=['## 7. Scheduling policy and XZ4: measure cluster cost first', '',
            'The follow-up first compares explicit Spread and LoadBalancing, then runs',
            'C0 for Z2, XZ4 and YZ6. Every C0 uses the unchanged D-V2 device function.',
            'LoadBalancing materially reduces Z2 C0 cost; the required Z2 C1/V3 rerun',
            'still does not beat ordinary D-V2. XZ4’s C0 cost is close enough to Z2 to',
            'justify a four-block prototype, so it was implemented with 850 uniquely',
            'owned producers/block and explicit X/Z/XZ remote-write validation.', '',
            '**Fresh final campaign; all cluster rows below use LoadBalancing:**']
        table(summary,['Variant','µs/physical step','Time / ordinary D-V2'],[
            [n,f"{r['us']:.3f}",f"{r['ratio']:.3f}×"] for n,r in summaries['d'].items()])
        decision=json.loads((BASE/'decision-d.json').read_text())
        summary += [f"The best cluster still takes {100*(decision['best_cluster_ratio']-1):.2f}% longer than ordinary D-V2.",
            'XZ4 saves more producer work, but its DSM interface bytes grow about 2.6×',
            'versus Z2 for only 25% more logical remote payload. All correctness gates',
            'pass. Following the guide’s stop condition, Tt=2 cluster tuning stops here;',
            'pointer hoisting was not added. Deeper temporal blocking is a future experiment.', '',
            '[Full staged results](docs/CLUSTER_SWEEP_RESULTS.md) · [Kernels and reproduction](docs/CLUSTER_SWEEP_DESIGN.md)']
        begin='<!-- CLUSTER_SWEEP_SUMMARY_BEGIN -->';end='<!-- CLUSTER_SWEEP_SUMMARY_END -->'
        readme=ROOT/'README.md';text=readme.read_text();block=begin+'\n'+'\n'.join(summary)+'\n'+end
        if begin in text:
            a=text.index(begin);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
        else:
            anchor='## Correctness is part of the transformation';assert anchor in text
            text=text.replace(anchor,block+'\n\n'+anchor,1)
        readme.write_text(text)
    print(ROOT/'docs/CLUSTER_SWEEP_RESULTS.md')

if __name__=='__main__':main()
