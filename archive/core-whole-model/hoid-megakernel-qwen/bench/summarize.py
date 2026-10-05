"""Summarize a results directory: throughput per engine, Hoid's multiple over each, and the gate.

    python3 bench/summarize.py results/latest

Throughput is the median over fresh processes of aggregate decode tokens/s (all rows). The
correctness gate passes when, in every process, each gate row's first ten generated tokens equal
the FP32 reference exactly. Every engine is judged by the same rule.
"""
import json
from pathlib import Path
import statistics
import sys

from workload import BATCHES, GATED, PROMPT_TOKENS

ENGINES = ('hoid', 'stock', 'sglang', 'trtllm')
LABELS = {'hoid': 'Hoid', 'stock': 'vLLM', 'sglang': 'SGLang', 'trtllm': 'TRT-LLM'}


def gate(report, reference, prefix):
    """Rows of one process whose first GATED tokens differ from the reference."""
    source = report['gate_tokens'] if prefix == 'gate' else report['tokens']
    failed = []
    for index in range(report['batch']):
        name = f'gate{index}' if prefix == 'gate' else f'row{index}'
        expected = reference['rows'][f'{prefix}/{report["context"]}/row{index}']['tokens']
        if source[name][:GATED] != expected:
            failed.append(index)
    return failed


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def version(report):
    return report.get('engine_version') or report.get('vllm')


def main():
    out = Path(sys.argv[1])
    reference = json.loads((out / 'reference.json').read_text())
    reports = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))]
    engines = [e for e in ENGINES if any(r['engine'] == e for r in reports)]
    cells = []
    for context in PROMPT_TOKENS:
        for batch in BATCHES:
            arms = {engine: [r for r in reports if (r['context'], r['batch'], r['engine']) == (context, batch, engine)]
                    for engine in engines}
            if not all(arms.values()):
                continue
            cell = dict(context=context, prompt_tokens=PROMPT_TOKENS[context], batch=batch)
            for engine, runs in arms.items():
                rates = [r['tokens_per_second'] for r in runs]
                cell[engine] = dict(
                    processes=len(runs), tokens_per_second=statistics.median(rates),
                    min=min(rates), max=max(rates), spread=(max(rates) - min(rates)) / statistics.median(rates),
                    tpot_ms_p50=statistics.median(r['tpot_ms_p50'] for r in runs),
                    gate_failures={i: gate(r, reference, 'gate') for i, r in enumerate(runs) if gate(r, reference, 'gate')},
                    bench_first10_failures={i: gate(r, reference, 'bench') for i, r in enumerate(runs)
                                            if gate(r, reference, 'bench')},
                    tokens_sha256=sorted({r['tokens_sha256'] for r in runs}))
                cell[engine]['gate_passed'] = not cell[engine]['gate_failures']
            if 'hoid' in arms:
                cell['multiple'] = {e: cell['hoid']['tokens_per_second'] / cell[e]['tokens_per_second']
                                    for e in arms if e != 'hoid'}
                hoid = arms['hoid'][0]['tokens']
                cell['hoid_vs_tokens'] = {e: {row: first_divergence(hoid[row], arms[e][0]['tokens'][row]) for row in hoid}
                                          for e in arms if e != 'hoid'}
            cells.append(cell)
    first = {e: next(r for r in reports if r['engine'] == e) for e in engines}
    summary = dict(schema='qwen3-4b-vllm-hoid.summary.v2', generated_tokens_per_row=reports[0]['generated_tokens_per_row'],
                   gated_tokens=GATED, engines={e: dict(version=version(first[e]), torch=first[e]['torch']) for e in engines},
                   gpu=reports[0]['gpu'], gate_passed=all(c[e]['gate_passed'] for c in cells for e in engines),
                   cells=cells)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    others = [e for e in engines if e != 'hoid']
    names = ', '.join(f'{LABELS[e]} {summary["engines"][e]["version"]}' for e in others)
    processes = cells[0][engines[0]]['processes'] if cells else 0
    lines = [f'# Qwen3-4B-Instruct-2507 decode: Hoid megakernel vs {names}', '',
             f'{summary["gpu"]}, BF16, TP1. Every request generates {summary["generated_tokens_per_row"]} tokens '
             f'greedily; tok/s is aggregate decode throughput (all rows) over the decode window, median of '
             f'{processes} fresh processes per engine.', '']
    header = ['context', 'batch', *(f'{LABELS[e]} tok/s' for e in engines)]
    if 'hoid' in engines:
        header += [f'Hoid / {LABELS[e]}' for e in others]
    header += [*(f'{LABELS[e]} TPOT p50 ms' for e in engines), 'spread ' + ' / '.join(LABELS[e] for e in engines),
               'gate (first 10 tokens vs FP32)']
    lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * 2 + '---:|' * (len(header) - 4) + '---|---|']
    for c in cells:
        row = [c['context'], str(c['batch']), *(f'{c[e]["tokens_per_second"]:.1f}' for e in engines)]
        if 'hoid' in engines:
            row += [f'**{c["multiple"][e]:.3f}x**' for e in others]
        row += [f'{c[e]["tpot_ms_p50"]:.3f}' for e in engines]
        row.append(' / '.join(f'{100 * c[e]["spread"]:.1f}%' for e in engines))
        failing = {LABELS[e]: c[e]['gate_failures'] for e in engines if not c[e]['gate_passed']}
        row.append('PASS' if not failing else f'FAIL {failing}')
        lines.append('| ' + ' | '.join(row) + ' |')
    if 'hoid' in engines:
        lines += ['', 'Hoid vs each engine on the timed benchmark rows (first differing generated token, or "same" '
                  f'for all {summary["generated_tokens_per_row"]}):', '']
        for c in cells:
            for e in others:
                rows = ', '.join(f'{row}: {"same" if d is None else d}' for row, d in c['hoid_vs_tokens'][e].items())
                lines.append(f'- {c["context"]} b{c["batch"]} vs {LABELS[e]}: {rows}')
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
