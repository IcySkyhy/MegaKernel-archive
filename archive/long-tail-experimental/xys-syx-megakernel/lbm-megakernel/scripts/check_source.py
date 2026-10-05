#!/usr/bin/env python3
"""The temporal collision must preserve every B0 expression, not just tolerances."""
from common import ROOT

baseline = (ROOT / 'kernels/cell.cuh').read_text()
start = '\t//Test whether the cell is fluid or obstacle'
body = baseline[baseline.index(start):baseline.index('\n\t//Write the results computed above')]
body = body.replace('TEST_FLAG_SWEEP( srcGrid, OBSTACLE )', '(flags & OBSTACLE)')
body = body.replace('TEST_FLAG_SWEEP( srcGrid, ACCEL )', '(flags & ACCEL)')
candidate = (ROOT / 'kernels/collision.cuh').read_text()
actual = candidate[candidate.index(start):candidate.rindex('\n    }')]
if actual != body:
    raise SystemExit('FAIL: temporal collision expressions differ from B0')
print('PASS: collision expressions match B0; only flag access is rebound')
