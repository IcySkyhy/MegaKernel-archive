#!/usr/bin/env python3
"""Reuse the existing driver verbatim except for its setup/launch functions."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=(ROOT/'src/driver.cuh').read_text()
a=s.index('static const Variant *selected = nullptr;')
b=s.index('static void pair(')
out=s[:a]+'#include "launch.cuh"\n'+s[b:]
p=ROOT/'build/sweep/driver.cuh';p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists() or p.read_text()!=out:p.write_text(out)
