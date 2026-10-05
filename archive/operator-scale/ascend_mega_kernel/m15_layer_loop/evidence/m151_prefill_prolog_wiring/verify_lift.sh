#!/usr/bin/env bash
# M151：见证 m15_prefill_prolog.h 的 M15PM 命名空间体 = m9_gdn_prolog.asc 第 53..842 行的逐字抽取。
# 失败即退出非零（可传播失败）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
python3 - "$ROOT" <<'PY'
import sys
root = sys.argv[1]
hdr = open(root + '/m15_layer_loop/m15_prefill_prolog.h').read()
s = hdr.index('namespace M15PM {\n// ====') + len('namespace M15PM {\n')
e = hdr.index('\n}  // namespace M15PM', s) + 1
body = hdr[s:e]
src = ''.join(open(root + '/m9_gdn_prolog/m9_gdn_prolog.asc').read().splitlines(keepends=True)[52:842])
import hashlib
print('m9[53..842]  sha256', hashlib.sha256(src.encode()).hexdigest(), len(src), 'bytes')
print('hdr M15PM    sha256', hashlib.sha256(body.encode()).hexdigest(), len(body), 'bytes')
if body != src:
    print('LIFT MISMATCH'); sys.exit(1)
print('LIFT OK: M15PM == m9[53..842] verbatim')
PY
