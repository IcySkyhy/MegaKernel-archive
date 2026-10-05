#!/usr/bin/env bash
# [M114] 逆证：把 M114 的判定分别「退回改前 / 放宽成永真 / 放宽成永假」，核 `--selftest` **确实会咬**。
# 用法（在仓根跑，零设备）：bash docs/evidence/scan_doc_refs/check_inverse_controls.sh
# 期望：三种改法下 NC-6 都报 FAIL、整体 rc=1（改前行为 = legacy 那一档）。
set -u
for mode in legacy always_true always_false; do
  echo "===== 逆证 $mode ====="
  LC_ALL=C python3 - "$mode" <<'PY'
import importlib.util
import sys

mode = sys.argv[1]
spec = importlib.util.spec_from_file_location('s', 'docs/scan_doc_refs.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
if mode == 'legacy':              # 把 M114 的改动"退回"（= 改前判定）
    S._has_section = S._has_section_legacy
elif mode == 'always_true':       # 放宽成永真（"改宽了"的坏形态）
    S._has_section = lambda h, sec: True
else:                             # 放宽成永假（另一半的逆证）
    S._has_section = lambda h, sec: False
sys.argv = ['scan_doc_refs.py', '--selftest']
try:
    rc = S.main()
except SystemExit as e:
    rc = e.code
print('rc =', rc)
PY
done
