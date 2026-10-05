#!/usr/bin/env bash
# M195（m19 同步卫生清理）证据复算脚本。
#
#   bash m19_qsa_indexer/evidence/sync_cleanup_m195/reproduce.sh            # 离线断言
#   M195_DEVICE=1 bash m19_qsa_indexer/evidence/sync_cleanup_m195/reproduce.sh   # 先重跑设备再断言
#
# 设备纪律（塔口径，同 m15/m11 evidence 惯例）：每档 flock -w 300 /tmp/npu0.lock、
# 进锁先 npu-smi（快照落该档日志）、timeout 在锁内；等锁未取得读数即记「未取得读数」。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ASC="$ROOT/m19_qsa_indexer/m19_qsa_indexer.asc"
DK5="$ROOT/docs/05-megakernel-design.md"
RD="$ROOT/m19_qsa_indexer/README.md"
PY=/usr/local/python3.12.13/bin/python3.12
BASE="${M195_BASE:-main}"
rc=0
say(){ echo "[m195] $*"; }
chk(){ if eval "$1"; then say "OK   $2"; else say "FAIL $2"; rc=1; fi; }

# ---------- 设备档（可选） ----------
if [ "${M195_DEVICE:-0}" = "1" ]; then
  D="$HERE/dumps_rerun"; rm -rf "$D"; mkdir -p "$D"
  if flock -w 300 /tmp/npu0.lock bash -c "
      { echo '--- npu-smi (lock entry) ---'; npu-smi info | head -12;
        cd '$ROOT' && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null;
        timeout 280 env M19_OUT='$D' ./m19_qsa_indexer/build/m19_qsa_indexer;
        echo '--- exit='\$?' ---'; } > '$HERE/device_rerun_default4.log' 2>&1"; then
    say "设备重跑完成：device_rerun_default4.log"
  else
    say "未取得读数：锁未拿到"
  fi
  "$PY" "$ROOT/m19_qsa_indexer/check_ref.py" "$D" || rc=1
  "$PY" "$ROOT/m19_qsa_indexer/check_select.py" "$D" || rc=1
fi

# ---------- 离线断言 ----------
# 1) 改后事件清点：非 CrossCore 的 SetFlag/WaitFlag 只剩 VSync 的 V_S 一对
n_vs=$(grep -c 'SetFlag<HardEvent::V_S>\|WaitFlag<HardEvent::V_S>' "$ASC")
n_other=$(grep -nE 'SetFlag<|WaitFlag<' "$ASC" | grep -vE 'CrossCoreSetFlag|CrossCoreWaitFlag|HardEvent::V_S' | wc -l)
chk "[ \"$n_vs\" -eq 2 ]" "V_S 保留一对（行数=$n_vs）"
chk "[ \"$n_other\" -eq 0 ]" "无其它非 CrossCore 事件对（行数=$n_other）"
chk "grep -q 'BufAcq<PIPE_S>(AIV_BUF_CNT)' '$ASC'" "WriteToken 走 PIPE_S 阻塞释放"
chk "grep -q 'BufAcq<PIPE_S>(AIV_BUF_C2)' '$ASC'" "DumpStats 走 PIPE_S 阻塞释放"
chk "grep -q 'BufRel<PIPE_S>(AIV_BUF_CNT)' '$ASC'" "WriteToken PIPE_S release 成对"
chk "grep -q 'BufRel<PIPE_S>(AIV_BUF_C2)' '$ASC'" "DumpStats PIPE_S release 成对"

# 2) 文档裁定落库（(B) 有值依赖例外）
chk "grep -q '有值依赖例外' '$DK5'" "docs/05 §6.1 ⓔ 落有值依赖例外子条"
chk "grep -q 'm13_moe_layer.asc:1082-1083' '$DK5'" "docs/05 引 m13 允许例子"
chk "grep -q 'm17_moe_layer.asc:1267-1268' '$DK5'" "docs/05 引 m17 允许例子"
# 3) 文件头不再把 S 管事件说成套用 m0/docs06
chk "! grep -q '不挂 PIPE_S（唯一的 S 管事件是核尾 MTE3 排空' '$ASC'" "m19 文件头旧表述已改"
chk "grep -q 'docs/05 §6.1 ⓔ' '$ASC'" "m19 文件头指向 docs/05 ⓔ"

# 4) 禁用词（六词）——只扫本 mission 的 diff 新增行（§9.15 定义块不在 diff 范围）
#    六词 pattern 由 Unicode 码点转义拼出，使本脚本自身不出现这六个字面（同 §9.15 定义块的豁免精神）
fw=$(python3 -c "print('|'.join(['\u5df2\u5168\u90e8','\u65e0\u6b8b\u7559','0 \u547d\u4e2d','\u96f6\u547d\u4e2d','\u7edd\u4e0d','\u5b8c\u5168\u6b63\u786e']))")
n_fw=$(git -C "$ROOT" diff -U0 "$BASE" -- m19_qsa_indexer docs 2>/dev/null | grep '^+' | grep -vE '^\+\+\+' | grep -cE "$fw")
chk "[ \"$n_fw\" -eq 0 ]" "新增加行在六词 pattern 上未命中（$n_fw）"

# 5) 文档引用扫描读数（与基线对比，见本目录 README）
echo "--- docs/scan_doc_refs.py ---"; ( cd "$ROOT" && python3 docs/scan_doc_refs.py ) | tail -2
echo "--- docs/scan_quote_refs.py ---"; ( cd "$ROOT" && python3 docs/scan_quote_refs.py ) >/tmp/m195_sqr.txt 2>&1; echo "scan_quote_refs rc=$? ; FIND=$(grep -c FIND /tmp/m195_sqr.txt)"

# 6) m19 现有可复算守卫
bash "$ROOT/m19_qsa_indexer/check_probe_retraction.sh" || rc=1

# 7) dump_compare.txt 生成/校验脚本（离线：报告须与内置期望逐字节一致）
"$PY" "$HERE/compare_dumps.py" || rc=1

[ "$rc" -eq 0 ] && say "===== ALL PASS =====" || say "===== FAILURES PRESENT ====="
exit "$rc"
