#!/usr/bin/env bash
# M116（Wave B2）读数一键复跑：m24 全档 + M98 探针的零回归档（`runs=kv`）
#
# 纪律（任务书 + 塔的补充第 7 条）：
#   · 设备槽 `flock -w 900 /tmp/npu0.lock <命令>`（-w 在锁文件之前）
#   · **进锁后**再 `npu-smi` 复查；写文件前 `df -h /`
#   · 内层再套一个 `timeout` 兜底：万一被测段挂住，不会把设备槽一直占着（本仓先例：M88 探针的 rc=124）
# 用法（在仓库根目录或本目录都能跑）：
#   flock -w 1800 /tmp/npu0.lock bash m24_attn_prefill/evidence/run_evidence.sh <wt-root>
# 其中 <wt-root> = 本 worktree 的根（默认 = 脚本上两级目录）。
set -u
WT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$WT" || exit 1

echo "# M116（Wave B2）读数复跑：$(date -Is)"
echo "# worktree = $WT"
echo "--- 进锁后的复查 ---"
df -h / | tail -1
npu-smi info | sed -n '2,14p'

BIN_M24="$WT/m24_attn_prefill/build/m24_attn_prefill"
BIN_M15="$WT/m15_layer_loop/build/m15_layer_loop"
MAN="$WT/m15_layer_loop/weights_manifest.txt"

if [ -x "$BIN_M24" ]; then
    echo "=== [1/2] m24 全档（契约档 + 变异档）==="
    echo "binary sha256 = $(sha256sum "$BIN_M24" | cut -d' ' -f1)"
    timeout 900 "$BIN_M24"
    echo "m24 rc=$?"
else
    echo "[skip] 未找到 $BIN_M24（先 cmake --build m24_attn_prefill/build）"
fi

if [ -x "$BIN_M15" ]; then
    echo "=== [2/2] M98 探针零回归档：m15_attn_cache.h 改动后重跑 runs=kv ==="
    echo '[本档覆盖 M116 对 m15_attn_cache.h 的改动：AC_MAX_ROWS 8->128 / AC_MAX_GROUPS 2->32 /'
    echo ' 门控 lane 的标量落盘通路 / 主 KV 落页抽成 MainKvFillChunk(rows=1)]'
    echo "binary sha256 = $(sha256sum "$BIN_M15" | cut -d' ' -f1)"
    timeout 900 "$BIN_M15" "$MAN" kv | tail -40
    echo "m15 runs=kv rc=${PIPESTATUS[0]}"
else
    echo "[skip] 未找到 $BIN_M15"
fi
echo "# done $(date -Is)"
