#!/usr/bin/env bash
# ============================================================
# M178 reproduce —— kv geometry doc sync / DECODE_CTX 留档
# M184（同一证据目录）：DECODE_CTX 4096 → 4097；改前/改后设备读数见 logs/m184_*.log。
#   本脚本的 device 步（runs=kv）现直接见证本修复：decode 打印从 m15_attn_kv_host.h 的
#   DECODE_CTX 派生 ⇒ 257 页 / 8421376 B/层（断言见下方 device 段）。
#
# 本脚本一键复算 M178 的交付，任一步失败都**传播非零退出**：
#   1. build   ：用 ASC 工具链构建 m15_layer_loop（rc 必须 = 0）
#   2. static  ：README 里被改过的每个几何数字 <-> m15_attn_kv.h 常量的一致性
#                （期望值**内建在本脚本**；README 或 header 再漂移 ⇒ 脚本变红）
#   3. device  ：`runs=kv` 在 `flock -w 300 /tmp/npu0.lock` 内跑
#                进锁先打 `=== lock acquired ===`、锁内 `npu-smi` 快照、
#                逐字命令、末尾 `exit=`（都在日志里）
#   4. assert  ：设备日志含 `ALL PASS（checks=215, guards=191, fails=0）`、
#                `Kv.cap` / `Kv.align` 两行关键打印逐字匹配、
#                两条 `Kv.neg.oldgeom*` 负重对照、且无任何 `FAIL`
#
# 用法：  bash m15_layer_loop/evidence/m178_kv_geom/reproduce.sh
# 环境：  ASCEND_ENV   默认 /usr/local/Ascend/ascend-toolkit/set_env.sh
#         M178_LOGDIR  默认 <本目录>/logs（日志落盘处）
#         M178_JOBS    默认 32（cmake --build -j）
# ============================================================
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
M15="$REPO_ROOT/m15_layer_loop"
README="$M15/README.md"
KVH="$M15/m15_attn_kv.h"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
LOGDIR="${M178_LOGDIR:-$HERE/logs}"
JOBS="${M178_JOBS:-32}"
LOGF="$LOGDIR/m178_runs_kv.log"
BUILDF="$LOGDIR/m178_build.log"
CMD="./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt kv"
mkdir -p "$LOGDIR"

fails=0
ok()   { printf '[m178][ok]   %s\n' "$*"; }
fail() { printf '[m178][FAIL] %s\n' "$*" >&2; fails=$((fails + 1)); }

# 固定字符串匹配（-F）；命中=ok，未命中=FAIL
chk_has() { # <file> <fixed-string> <label>
  if grep -Fq -- "$2" "$1"; then ok "$3"; else fail "$3 :: 未命中 <$2>（$1）"; fi
}
# 固定字符串必须【不】出现（挡旧值）
chk_absent() { # <file> <fixed-string> <label>
  if grep -Fq -- "$2" "$1"; then fail "$3 :: 旧值 <$2> 仍在（$1）"; else ok "$3"; fi
}
# 正则匹配（用于对不齐的空白）
chk_re() { # <file> <ere> <label>
  if grep -Eq -- "$2" "$1"; then ok "$3"; else fail "$3 :: 未命中 /$2/（$1）"; fi
}

echo "[m178] repo root = $REPO_ROOT"
echo "[m178] log dir   = $LOGDIR"

# ------------------------------------------------------------
# 1. build
# ------------------------------------------------------------
if [ -f "$ASCEND_ENV" ]; then
  # shellcheck disable=SC1090
  . "$ASCEND_ENV"
else
  fail "找不到 ASCEND_ENV：$ASCEND_ENV"
fi
{
  echo "# cmake -B $M15/build -S $M15 -DCMAKE_BUILD_TYPE=Release"
  cmake -B "$M15/build" -S "$M15" -DCMAKE_BUILD_TYPE=Release
  echo "# cmake --build $M15/build -j$JOBS --target m15_layer_loop"
  cmake --build "$M15/build" -j"$JOBS" --target m15_layer_loop
} >"$BUILDF" 2>&1
brc=$?
if [ "$brc" -eq 0 ]; then ok "build rc=0（日志 $BUILDF）"; else fail "build rc=$brc（日志 $BUILDF）"; fi
BIN="$M15/build/m15_layer_loop"
if [ -x "$BIN" ]; then ok "binary 存在：$BIN"; else fail "binary 不存在：$BIN"; fi

# ------------------------------------------------------------
# 2. static：README <-> header 几何一致性（期望值内建）
# ------------------------------------------------------------
echo "[m178] ---- 静态一致性（header 常量 / README 文本）----"
# header 侧（m15_attn_kv.h）—— 权威
chk_has "$KVH" 'static_assert(KV_BLOCK_BYTES == 32768u' 'header: 主 KV 页 = 32,768 B'
chk_has "$KVH" 'static_assert(KV_TOKEN_STRIDE == 1024u && KV_HEAD_PLANE_BYTES == 16384u' 'header: 同 head token 步长 1,024 B / head 平面 16,384 B'
chk_has "$KVH" 'static_assert(KV_LAYER_STRIDE == 8421376u' 'header: 层 stride 8,421,376 B'
chk_has "$KVH" 'static_assert(PREFILL_BLOCKS == 257u && DECODE_BLOCKS == 257u' 'header: 257 页(prefill) / 257 页(decode ctx=4097；M184)'
chk_has "$KVH" 'constexpr uint32_t DECODE_CTX = 4097;' 'header: DECODE_CTX = 4097（M184）'
chk_has "$KVH" 'static_assert(PREFILL_COMP_ROWS == 1028u && DECODE_COMP_ROWS == 1028u' 'header: DECODE_COMP_ROWS = 1,028（M184）'
chk_has "$KVH" 'static_assert(KV_PLANE_BYTES == 101056512u, "12 × 8,421,376 = 96.38 MiB")' 'header: 主 KV 平面 101,056,512 B = 96.38 MiB'
chk_has "$KVH" '// 2,048 B/token（2 head × (K512+V512)）' 'header: 一个 token 合计 2,048 B'
# README 侧（m15_layer_loop/README.md）
chk_has "$README" '页 **32,768 B**；token **2,048 B**（2 头合计）' 'README: 页 32,768 B / token 2,048 B'
chk_has "$README" '**8,421,376 B**（257 页 × 32,768 B）' 'README: 主 KV 层 stride 8,421,376 B'
chk_has "$README" '**同一 head 内** token 步长 1,024 B' 'README: 同 head token 步长 1,024 B'
chk_has "$README" '| 32,768 / 16,384 / 1,024 | 0 |' 'README: 对齐表 32,768 / 16,384 / 1,024'
chk_has "$README" '`KvInBlockOffset(slot,0,0) + KV_TOKEN_STRIDE` = +1,024 B' 'README: naive 落点 +1,024 B'
chk_has "$README" '257 页 × 32,768 = **8,421,376 B/层**' 'README: prefill 容量 8,421,376 B/层'
# M184：decode 列已随 DECODE_CTX = 4097 同步（256 页 / 8,388,608 B → 257 页 / 8,421,376 B）
chk_has "$README" 'decode m=1 / ctx=4097' 'README: decode 列头 ctx=4097（M184）'
chk_has "$README" '主 KV **101,056,512 B（96.38 MiB）**' 'README: 12 层合计 101,056,512 B（96.38 MiB）'
chk_has "$README" '主 KV 12 层 × 8.421 MB/层 = 257 页/层' 'README: H_Alloc 打印示例 8.421 MB/层'
# M184：host 的 decode 打印从权威 DECODE_CTX 派生（不再手写字面 4096）
chk_has "$M15/m15_attn_kv_host.h" 'const uint64_t decodeCtx = DECODE_CTX;' 'host: decode 打印从 DECODE_CTX 派生（M184）'
# 旧值必须消失
for s in '4,210,688' '50,528,256' '4,194,304' '4.211' '48.19' '各 16×256' 'token 步长 512' '两个 8,192 B' '两次 512 B' '8,388,608' 'decode m=1 / ctx=4096'; do
  chk_absent "$README" "$s" "README: 旧值 <$s> 已清除"
done

# ------------------------------------------------------------
# 3. device：runs=kv under flock（进锁先 npu-smi，timeout 在锁内）
# ------------------------------------------------------------
echo "[m178] ---- 设备档 runs=kv（flock -w 300 /tmp/npu0.lock）----"
INNER="$(mktemp)"
cat >"$INNER" <<INNER_EOF
#!/usr/bin/env bash
echo "=== lock acquired ==="
echo "# verbatim (canonical): flock -w 300 /tmp/npu0.lock bash -c 'source $ASCEND_ENV && cd $REPO_ROOT && timeout 240 $CMD'"
echo "# verbatim (as run, inner script): flock -w 300 /tmp/npu0.lock bash $INNER"
echo "# binary sha256 (built just before this run):"
sha256sum "$M15/build/m15_layer_loop" 2>/dev/null || echo "  (binary missing)"
echo "# npu-smi snapshot (captured inside the lock, before the run):"
npu-smi info
source "$ASCEND_ENV"
cd "$REPO_ROOT" || { echo "exit=9"; exit 9; }
timeout 240 $CMD
echo "exit=\$?"
INNER_EOF
chmod +x "$INNER"
flock -w 300 /tmp/npu0.lock bash "$INNER" >"$LOGF" 2>&1
frc=$?
rm -f "$INNER"
if [ "$frc" -ne 0 ]; then
  fail "flock 未取得读数（rc=$frc）—— 见 $LOGF"
fi

# ------------------------------------------------------------
# 4. assert 设备日志
# ------------------------------------------------------------
chk_has "$LOGF" '=== lock acquired ===' 'device: 日志含 === lock acquired ==='
chk_has "$LOGF" '# verbatim (canonical): flock -w 300 /tmp/npu0.lock bash -c' 'device: 逐字命令'
chk_has "$LOGF" '# binary sha256' 'device: 二进制 sha256 已记录'
chk_has "$LOGF" 'npu-smi' 'device: 锁内 npu-smi 快照'
chk_has "$LOGF" 'exit=0' 'device: 二进制 exit=0'
chk_has "$LOGF" 'ALL PASS（checks=215, guards=191, fails=0）' 'device: ALL PASS tally'
chk_has "$LOGF" '主 KV   : ceil(4097/16)=257 页 × 32768 B/页 = 8421376 B/层；×12 层 = 101056512 B' 'device: Kv.cap 主 KV 逐字'
chk_has "$LOGF" 'decode(1,ctx=4097)：主 KV 257 页 = 8421376 B/层；compressed 1028 行 = 263168 B/层' 'device: Kv.cap decode 行 257 页（M184）'
chk_has "$LOGF" '主 KV 页 32768 / 页内 token 步长 1024 / head 平面 16384 / head 槽内 K‖V = 512|512' 'device: Kv.align 逐字'
chk_re "$LOGF" 'Kv\.neg\.oldgeom +PASS' 'device: Kv.neg.oldgeom（host 负重对照）'
chk_re "$LOGF" 'Kv\.neg\.oldgeom\.dev +PASS' 'device: Kv.neg.oldgeom.dev（device 负重对照）'
if grep -Fq 'FAIL' "$LOGF"; then fail "device: 日志出现 FAIL"; else ok 'device: 日志 0 FAIL'; fi

# ------------------------------------------------------------
# 5. 禁用词自查（六条绝对化措辞；按塔口径只声明"该 pattern 与这个范围"）
#    字面量用相邻字符串拼接，避免自查脚本自身命中
# ------------------------------------------------------------
echo "[m178] ---- 禁用词自查（范围：本目录 README.md / reproduce.sh / .gitignore）----"
banned=('已''全部' '无''残留' '0'' 命中' '零''命中' '绝''不' '完全''正确')
for scan in "$HERE/README.md" "$HERE/reproduce.sh" "$HERE/.gitignore"; do
  [ -f "$scan" ] || continue
  for w in "${banned[@]}"; do
    n=$(grep -Fc -- "$w" "$scan" 2>/dev/null || true)
    if [ "${n:-0}" -ne 0 ]; then fail "禁用词 <$w> 出现在 $scan（${n} 行）"; fi
  done
done
ok '禁用词自查：本目录范围内未命中'

# ------------------------------------------------------------
echo "[m178] ================= 汇总 ================="
if [ "$fails" -eq 0 ]; then
  echo "[m178] REPRODUCE OK（0 failure）"
  exit 0
fi
echo "[m178] REPRODUCE FAILED：$fails 项" >&2
exit 1
