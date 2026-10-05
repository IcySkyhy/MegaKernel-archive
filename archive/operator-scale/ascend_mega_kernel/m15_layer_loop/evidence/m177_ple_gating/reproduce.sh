#!/usr/bin/env bash
# ============================================================
# M177 —— PLE 装载门控加固：证据复算（一键）
#
# 用法：
#   bash m15_layer_loop/evidence/m177_ple_gating/reproduce.sh
#   bash m15_layer_loop/evidence/m177_ple_gating/reproduce.sh --with-negctl   # 追加负向档 ③
#
# 覆盖（失败一律以非零退出传播）：
#   ① 坏配置档（`runs=chain` + `M15_PLE_WIRE=1`，修复后，M15_LAYERS=2 缩短）：
#        期望日志出现 `[m15][WARN] H_PleArgsOf`（装载未跑 ⇒ 门控关闭）
#        **且** 该进程 `ALL PASS`、exit=0（坏配置不再把垃圾实参送进设备）。
#   ② 合法档 `runs=plewire`（`M15_PLE_WIRE=1 M15_PLE_STAGE=31`，默认 48 层）：
#        逐条把归档 `evidence/ple_wire/B_full_stage31.log` 的 `Pw.*` 行在本次日志里
#        **逐字**比对（归档 11 行 = 10 条判定项 + 1 条 `wired.delta` 读数行），另要求 main 上
#        M124 新增的 `Pw.kv.T3`/`Pw.kv.nonvac` 亦 PASS（见 README §3 的偏差说明）。
#   ③ 负向档（仅 `--with-negctl`）：把门控**临时回退**到「分配即放行」→ 重建 →
#        跑与 ① **同一条**命令，期望 exit≠0 且日志出现 `aicore error exception`
#        （旧门控下坏配置会咬）。跑完**无条件**还原源码并重建。
#
# 设备纪律（每一步）：进锁 `flock -w 300 /tmp/npu0.lock`；锁内先 `npu-smi` 快照落盘；
#                    `timeout` 在锁内；一次进锁一条命令。
# 预计耗时：默认两档约 2 分钟（含一次构建）；`--with-negctl` 约 4 分钟（多两次构建 + 一档设备）。
# ============================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
[ -f "$REPO/m15_layer_loop/m15_hc_host.h" ] || { echo "[FAIL] 推不出仓库根：$REPO"; exit 2; }
cd "$REPO" || exit 2

SRC=m15_layer_loop/m15_hc_host.h
BIN=./m15_layer_loop/build/m15_layer_loop
MAN=m15_layer_loop/weights_manifest.txt
ARCH=m15_layer_loop/evidence/ple_wire/B_full_stage31.log
OUT="$HERE"
WITH_NEGCTL=0
for a in "$@"; do
    case "$a" in
    --with-negctl) WITH_NEGCTL=1 ;;
    *) echo "[FAIL] 未知参数：$a"; exit 2 ;;
    esac
done

FAILS=0
fail() { printf '[FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }
say() { printf '[m177] %s\n' "$*"; }

# ---- 构建（修复版）----
build() {
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
    set -u
    cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release >/tmp/m177_repro_cmake.log 2>&1 || return 1
    cmake --build m15_layer_loop/build -j4 --target m15_layer_loop >/tmp/m177_repro_build.log 2>&1 || return 1
    return 0
}

# ---- 进锁跑一条命令，日志自含：锁标记 + npu-smi 快照 + 逐字命令 + 输出 + exit ----
run_locked() {   # run_locked <log> <timeout_s> <cmd...>
    local log="$1" tmo="$2"
    shift 2
    flock -w 300 /tmp/npu0.lock bash -c '
        log="$1"; tmo="$2"; shift 2
        {
            echo "=== lock acquired $(date -u +%FT%TZ) ==="
            echo "--- npu-smi info (in lock) ---"
            npu-smi info 2>&1
            echo "--- npu-smi rc=$? ---"
            echo "--- command (verbatim) ---"
            printf "%q " timeout "$tmo" "$@"; echo
            echo "--- binary output ---"
            timeout "$tmo" "$@"
            rc=$?
            echo "exit=$rc"
        } > "$log" 2>&1
        exit $rc
    ' _ "$log" "$tmo" "$@"
    return $?
}

# ---- 0. 构建 + 记录二进制 sha ----
say "构建 m15_layer_loop …"
if ! build; then
    echo "[FAIL] 构建失败（见 /tmp/m177_repro_build.log）"
    exit 2
fi
SHA_FIX="$(sha256sum "$BIN" | cut -d' ' -f1)"
say "修复版二进制 sha256 = $SHA_FIX"
{
    echo "# M177 复算记录的二进制（reproduce.sh 生成）"
    echo "binary_sha256(fixed) = $SHA_FIX"
} > "$OUT/binary_sha256.txt"
[ "$SHA_FIX" = "5f656445f06abdd795dd0c737b5ff82efe81bc0e4446d1051b892c7c54e5b407" ] ||
    say "注：二进制 sha 与归档值不同（预期 — 归档值只对 M177 tip ff916fb 的源码成立）"

# ---- ① 坏配置（修复后）----
say "① 坏配置档 runs=chain + M15_PLE_WIRE=1（修复版） …"
run_locked "$OUT/T1_bad_config_fixed.log" 240 \
    env M15_LAYERS=2 M15_PLE_WIRE=1 "$BIN" "$MAN" chain
rc=$?
grep -q '\[m15\]\[WARN\] H_PleArgsOf' "$OUT/T1_bad_config_fixed.log" ||
    fail "① 未出现 [m15][WARN] H_PleArgsOf（门控未按预期关闭）"
grep -q 'ALL PASS' "$OUT/T1_bad_config_fixed.log" ||
    fail "① 进程未 ALL PASS（坏配置本应被响亮跳过）"
[ "$rc" -eq 0 ] || fail "① exit=$rc ≠ 0"
grep -q 'pleTableDev=(nil)' "$OUT/T1_bad_config_fixed.log" ||
    fail "① WARN 未显示装载见证 pleTableDev=(nil)"

# ---- ② 合法 plewire 档 + 与归档逐字比对 ----
say "② 合法档 runs=plewire（M15_PLE_WIRE=1 M15_PLE_STAGE=31） …"
[ -f "$ARCH" ] || fail "② 缺归档 $ARCH（无法做逐字比对）"
run_locked "$OUT/T2_plewire_fixed.log" 280 \
    env M15_PLE_WIRE=1 M15_PLE_STAGE=31 "$BIN" "$MAN" plewire
rc=$?
[ "$rc" -eq 0 ] || fail "② exit=$rc ≠ 0"
grep -qE 'Pw\.emb\.T1 +PASS bad=0 miss=0/16' "$OUT/T2_plewire_fixed.log" ||
    fail "② Pw.emb.T1 未 PASS bad=0 miss=0/16"
grep -qE 'Pw\.ids\.T1 +PASS bad=0/16' "$OUT/T2_plewire_fixed.log" ||
    fail "② Pw.ids.T1 未 PASS bad=0/16"
grep -q 'wired.delta.*5636/10240' "$OUT/T2_plewire_fixed.log" ||
    fail "② Pw.wired.delta 读数与归档（5636/10240）不符"
grep -q '验证 Pw：判定项 12 条 / FAIL 0 条' "$OUT/T2_plewire_fixed.log" ||
    fail "② 验证 Pw 未达 12 条 / FAIL 0 条"
grep -qE '^\[m15\] +Pw\.kv\.T3 +PASS' "$OUT/T2_plewire_fixed.log" ||
    fail "② M124 的 Pw.kv.T3 未 PASS"
grep -qE '^\[m15\] +Pw\.kv\.nonvac +PASS' "$OUT/T2_plewire_fixed.log" ||
    fail "② M124 的 Pw.kv.nonvac 未 PASS"
# 归档的 `Pw.*` 行逐字复现（10 条判定项 + 1 条 wired.delta 读数行 = 11 行）
n_arch=0
while IFS= read -r line; do
    n_arch=$((n_arch + 1))
    grep -Fqx -- "$line" "$OUT/T2_plewire_fixed.log" ||
        fail "② 归档行未逐字复现：$line"
done < <(grep -E '^\[m15\] +Pw\.' "$ARCH")
[ "$n_arch" -eq 11 ] || fail "② 归档 Pw.* 行数 = $n_arch（期望 11 = 10 判定项 + 1 读数行）"
say "② 归档 $n_arch 行 Pw.* 逐字比对完成"

# ---- ③ 负向档（临时回退门控）----
if [ "$WITH_NEGCTL" -eq 1 ]; then
    say "③ 负向档：临时把门控回退到「分配即放行」→ 重建 → 跑同一条坏配置 …"
    BAK="$(mktemp /tmp/m177_hc_host.XXXXXX.h)"
    cp "$SRC" "$BAK"
    restore_src() { cp "$BAK" "$SRC" && rm -f "$BAK"; }
    trap 'restore_src' EXIT
    set +u
    python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = "    if (C.pleWDev == nullptr || C.pleTableDev == nullptr) {"
new = "    if (C.pleWDev == nullptr) {   // M177 negctl：临时回退（reproduce.sh 还原）"
if s.count(old) != 1:
    raise SystemExit("patch anchor count=%d (expect 1)" % s.count(old))
open(p, "w", encoding="utf-8").write(s.replace(old, new))
PY
    prc=$?
    set -u
    [ "$prc" -eq 0 ] || fail "③ 回退 patch 失败（rc=$prc）"
    if [ "$prc" -eq 0 ]; then
        build || fail "③ 回退版构建失败"
        run_locked "$OUT/T3_negctl_oldgate.log" 240 \
            env M15_LAYERS=2 M15_PLE_WIRE=1 "$BIN" "$MAN" chain
        rc=$?
        [ "$rc" -ne 0 ] || fail "③ 旧门控下坏配置竟然 exit=0（负向档失效）"
        grep -q 'aicore error exception' "$OUT/T3_negctl_oldgate.log" ||
            fail "③ 未见 aicore exception（旧门控咬的证据）"
        grep -q 'sync after ch_layer failed (err 507015)' "$OUT/T3_negctl_oldgate.log" ||
            fail "③ 未见 sync after ch_layer failed (err 507015)"
        printf 'binary_sha256(negctl) = %s\n' "$(sha256sum "$BIN" | cut -d' ' -f1)" >> "$OUT/binary_sha256.txt"
    fi
    restore_src
    trap - EXIT
    say "③ 源码已还原，重建修复版 …"
    build || fail "③ 还原后重建失败"
    SHA_FIX2="$(sha256sum "$BIN" | cut -d' ' -f1)"
    say "还原后修复版 sha256 = $SHA_FIX2"
    [ "$SHA_FIX2" = "$SHA_FIX" ] || fail "③ 还原后二进制与修复版不一致（$SHA_FIX2 ≠ $SHA_FIX）"
fi

say "----------------------------------------"
if [ "$FAILS" -eq 0 ]; then
    say "全部复算通过（日志见 $OUT/）"
    exit 0
fi
say "复算失败 $FAILS 条"
exit 1
