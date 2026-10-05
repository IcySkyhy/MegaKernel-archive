#!/usr/bin/env bash
# probe_cube_fp32/run_probes.sh —— M106 一键复现：构建 → 逐 mode 独立进程（锁内跑设备）→ 归档 evidence/ → 独立判据
#
# 用法（本目录下）：bash run_probes.sh
#
# 跑什么：prec ×1、hf32 ×1、perf ×3、perf_hf32 ×3、dump ×1（吞吐要跨进程重复，
#         README 引的"多次运行"必须能在 evidence/logs/run_perf_r*.log 里找到出处）；
#         之后跑 check_ref.py（rc 覆盖全部 7 条判据）+ 两条负向对照（d6/d3 弄坏 ⇒ rc 必须 = 1）。
#
# 纪律：
#   - 每个 mode 一个独立进程（故障不污染后续；每条有独立 rc）；
#   - 设备槽用 flock 自助排队：flock -w 900 /tmp/npu0.lock <命令>（-w 在锁文件之前）；
#     编译/分析在锁外，锁内只放设备那段；**绝不并发**；
#   - 跑前/跑后各记一次 npu-smi（如实记录并发背景）；
#   - 源码 sha256 落 commands.txt：证据与「哪份源码」内容寻址绑定（不靠 HEAD）。

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"
mkdir -p "$LOGS" "$DUMPS"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

{
  echo "# probe_cube_fp32（M106）证据生成记录"
  echo "date: $(date -Is)"
  echo "pwd: $ROOT"
  echo "HEAD_at_run: $(git rev-parse HEAD 2>/dev/null)  # 可能早于把本批证据提交进去的 commit（见 sha256）"
  echo "dirty_at_run: $(git status --porcelain 2>/dev/null | wc -l) files"
  echo "CANN: $(tr '\n' ' ' < /usr/local/Ascend/cann-9.1.0/compiler/version.info 2>/dev/null)"
  echo "ASC arch: dav-3510"
  echo
  echo "## 源码内容指纹（内容寻址 ⇒ 用它核对「本 log 由哪份源码产出」）"
  sha256sum probe_cube_fp32.asc CMakeLists.txt run_probes.sh check_ref.py 2>/dev/null
  echo
  echo "## 复现命令（本目录执行）"
  echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
  echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4"
  echo "flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 prec       # 默认（未开 HF32）"
  echo "flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 hf32       # asc_enable_hf32()"
  echo "flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 perf       # 吞吐 fp32 vs bf16"
  echo "flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 perf_hf32  # 吞吐 HF32 vs bf16"
  echo "flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 dump $DUMPS"
  echo "$PY check_ref.py $DUMPS   # 独立 numpy fp64 复核：rc 0 = d0/d1/d2/d3/d4/d5/d6 **七条判据全部成立**；1 = 有任一条不成立"
  echo "bash run_probes.sh            # 本文（含上面全部 + 两条负向对照）"
  echo
  echo "## 跑前 npu-smi（并发背景）"
  npu-smi info 2>&1
} > "$LOGS/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
  echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }
echo "== 构建（编译器：$(grep -m1 CMAKE_ASC_COMPILER "$LOGS/cmake_configure.log" | sed 's/.*: //')）=="
cmake --build "$BUILD" -j4 > "$LOGS/build.log" 2>&1
BUILD_RC=$?
echo "build rc=$BUILD_RC" >> "$LOGS/build.log"
if [ ! -f "$BUILD/probe_cube_fp32" ]; then
  echo "构建失败（rc=$BUILD_RC），逐字诊断见 $LOGS/build.log"; tail -30 "$LOGS/build.log"; exit 1
fi
echo "构建 rc=$BUILD_RC（产物 $BUILD/probe_cube_fp32 存在）"

rm -f "$DUMPS"/prec_*.bin
rm -f "$LOGS"/run_*.log "$LOGS"/negctl_*.log   # 清掉上一次的 run 日志，避免新旧命名混在归档里（M106 r1 P2-1 的教训）
rm -rf "$EV"/negctl_d6 "$EV"/negctl_d3
: > "$LOGS/matrix.txt"

runone() {   # runone <mode> <rep_idx> [extra args]
  local mode="$1"; shift
  local rep="$1"; shift
  local tag="${mode}_r${rep}"
  local log="$LOGS/run_${tag}.log"
  echo "== run $tag =="
  local attempt=0 rc=1 rcs=""
  # 设备可能被别人长时间占用：flock -w 900 排队失败就重试（设备忙不算 blocker，见 README §9）
  while [ "$attempt" -lt 6 ]; do
    attempt=$((attempt + 1))
    set +e
    timeout 600 flock -w 900 /tmp/npu0.lock "$BUILD/probe_cube_fp32" "$mode" "$@" > "$log" 2>&1
    rc=$?
    set -e
    rcs="$rcs$rc "
    [ "$rc" = "0" ] && break
    echo "  ($tag attempt $attempt rc=$rc —— 多为 flock 排队超时；重试)" >&2
    sleep 5
  done
  echo "rc=$rc" >> "$log"
  echo "attempts=$attempt rc_seq=$rcs" >> "$log"
  printf '%-16s rc=%-4s attempts=%-3s bytes=%-8s\n' "$tag" "$rc" "$attempt" "$(stat -c%s "$log")" | tee -a "$LOGS/matrix.txt"
  tail -n 24 "$log"
}

runrep() {   # runrep <mode> <reps> [extra args]
  local mode="$1"; local reps="$2"; shift 2
  local i
  for i in $(seq 1 "$reps"); do runone "$mode" "$i" "$@"; done
}

# 精度/落盘各 1 次；吞吐与比值要跨进程重复（P2-1 修复：README 引的"多次运行"必须能在归档里找到出处）
set -e
runrep prec 1
runrep hf32 1
runrep perf 3
runrep perf_hf32 3
runrep dump 1 "$DUMPS"

echo "== 独立复核（numpy fp64）=="
set +e
"$PY" check_ref.py "$DUMPS" > "$LOGS/check_ref.log" 2>&1
CR_RC=$?
set -e
echo "check_ref rc=$CR_RC" >> "$LOGS/check_ref.log"
printf '%-16s rc=%-4s\n' "check_ref" "$CR_RC" | tee -a "$LOGS/matrix.txt"
cat "$LOGS/check_ref.log"

# ---- 负向对照（判据非空洞）：把载重判别项故意弄坏，rc 必须变红（必须 = 1）----
negctl() {   # negctl <name> <python-patch>
  local name="$1"; local patch="$2"
  local nd="$EV/negctl_$name"
  mkdir -p "$nd"
  cp "$DUMPS"/prec_a.bin "$DUMPS"/prec_b.bin "$DUMPS"/prec_c.bin "$nd"/
  "$PY" -c "$patch" "$nd"
  set +e
  "$PY" check_ref.py "$nd" > "$LOGS/negctl_$name.log" 2>&1
  local rc=$?
  set -e
  echo "rc=$rc" >> "$LOGS/negctl_$name.log"
  if [ "$rc" = "1" ]; then
    printf '%-16s rc=%-4s  (期望 1：判据抓住了)\n' "negctl_$name" "$rc" | tee -a "$LOGS/matrix.txt"
  else
    printf '%-16s rc=%-4s  <== 空洞判据！载重项被弄坏却没变红\n' "negctl_$name" "$rc" | tee -a "$LOGS/matrix.txt"
  fi
  grep -m1 'FAIL' "$LOGS/negctl_$name.log" || true
  eval "NEG_${name}_RC=$rc"
  return 0
}
# ① 把 d6 的 C 全部改成 1.0（模拟"操作数被舍成 hf32/bf16 ⇒ 2^-23 位丢失"）
negctl d6 'import sys,numpy as np,os;p=os.path.join(sys.argv[1],"prec_c.bin");c=np.fromfile(p,dtype="<f4").reshape(7,16,16);c[6]=np.float32(1.0);c.tofile(p)'
# ② 把 d3 的 C 改成 2^24+255 的 fp32 舍入值（模拟"累加器变成按 8 分块精确和 / cube_k=8"）
negctl d3 'import sys,numpy as np,os;p=os.path.join(sys.argv[1],"prec_c.bin");c=np.fromfile(p,dtype="<f4").reshape(7,16,16);c[3]=np.float32(2**24+255);c.tofile(p)'
R6="${NEG_d6_RC:-NA}"; R3="${NEG_d3_RC:-NA}"

{
  echo "## 跑后 npu-smi（并发背景）"
  npu-smi info 2>&1
} >> "$LOGS/commands.txt"

echo "== 归档 sha256 =="
( cd "$EV" && find logs dumps negctl_d6 negctl_d3 -type f | sort | xargs sha256sum 2>/dev/null ) > "$EV/sha256.txt"
wc -l < "$EV/sha256.txt" | xargs -I{} echo "evidence/sha256.txt: {} files"

echo "== 汇总 =="
cat "$LOGS/matrix.txt"
echo "build_rc=$BUILD_RC check_ref_rc=$CR_RC negctl_d6_rc=$R6 negctl_d3_rc=$R3"
[ "$BUILD_RC" = "0" ] && [ "$CR_RC" = "0" ] && [ "$R6" = "1" ] && [ "$R3" = "1" ] \
  && echo "RESULT: OK" || echo "RESULT: 需人看（见上三行 rc）"
