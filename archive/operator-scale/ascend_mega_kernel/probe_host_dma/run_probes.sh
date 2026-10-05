#!/usr/bin/env bash
# probe_host_dma/run_probes.sh —— M86 一键复现：构建 → 逐 mode 独立进程运行 → 归档 evidence/
# 用法（本目录下）：bash run_probes.sh [REPS]
#   REPS 默认 5：正例（read/hbm/empty）各自独立进程重复次数（带宽类结论要「多次独立进程」）
#
# 纪律：
#   - 每个 mode 一次独立进程（故障可能污染上下文）；
#   - 跑设备前/后各记一次 npu-smi（如实记录并发背景）；
#   - 所有命令与读数落在 evidence/，可复核。

set -u
cd "$(dirname "$0")"

ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
BUILD="$ROOT/build"
REPS="${1:-5}"

mkdir -p "$LOGS"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

{
  echo "# probe_host_dma 复现记录"
  echo "date: $(date -Is)"
  echo "pwd: $ROOT"
  echo "HEAD_at_run: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null)  # 本次跑之前的分支 tip——它可能**早于**把本批证据提交进去的那个 commit"
  echo "dirty_at_run: $(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l) files"
  echo "HEAD_dirty_list:"
  git -C "$ROOT" status --porcelain 2>/dev/null | head -20 | sed 's/^/  /'
  echo "CANN: $(cat /usr/local/Ascend/cann-9.1.0/compiler/version.info 2>/dev/null | tr '\n' ' ')"
  echo "CANN_home: ${ASCEND_TOOLKIT_HOME:-unset}"
  echo "REPS: $REPS"
  echo
  echo "## 源码内容指纹（与 git tip 无关，内容寻址 ⇒ 可用它核对「本 log 由哪份源码产出」）"
  (cd "$ROOT" && sha256sum probe_host_dma.asc CMakeLists.txt run_probes.sh mmap_ngram_plan.py check_locale.sh check_fingerprints.sh 2>/dev/null)
  echo
  echo "## 跑前 npu-smi（并发背景）"
  npu-smi info 2>&1
} > "$LOGS/commands.txt"

echo "== 构建 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/build_configure.log" 2>&1
cmake --build "$BUILD" -j8 > "$LOGS/build.log" 2>&1 || { echo "构建失败，见 $LOGS/build.log"; exit 1; }

run() {  # run <logname> <args...>
  local name="$1"; shift
  echo "  -> $name : $*"
  timeout 300 "$BUILD/probe_host_dma" "$@" > "$LOGS/$name.log" 2>&1
  echo "rc=$?" >> "$LOGS/$name.log"
}

echo "== caps（能力面，无 kernel）=="
run caps caps

echo "== 空跑基线 empty × $REPS（独立进程）=="
for i in $(seq 1 "$REPS"); do run "empty_run${i}" empty --launches 20; done

echo "== HBM 对照 hbm × $REPS（独立进程）=="
for i in $(seq 1 "$REPS"); do run "hbm_run${i}" hbm --launches 20; done

echo "== 正向 read（host-mapped）× $REPS（独立进程）=="
for i in $(seq 1 "$REPS"); do run "read_run${i}" read --launches 20; done

echo "== read 尺寸扫描（每点独立进程）=="
for R in 2 4 8 16 22; do run "read_sweep_r${R}" read --tile 8192 --repeats "$R" --launches 10; done
run "read_tile2048_r16" read --tile 2048 --repeats 16 --launches 10
run "read_tile4096_r16" read --tile 4096 --repeats 16 --launches 10
run "read_tile8192_r14" read --tile 8192 --repeats 14 --launches 10

echo "== 负向对照 =="
run unreg_unregistered_ptr unreg --tile 8192 --repeats 8 --launches 3
run page_partial_registration page --tile 8192 --repeats 8 --launches 3
run align_off1 align --off 1 --repeats 8 --launches 3
run align_off32 align --off 32 --repeats 8 --launches 3
run hbm_off1_contrast hbm --off 1 --repeats 8 --launches 5

echo "== ngram 真实路径：注册来源的能力边界（file/anon/malloc）=="
run filemmap_file   filemmap --mapmode file   --tile 8192 --repeats 8 --launches 5 --path /tmp/m86_ngram_file.bin
run filemmap_filerd filemmap --mapmode filerd --tile 8192 --repeats 8 --launches 5 --path /tmp/m86_ngram_filerd.bin
run filemmap_filepriv filemmap --mapmode filepriv --tile 8192 --repeats 8 --launches 5 --path /tmp/m86_ngram_filepriv.bin
for i in 1 2 3; do run "filemmap_anon_run${i}"   filemmap --mapmode anon   --tile 8192 --repeats 8 --launches 10 --path /tmp/m86_ngram_anon.bin; done
for i in 1 2 3; do run "filemmap_malloc_run${i}" filemmap --mapmode malloc --tile 8192 --repeats 8 --launches 10 --path /tmp/m86_ngram_malloc.bin; done

echo "== 注册/注销代价（选窗口大小的输入）—— ≥5 次独立进程，报中位+区间 =="
# 可选：跑 regbench 前等一个「卡上无别的进程」的窗口，以把「自身方差」与「并发干扰」分开。
#   WAIT_IDLE=<秒>（默认 0 = 不等）。无论等没等到，都把跑前/跑后快照写进 regbench_idle_window.txt。
WAIT_IDLE="${WAIT_IDLE:-0}"
rm -f "$LOGS"/regbench*.log     # 清掉上一批（含旧的单次 regbench.log），避免陈值与新值混在同一归档里
rm -f "$EV/regbench_idle_window.txt"
{
  echo "# regbench 的并发窗口记录（$(date -Is)）"
  echo "WAIT_IDLE=${WAIT_IDLE}s；判定「空闲」= npu-smi 输出含 'No running processes found'"
} > "$EV/regbench_idle_window.txt"
if [ "$WAIT_IDLE" -gt 0 ]; then
  t_end=$(( SECONDS + WAIT_IDLE ))
  got_idle=0
  while [ "$SECONDS" -lt "$t_end" ]; do
    if npu-smi info 2>&1 | grep -q "No running processes found"; then got_idle=1; break; fi
    sleep 10
  done
  echo "  [idle] waited=$(( WAIT_IDLE - (t_end - SECONDS) ))s got_idle_window=$got_idle"
  echo "got_idle_window=$got_idle（等了 $(( WAIT_IDLE - (t_end - SECONDS) ))s）" >> "$EV/regbench_idle_window.txt"
fi
{ echo; echo "## regbench 前快照"; npu-smi info 2>&1; } >> "$EV/regbench_idle_window.txt"
for i in $(seq 1 "$REPS"); do run "regbench_run${i}" regbench; done
{ echo "## regbench 后快照"; npu-smi info 2>&1; } >> "$EV/regbench_idle_window.txt"

{
  echo "## 跑后 npu-smi（并发背景）"
  npu-smi info 2>&1
} >> "$LOGS/commands.txt"

echo "== 汇总（从各 log 抽取关键行）=="
SUMMARY="$EV/summary.txt"
: > "$SUMMARY"
for f in "$LOGS"/*_run*.log "$LOGS"/read_sweep_*.log "$LOGS"/read_tile*.log \
         "$LOGS"/unreg_*.log "$LOGS"/page_*.log "$LOGS"/align_*.log "$LOGS"/hbm_off1_contrast.log \
         "$LOGS"/filemmap_*.log "$LOGS"/regbench*.log; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"
  bl="$(grep -m1 'BANDWIDTH' "$f" | sed 's/.*host-wall: //')"
  tm="$(grep -m1 '\[TIME\]' "$f" | sed 's/.*perLaunch=\([0-9.-]*\) ns.*/\1/')"
  ae="$(grep -m1 'aclError_after_last' "$f" | sed 's/.*=//')"
  ve="$(grep -m1 '^\[VERIFY\] ' "$f" | sed 's/^\[VERIFY\] //')"
  cy="$(grep -m1 '\[CYCLE\]' "$f" | sed 's/.*aggSpan(all blocks)=//; s/ syscnt//')"
  rg="$(grep -m1 'HostRegisterV2' "$f" | sed 's/.*rc=//; s/ .*//')"
  ov="$(grep -m1 '^== done' "$f" | sed 's/== done mode=[a-z]* overall=//; s/ ==//')"
  printf '%-30s perLaunch=%-12s ns aclError=%-8s regRC=%-7s overall=%-18s cycSpan=%-10s %s | %s\n' \
     "$b" "${tm:-NA}" "${ae:-NA}" "${rg:-NA}" "${ov:-NA}" "${cy:-NA}" "$bl" "$ve" >> "$SUMMARY"
done
cat "$SUMMARY"
echo
echo "== 中位数汇总（共享卡上单次易受并发干扰；核心结论按中位 + 区间）==" | tee -a "$SUMMARY"
loop_med() {  # loop_med <name-prefix>
  local pre="$1"
  local vals
  vals=$(for f in "$LOGS/${pre}"*.log; do
           grep -m1 'perLaunch=' "$f" | sed 's/.*perLaunch=\([0-9.-]*\) ns.*/\1/'
         done | sort -n)
  local n; n=$(echo "$vals" | grep -c .)
  local m; m=$(echo "$vals" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  printf '  %-24s n=%-3s median_perLaunch=%s ns\n' "$pre" "$n" "$m" | tee -a "$SUMMARY"
}
loop_med empty_run
loop_med hbm_run
loop_med read_run
loop_med filemmap_anon_run
loop_med filemmap_malloc_run
echo "  注：上表每个 log 都可在 evidence/logs/ 复核；单点离群（共享卡）已在上方逐行列出。" | tee -a "$SUMMARY"

# ---- regbench：按 size 汇总（每次 size 的 register 毫秒数，跨进程取中位 + 区间）----
{
  echo "== regbench 汇总（register ms，跨独立进程；中位 + [min,max] + n）=="
  # 正对照：把被抽取的原始行与抽出的值并排打印——防止正则咬错字段
  # （曾踩过：`.*register rc=` 贪婪匹配到 `unregister rc=`，抽出的是**注销**耗时）
  echo "  [正对照] 原始行 -> 抽取值（须等于该行 register 字段，不是 unregister 字段）："
  grep -h "MB register" "$LOGS"/regbench_run1.log 2>/dev/null | head -3 | while read -r line; do
    reg=$(echo "$line" | sed 's/.* MB register rc=[0-9]* \([0-9.]*\) ms.*/\1/')
    unreg=$(echo "$line" | sed 's/.*unregister rc=[0-9]* \([0-9.]*\) ms.*/\1/')
    printf '    register_extracted=%-10s unregister_field=%-10s <= %s\n' "$reg" "$unreg" "$line"
  done
  for SZ in 4 64 256; do
    vals=$(grep -h "size= *${SZ} MB register" "$LOGS"/regbench*.log 2>/dev/null \
           | sed 's/.* MB register rc=[0-9]* \([0-9.]*\) ms.*/\1/' | sort -n)
    n=$(echo "$vals" | grep -c .)
    if [ "$n" -gt 0 ]; then
      med=$(echo "$vals" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
      mn=$(echo "$vals" | head -1); mx=$(echo "$vals" | tail -1)
      printf '  %3s MB register: median=%-10s ms range=[%s, %s] n=%s\n' "$SZ" "$med" "$mn" "$mx" "$n"
    else
      printf '  %3s MB register: NO-EVIDENCE\n' "$SZ"
    fi
  done
} | tee -a "$SUMMARY"
echo

echo "== 判据命令 locale 一致性 + 扫描正对照 =="
bash "$ROOT/check_locale.sh" > /dev/null 2>&1 && echo "  -> evidence/locale_check.txt"

echo "== ngram mmap 方案（真实几何 + 小几何全链路 demo）=="
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" "$ROOT/mmap_ngram_plan.py" --real > "$EV/mmap_plan_real.txt" 2>&1
echo "  -> evidence/mmap_plan_real.txt (rc=$?)"
"$PY" "$ROOT/mmap_ngram_plan.py" --demo --dir /tmp/m86_mmap_demo --shards 8 --rows 1000 \
      --window-kib 64 --windows 4 --sample 256 > "$EV/mmap_plan_demo.txt" 2>&1
echo "  -> evidence/mmap_plan_demo.txt (rc=$?)"
tail -1 "$EV/mmap_plan_demo.txt"

echo "== 源码指纹自检（commands.txt 记的指纹 vs 当前工作树）=="
bash "$ROOT/check_fingerprints.sh" > "$EV/fingerprint_check.txt" 2>&1
tail -3 "$EV/fingerprint_check.txt"

echo
echo "归档完成：$EV"
