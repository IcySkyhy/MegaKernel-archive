#!/usr/bin/env bash
# probe_sync_quirks/run_probes.sh —— 一条命令重跑全部探针并归档原始证据
#
# 用法（本目录下）：bash run_probes.sh [reps]
#   reps  探针 B 每个变体的重复次数（默认 50；3510 上"标量->向量"通路有间歇丢写，
#         重复多次才能区分"编译不过"与"编译过但偶发取值错"）
#
# 产物（全部进 git，供 review 追溯）：
#   evidence/logs/build_<target>.log    每个 target 完整构建输出（含 stderr 原文）
#   evidence/logs/run_<target>.log      每个 target 完整运行输出
#   evidence/logs/compile_matrix.txt    编译通过/失败矩阵（逐 target 一行）
#   evidence/logs/run_matrix.txt        运行结果矩阵（逐 target 一行）
#   evidence/dumps/                     probe A/A2 的 UB 快照 .bin、probe B 的输出 .bin
#   evidence/dumps_txt/                 probe A 逐元素（value/index 分列）文本表
#   evidence/decode_A_summary.txt       probe A 判读摘要（独立复算）
#   evidence/commands.txt               本次运行用到的确切命令与版本信息

set -u
cd "$(dirname "$0")"

REPS="${1:-50}"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
DUMP_TXT="$EV/dumps_txt"
BUILD="$ROOT/build"

mkdir -p "$LOGS" "$DUMPS" "$DUMP_TXT"
rm -f "$LOGS"/*.log "$DUMPS"/*.bin "$DUMP_TXT"/*.txt

source /usr/local/Ascend/ascend-toolkit/set_env.sh
BISHENG="$(command -v bisheng || echo /usr/local/Ascend/cann-9.1.0/bin/bisheng)"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

{
    echo "# probe_sync_quirks 证据生成记录"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu             : $(npu-smi info 2>/dev/null | sed -n '6p' | tr -s ' ')"
    echo "CANN/compiler   : $BISHENG"
    echo "bisheng version : $($BISHENG --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "ASC arch        : dav-3510"
    echo "cmake           : $(cmake --version | head -1)"
    echo "reps (probe B)  : $REPS"
    echo
    echo "## 复现命令（本目录执行）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release"
    echo "cmake --build build -j8                     # 默认 target（不含预期编译失败的 form 5/8 与 u32loop 靶子）"
    echo "bash run_probes.sh $REPS                     # 本文；含 form 5/8/u32loop 的单独构建与 stderr 归档、form 9 的运行归档"
    echo
    echo "## 单变体手工复现（不经 CMake，直接调编译器——form 5 的错误原文就是这样抓的）"
    echo "# $BISHENG -DNPU_ARCH_DAV_3510 -DPROBE_B_IDX_T=int32_t -DPROBE_B_FORM=5 \\"
    echo "#     -DPROBE_B_TAG=probe_b_idx_s32_f5_gmget -std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG \\"
    echo "#     -c --asc-aicore-lang probe_b_vec_idx.asc -o /tmp/o.o"
} > "$EV/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
    echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }

# 逐 target 构建（含 EXCLUDE_FROM_ALL 的预期失败变体），分别归档输出
# EXTRA_TARGETS：不走 "target=type=form" 三元组的归档靶子（如带额外编译宏的归纳变量变体）
EXTRA_TARGETS="probe_b_idx_s32_f0_u32loop"
TARGETS=$(grep -hoE '"[a-z0-9_]+=[a-z0-9_]+=[0-9]"' CMakeLists.txt | tr -d '"' | cut -d= -f1)
: > "$LOGS/compile_matrix.txt"
echo "== 逐 target 构建（$(echo "$TARGETS $EXTRA_TARGETS" | wc -w) 个变体）=="
for t in probe_a_mrgsort4 probe_a2_mrgsort4_api $TARGETS $EXTRA_TARGETS; do
    log="$LOGS/build_$t.log"
    cmake --build "$BUILD" --target "$t" -j4 > "$log" 2>&1
    rc=$?
    case "$t" in
        *f5_gmget* | *f8_gmraw* | *u32loop*) note="  [预期失败：归档靶子，见 README §5]" ;;
        *) note="" ;;
    esac
    if [ $rc -eq 0 ]; then
        printf 'COMPILE-OK    %s%s\n' "$t" "$note" | tee -a "$LOGS/compile_matrix.txt"
    else
        printf 'COMPILE-FAIL  %s  (exit=%d, 见 logs/build_%s.log)%s\n' "$t" "$rc" "$t" "$note" \
            | tee -a "$LOGS/compile_matrix.txt"
    fi
done

echo "== 运行 probe A / A2（dump 到 evidence/dumps）=="
"$BUILD/probe_a_mrgsort4" "$DUMPS" > "$LOGS/run_probe_a_mrgsort4.log" 2>&1
"$BUILD/probe_a2_mrgsort4_api" "$DUMPS" > "$LOGS/run_probe_a2_mrgsort4_api.log" 2>&1
tail -3 "$LOGS/run_probe_a_mrgsort4.log"
tail -3 "$LOGS/run_probe_a2_mrgsort4_api.log"

echo "== 运行 probe B（reps=$REPS）=="
: > "$LOGS/run_matrix.txt"
BINS=$(ls "$BUILD" | grep -E '^probe_b_idx_' | sort)
for b in $BINS; do
    log="$LOGS/run_$b.log"
    "$BUILD/$b" "$DUMPS" "$REPS" > "$log" 2>&1
    rc=$?
    case "$b" in
        *misalign_demo*)
            printf '%-34s EXPECTED-FAIL（归档靶子）：运行 %s —— 见 logs/run_%s.log\n' "$b" \
                "$(grep -oE 'aclError=[0-9]+' "$log" | head -1)" "$b" | tee -a "$LOGS/run_matrix.txt"
            ;;
        *)
            printf '%-34s %s\n' "$b" "$(grep -oE 'RUN (PASS|FAIL)[^：]*' "$log" | head -1) $(grep -oE 'RUN FAIL[^：]*：[0-9]+/[0-9]+ 点不符' "$log" | head -1)" \
                | tee -a "$LOGS/run_matrix.txt"
            ;;
    esac
done

echo "== 解析 probe A 的 UB dump -> evidence/dumps_txt + decode_A_summary.txt =="
"$PY" tools/decode_mrgsort_dump.py "$DUMPS" tools > "$LOGS/decode_A.log" 2>&1
tail -2 "$LOGS/decode_A.log"

echo
echo "完成。证据目录：$EV"
echo "  compile_matrix.txt / run_matrix.txt 见 $LOGS；逐元素表格见 $DUMP_TXT"
