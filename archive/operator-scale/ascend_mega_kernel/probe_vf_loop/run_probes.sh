#!/usr/bin/env bash
# probe_vf_loop/run_probes.sh —— 一条命令重跑全部探针并归档原始证据
#
# 用法（本目录下）：bash run_probes.sh [reps]
#   reps  每个 target 的**独立进程运行**次数（默认 5；mission 硬要求 ≥5）
#   环境变量 ONLY=<正则>：只做名字匹配的 target（分块跑用；只清这些 target 的旧日志）。
#     第一次跑（ONLY 为空）会清空 evidence/logs 与 evidence/dumps 并 clean 重建全部 target。
#     同一张卡上多 worker 共用，长驻 run 可能被别人的清场动作打断 ⇒ 需要时可分块：
#       ONLY='nest_o0i0b' bash run_probes.sh 5
#
# 产物（全部进 git，供 review 追溯）：
#   evidence/commands.txt               环境/编译器版本 + 完整复现命令 + 逐 target 完整编译选项
#   evidence/logs/build_<target>.log    每个 target 的完整构建输出（VERBOSE=1 ⇒ 含完整编译命令行）
#   evidence/logs/run_<target>_rep<k>.log   第 k 次独立进程运行的完整 stdout
#   evidence/logs/compile_matrix.txt    编译通过/失败矩阵（逐 target 一行）
#   evidence/logs/compile_options.txt   逐 target 编译选项摘要
#   evidence/logs/run_matrix.txt        运行结果矩阵（**由 tools/summarize.py reindex 从 run_*.log 重建**；
#                                       逐 target 两行：RUN PASS/FAIL + 「rows 档位 N，FAIL M，判据列跨 rep 一致/不一致」）
#   evidence/logs/sha256.txt            逐 (target, rep) 摘要（同上重建）：日志 sha256 + exit + **判据列 sha16** + JUDGE-SAME/DIFF
#   evidence/logs/summarize_verify.txt  跨 rep 一致性 + 判据列独立复算（三态退出码 0/1/2）
#   evidence/logs/summarize_selftest.txt 负向对照：在 /tmp 副本里改字节，验证核验会报差异（不改工作区）
#   evidence/dumps/                     probe ② (gather 扫描) 的 64 lane 输出 .bin

set -u
cd "$(dirname "$0")"

REPS="${1:-5}"
ONLY="${ONLY:-}"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"

mkdir -p "$LOGS" "$DUMPS"
if [ -z "$ONLY" ]; then
    rm -f "$LOGS"/*.log "$LOGS"/*.txt "$DUMPS"/*.bin
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
BISHENG="$(command -v bisheng || echo /usr/local/Ascend/cann-9.1.0/bin/bisheng)"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

{
    echo "# probe_vf_loop 证据生成记录"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu             : $(npu-smi info 2>/dev/null | sed -n '6p' | tr -s ' ')"
    echo "CANN/compiler   : $BISHENG"
    echo "bisheng version : $($BISHENG --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "ASC arch        : dav-3510"
    echo "cmake           : $(cmake --version | head -1)"
    echo "reps            : $REPS（每 target 独立进程运行次数）"
    echo
    echo "## 复现命令（本目录执行）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release"
    echo "cmake --build build -j4                     # 默认 target（不含预期编译失败的靶子）"
    echo "bash run_probes.sh $REPS                     # 本文；含失败靶子的单独构建与 stderr 归档"
    echo
    echo "## 单变体手工复现（不经 CMake 直接调编译器；失败靶子的错误原文就是这样抓的）"
    echo "# $BISHENG -DNPU_ARCH_DAV_3510 -DPROBE_OUTER=0 -DPROBE_INNER=0 -DPROBE_BODY=1 \\"
    echo "#     -DPROBE_CT_ROWS=0 -DPROBE_TAG=nest_o0i0b1 -std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG \\"
    echo "#     -c --asc-aicore-lang probe_vf_nest.asc -o /tmp/o.o"
    echo
    echo "## 逐 target 完整编译选项：见 evidence/logs/compile_options.txt 与各 build_<target>.log（VERBOSE=1）"
    echo "## 驱动内部 cc1 开关全文：见 evidence/logs/bisheng_verbose_cc1_flags.txt（bisheng -v 输出）"
} > "$EV/commands.txt"

# 归档驱动内部 cc1 命令行（含全部 -mllvm 开关）——"结论能否被复现"的关键证据之一
"$BISHENG" -v -DNPU_ARCH_DAV_3510 -DPROBE_OUTER=0 -DPROBE_INNER=0 -DPROBE_BODY=0 -DPROBE_CT_ROWS=0 \
    -DPROBE_TAG=nest_o0i0b0 -std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG -c --asc-aicore-lang \
    probe_vf_nest.asc -o /tmp/probe_vf_loop_verbose.o > "$LOGS/bisheng_verbose_cc1_flags.txt" 2>&1
rm -f /tmp/probe_vf_loop_verbose.o

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=Release > "$LOGS/cmake_configure.log" 2>&1 || {
    echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }
cp -f "$BUILD/compile_commands.json" "$EV/compile_commands.json" 2>/dev/null || true

# 逐 target 构建（含 EXCLUDE_FROM_ALL 的预期失败靶子），分别归档输出。
# VERBOSE=1：把**完整编译命令行（全部编译选项）**留在 build_<target>.log 里 —— 结论能否复现的关键。
EXTRA_TARGETS="nest_o0i0b2_scal nest_o0i0b2s_scalstore nest_o0i0b4_marker_ctl ag_arange_u32"
# 只做编译归档、**不运行**的 target（正对照：证明「向量 store 版」能编过；跑出来的读数异常另见 negmark）
BUILD_ONLY_TARGETS="nest_o0i0b4_marker_ctl"
# 注意：行尾允许有注释（.*"），首字符 '#' 的注释行不匹配（CMakeLists 里被刻意排除的 body-4 靶子
# 就是注释形式）—— 早先版本漏了行尾注释，导致 A 组 8 个 target 没跑（已修）。
TARGETS=$(sed -n 's/^[[:space:]]*"\([A-Za-z0-9_]*\)=[0-9]=[0-9]=[0-9]=[0-9]*".*$/\1/p' CMakeLists.txt | sort -u)
O2TARGETS=$(sed -n 's/^foreach(O2SPEC "\([A-Za-z0-9_]*\)=[0-9]=[0-9]" "\([A-Za-z0-9_]*\)=[0-9]=[0-9]")$/\1\n\2/p' CMakeLists.txt | sort -u)
# 诊断靶子（关编译器内部 pass，用于归属取证）：从 set(DIAG_TARGETS ...) 块里取
DIAGTARGETS=$(sed -n '/^set(DIAG_TARGETS/,/^$/p' CMakeLists.txt | sed -n 's/^[[:space:]]*"\([A-Za-z0-9_]*\)=.*$/\1/p')
ALL_TARGETS="ag_gather_sweep $TARGETS $O2TARGETS $DIAGTARGETS $EXTRA_TARGETS"
if [ -n "$ONLY" ]; then
    # 分块模式：只做匹配的 target，且只清这些 target 的旧日志。
    # 注意两个精确性问题：① `run_<t>_rep*.log`（而不是 `run_<t>_*.log`），否则
    #   nest_o0i0b1 会连 nest_o0i0b1_O2 的日志一起删；② ag_gather_sweep 的逐档位日志
    #   （run_..._s<start>_x<stride>.log）保留 —— 由 AG_ONLY 决定这次要重跑哪几档，
    #   在运行循环里逐档覆盖，免得没被选中的档位记录被误删。
    KEEP=""
    for t in $ALL_TARGETS; do
        if echo "$t" | grep -qE "$ONLY"; then KEEP="$KEEP $t"; fi
    done
    ALL_TARGETS="$KEEP"
    for t in $ALL_TARGETS; do
        rm -f "$LOGS/build_$t.log" "$LOGS"/run_${t}_rep*.log
        [ "$t" = "ag_gather_sweep" ] || rm -f "$LOGS"/run_${t}_s*.log
    done
    echo "== ONLY='$ONLY' 分块模式：$(echo $ALL_TARGETS | wc -w) 个 target =="
else
    # 全量模式：clean 后重建，保证每个 target 的 build 日志都含完整编译命令行（VERBOSE=1）
    cmake --build "$BUILD" --target clean > /dev/null 2>&1 || true
fi
# 汇总文件：全量模式清空重写；分块模式追加（保留前几块的记录）
RESET() { if [ -z "$ONLY" ]; then : > "$1"; fi; }
RESET "$LOGS/compile_matrix.txt"
RESET "$LOGS/compile_options.txt"
RESET "$LOGS/run_matrix.txt"
RESET "$LOGS/sha256.txt"
echo "== 逐 target 构建（$(echo $ALL_TARGETS | wc -w) 个变体）=="
for t in $ALL_TARGETS; do
    log="$LOGS/build_$t.log"
    cmake --build "$BUILD" --target "$t" -j4 -- VERBOSE=1 > "$log" 2>&1
    rc=$?
    {
        printf '### %s\n' "$t"
        grep -oE -- '--npu-arch=dav-3510.*probe_[A-Za-z_]+\.asc' "$log" | head -1
        grep -oE '(/usr/local[A-Za-z0-9_./-]*bisheng|/bin/bisheng)[^|]*' "$log" | head -1
        grep -oE -- '-O[0-9]|-DNDEBUG|-DPROBE_[A-Z_]+=[A-Za-z0-9_]+|-DAG_[A-Z_]+=[A-Za-z0-9_]+|-DNPU_ARCH_DAV_3510' "$log" \
            | sort -u | tr '\n' ' '
        printf '\n'
    } >> "$LOGS/compile_options.txt"
    case "$t" in
        *b2_scal* | *b2s_scalstore* | *ag_arange_u32*) note="  [预期失败：归档靶子，见 README §变体矩阵/§3.1 C15]" ;;
        *) note="" ;;
    esac
    if [ $rc -eq 0 ]; then
        printf 'COMPILE-OK    %s%s\n' "$t" "$note" | tee -a "$LOGS/compile_matrix.txt"
    else
        printf 'COMPILE-FAIL  %s  (exit=%d, 见 logs/build_%s.log)%s\n' "$t" "$rc" "$t" "$note" \
            | tee -a "$LOGS/compile_matrix.txt"
    fi
done

echo "== 运行（每 target $REPS 次独立进程）=="
for t in $ALL_TARGETS; do
    [ -x "$BUILD/$t" ] || continue
    case " $BUILD_ONLY_TARGETS " in *" $t "*) echo "（$t：只做编译归档，不运行）"; continue ;; esac
    case "$t" in
        ag_gather_sweep)
            # probe ②：host 传 (start, stride)，每个档位一个独立进程 —— 边界扫描。
            # 索引语义：idx[lane] = start + lane*stride（元素），源数组 NEL=16384 fp32（64KB @ UB 0）。
            # 实测结论（见 README §②）：数组范围内全对；越出数组但留在 UB 内 = 静默读该处 UB；
            # idx*4 ≥ 256KB（fp32、base=0 ⇒ start+63 ≥ 65536）→ aclError 507035。
            for spec in "0 1" "1024 1" "4096 1" "8064 1" "8191 1" "8192 1" "8193 1" "12000 1" "16000 1" \
                        "16320 1" "16321 1" "16383 1" "16384 1" "20000 1" "32000 1" "60000 1" "65000 1" \
                        "65400 1" "65472 1" "65473 1" "65500 1" "65535 1" "65536 1" "65537 1" "70000 1" \
                        "131072 1" "1048576 1" \
                        "0 2" "16000 2" "16321 2" \
                        "0 8" "16000 8" "16321 8" \
                        "0 64" "16000 64" "16321 64" \
                        "0 128" "1 128" "64 128" "8064 128" "8191 128" "8192 128" "8193 128" \
                        "16000 128" "16321 128" "20000 128" "1048576 128"; do
                set -- $spec
                # 可选：AG_ONLY=<正则> 只跑匹配的档位（匹配 "s<start>_x<stride>"；分块跑用）
                if [ -n "${AG_ONLY:-}" ] && ! echo "s$1_x$2" | grep -qE "$AG_ONLY"; then
                    continue
                fi
                log="$LOGS/run_${t}_s$1_x$2.log"
                "$BUILD/$t" "$DUMPS" "$1" "$2" > "$log" 2>&1
                printf '%-42s %s\n' "$t s=$1 x=$2" \
                    "$(grep -oE 'in_rng_bad=[0-9]+ oob_lanes=[0-9]+ oob_zero=[0-9]+ oob_first=[^ ]+|aclError=[0-9]+' "$log" | head -1)"
            done
            ;;
        *)
            for r in $(seq 1 "$REPS"); do
                log="$LOGS/run_${t}_rep${r}.log"
                "$BUILD/$t" "$DUMPS" "$((r - 1))" > "$log" 2>&1
                rc=$?
                h="$(sha256sum "$log" | cut -d' ' -f1)"
                # judge=<判据列 sha16>：只取 cnt/expect/tri/MISMATCH/lane_mismatch/verdict 这些行，
                # 并把会随 UB 残留变化的 hash=0x… 抹掉 —— 这样"非确定性"能区分「判据列变了」还是
                # 「只有数据面列变了」。汇总文件不在这里写：跑完由 tools/summarize.py reindex 统一重建
                # （带 target 标签 + FAIL 计数 + 判据列结论），避免同一格式有两份实现。
                judge="$(grep -E '^([[:space:]]+(cnt|expect|tri|sumj)[[:space:]]*:[[:space:]]*$|  rows=[0-9]+ verdict=|.*(MISMATCH|JSET-ODD|SUMJ-ODD|TRI-ODD)|.*lane_mismatch=)' "$log" |
                    sed -E 's/hash=0x[0-9a-fA-F]+/hash=<strip>/' | sha256sum | cut -c1-16)"
                if [ "$r" = "1" ]; then
                    judgeFirst="$judge"
                    judgeState="JUDGE-FIRST"
                elif [ "$judge" = "$judgeFirst" ]; then
                    judgeState="JUDGE-SAME"
                else
                    judgeState="JUDGE-DIFF"
                fi
                printf '%-24s rep%s  exit=%d  judge=%s  %s  %s\n' "$t" "$r" "$rc" "$judge" "$judgeState" "${h:0:16}"
            done
            ;;
    esac
done

# 汇总文件由工具从 run_*.log **统一重建**（含 target 标签、FAIL 计数、判据列跨 rep 结论；
# 这样 p2-1/p2-2 那类"叙述与归档不一致"在结构上不会再出现）
echo "== 从原始日志重建 run_matrix.txt / sha256.txt =="
"$PY" tools/summarize.py reindex --logs "$LOGS"

# 跨 rep 核验 + 判据列独立复算（三态退出码：0 通过 / 1 有差异 / 2 没得比）→ 归档 + 非零退出即报错
echo "== 跨 rep 核验 + 判据列独立复算 =="
"$PY" tools/summarize.py verify --logs "$LOGS" > "$LOGS/summarize_verify.txt" 2>&1
vrc=$?
tail -4 "$LOGS/summarize_verify.txt"
[ "$vrc" = "0" ] || echo "⚠ verify 退出码=$vrc（0=通过 / 1=有差异 / 2=没得比），见 logs/summarize_verify.txt"
echo "== 负向对照 selftest（在 /tmp 副本里改字节，验证核验能报差异）=="
"$PY" tools/summarize.py selftest > "$LOGS/summarize_selftest.txt" 2>&1
tail -3 "$LOGS/summarize_selftest.txt"

echo
echo "完成。证据目录：$EV"
echo "  compile_matrix.txt / run_matrix.txt / sha256.txt / summarize_verify.txt / summarize_selftest.txt 见 $LOGS"
