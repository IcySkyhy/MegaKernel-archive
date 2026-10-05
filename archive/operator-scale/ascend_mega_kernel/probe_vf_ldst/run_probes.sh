#!/usr/bin/env bash
# probe_vf_ldst/run_probes.sh —— 一条命令重跑全部变体并归档原始证据
#
# 用法（本目录下）：bash run_probes.sh [procs]
#   procs  每个变体的**独立进程**运行次数（默认 5；mission 硬要求 ≥5）
#
# 硬要求落地：
#   ① 每个变体 ≥5 次**独立进程**运行（不是同进程重复 launch）——外层 for 起新进程；
#   ② 逐次 dump 原始字节 + sha256（evidence/logs/sha256_manifest.txt），
#      非确定性 = "同变体不同进程的判定项 dump sha256 不一致 / 判读不一致"，由汇总表量化；
#   ③ 归档完整编译选项（evidence/commands.txt + 逐 target build log）；
#   ④ **声明集与实跑集同源**：本脚本在跑变体的同一组循环里逐条 echo
#      `evidence/logs/declared_groups.txt`，summarize.py 拿它当覆盖基准 ⇒ 「声明了多少组」
#      不可能与「实际跑了多少组」各写一份而漂移（tower 规则：计数与匹配器同源）。
#
# 产物（全部进 git，供 review 追溯）：
#   evidence/commands.txt                  复现命令、CANN/编译器版本、日期、NPU、完整编译选项
#   evidence/logs/build_<target>.log       每个 target 完整构建输出
#   evidence/logs/compile_matrix.txt       编译通过/失败矩阵
#   evidence/logs/declared_groups.txt      **声明集**（与实跑同源产出）
#   evidence/logs/run_<target>_p<k>.log    每个（变体 × 独立进程）完整运行输出
#   evidence/logs/run_matrix.txt           逐（变体 × 进程）判读一行
#   evidence/logs/sha256_manifest.txt      每个 dump 文件的 sha256
#   evidence/logs/ub_region_consistency.txt  判定项 dump 一致 / scratch 区不一致的**分区**证据
#   evidence/run_summary.txt               跨进程汇总（含覆盖三栏与对照自检）

set -u
cd "$(dirname "$0")"

PROCS="${1:-5}"
ROOT="$PWD"
EV="$ROOT/evidence"
LOGS="$EV/logs"
DUMPS="$EV/dumps"
BUILD="$ROOT/build"

mkdir -p "$LOGS" "$DUMPS"
rm -f "$LOGS"/*.log "$LOGS"/*.txt "$DUMPS"/*.bin "$EV/run_summary.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
BISHENG="$(command -v bisheng || echo /usr/local/Ascend/cann-9.1.0/bin/bisheng)"
PY="/usr/local/python3.12.13/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

ASC_OPTS="-std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG -c --asc-aicore-lang"
CM_OPTS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_ASC_ARCHITECTURES=dav-3510"

# ---- 变体集（与 CMakeLists.txt 的 spec 同源：B/C 由规格串 grep 而来）----
A_CONTIG_TARGETS="probe_a_first_elem probe_a_norm_b32"
A_NC_TARGETS="probe_a_first_elem_nc probe_a_norm_b32_nc"
A_MASKS_CONTIG="0 1 8 64"        # UpdateMask(count)：全 0 / 单 lane / 低 8 lane / 全 1
A_MASKS_NC="0 1 2 3"             # M4 / M3 / 交替(0x5555…) / 交替反相(0xAAAA…)
B_TARGETS=$(grep -hoE '"probe_b_[a-z0-9_]+=' CMakeLists.txt | tr -d '"=')
# C 的 target 在 CMakeLists 里是 `probe_add_c(<target> <order> <shape>)` 调用（不带引号），
# 所以从 add_executable 的名字解析，保持"声明集来自同一个 CMakeLists"这一同源性质。
C_TARGETS=$(grep -hoE 'probe_add_c\(probe_c_rt_[a-z0-9_]+' CMakeLists.txt | sed 's/.*(//')

{
    echo "# probe_vf_ldst 证据生成记录"
    echo "date            : $(date -Is)"
    echo "host            : $(uname -srm)"
    echo "npu             : $(npu-smi info 2>/dev/null | sed -n '6p' | tr -s ' ')"
    echo "CANN/compiler   : $BISHENG"
    echo "bisheng version : $($BISHENG --version 2>&1 | head -2 | tr '\n' ' ')"
    echo "ASC arch        : dav-3510 (Ascend 950PR / NPU 0)"
    echo "cmake           : $(cmake --version | head -1)"
    echo "procs/variant   : $PROCS（独立进程）"
    echo
    echo "## 复现命令（本目录执行）"
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "cmake -B build -S . $CM_OPTS"
    echo "cmake --build build -j8"
    echo "bash run_probes.sh $PROCS"
    echo
    echo "## 单变体手工复现（不经 CMake，完整编译选项在这里）"
    echo "# 编译某变体（以 probe_b_backlog_rel0 为例；宏取值见 CMakeLists.txt 的规格串）"
    echo "# $BISHENG -DNPU_ARCH_DAV_3510 -DPROBE_B_REL=0 -DPROBE_B_READ=0 -DPROBE_B_GRAN=0 \\"
    echo "#     -DPROBE_B_MID=0 -DPROBE_B_LAYOUT=0 -DPROBE_B_PROD=0 -DPROBE_B_OFF=0 \\"
    echo "#     -DPROBE_B_VREL=0 -DPROBE_B_BACKLOG=1 -DPROBE_B_TAG=probe_b_backlog_rel0 \\"
    echo "#     $ASC_OPTS probe_b_ldst_handover.asc -o /tmp/o.o"
    echo "# 运行（dumpdir procIdx [maskSel]）"
    echo "# ./build/probe_b_backlog_rel0    evidence/dumps 3"
    echo "# ./build/probe_a_first_elem      evidence/dumps 3 8     # 第 4 参 = 连续掩码的有效 lane 数"
    echo "# ./build/probe_a_norm_b32_nc     evidence/dumps 3 2     # 第 4 参 = 0:M4 1:M3 2/3:交替"
    echo
    echo "## CMake 为每个 target 实际加入的编译选项（autogen 生成物原文）"
    for t in $(ls "$BUILD"/CMakeFiles 2>/dev/null | grep -E '^probe_.*\.dir$' | head -4); do
        f="$BUILD/CMakeFiles/$t/flags.make"
        [ -f "$f" ] && { echo "--- $t"; grep -E '^CXX_FLAGS|^CXX_DEFINES' "$f"; }
    done
} > "$EV/commands.txt"

echo "== cmake 配置 =="
cmake -B "$BUILD" -S . $CM_OPTS > "$LOGS/cmake_configure.log" 2>&1 || {
    echo "cmake 配置失败，见 $LOGS/cmake_configure.log"; exit 1; }

echo "== 逐 target 构建（$(echo $A_CONTIG_TARGETS $A_NC_TARGETS $B_TARGETS $C_TARGETS | wc -w) 个变体）=="
: > "$LOGS/compile_matrix.txt"
for t in $A_CONTIG_TARGETS $A_NC_TARGETS $B_TARGETS $C_TARGETS; do
    log="$LOGS/build_$t.log"
    cmake --build "$BUILD" --target "$t" -j4 > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        printf 'COMPILE-OK    %s\n' "$t" | tee -a "$LOGS/compile_matrix.txt"
    else
        printf 'COMPILE-FAIL  %s  (exit=%d, 见 logs/build_%s.log)\n' "$t" "$rc" "$t" \
            | tee -a "$LOGS/compile_matrix.txt"
    fi
done

# ---- 声明集：在跑变体的同一组循环里产出，与实跑同源 ----
DECLARED="$LOGS/declared_groups.txt"
{
    echo "# 声明集（由 run_probes.sh 的跑变体循环逐条产出；summarize.py 以此为覆盖基准）"
    for t in $A_CONTIG_TARGETS; do for m in $A_MASKS_CONTIG; do echo "$t mask=$m"; done; done
    for t in $A_NC_TARGETS; do for m in $A_MASKS_NC; do echo "$t mask=$m"; done; done
    for t in $B_TARGETS; do echo "$t"; done
    for t in $C_TARGETS; do echo "$t"; done
} > "$DECLARED"
echo "== 声明集：$(($(wc -l < "$DECLARED") - 1)) 组 =="

: > "$LOGS/run_matrix.txt"
{
    echo "# probe_vf_ldst run_matrix.txt —— 逐（变体组 × 独立进程）判读一行"
    echo "# 列：组名 / mask / 进程号 / rc / 判读 / 读数"
    echo "# rc 列来源：正常全量跑时由当次 \`\$?\` 直接记录；若本文件按既有 run_*.log 重生成，"
    echo "#   则 rc 由二进制自己的退出码契约推得（RUN PASS→0 / RUN FAIL→1 / 其它→2，"
    echo "#   契约见各 .asc 末尾的 \`return\`）。两种来源在通过/失败两种路径上实测一致。"
} >> "$LOGS/run_matrix.txt"

echo "== 运行 probe A：连续掩码族（每个 target × 每个 mask × $PROCS 个独立进程）=="
for t in $A_CONTIG_TARGETS; do
    for m in $A_MASKS_CONTIG; do
        for ((p = 0; p < PROCS; ++p)); do
            log="$LOGS/run_${t}_mask${m}_p${p}.log"
            "$BUILD/$t" "$DUMPS" "$p" "$m" > "$log" 2>&1
            rc=$?
            verdict=$(grep -oE 'RUN (PASS|FAIL)' "$log" | head -1)
            detail=$(grep -oE '写出 [0-9]+ lane（期望 [0-9]+）、未触碰 [0-9]+ lane' "$log" | head -1)
            printf '%-40s mask=%-3s p=%s rc=%d  %-9s %s\n' "$t" "$m" "$p" "$rc" "$verdict" "$detail" \
                >> "$LOGS/run_matrix.txt"
        done
    done
done

echo "== 运行 probe A：非连续/交替掩码族（补 mission 明列的「交替」）=="
for t in $A_NC_TARGETS; do
    for m in $A_MASKS_NC; do
        for ((p = 0; p < PROCS; ++p)); do
            log="$LOGS/run_${t}_mask${m}_p${p}.log"
            "$BUILD/$t" "$DUMPS" "$p" "$m" > "$log" 2>&1
            rc=$?
            verdict=$(grep -oE 'RUN (PASS|FAIL)' "$log" | head -1)
            detail=$(grep -oE '写出 [0-9]+ lane（期望 [0-9]+）、未触碰 [0-9]+ lane' "$log" | head -1)
            # 真实句子是「…交替掩码被严格遵守（32 个 lane、全偶；其余全为 POISON）」。
            # 两处坑：① 不能要求 `全.` 后紧跟 `）`（否则该组 detail 列为空）；
            # ② **不要用含多字节字符的 grep 括号表达式**（`[偶奇]` 会把「偶/奇」截断成半个
            #    字符、写出非法 UTF-8）—— 这里拆成 ASCII 前缀 + 定长串判断，字节安全。
            if [ -z "$detail" ]; then
                d=$(grep -oE '交替掩码被严格遵守（[0-9]+ 个 lane' "$log" | head -1)
                if [ -n "$d" ]; then
                    if grep -q '全偶' "$log"; then detail="$d、全偶"; else detail="$d、全奇"; fi
                fi
            fi
            printf '%-40s mask=%-3s p=%s rc=%d  %-9s %s\n' "$t" "$m" "$p" "$rc" "$verdict" "$detail" \
                >> "$LOGS/run_matrix.txt"
        done
    done
done
tail -3 "$LOGS/run_matrix.txt"

echo "== 运行 probe B（每个变体 $PROCS 个独立进程）=="
: > "$LOGS/run_matrix_b.txt"
for t in $B_TARGETS; do
    for ((p = 0; p < PROCS; ++p)); do
        log="$LOGS/run_${t}_p${p}.log"
        "$BUILD/$t" "$DUMPS" "$p" > "$log" 2>&1
        rc=$?
        verdict=$(grep -oE 'RUN (PASS|FAIL)' "$log" | head -1)
        cls=$(grep -oE 'ok=[0-9]+ stale_prev=[0-9]+ ahead=[0-9]+ poison=[0-9]+ zero=[0-9]+ other=[0-9]+' \
                  "$log" | head -1)
        first=$(grep -oE '首个不符 chunk=[0-9]+ 得到 [^ ]+ 期望 [^ ]+' "$log" | head -1)
        printf '%-28s p=%s rc=%d  %-9s %s %s\n' "$t" "$p" "$rc" "$verdict" "$cls" "$first" \
            >> "$LOGS/run_matrix_b.txt"
    done
done
tail -3 "$LOGS/run_matrix_b.txt"

echo "== 运行 probe C（每个变体 $PROCS 个独立进程）=="
: > "$LOGS/run_matrix_c.txt"
for t in $C_TARGETS; do
    for ((p = 0; p < PROCS; ++p)); do
        log="$LOGS/run_${t}_p${p}.log"
        "$BUILD/$t" "$DUMPS" "$p" > "$log" 2>&1
        rc=$?
        verdict=$(grep -oE 'RUN (PASS|FAIL)' "$log" | head -1)
        cls=$(grep -oE 'ok=[0-9]+ stale_prev=[0-9]+ ahead=[0-9]+ poison=[0-9]+ zero=[0-9]+ other=[0-9]+' \
                  "$log" | head -1)
        first=$(grep -oE '首个不符 chunk=[0-9]+ 得到 [^ ]+ 期望 [^ ]+' "$log" | head -1)
        printf '%-22s p=%s rc=%d  %-9s %s %s\n' "$t" "$p" "$rc" "$verdict" "$cls" "$first" \
            >> "$LOGS/run_matrix_c.txt"
    done
done
tail -3 "$LOGS/run_matrix_c.txt"

cat "$LOGS/run_matrix_b.txt" "$LOGS/run_matrix_c.txt" >> "$LOGS/run_matrix.txt"

echo "== sha256 归档 =="
( cd "$EV" && find dumps -name '*.bin' | sort | xargs sha256sum ) > "$LOGS/sha256_manifest.txt"
wc -l < "$LOGS/sha256_manifest.txt" | xargs printf '  dump 文件数: %s\n'

echo "== 判定项 dump 的跨进程一致性（分区）=="
"$PY" tools/ub_regions.py "$EV" "$PROCS" "$A_MASKS_CONTIG" "$A_MASKS_NC" "$B_TARGETS" "$C_TARGETS" \
    > "$LOGS/ub_region_consistency.txt" 2>&1
tail -6 "$LOGS/ub_region_consistency.txt"

echo "== 负向对照（summarize.py 的覆盖率自校验是否真的会咬）=="
bash tools/negative_controls.sh "$PROCS"
echo "  负向对照退出码：$?（0 = 三条对照全部符合预期）"

echo "== 跨进程汇总 =="
"$PY" tools/summarize.py "$EV" "$PROCS" > "$EV/run_summary.txt" 2>&1
SUM_RC=$?
cat "$EV/run_summary.txt"

echo
echo "完成。证据目录：$EV"
echo "  compile_matrix.txt / declared_groups.txt / run_matrix.txt / sha256_manifest.txt /"
echo "  ub_region_consistency.txt 见 $LOGS"
echo "  跨进程汇总：evidence/run_summary.txt"
# 三态退出码（tower 规则）：0=比过且通过  1=比过有差异  2=没得比/输入缺失
if [ $SUM_RC -eq 0 ]; then
    echo "本脚本退出码：0（RESULT: OK）"
elif [ $SUM_RC -eq 1 ]; then
    echo "本脚本退出码：1（RESULT: DIFF —— 缺组/进程数不足/跨进程不一致/对照自检不符）"
elif [ $SUM_RC -eq 2 ]; then
    echo "本脚本退出码：2（RESULT: SKIPPED —— 没有可比较的输入或没有声明清单）"
else
    echo "本脚本退出码：$SUM_RC（summarize.py 异常）"
fi
exit $SUM_RC
