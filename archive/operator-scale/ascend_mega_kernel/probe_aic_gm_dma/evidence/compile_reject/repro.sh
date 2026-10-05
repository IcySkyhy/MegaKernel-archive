#!/usr/bin/env bash
# repro.sh —— M141 编译期取证（零设备）：逐个编译小探针，记录 bisheng 对每一条腿的接受/拒绝。
#
# 为什么要单独做这一档：`ub2gm_raw` 那条腿被编译器**直接拒绝**，进不了可执行文件，
# 所以它无法作为运行期变体；本脚本把「编译期拒绝」的原样诊断落盘，作为一条独立读数。
#
# 用法（本目录下）：bash repro.sh
# 产物：同目录 *.log（每份 .asc 的 bisheng 完整输出）、compile_matrix.txt（汇总）
set -u
cd "$(dirname "$0")"
BH=/usr/local/Ascend/cann-9.1.0/bin/bisheng
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

# 与 probe_crosscore_tail/CMakeLists.txt:22-27 同源的编译参数
FLAGS="-DNPU_ARCH_DAV_3510 -std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG --asc-aicore-lang"

echo "### bisheng: $BH" > compile_matrix.txt
"$BH" --version 2>&1 | head -2 >> compile_matrix.txt
echo "### flags: $FLAGS" >> compile_matrix.txt
echo "### note: ACCEPT 只表示『编译通过』；运行期是否落盘由 probe_aic_gm_dma 的对应档回答。" >> compile_matrix.txt
echo >> compile_matrix.txt

for f in raw_ub2gm_aic raw_ub2l1_aic raw_gm2ub_aic raw_l12ub_aic api_ub2gm_aic; do
    log="$f.log"
    if "$BH" $FLAGS -c -o "/tmp/${f}.o" "$f.asc" > "$log" 2>&1; then
        first=$(grep -m1 'error:' "$log" || true)
        printf '%-18s %s\n' "$f" "ACCEPT (rc=0) ${first:+；注意: $first}" >> compile_matrix.txt
    else
        rc=$?
        # 取第一条 error 行，保留到分号前的关键判断
        first=$(grep -m1 'error:' "$log" | cut -c1-200)
        printf '%-18s %s\n' "$f" "REJECT (rc=$rc) :: $first" >> compile_matrix.txt
    fi
done

cat compile_matrix.txt
