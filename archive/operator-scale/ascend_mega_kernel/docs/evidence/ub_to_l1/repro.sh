#!/usr/bin/env bash
# M120：`docs/15 §6.2` / `docs/19 §7 第 4 条` 更正的复现脚本。
#
# 做三件事，全部是 **host 侧**（不碰 NPU、不跑设备档）：
#   ① 记录两处证据来源的不可变锚点：本机 CANN 安装路径 + asc-devkit 的 commit；
#   ② 把「官方文档逐字 + 本机头文件逐字」按 `文件:行` 打出来；
#   ③ 用三个最小探针判定：C API 的 UB→L1 硬通道能否编过、基础 API 走的是哪条分支。
#
# 用法（任意 cwd）：
#   bash docs/evidence/ub_to_l1/repro.sh
# 输出：全部打到 stdout；调用方负责重定向到 logs/ 下的文件（README §5 就是这么取的读数）。
#
# 依赖：/usr/local/Ascend 下的 CANN 9.1.0（bisheng 编译器）+ 本机 /workspace/asc-devkit 工作副本。
# 两者都是**外部快照**（本仓不持有），故下面凡引用它们的地方都写成变量并在输出里回显取值。

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
BISHENG="$CANN/bin/bisheng"
DEVKIT="${ASC_DEVKIT:-/workspace/asc-devkit}"
WORK="$(mktemp -d /tmp/m120_repro.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "### 0. 锚点（外部快照的取值，读的人自行核对）"
echo "CANN_HOME   = $CANN"
echo "BISHENG     = $BISHENG"
echo "ASC_DEVKIT  = $DEVKIT"
if [ -d "$DEVKIT/.git" ]; then
    echo "asc-devkit rev = $(git -C "$DEVKIT" rev-parse HEAD)"
else
    echo "asc-devkit rev = (不是 git 工作副本，无法取 commit)"
fi
echo

echo "### 1. 官方文档逐字（asc-devkit，行号以 rev 648a6018207d75af44c6865f96511bafadd90630 为准）"
echo "--- 1a. 四条 UBToL1 基础 API 的『软件仿真』NOTE（这四条都在） ---"
grep -n "本接口为软件仿真实现" \
    "$DEVKIT/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load"/DataCopy*_UBToL1*.md
echo "--- 1b. 3510 新增 UB→L1 硬件通道（同一批文档的另一半） ---"
grep -n "新增UB到L1 Buffer搬运数据通路\|UB向L1 Buffer搬运数据不再需要通过GM中转\|无需配置编译选项\|开启该编译选项后\|未开启该编译选项时" \
    "$DEVKIT/docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md"
echo "--- 1c. SSBuffer 指南：硬通道 vs GM 仿真，C API 不受注册要求约束 ---"
grep -n "硬通道路径与GM软件仿真路径不同" \
    "$DEVKIT/docs/zh/guide/programming_guide/advanced_programming/inter_core_communication/ssbuffer_inter_core_memory_feature.md"
echo "--- 1d. 编译宏 ENABLE_CV_COMM_VIA_SSBUF 的定义处 ---"
echo "    （定义句 :299 里宏名是**转义下划线** ENABLE\\_CV\\_COMM\\_VIA\\_SSBUF，字面量 grep 会漏掉它、只命中 :301；"
echo "      故 pattern 让那个反斜杠可选 —— M120 r1 复审 P2-1 的教训）"
grep -nE 'ENABLE\\?_CV\\?_COMM\\?_VIA\\?_SSBUF' \
    "$DEVKIT/docs/zh/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.md"
echo "--- 1e. 官方样例 README：版本声明 + 为什么它不用 Nd2NzParams 版本 ---"
grep -n "CANN 9.2.0\|硬件本身不支持该能力\|UBNZ\|__mix__" \
    "$DEVKIT/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_ub2l1/README.md"
echo "--- 1f. ND2NZ 文档里的 950 特例（1:2 硬通道 + 1:1 兼容模式经 GM） ---"
grep -n "1:2硬通道" \
    "$DEVKIT/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_ND2NZ.md"
echo

echo "### 2. 本机 CANN 头/实现逐字"
echo "--- 2a. C API 公开声明 ---"
grep -n "asc_copy_ub2l1" "$CANN/x86_64-linux/asc/include/c_api/vector_datamove/vector_datamove.h"
echo "--- 2b. C API 实现：直接下发硬件 intrinsic，不碰 TPipe/KFC/Matmul ---"
grep -n "copy_ubuf_to_cbuf\|ASC_IS_AIV" \
    "$CANN/x86_64-linux/asc/impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h"
echo "--- 2c. 硬件 intrinsic 本体（Bisheng 自带头） ---"
grep -n "copy_ubuf_to_cbuf" \
    "$CANN/tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h"
echo "--- 2d. KFC_C310_SSBUF 的定义（探针 3 引用的那个宏） ---"
grep -n "KFC_C310_SSBUF" "$CANN/x86_64-linux/asc/impl/basic_api/kernel_utils.h"
echo "--- 2e. 基础 API 的两条分支（硬件 vs GM/KFC + Matmul workspace） ---"
grep -n "KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1\|CopyUbufToCbuf(dst, src\|GetKfcClient()->AllocUB\|ScmDataCopyMsg((__cbuf__ void\*)dst" \
    "$CANN/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h"
echo

echo "### 3. 编译探针（零设备；只编不跑）"
echo "COMMON FLAGS: --npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c"
echo "--- 3a. C API asc_copy_ub2l1（__mix__(1,2) kernel） ---"
"$BISHENG" --npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c "$HERE/probe_capi_ub2l1.asc" -o "$WORK/capi.o"
echo "rc=$?   (stderr 空 ⇒ 编过；产物 $WORK/capi.o)"
echo "--- 3b. 基础 API DataCopy(L1_TSCM, UB_VECIN, count)，不带编译宏 ---"
"$BISHENG" --npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c "$HERE/probe_basic_ub2l1.asc" -o "$WORK/basic_nomacro.o"
echo "rc=$?"
echo "--- 3c. 同上，带 -DENABLE_CV_COMM_VIA_SSBUF=true（官方样例的配置） ---"
"$BISHENG" --npu-arch=dav-3510 -DENABLE_CV_COMM_VIA_SSBUF=true -std=c++17 --asc-aicore-lang -c "$HERE/probe_basic_ub2l1.asc" -o "$WORK/basic_macro.o"
echo "rc=$?"
echo "--- 3d. 分支判定（把 DataCopyUB2L1Impl:582 的 #if 抄成 #error） ---"
echo "    不带编译宏："
"$BISHENG" --npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c "$HERE/probe_branch.asc" -o "$WORK/br1.o" 2>&1 | grep -m1 -E "BRANCH_|MIX_CORE"
echo "    带 -DENABLE_CV_COMM_VIA_SSBUF=true："
"$BISHENG" --npu-arch=dav-3510 -DENABLE_CV_COMM_VIA_SSBUF=true -std=c++17 --asc-aicore-lang -c "$HERE/probe_branch.asc" -o "$WORK/br2.o" 2>&1 | grep -m1 -E "BRANCH_|MIX_CORE"
echo "--- 3e. \`__MIX_CORE_AIC_RATION__\` 到底有没有被定义（3d 结论的前提） ---"
"$BISHENG" --npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c "$HERE/probe_mixmacro.asc" -o "$WORK/mix.o" 2>&1 | grep -m1 -E "MIX_RATIO_"
echo
echo "### 4. 版本差取证：官方样例在本机 CANN 9.1.0 上能不能整编"
echo "    官方样例 README:11 声明『>= CANN 9.2.0』；本机是 9.1.0。把样例（basic API 版，scenario 1）"
echo "    拷到临时目录整编一次，看它卡在哪 —— 这决定『硬通道在本机是否已就绪』能说到哪一步。"
EXDIR="$WORK/basic_ub2l1_example"
cp -r "$DEVKIT/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_ub2l1" "$EXDIR" 2>/dev/null
if [ -d "$EXDIR" ]; then
    mkdir -p "$EXDIR/build" && cd "$EXDIR/build"
    cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=1 .. > cmake.log 2>&1
    echo "    cmake rc=$?"
    make -j4 > make.log 2>&1
    echo "    make  rc=$?  （非 0 = 整编不通过；首个 error 行如下）"
    grep -m2 "error:" make.log
    cd "$HERE"
else
    echo "    (样例目录不存在，跳过)"
fi
echo

echo "### 5. 说明"
echo "本脚本只回答『接口在不在、走哪条分支、能不能编过』。"
echo "**『真机上这条硬通道是否可用、带宽多少、AIC 怎么等到数据』不在本脚本覆盖范围内** —— 需要设备档。"
