#!/usr/bin/env bash
# probe_ub2l1/evidence/version_diff.sh —— M121 版本差取证（host 侧，零设备）
#
# 回答的问题：官方样例（asc-devkit）声明「>= CANN 9.2.0」，本机是 9.1.0 —— 到底卡在哪？
# 全部是**只编不跑**的 host 侧操作；把逐字诊断打出来（人类反复要求：不能只说"不支持"）。
#
# 用法（任意 cwd）：bash probe_ub2l1/evidence/version_diff.sh > probe_ub2l1/evidence/logs/version_diff.log 2>&1
#
# 依赖（外部快照，本仓不持有；取值在输出里回显，读的人自行核对）：
#   CANN 9.1.0   —— /usr/local/Ascend/cann-9.1.0（bisheng 编译器 + C API 头）
#   asc-devkit   —— 官方样例与 SDD 文档的本地工作副本（默认 /workspace/asc-devkit）

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
BISHENG="$CANN/bin/bisheng"
DEVKIT="${ASC_DEVKIT:-/workspace/asc-devkit}"
WORK="$(mktemp -d /tmp/m121_verdiff.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
BFLAGS="--npu-arch=dav-3510 -std=c++17 --asc-aicore-lang -c"

echo "### 0. 锚点"
echo "date        = $(date -Is)"
echo "CANN_HOME   = $CANN"
if [ -f "$CANN/compiler/version.info" ]; then echo "version.info= $(tr '\n' ' ' < "$CANN/compiler/version.info")"; fi
echo "BISHENG     = $BISHENG"
echo "bisheng     = $("$BISHENG" --version 2>&1 | head -1)"
echo "ASC_DEVKIT  = $DEVKIT"
if [ -d "$DEVKIT/.git" ]; then echo "devkit rev  = $(git -C "$DEVKIT" rev-parse HEAD)"; fi
echo

echo "### 1. 官方样例的版本声明（逐字）"
grep -n "CANN 9.2.0\|>= CANN" "$DEVKIT/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_ub2l1/README.md" || true
echo

echo "### 2. 官方【基础 API】样例 scenario 1，逐字整编（期望：卡在 ceil_div）"
EX1="$WORK/basic_ex"
cp -r "$DEVKIT/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_ub2l1" "$EX1"
# 取 rc 的方式：**先把完整 stderr 落到文件**（不接管道），再 head 出来看。若直接接 `head`，
# 一是读到的 rc 会是 head/SIGPIPE 的状态而不是 bisheng 的，二是"6 errors generated."会被读成 rc=0。
( cd "$EX1" && "$BISHENG" $BFLAGS -DSCENARIO_NUM=1 data_copy_ub2l1.asc -o "$WORK/basic_verbatim.o" ) \
    > "$WORK/basic_verbatim.txt" 2>&1
BRC=$?
head -30 "$WORK/basic_verbatim.txt"
echo "bisheng rc=$BRC   （非 0 = 整编不通过；上面是前 30 行）"
echo

echo "### 3. 绕开缺符号的最小探针：把 AscendC::Std::ceil_div 换成等价本地 constexpr，其余一字不改"
EX2="$WORK/basic_shim"
cp -r "$EX1" "$EX2"
python3 - "$EX2/data_copy_ub2l1.asc" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("AscendC::Std::ceil_div(", "m121_ceil_div(")
anchor = '#include "data_utils.h"\n'
helper = anchor + "\n// M121 版本差最小探针：9.1.0 无 AscendC::Std::ceil_div，用等价本地 constexpr 顶掉。\n__aicore__ constexpr uint32_t m121_ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }\n"
assert anchor in s
s = s.replace(anchor, helper, 1)
open(p, "w").write(s)
PY
( cd "$EX2" && "$BISHENG" $BFLAGS -DSCENARIO_NUM=1 data_copy_ub2l1.asc -o "$WORK/basic_shim.o" ) \
    > "$WORK/basic_shim.txt" 2>&1
BRC=$?
head -20 "$WORK/basic_shim.txt"
echo "bisheng rc=$BRC   （0 = 只差这一个符号；绕开后可编过）"
echo

echo "### 4. 官方【C API】样例 scenario 1，逐字整编（期望：先卡 ASCENDC_HOST_AICORE，再卡 ceil_div）"
EX3="$WORK/capi_ex"
cp -r "$DEVKIT/examples/02_simd_c_api/03_c_api/00_data_movement/data_copy_ub2l1" "$EX3"
( cd "$EX3" && "$BISHENG" $BFLAGS -DSCENARIO_NUM=1 data_copy_ub2l1.asc -o "$WORK/capi_verbatim.o" ) \
    > "$WORK/capi_verbatim.txt" 2>&1
BRC=$?
head -20 "$WORK/capi_verbatim.txt"
echo "bisheng rc=$BRC   （非 0 = 整编不通过）"
echo

echo "### 5. 拿本探针源码本体做同样的只编不跑（证明「我们自己的最小实现能编过」）"
( cd "$HERE/.." && "$BISHENG" $BFLAGS probe_ub2l1.asc -o "$WORK/probe_ours.o" ) \
    > "$WORK/probe_ours.txt" 2>&1
BRC=$?
head -20 "$WORK/probe_ours.txt"
echo "bisheng rc=$BRC   （0 = 本探针源码可编过）"
echo

echo "### 6. 分支判定：DataCopyUB2L1Impl 自身的 #if 到底走了哪条（本机 9.1.0，两条编译选项各一次）"
echo "    6a 文件作用域 vs mix kernel 体内（同一个 #if 抄两处）："
cat > "$WORK/brscope.asc" <<'EOF'
#include "kernel_operator.h"
#if KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1
#error SCOPE_FILE_BRANCH_HARDWARE
#else
#error SCOPE_FILE_BRANCH_GM
#endif
__global__ __mix__(1, 2) void k(__gm__ float* p) {
#if KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1
#error SCOPE_BODY_BRANCH_HARDWARE
#else
#error SCOPE_BODY_BRANCH_GM
#endif
#if defined(__MIX_CORE_AIC_RATION__)
#error BODY_MIX_RATION_DEFINED
#else
#error BODY_MIX_RATION_UNDEFINED
#endif
    p[0] = 1.0f;
}
EOF
for opt in "" "-DENABLE_CV_COMM_VIA_SSBUF=true"; do
  echo "  [选项: ${opt:-（无）}]"
  "$BISHENG" --npu-arch=dav-3510 $opt -std=c++17 --asc-aicore-lang -c "$WORK/brscope.asc" -o "$WORK/brs.o" 2>&1 \
      | grep -E "^$WORK/brscope.asc:[0-9]+:[0-9]+: error: (SCOPE|BODY)" | sort -u | sed 's/^/    /'
done
echo "    6b 宏的实际取值（static_assert 故意写错，让诊断把展开后的值打出来）："
cat > "$WORK/brval.asc" <<'EOF'
#include "kernel_operator.h"
__global__ __mix__(1, 2) void k(__gm__ float* p) {
    static_assert(KFC_C310_SSBUF == 0, "PROBE: KFC_C310_SSBUF == 0");
    static_assert(KFC_C310_SSBUF == 1, "PROBE: KFC_C310_SSBUF == 1");
    p[0] = 1.0f;
}
EOF
for opt in "" "-DENABLE_CV_COMM_VIA_SSBUF=true"; do
  echo "  [选项: ${opt:-（无）}]"
  "$BISHENG" --npu-arch=dav-3510 $opt -std=c++17 --asc-aicore-lang -c "$WORK/brval.asc" -o "$WORK/brv.o" 2>&1 \
      | grep -E "static assertion failed" | sort -u | sed 's/^/    /'
done
echo "    读法：默认构建 KFC_C310_SSBUF=0，但 #if 仍成立 —— 因为 __MIX_CORE_AIC_RATION__ 在本机 9.1.0 的"
echo "    mix kernel 体内**未定义**（#if 里当 0 用）⇒ '0 != 1' 为真 ⇒ 落硬件分支。带宏时则是 KFC_C310_SSBUF=1 成立。"
echo

echo "### 7. 本机 9.1.0 里这几个官方样例用到的符号的落点"
echo "--- 7a. AscendC::Std::cmath 只有 ceil_division / ceil_align，**没有 ceil_div**："
grep -n "ceil_div\|ceil_division\|ceil_align" "$CANN/x86_64-linux/asc/include/utils/std/cmath.h" || true
echo "--- 7b. asc_unit_flag_mode / asc_store_l2_cache_mode / asc_relu_pre_mode 三个枚举包装在 9.1.0 头里不存在："
grep -rn "asc_unit_flag_mode\|asc_store_l2_cache_mode\|asc_relu_pre_mode" "$CANN/x86_64-linux/asc/include/" | head -5 || echo "    （无匹配 ⇒ 不存在）"
echo "--- 7c. 底层枚举本体在编译器自带头（bisheng）里，故传裸值等价："
sed -n '160,170p;215,226p' "$CANN/tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h"
echo

echo "### 8. 结论口径（不外推）"
echo "  ① 官方样例（基础 API 与 C API 两版）在 9.1.0 上**整编不通过**，缺的是 9.2 新增的符号/枚举包装；"
echo "  ② 本探针用「本地 constexpr 代替 ceil_div + 传裸值代替枚举包装」绕开，两个 executable 都编过 rc=0；"
echo "  ③ 这**只证明接口/编译面可用**，**不证明**「官方样例本身能在 9.1.0 跑通」——绕开处改变了源码；"
echo "  ④ 本探针真机读数（见 README 主体）与是否绕开无关：它直接调用 9.1.0 头里就有的"
echo "     asc_copy_ub2l1 / 基础 API DataCopy，绕开只影响 L0/L0C 那几行的参数写法。"
