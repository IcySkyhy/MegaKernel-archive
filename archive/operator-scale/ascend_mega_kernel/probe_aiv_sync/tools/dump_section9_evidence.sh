#!/usr/bin/env bash
# dump_section9_evidence.sh —— 生成 README §9「核验（实跑命令 → 输出）」列的**完整原始输出**
#
# 为什么需要它：§9 的核验列必须能"逐格复现"。凡标成"命令 → 输出"的格子，要么是**完整输出**，
# 要么是**明确标注过的子集**（给出完整输出行数 + 指向本文件）。本脚本把 §9 引用的每条命令跑一遍，
# 原样落盘到 evidence/section9_greps.txt，带 tag；README 每格用 `#tag` 指向它。
#
# 用法（本目录下）：bash tools/dump_section9_evidence.sh
#   产物：evidence/section9_greps.txt（进 git）
#   自检：LC_ALL=C 与 LC_ALL=C.UTF-8 各跑一次，两份输出应逐字节相同（locale 稳定性）
#
# 树的口径：m15/m20 这几个文件本 mission **未改动**（`git diff --name-only <base> HEAD -- m15_layer_loop m20_hyperconn` 为空），
# 故在 wt-64 工作树里跑 == 在基快照 89ce319 上跑；`m15_chain_host.h` **不在基快照里**（M6x 之后新增），
# 只能在 main checkout 上跑 —— 每段输出里显式标注 tree 与 commit。
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"                                     # probe_aiv_sync/（本工程）
WT="$(git -C "$ROOT" rev-parse --show-toplevel)" # wt-64 工作树根（m15/m20 在这里）
MAIN="/workspace/ascend_mega_kernel"            # main checkout（只读）
CANN="/usr/local/Ascend/cann-9.1.0"             # CANN 安装树（只读；编译器实际用的头文件）
OUT="$ROOT/evidence/section9_greps.txt"
TMP="$(mktemp)"
WT_COMMIT="$(git -C "$WT" rev-parse --short=12 89ce319)"
MAIN_COMMIT="$(git -C "$MAIN" rev-parse --short=12 HEAD)"

{
    echo "# §9 核验列 —— 完整原始输出（由 tools/dump_section9_evidence.sh 生成，勿手改）"
    echo "# 生成时间      : $(date -Is)"
    echo "# wt-64 工作树   : $WT（§9 引用的 m15/m20 文件本 mission 未改动 ⇒ 内容 == 基快照 $WT_COMMIT）"
    echo "# main checkout : $MAIN（commit $MAIN_COMMIT；仅用于基快照里不存在的文件）"
    echo "# CANN install  : $CANN（只读；IsSplitCubePipe 规则的权威头文件）"
    echo "# 每段格式      : [tag] + tree/commit + LC_ALL + 命令 + 完整 stdout + exit + 输出行数"
    echo "# 复现          : LC_ALL=C 与 LC_ALL=C.UTF-8 各跑一次，输出应逐字节相同"
    echo
} > "$OUT"

nseg=0
NF=0
run() {   # run <tag> <tree: WT|MAIN|DEVKIT> <cmd...>
    local tag="$1"; local tree="$2"; shift 2
    local dir commit
    case "$tree" in
        WT)     dir="$WT";     commit="$WT_COMMIT(基快照内容)";;
        MAIN)   dir="$MAIN";   commit="$MAIN_COMMIT(main)";;
        CANN)   dir="$CANN";   commit="(CANN 9.1.0 只读树)";;
    esac
    ( cd "$dir" && "$@" ) > "$TMP" 2>&1
    local rc=$?
    local n; n=$(wc -l < "$TMP")
    {
        echo "=== [$tag] tree=$tree commit=$commit LC_ALL=${LC_ALL:-<unset>}"
        echo "\$ $*"
        cat "$TMP"
        echo "[exit=$rc] [完整输出行数=$n]"
        echo
    } >> "$OUT"
    nseg=$((nseg + 1))
    if [ "$rc" -ne 0 ]; then
        NF=$((NF + 1))
        echo "!! 注意：段 [$tag] exit=$rc（非零退出 = 该段不是有效证据，见 README §9 口径）" >> "$OUT"
    fi
}

# ---- m20：ProcessAiv 的调用次数、dWS 生命周期、BarrierAiv 定义与调用点 ----
run m20_processaiv_dws   WT grep -n -e "ProcessAiv()" -e "dWS" m20_hyperconn/m20_hyperconn.asc
run m20_barrierAiv_def   WT grep -n -A6 "inline void BarrierAiv" m20_hyperconn/m20_hyperconn.asc
run m20_barrierAiv_calls WT grep -n "BarrierAiv<" m20_hyperconn/m20_hyperconn.asc
run m20_barrierAic_def   WT grep -n -A5 "inline void BarrierAic" m20_hyperconn/m20_hyperconn.asc
run m20_barrierAic_calls WT grep -n "BarrierAic<" m20_hyperconn/m20_hyperconn.asc
# ---- m15：层循环结构（m15_chain_host.h 只存在于 main）----
run m15_chain_loop       MAIN grep -n -e "for (uint32_t L = 0; L < nL" -e "H_LaunchChainLayer" m15_layer_loop/m15_chain_host.h
run m15_chain_sync_calls MAIN grep -n "H_Sync(C" m15_layer_loop/m15_chain_host.h
run m15_nl_def           WT grep -n "NL = 48" m15_layer_loop/m15_loop_layout.h
# ---- m15：S4 递推段 head 属主与 state 访问 ----
run m15_head_loop        WT grep -n "for (uint32_t h = bid" m15_layer_loop/m15_gdn_layer.h
run m15_state_access     WT grep -n "stateGm_" m15_layer_loop/m15_gdn_layer.h
run m15_head_loop_ctx    WT grep -n -B2 -A2 "for (uint32_t h = bid" m15_layer_loop/m15_gdn_layer.h
run m15_phase_boundary   WT grep -n "M15L_PhaseBoundaryAiv" m15_layer_loop/m15_layer_kernel.h
run m15_phase_body       WT grep -n -A8 "inline void M15L_PhaseBoundaryAiv" m15_layer_loop/m15_layer_kernel.h
# ---- m15：ws 分段偏移表（按构造不重叠）----
run m15_ws_segments      WT sed -n 284,297p m15_layer_loop/m15_gdn_resources.h
# ---- m15：AIC 侧会合（注意：m15 里是**内联** CrossCore 调用，没有 BarrierAic 包装；
#      名字叫 BarrierAic 的包装在 m20 —— 见 m20_barrierAic_* 两段）----
run m15_aic_inline WT grep -n -e "CrossCoreSetFlag<CC_MODE0" -e "CrossCoreWaitFlag<CC_MODE0" m15_layer_loop/m15_gdn_layer.h
# ---- 机制佐证：AIC 侧 pipe 白名单（devkit 头文件）----
run aic_splitcube_pipe   CANN grep -n -A8 "IsSplitCubePipe" asc/impl/basic_api/kernel_event.h
# ---- 机制佐证：消费型计数器的文档口径（本仓 + 厂商文档）----
run doc_counter_repo     WT grep -n "计数 15 次" docs/05-megakernel-design.md
run doc_counter_tpl      WT grep -n -e "计数范围为0-15" -e "减去1进行还原" /workspace/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md
run doc_counter_keyfeat  WT grep -n -e "计数器值增加为1" -e "计数器值减去1" /workspace/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md

echo "生成完毕：$OUT（$nseg 段，其中非零退出的段：$NF）"
[ "$NF" -eq 0 ] || { echo "有 $NF 段非零退出 —— 请修命令后重新生成（失败命令不得当证据）"; exit 1; }
