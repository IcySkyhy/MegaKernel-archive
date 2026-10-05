#!/usr/bin/env bash
# 复现三档 dump 并校验 sha256 —— 补 P2-7 的"生成命令 + 校验和"证据，并在校验失败时**非零退出**。
#
# 背景：data/ 归档了 q/out/gp/ws*/dbg* 等，但 **seq=4096 的 k/v 各 4MB 未入库**（避免仓库膨胀）；
# 它们由 host 侧确定性生成（Hash3、无 rand 状态），故"生成命令 + 校验和"即是等价证据：
#   data/dump_sha256.txt              —— 已归档文件的 sha256
#   data/dump_sha256_regenerable.txt  —— 未归档但可确定性重跑得到的文件（seq=4096 的 k/v）
#
# 用法：
#   data/regen_4096_kv.sh                     # 新建构建目录 /tmp/m10_regen，跑默认三档
#   data/regen_4096_kv.sh /tmp/m10            # 复用已有构建目录
#   CASES=4096 data/regen_4096_kv.sh /tmp/m10 # 只跑某档（清单会按本次真正生成的 case 过滤）
#
# 语义（reviewer r3 的 P2-4）：校验**只覆盖本次真正生成的那些 case**（避免"拿没生成的文件去 -c 而报一堆
# open or read 失败"的误报），且**任何一项校验失败都会 exit 非零**（不允许"报着失败还退 0"）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-/tmp/m10_regen}"
CASES="${CASES:-256,300,4096}"
DUMPDIR="${DUMPDIR:-${BUILD}/dumps}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [[ ! -x "${BUILD}/m10_attn_decode" ]]; then
    echo "[regen] 构建 ${BUILD}"
    cmake -B "${BUILD}" -S "${ROOT}" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "${BUILD}" -j4
fi

mkdir -p "${DUMPDIR}"
cd "${DUMPDIR}"
echo "[regen] M10_CASES=${CASES} → dump 到 ${DUMPDIR}"
M10_CASES="${CASES}" "${BUILD}/m10_attn_decode" | tail -3

# 按本次跑的 case 过滤清单（文件名含 _s<case>_）；返回过滤后的清单路径与项数
filter_manifest() {   # $1=清单  $2=输出临时文件
    : > "$2"
    while IFS= read -r line; do
        [[ -z "${line}" || "${line:0:1}" == "#" ]] && continue
        fname="${line##* }"
        for c in ${CASES//,/ }; do
            if [[ "${fname}" == *"_s${c}_"* ]]; then printf '%s\n' "${line}" >> "$2"; break; fi
        done
    done < "$1"
}

FAILED=0
for man in dump_sha256.txt dump_sha256_regenerable.txt; do
    tmp="$(mktemp)"
    filter_manifest "${ROOT}/data/${man}" "${tmp}"
    n="$(grep -c . "${tmp}" || true)"
    if [[ "${n}" -eq 0 ]]; then
        echo "[regen] ❌ ${man}：按 CASES=${CASES} 过滤后没有任何可校验项——检查 CASES 是否正确"
        FAILED=1
    elif sha256sum -c "${tmp}" --quiet; then
        echo "[regen] ✅ ${man}（${n} 项一致）"
    else
        echo "[regen] ❌ ${man}：校验失败（上面列出的文件与清单不符）"
        FAILED=1
    fi
    rm -f "${tmp}"
done

echo "[regen] dump 目录：${DUMPDIR}"
if [[ "${FAILED}" -ne 0 ]]; then
    echo "[regen] 结论：❌ 校验未全部通过"
    exit 1
fi
echo "[regen] 结论：✅ 全部一致"
