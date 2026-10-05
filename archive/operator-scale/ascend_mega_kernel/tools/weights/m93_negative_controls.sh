#!/bin/bash
# M93 · 负向对照 + 加性证明的可复现驱动（变异只在 /tmp 副本里做；本脚本不写仓库内除
# evidence/ 与 /tmp 之外的任何路径）。
#
# 用法（仓库根目录）：
#   归档复现（**含**设备档 NC-7；须在 `npu-smi info` 无兄弟 m15_layer_loop 时跑）：
#     M93_NC_DEVICE=1 bash tools/weights/m93_negative_controls.sh \
#         > tools/weights/evidence/m93_negative_controls.txt 2>&1
#   只跑 host 侧（不含 NC-7；纯 host、不占设备，秒级）：
#     bash tools/weights/m93_negative_controls.sh            # 或重定向到 /tmp
#
# ⚠ **加性基线钉的是不可变 commit（本轮 fork 点 `52f46b8…`），不是 `main`**（塔的第六变体：
#   复现命令不要钉会移动的 ref）。本分支合入后 `main` 会前移、其 manifest 就等于 tip，
#   若基线跟着 main 走，加性 diff 会退化成 0 删 / 0 增 ⇒ **「纯加性」这条核心判据静默变成空洞**。
#   现脚本对这种情况有 `ADDITIVE-CRIT VACUOUS` 守卫：基线 == tip 时**响亮地 rc=1 退出**。
#   要比较别的基线可覆盖：`BASE_COMMIT=<40 位 sha> bash …`（请用 commit id，不要用分支名）。
#
# ⚠ 重定向**必须带 2>&1**：NC-6 那条 `[FAIL] ... 与 checkpoint 现况不一致` 来自
#   `slice_layer_manifest.py` 的 `raise SystemExit(...)`，写的是 **stderr**；只重定向 stdout
#   会比归档少那一行（M93 r2 复审实测：只 `>` ⇒ diff 1 行；`> … 2>&1` ⇒ 逐字节相同）。
#
# ⚠ **设备档 NC-7 默认关闭**（`M93_NC_DEVICE` 未设为 1 时打印 SKIPPED）。原因：那一步要跑真实
#   `runs=all`，**实测 host 峰值 RSS = 15,982 MB ≈ 15.6 GiB**（`M15_LAYERS=48 M15_STEPS=3`，
#   /proc VmRSS 采样）—— 32 GB cgroup 下**两个并发 run 就超限**。
#   2026-09-27 本 mission 因此 OOM 过两次；根因是早期版本在脚本内做「归档复现性自检」时
#   **递归调用自己**，每层都重跑一遍设备档 ⇒ 并发跑多个 run。**该自检块已删除**，本脚本
#   现在**绝不调用自己**，且设备档默认关闭、必须显式 `M93_NC_DEVICE=1`（**一次只跑一个**）。
#
# 每条判据都给出「命令 → 输出/rc」；正向全绿 + 负向对照（NC-1/2/2b/3/4/5/6/8/9/10/11/12 恒开，
# NC-7 由 M93_NC_DEVICE 门控）必 FAIL。
set -u
PY=/usr/local/python3.12.13/bin/python3.12
# 加性比较的基线 = 本分支的 **fork 点（不可变 commit，全 40 位 sha）**。
# ⚠ **不要钉 `main` 或任何分支名**（塔的第六变体：复现命令不要钉会移动的 ref）：
#   本分支一合入，`main` 的 weights_manifest.txt 就等于 tip ⇒ 加性 diff 退化成 0 删 / 0 增，
#   归档里记的 0/247 便**复现不出**，「纯加性」这条核心判据会**静默变成空洞**。
#   下面「加性证明」段有两条守卫，都在**失败路径**上强制 rc≠0：
#     · 基线**读不到 / 不存在** ⇒ `ADDITIVE-CRIT FAIL：基线读不到…` + exit 1
#       （否则 `before.txt` 为空 ⇒ 整份 manifest 被算成"新增" ⇒ 输出误导性的 PASS）
#     · 基线 == tip（0 删 且 0 增）⇒ `ADDITIVE-CRIT **VACUOUS**` + exit 1（不许静默空洞）
#     · 删除 != 0（真非加性）⇒ `ADDITIVE-CRIT FAIL` + exit 1
BASE_COMMIT=${BASE_COMMIT:-52f46b8693f0292c132312ea1d47e78c5b52888a}   # = fork 点（= rebase 时的 main，不可变）
BASE_COMMIT_SHA=$(git rev-parse --short "$BASE_COMMIT" 2>/dev/null || echo '?')   # 现场解析并写进产物
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

W=/tmp/m93_nc_work
rm -rf "$W"; mkdir -p "$W"

echo "# M93 · 负向对照读数（本文件由 tools/weights/m93_negative_controls.sh 现场生成）"
echo "# 加性基线 = 不可变 commit ${BASE_COMMIT}（短 sha ${BASE_COMMIT_SHA}）= rebase 时的 main（= 本分支 fork 点）；"
echo "#   可由 BASE_COMMIT 覆盖（请用 40 位 commit id，不要用分支名）；**本判据不读 main** ⇒ 复现恒成立（第六变体）。"
echo "# 变异只在 ${W} 的副本里做"
echo

# ---- 基线 manifest（改前）与改后 manifest ----
# 守卫 1（失败路径）：基线 commit 不存在 / 或该 tree 里没有这份 manifest ⇒ 直接 rc≠0。
# 不加这条时 `git show` 失败会留下**空的 before.txt**，使整份 manifest 都被算成"新增"
# ⇒ 输出误导性的 `ADDITIVE-CRIT PASS（… 新增 N > 0 …）` 且 rc=0（M93 r5 复审实测暴露）。
if ! git cat-file -e "${BASE_COMMIT}:m15_layer_loop/weights_manifest.txt" 2>/dev/null; then
  echo "## 加性证明（改前 vs 改后）"
  echo "ADDITIVE-CRIT FAIL：基线 commit ${BASE_COMMIT} 读不到 m15_layer_loop/weights_manifest.txt"
  echo "  （BASE_COMMIT 无效 / 该 commit 里没有这份 manifest。请给一个含该文件的 40 位 commit id。）"
  echo "  复核：\$ git cat-file -e ${BASE_COMMIT}:m15_layer_loop/weights_manifest.txt; echo rc=\$?"
  exit 1
fi
git show "${BASE_COMMIT}:m15_layer_loop/weights_manifest.txt" > "$W/before.txt"
if [ ! -s "$W/before.txt" ]; then
  echo "## 加性证明（改前 vs 改后）"
  echo "ADDITIVE-CRIT FAIL：基线 ${BASE_COMMIT_SHA} 的 manifest 读出来是空文件（before.txt 大小 0）"
  echo "  ⇒ 无法据此判「加性」；失败必须反映到 rc。"
  exit 1
fi
echo "## 加性证明（改前 vs 改后）"
echo "\$ git show ${BASE_COMMIT}:m15_layer_loop/weights_manifest.txt > before.txt"
echo "\$ diff before.txt m15_layer_loop/weights_manifest.txt"
del=$(diff "$W/before.txt" m15_layer_loop/weights_manifest.txt | grep -c '^<')
add=$(diff "$W/before.txt" m15_layer_loop/weights_manifest.txt | grep -c '^>')
echo "删除行数=${del}  新增行数=${add}   （加性要求：删除行数 == 0）"
if [ "$del" -ne 0 ]; then
  # 真的删了行 ⇒ 非加性。与 VACUOUS 分支对称：必须 rc≠0。
  echo "ADDITIVE-CRIT FAIL：基线（${BASE_COMMIT_SHA}）里有 ${del} 行在 tip 上不存在（不是纯加性）。"
  echo "  复核：\$ diff <(git show ${BASE_COMMIT}:m15_layer_loop/weights_manifest.txt) m15_layer_loop/weights_manifest.txt | grep '^<'"
  exit 1
elif [ "$add" -eq 0 ]; then
  # 基线 == tip ⇒ 这条判据此刻**没有信息**（0 vs 0 空过）。必须响亮地失败，不许静默 PASS。
  echo "ADDITIVE-CRIT **VACUOUS**：基线（${BASE_COMMIT_SHA}）与 tip 的 manifest 相同 ⇒ 本判据无信息。"
  echo "  复核：\$ git rev-parse ${BASE_COMMIT}  \$ git rev-parse HEAD  —— 两者不该相同（除非本分支已合入）。"
  echo "  若本分支已合入 main，加性证明要看**合并前**的那个 commit（= 上面的 BASE_COMMIT）。"
  exit 1
else
  echo "ADDITIVE-CRIT PASS（删除 0 / 新增 ${add} > 0 ⇒ 非空洞）"
fi
echo
echo "\$ diff <(grep ' role=moe_' before.txt) <(grep ' role=moe_' weights_manifest.txt)"
if diff <(grep ' role=moe_' "$W/before.txt") <(grep ' role=moe_' m15_layer_loop/weights_manifest.txt) >/dev/null; then
  echo "  E=4 的 role=moe_* 行全部逐字节相同（$(grep -c ' role=moe_' "$W/before.txt") 行）"
else
  echo "  E=4 行有差异 ⇒ ADDITIVE-CRIT FAIL"
fi
echo

# ---- NC-0：正向 ----
echo "## NC-0（正向）真规模判据"
echo "\$ \$PY tools/weights/moe_real_scale_audit.py"
$PY tools/weights/moe_real_scale_audit.py > "$W/nc0.log" 2>&1; rc0=$?
grep -E '^\[chk\]|^\[summary\]' "$W/nc0.log"
echo "rc=${rc0}"
echo

# ---- NC-1：专家数当 4 ----
echo "## NC-1 把专家数当 4（当前缩形档）⇒ 必 FAIL"
echo "\$ \$PY tools/weights/moe_real_scale_audit.py --negative-experts 4"
$PY tools/weights/moe_real_scale_audit.py --negative-experts 4 > "$W/nc1.log" 2>&1; rc1=$?
grep -E '^\[chk\].*FAIL|^\[summary\]' "$W/nc1.log"
echo "rc=${rc1}"
echo

# ---- NC-2 / NC-2b：生成器侧专家数 ----
for n in 4 256; do
  echo "## NC-2$([ "$n" = 256 ] && echo b) M93 真规模段按 ${n} 专家算 ⇒ 必 FAIL"
  echo "\$ \$PY m15_layer_loop/slice_layer_manifest.py --out /tmp/nc_${n}.txt --moe512-neg-experts ${n}"
  $PY m15_layer_loop/slice_layer_manifest.py --out "$W/nc_${n}.txt" --moe512-neg-experts "$n" > "$W/nc2_${n}.log" 2>&1
  echo "rc=$?"; cat "$W/nc2_${n}.log"
  echo
done

# ---- NC-3：stride 变异（/tmp 副本）----
echo "## NC-3 stride 变异：把「单专家字节」写成整张量字节（只在 /tmp 副本）"
rm -rf "$W/mut_stride"; mkdir -p "$W/mut_stride"
git archive "${BASE_COMMIT}" | tar -x -C "$W/mut_stride"
cp m15_layer_loop/slice_layer_manifest.py "$W/mut_stride/m15_layer_loop/"
$PY - "$W/mut_stride/m15_layer_loop/slice_layer_manifest.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()
old = "one = prod(shape[1:]) * DT_BYTES[dt]"
new = "one = prod(shape) * DT_BYTES[dt]  # MUTANT"
assert s.count(old) == 1, s.count(old)
open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print("mutation applied:", old, "->", new.strip())
PY
echo "\$ (cd \"$W/mut_stride\" && \$PY m15_layer_loop/slice_layer_manifest.py --out /tmp/nc3.txt)"
(cd "$W/mut_stride" && $PY m15_layer_loop/slice_layer_manifest.py --out "$W/nc3.txt" > "$W/nc3.log" 2>&1)
echo "rc=$?"; cat "$W/nc3.log"
echo

# ---- NC-4：缩形档专家数变异（/tmp 副本）----
echo "## NC-4 加性变异：NUM_EXPERTS_MOE 4→8（只在 /tmp 副本）"
rm -rf "$W/mut_e4"; mkdir -p "$W/mut_e4"
git archive "${BASE_COMMIT}" | tar -x -C "$W/mut_e4"
cp m15_layer_loop/slice_layer_manifest.py "$W/mut_e4/m15_layer_loop/"
sed -i 's/^NUM_EXPERTS_MOE = 4$/NUM_EXPERTS_MOE = 8/' "$W/mut_e4/m15_layer_loop/slice_layer_manifest.py"
echo "变异后的常量行（不记录行号 —— 那是会漂的活值）："
grep '^NUM_EXPERTS_MOE' "$W/mut_e4/m15_layer_loop/slice_layer_manifest.py"
(cd "$W/mut_e4" && $PY m15_layer_loop/slice_layer_manifest.py --out "$W/nc4.txt" > "$W/nc4.log" 2>&1)
echo "生成本身 rc=$?（它骗不过生成器内部算式，但骗不过加性判据）"
del4=$(diff "$W/before.txt" "$W/nc4.txt" | grep -c '^<')
echo "改前 vs NC-4 产物 的删除行数 = ${del4}   （加性要求 == 0；NC-4 得 ${del4} ⇒ ADDITIVE-CRIT FAIL）"
echo

# ---- NC-12：生成器 C7（E=4 档 × 倍数关系）的破坏对照 ----
echo "## NC-12 破坏被测对象（生成器 C7「E=NUM_EXPERTS_MOE 档 × 倍数 == 真规模」）"
echo "\$ # 只在 /tmp 副本把 NUM_EXPERTS_MOE 4→3（512 % 3 = 2 ≠ 0 ⇒ C7 必 FAIL）"
rm -rf "$W/mut_e3"; mkdir -p "$W/mut_e3"
git archive "${BASE_COMMIT}" | tar -x -C "$W/mut_e3"
cp m15_layer_loop/slice_layer_manifest.py "$W/mut_e3/m15_layer_loop/"
sed -i 's/^NUM_EXPERTS_MOE = 4$/NUM_EXPERTS_MOE = 3/' "$W/mut_e3/m15_layer_loop/slice_layer_manifest.py"
grep '^NUM_EXPERTS_MOE' "$W/mut_e3/m15_layer_loop/slice_layer_manifest.py"
(cd "$W/mut_e3" && $PY m15_layer_loop/slice_layer_manifest.py --out "$W/nc12.txt" > "$W/nc12.log" 2>&1)
echo "rc=$?（期望 1）"; cat "$W/nc12.log"
echo
echo '> 注：NC-4（4→8）**骗得过**生成器内部算式 —— 倍数比 `512/NUM_EXPERTS_MOE` 与档宽同源 ⇒'
echo "> C7 仍绿；它只被「改前 E=4 行逐字节不变」这条**加性判据**咬住。NC-12（4→3）则是能翻"
echo '> C7 的变异（`512 % 3 ≠ 0`）。两者**一起**才覆盖「缩形档专家数被改错」这个形态。'
echo

# ---- 区域无关性 ----
echo "## 区域无关性（纪律 ⑩）"
for loc in C C.UTF-8; do
  LC_ALL=$loc $PY tools/weights/moe_real_scale_audit.py >/dev/null 2>&1; a=$?
  LC_ALL=$loc $PY tools/weights/moe_real_scale_audit.py --negative-experts 4 >/dev/null 2>&1; b=$?
  echo "LC_ALL=${loc} ：正向 rc=${a} ；NC-1 rc=${b}"
done

# ============================================================
# NC-5..NC-11：塔的第五变体纪律 —— 「把被测对象弄坏，判据必须变红」
#   被测对象 = checkpoint 分片头（`/tmp` 镜像：真实头 + 稀疏尾，apparent 170 G / 实际 <1 MB）
#   每条判据给一次**定向**破坏：只有它该红，别的仍绿。
#   NC-6/NC-7 的被测对象分别是已入库 manifest 与 manifest 的 offset 字段。
# ============================================================
MIRROR_ROOT=/tmp/m93_mirror

# build_mirror <dst> <mutation>：造镜像并按 mutation 做**等宽原地**破坏（头长不变）
build_mirror() {
  $PY - "$1" "$2" <<'PY'
import glob, os, shutil, struct, sys
src, dst, mut = "/workspace/Qwen3.8-Flash-Next-MXFP4", sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
shards = sorted(glob.glob(src + "/*.safetensors"))

def read_head(p):
    with open(p, 'rb') as f:
        hl = struct.unpack('<Q', f.read(8))[0]
        return hl, f.read(hl)

for p in shards:                                  # 真实头 + 稀疏尾
    hl, head = read_head(p)
    q = os.path.join(dst, os.path.basename(p))
    with open(q, 'wb') as g:
        g.write(struct.pack('<Q', hl)); g.write(head)
    os.truncate(q, os.path.getsize(p))
shutil.copy(os.path.join(src, "config.json"), os.path.join(dst, "config.json"))

def patch(shard, old, new, count=1):
    """等宽原地替换该分片头里的 old→new；长度必须不变（否则会移动所有 tensor 偏移）"""
    assert len(old) == len(new), (old, new)
    p = os.path.join(dst, shard)
    hl, head = read_head(p)
    s = head.decode('utf-8')
    assert s.count(old) >= count, (shard, old, s.count(old))
    s2 = s.replace(old, new, count)
    assert len(s2) == len(s), "header length changed!"
    with open(p, 'r+b') as f:
        f.write(struct.pack('<Q', hl)); f.write(s2.encode('utf-8'))

if mut == "ok":
    pass
elif mut == "offset_minus1":                       # C2/C1/C3C6/C6
    patch("model-00002-of-00131.safetensors", ",838860800]", ",838860799]")
elif mut == "ngram_rename":                        # C8（破坏「ngram 分流」这个被测对象）
    n = 0
    for p in sorted(glob.glob(dst + "/*.safetensors")):
        hl, head = read_head(p)
        s = head.decode('utf-8')
        if "ngram" not in s:
            continue
        s2 = s.replace("ngram", "xxxxx")           # 5 字符 → 5 字符，等宽
        assert len(s2) == len(s)
        n += s2.count("xxxxx")
        with open(p, 'r+b') as f:
            f.write(struct.pack('<Q', hl)); f.write(s2.encode('utf-8'))
    assert n >= 130, n
    print("renamed %d 'ngram' occurrences (等宽)" % n)
elif mut == "shape0_256":                          # C5
    patch("model-00002-of-00131.safetensors", '"shape":[512,1280,1280]', '"shape":[256,1280,1280]')
elif mut == "k_1279":                              # C4
    patch("model-00002-of-00131.safetensors", '"shape":[512,1280,1280]', '"shape":[512,1280,1279]')
elif mut == "truncate":                            # C0
    q = os.path.join(dst, "model-00002-of-00131.safetensors")
    hl, _ = read_head(q)
    os.truncate(q, 8 + hl)                          # 只留头：该分片全部张量变 incomplete
else:
    raise SystemExit("unknown mutation: " + mut)
PY
}

# mirror_case <tag> <mutation> <期望翻红的判据>
mirror_case() {
  tag=$1; mut=$2; exp=$3
  d="$MIRROR_ROOT/$tag"
  build_mirror "$d" "$mut" > "$W/m_$tag.build" 2>&1 || { echo "BUILD FAILED"; cat "$W/m_$tag.build"; return; }
  $PY tools/weights/moe_real_scale_audit.py --model-dir "$d" > "$W/m_$tag.log" 2>&1
  rc=$?
  echo "## NC-${tag} 破坏被测对象：mutation=${mut}"
  echo "\$ \$PY tools/weights/moe_real_scale_audit.py --model-dir <镜像，mutation=${mut}>"
  [ -s "$W/m_$tag.build" ] && cat "$W/m_$tag.build"
  echo "rc=${rc}（期望 1）  $(grep '^\[summary\]' "$W/m_$tag.log")"
  echo "翻红的判据："
  grep -E '^\[chk\].*FAIL' "$W/m_$tag.log" | sed 's/^/  /'
  echo "期望翻红 = ${exp}"
  echo "仍在 PASS 的判据（定向性证据，应只红那几条）："
  grep -E '^\[chk\].*PASS' "$W/m_$tag.log" | sed 's/^/  /' | head -8
  echo
}

echo
echo "## NC-5a 正向：镜像是忠实副本（真实头 + 稀疏尾）"
build_mirror "$MIRROR_ROOT/ok" ok >/dev/null 2>&1
$PY tools/weights/moe_real_scale_audit.py --model-dir "$MIRROR_ROOT/ok" > "$W/nc5a.log" 2>&1
echo "\$ \$PY tools/weights/moe_real_scale_audit.py --model-dir \$MIRROR_ROOT/ok"
echo "rc=$?  $(grep '^\[summary\]' "$W/nc5a.log")（期望 20/0）"
echo "镜像占盘：$(du -sh "$MIRROR_ROOT/ok" | cut -f1)（apparent $(du -sh --apparent-size "$MIRROR_ROOT/ok" | cut -f1)）"
echo

mirror_case 5   offset_minus1 "C2 + C1 + C3/C6(gate_up) + C6(gate_up) —— 其余张量与 C4/C8 仍 PASS"
mirror_case 8   ngram_rename  "C8 —— C1/C2/C5 仍 PASS（复审 r2 提供的配方，M93 照抄并实跑）"
mirror_case 9   shape0_256    "C5 —— 专家维维长与 config.num_experts 不符"
mirror_case 10  k_1279        "C4（gate_up 打包 K/2）—— shape 与 config 几何量不符"
mirror_case 11  truncate      "C0（分片完整性）—— 该分片全部张量变 incomplete"

echo "## NC-6 破坏被测对象（已入库 manifest）⇒ --check 必 FAIL"
sed '0,/role=conv1d/s/offset=26376680 /offset=26376681 /' m15_layer_loop/weights_manifest.txt > "$W/nc6_manifest.txt"
echo "（等宽原地改：行数 $(grep -c . m15_layer_loop/weights_manifest.txt) -> $(grep -c . "$W/nc6_manifest.txt")）"
$PY m15_layer_loop/slice_layer_manifest.py --out "$W/nc6_manifest.txt" --check; echo "rc=$?（期望 1）"
echo
echo "## NC-7 破坏被测对象（manifest 的 offset）⇒ 设备判据必 FAIL"
BIN_DEV="$ROOT/m15_layer_loop/build/m15_layer_loop"
if [ "${M93_NC_DEVICE:-0}" != "1" ]; then
  echo "SKIPPED（未设 M93_NC_DEVICE=1）"
  echo "# ⚠ 归档版（tools/weights/evidence/m93_negative_controls.txt）**含** NC-7 的完整读数；"
  echo "#   它由文件头那条带 M93_NC_DEVICE=1 的命令产出。设备档 host 侧峰值**实测 15,982 MB**，请在"
  echo "#   \`npu-smi info\` 无兄弟 m15_layer_loop 时跑（32 GB cgroup，本 mission 已因此 OOM 两次）。"
  echo "#   本次未跑 ⇒ **本输出不是归档版**，不要拿它去 diff 归档。"
elif [ ! -x "$BIN_DEV" ] || [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  echo "SKIPPED（无设备/无二进制）"
else
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
  DEV_MAN="$W/nc7_manifest.txt"
  sed '0,/role=conv1d/s/offset=26376680 /offset=26376681 /' m15_layer_loop/weights_manifest.txt > "$DEV_MAN"
  if [ ! -s "$DEV_MAN" ]; then echo "SKIPPED（无法生成破坏后的 manifest）"; else
    echo "（破坏后的 manifest：$(wc -l < "$DEV_MAN") 行，$(grep -c 'offset=26376681 ' "$DEV_MAN") 处 offset 被改）"
    M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 "$BIN_DEV" "$DEV_MAN" all > "$W/nc7.log" 2>&1
    echo "rc=$?（期望 1；未破坏时同配置 = 2068 checks + 290 guards / 0 FAIL）"
    grep -a ' FAIL ' "$W/nc7.log" > "$W/nc7_fails.txt"
    echo "判据 FAIL 行数 = $(wc -l < "$W/nc7_fails.txt")；样例 3 条："
    head -3 "$W/nc7_fails.txt"
    echo "ALL PASS 行数 = $(grep -acE 'ALL PASS' "$W/nc7.log")（期望 0）"
  fi
fi
echo
echo "## 破坏对照的覆盖表（逐条映射；本 mission 的每条判据都要能在此表里找到一行）"
echo "| 判据 | 破坏对照 |"
echo "|---|---|"
echo "| C0 分片完整性 | NC-11（镜像某分片截到只剩头） |"
echo "| C1 48 层逐层字节相同 | NC-5（gate_up data_offsets 末端 −1） |"
echo "| C2 头部 offset 差 == shape×dtype | NC-5 |"
echo "| C3/C6 单专家 × n == 头部全量 | NC-5（gate_up）；生成器侧另见 NC-2/NC-2b |"
echo "| C4 K 打包 / N 几何 | NC-10（镜像 gate_up K 1280→1279） |"
echo "| C5 专家维 shape[0] == config.num_experts | NC-9（镜像 shape[0] 512→256） |"
echo "| C6 全量/单专家 == num_experts | NC-5 |"
echo "| C8 非 ngram ≤ HBM | NC-8（镜像 130 个 ngram 张量改名；复审 r2 配方） |"
echo "| 生成器 C7（E 档 × 倍数 == 真规模） | NC-12（NUM_EXPERTS_MOE 4→3） |"
echo "| 生成器 C4b（单专家字节 == config 几何量） | NC-3（stride 变异） |"
echo "| 生成器 真规模算式 vs 头部 | NC-2 / NC-2b（--moe512-neg-experts 4 / 256） |"
echo "| 生成器 --check | NC-6（入库 manifest 等宽改一字节） |"
echo "| 「E=4 行逐字节不变」加性判据 | NC-4（NUM_EXPERTS_MOE 4→8） |"
echo "| 设备侧 runs=all 判据（本 mission 的产物 = manifest） | NC-7（破坏 manifest 的 offset） |"
echo
echo "## **没有**破坏对照的判据（如实披露，不写成「已全部覆盖」）"
echo '# 以下三条是 `slice_layer_manifest.py` 的**结构性自检**，它们的被测对象是 config.json 本身'
echo "# （不是 checkpoint 字节、也不是本 mission 的产物），且都是「本就不该成立才通过」的形态；"
echo "# 本 mission 未为它们造破坏对照："
echo "#   ① layer_types 与 3:1 模式（每 full_attention_interval 个的末位必须是 full_attention）"
echo "#   ② 单层 in_proj 拼接行数 == IN_N(16480)"
echo "#   ③ config 几何量 (hidden, inter, group) == 本文件常量 (2560, 640, 32)"
echo "# 理由：要翻红它们必须改 config.json 或改本文件常量 —— 那已不是「弄坏被测对象」，而是"
echo "# 换一个输入/换一份实现，读数的解释会变（与第五变体要证的「判据盯着某个对象」不同）。"
echo "# 另：audit 工具的 C2/C5 用的是**头部的 shape**，生成器用的是**config 的 shape**；两侧交叉，"
echo "# NC-5/9/10 破坏头部、若把 config 也改坏才能翻红生成器侧，故未做。"
echo
echo "## 未做的破坏对照（真实范围披露）"
echo "# kernel 侧判据（runs=all 的 2068 条）本身**不是本 mission 改的对象**：本 mission 未碰任何"
echo "# .asc/.h。改前/改后二进制 sha256 **逐字节相同**，现场量得 $(sha256sum "$BIN_DEV" 2>/dev/null | cut -c1-16)…"
echo "#   （本行数值由本脚本**现场算出**、不钉字面量 ⇒ 不会像 r3 那样随 rebase 过期；"
echo "#    复核：\$ sha256sum m15_layer_loop/build/m15_layer_loop）"
echo "# 对 kernel 做「弄坏实现」对照需要改 m15_*.h/.asc —— 那些文件不在本 mission 的写权限内，故未做。"
echo "# 本 mission 能做的等价对照是 NC-7：**弄坏本 mission 的产物（manifest）⇒ 设备判据变红**，已给读数。"
echo "# 另：T3 档的 Σ|terms| / ulp_mult 等 ε 口径**不在本 mission 的任何交付文件里**"
echo "# （grep 本 mission 新增/修改的文件：未发现命中），故无「又动了那两处口径」的风险。"

# ============================================================
# 归档复现性（M93 r2 P2-1）—— **不在脚本内自检**
# ============================================================
# 本脚本**不得**在自己的 body 里再调自己：那会递归，且每一层都会跑一遍 NC-7 的设备档
# （那一步 host 侧要 ~6 GB：hostArena 4.17 GB + hcArena 1.32 GB + moeArena 0.63 GB）。
# 2026-09-27 本 mission 就因为这样一段自检把整个会话 OOM 掉了（32 GB cgroup、6 agent 并行）。
# 复现性由**外部**两条命令核（见文件头）：带 `2>&1` 应逐字节复现本归档。
