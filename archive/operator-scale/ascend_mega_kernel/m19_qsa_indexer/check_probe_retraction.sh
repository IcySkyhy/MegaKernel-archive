#!/usr/bin/env bash
# M156 撤回守卫：断言 probe_aic_blockidx 的撤回状态成立；任一断言不成立即非零退出。
# 用法（仓库根目录）：bash m19_qsa_indexer/check_probe_retraction.sh
# 说明：本脚本不碰设备，只读源码/文档；「能传播失败」= 任一断言失败则 exit 非零。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M19="$ROOT/m19_qsa_indexer"
rc=0

fail() { echo "FAIL: $*"; rc=1; }
ok()   { echo "ok:   $*"; }

# 1. 目标不在构建
if grep -q 'add_executable(probe_aic_blockidx' "$M19/CMakeLists.txt"; then
  fail "CMakeLists.txt 仍含 add_executable(probe_aic_blockidx"
else
  ok "CMakeLists.txt 无 add_executable(probe_aic_blockidx"
fi
if grep -Eq 'foreach\(tgt .*probe_aic_blockidx' "$M19/CMakeLists.txt"; then
  fail "foreach 名单仍含 probe_aic_blockidx"
else
  ok "foreach 名单不含 probe_aic_blockidx"
fi

# 2. 源文件带撤回标记，且无任何非注释行（= 不再有可编译的 UB 暂存通路）
if grep -q 'RETRACTED' "$M19/probe_aic_blockidx.asc"; then
  ok "源文件含 RETRACTED 撤回标记"
else
  fail "源文件缺 RETRACTED 撤回标记"
fi
if grep -vE '^\s*(/\*|\*|//)' "$M19/probe_aic_blockidx.asc" | grep -q .; then
  fail "源文件仍含非注释行（可能有可编译代码）"
else
  ok "源文件为纯注释（无非注释行）"
fi

# 3. manifest 的旧措辞恰好 1 次，且只出现在含更正解读「摆位不成立」的那一行
#    （只判「带『更正』二字即豁免」会被「追加一行 + 带更正」绕过 —— M86 后的打标即豁免形态）
old_count=$(grep -c '全项目级负结果' "$M19/evidence/dump_manifest.md")
if [ "$old_count" -ne 1 ]; then
  fail "dump_manifest.md 旧措辞出现 $old_count 次（期望恰 1 次）"
elif grep '全项目级负结果' "$M19/evidence/dump_manifest.md" | grep -q '摆位不成立'; then
  ok "dump_manifest.md 旧措辞恰 1 次且在同行的更正语境"
else
  fail "dump_manifest.md 该旧措辞行不含更正解读「摆位不成立」"
fi
if grep -q '摆位不成立' "$M19/evidence/dump_manifest.md"; then
  ok "dump_manifest.md 含更正解读「摆位不成立」"
else
  fail "dump_manifest.md 缺更正解读「摆位不成立」"
fi

# 4. README §5.4 存在：标题须以「### 5.4 + 空白」收尾（排除 ### 5.4x / ### 5.40 这类前缀匹配），
#    且必须同时命中 §5.4 正文里的锚点句 —— 单查标题前缀会被「改标题名 + 删正文」绕过。
readme_anchor='本探针无法回答，已从构建移除'
if ! grep -qE '^### 5\.4[[:space:]]' "$M19/README.md"; then
  fail "README 缺 §5.4 标题（须匹配 ^### 5\\.4[[:space:]]）"
elif ! grep -qF "$readme_anchor" "$M19/README.md"; then
  fail "README §5.4 正文锚点缺失：$readme_anchor"
else
  ok "README 含 §5.4（标题边界 + 正文锚点）"
fi

if [ "$rc" -ne 0 ]; then echo "== RETRACTION GUARD: FAIL =="; else echo "== RETRACTION GUARD: PASS =="; fi
exit "$rc"
