#!/bin/bash
# M30 (round-2 fix): archive the upstream vllm-ascend / vLLM reference facts.
#
# Why this exists: round-1 of the M30 review found that README §6.1 claimed a branch
# listing for github.com/vllm-project/vllm-ascend that no archived evidence backed, and
# that the conclusion drawn from it ("no vllm-ascend targets vLLM >= 0.29") was false.
# Root cause: a `grep -oE 'releases/v[0-9.]+$'` filter silently dropped every branch
# ending in "rc" (e.g. releases/v0.29.0rc). This script prints the RAW ref listings so
# no filter can hide a ref again, and pins every claim to a file in evidence/.
#
# Read-only: clones a shallow copy of the upstream repo into /tmp (never into the
# workspace) and only reads the /workspace checkouts.
#
#   bash baseline_env/scripts/collect_upstream_refs.sh [outdir]
set -u
OUT=${1:-baseline_env/evidence}
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd) || exit 1
CLONE=${CLONE:-/tmp/va-refs}
UPSTREAM=https://github.com/vllm-project/vllm-ascend.git
FORK=https://gitcode.com/liruixin_dvc/vllm-ascend.git
LANES="releases/v0.29.0rc releases/v0.28.0rc"

# A failed network call must never clobber a previously good archive: build into a temp
# file and only move it into place when the run completed. (Round-2: a github outage
# truncated evidence/11 mid-run, which is exactly what this guards against.)
net() {
  for _ in 1 2 3; do
    out=$(timeout 60 git "$@" 2>/dev/null) && [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
    sleep 8
  done
  echo "NETWORK_UNAVAILABLE: git $*"
  return 1
}
TMP=$(mktemp)

# shallow single-branch clone: downloads blobs in one pack (~83 MB, seconds), so the
# greps below run locally instead of fetching hundreds of blobs one by one.
if [ ! -d "$CLONE/.git" ]; then
  net clone --depth 1 --single-branch --branch main "$UPSTREAM" "$CLONE" || exit 1
fi
cd "$CLONE" || exit 1
for b in $LANES; do
  git rev-parse --verify -q "refs/heads/$b" >/dev/null || net fetch --depth 1 origin "$b:$b"
done

{
  echo "### collected: $(date -Is)"
  echo "### upstream: $UPSTREAM   fork: $FORK"
  echo "### local shallow copy: $CLONE (git clone --depth 1 --single-branch --branch main)"
  echo
  echo "## 1. upstream HEADS — RAW, unfiltered"
  HEADS=$(net ls-remote --heads "$UPSTREAM" | sed 's#refs/heads/##' | sort -k2)
  printf '%s\n' "$HEADS"
  echo "  (count = $(printf '%s\n' "$HEADS" | grep -cE '^[0-9a-f]{40}') heads — same command re-run later is"
  echo "   expected to give the same number; report this count, not a remembered one)"
  echo
  echo "## 2. upstream TAGS — RAW (peeled ^{} entries dropped)"
  TAGS=$(net ls-remote --tags "$UPSTREAM" | sed 's#refs/tags/##' | sort -V -k2)
  if printf '%s' "$TAGS" | grep -q NETWORK_UNAVAILABLE; then
    echo "  (github tags endpoint unreachable at collection time)"
    echo "  newest tag — taken from the archived listing, NOT from memory"
    echo "  (a hardcoded 'newest tag' here would go stale the next time upstream"
    echo "   releases; that was the M51 class this line is fixing):"
    grep -A4 'newest tags on the upstream project' "$OUT/02-version-matrix.txt" 2>/dev/null | head -7
  else
    printf '%s\n' "$TAGS" | grep -v '\^{}'
    echo "  (count = $(printf '%s\n' "$TAGS" | grep -v '\^{}' | grep -cE '^[0-9a-f]{40}') tags;"
    echo "   raw ls-remote --tags lines including peeled ^{} = $(printf '%s\n' "$TAGS" | grep -cE '^[0-9a-f]{40}'))"
  fi
  echo
  echo "## 3. fork HEADS — RAW"
  FORKHEADS=$(net ls-remote --heads "$FORK" | sed 's#refs/heads/##')
  printf '%s\n' "$FORKHEADS"
  echo "  (count = $(printf '%s\n' "$FORKHEADS" | grep -cE '^[0-9a-f]{40}') heads)"
  echo
  echo "## 4. what vLLM version each plugin lane targets"
  echo "### requirements.txt / [build-system].requires are BUILD-TIME pins. The target vLLM"
  echo "### version is VLLM_TAG in Dockerfile / .github/vllm-release-tag.commit."
  for base in main $LANES; do
    echo "--- $base  ($(git log --oneline -1 "$base" 2>/dev/null | cut -c1-60))"
    echo -n "  .github/vllm-release-tag.commit   : "; git show "$base:.github/vllm-release-tag.commit" 2>&1 | head -1
    echo -n "  .github/vllm-main-verified.commit : "; git show "$base:.github/vllm-main-verified.commit" 2>&1 | head -1
    echo -n "  Dockerfile ARG VLLM_TAG           : "; git show "$base:Dockerfile" 2>/dev/null | grep -m1 'ARG VLLM_TAG='
    echo -n "  requirements.txt torch pins       : "
    git show "$base:requirements.txt" 2>/dev/null | grep -E "^(torch|torch-npu|triton-ascend)=" | tr '\n' ' '; echo
    echo -n "  vllm_ascend/ files w/ qwen4_exp in the PATH : "
    git ls-tree -r --name-only "$base" 2>/dev/null | grep -ci "qwen4_exp"
    echo -n "  vllm_ascend/ FILES matching content qwen4_exp (git grep -l | wc -l) : "
    git grep -il "qwen4_exp" "$base" -- vllm_ascend 2>/dev/null | wc -l
    echo -n "  vllm_ascend/ OCCURRENCES of qwen4_exp (git grep -o | wc -l) : "
    git grep -io "qwen4_exp" "$base" -- vllm_ascend 2>/dev/null | wc -l
    echo -n "  vllm_version_is(\"0.29.0\") FILES matching (git grep -l | wc -l) : "
    git grep -l 'vllm_version_is("0.29.0")' "$base" -- vllm_ascend 2>/dev/null | wc -l
    echo -n "  vllm_version_is(\"0.29.0\") OCCURRENCES (git grep -o | wc -l) : "
    git grep -o 'vllm_version_is("0.29.0")' "$base" -- vllm_ascend 2>/dev/null | wc -l
  done
  echo
  echo "## 5. BOTTOM LINE — does ANY Ascend plugin tree implement qwen4_exp?"
  echo "### (all counts below are FILES matching, i.e. 'git grep -l | wc -l')"
  for base in main releases/v0.29.0rc; do
    echo -n "  upstream $base   whole-tree FILES matching qwen4_exp : "
    git grep -il "qwen4_exp" "$base" 2>/dev/null | wc -l
  done
  echo -n "  upstream main    whole-tree OCCURRENCES of qwen4_exp : "
  git grep -io "qwen4_exp" main 2>/dev/null | wc -l
  echo -n "  upstream main    FILES matching VLLM_ASCEND_ENABLE_QSA : "
  git grep -l "VLLM_ASCEND_ENABLE_QSA" main 2>/dev/null | wc -l
  echo "  (0 under vllm_ascend/ = the upstream 'QSA support' is documentation-first; no code carries it)"
  echo -n "  ... of those, FILES under vllm_ascend/ (code) : "
  git grep -l "VLLM_ASCEND_ENABLE_QSA" main -- vllm_ascend 2>/dev/null | wc -l
  echo
  echo "## 6. the vLLM commits those lanes pin — do they contain qwen4_exp?"
  V=/workspace/vllm
  for c in ced6857afa0ea7b2e3f0846a62e1394e90f15607; do
    echo "--- $c"
    echo -n "  object type in $V            : "; git -C "$V" cat-file -t "$c" 2>&1 | head -1
    echo -n "  git describe --tags          : "; git -C "$V" describe --tags "$c" 2>&1 | head -1
    echo -n "  files under vllm/models/qwen4_exp/ : "
    git -C "$V" ls-tree -r --name-only "$c" 2>/dev/null | grep -c '^vllm/models/qwen4_exp/'
  done
  echo "--- releases/v0.29.0rc pins (full sha, from markdown frontmatter above):"
  git show releases/v0.29.0rc:.github/vllm-main-verified.commit 2>/dev/null | head -1 | while read -r c; do
    echo -n "  describe --tags : "; git -C "$V" describe --tags "$c" 2>&1 | head -1
    echo -n "  files under vllm/models/qwen4_exp/ : "
    git -C "$V" ls-tree -r --name-only "$c" 2>/dev/null | grep -c '^vllm/models/qwen4_exp/'
  done
  echo
  echo "## 7. the local source checkouts (read-only) — what each one pins"
  echo "--- /workspace/vllm-ascend (fork, branch $(git -C /workspace/vllm-ascend branch --show-current))"
  for f in .github/vllm-release-tag.commit .github/vllm-main-verified.commit; do
    echo -n "  $f: "; cat "/workspace/vllm-ascend/$f" 2>/dev/null | head -1
    c=$(cat "/workspace/vllm-ascend/$f" 2>/dev/null | head -1)
    if [ -n "$c" ] && git -C /workspace/vllm cat-file -e "$c^{commit}" 2>/dev/null; then
      echo -n "    -> vLLM tag: "; git -C /workspace/vllm describe --tags "$c" 2>&1 | head -1
      echo -n "    -> is it an ancestor of vLLM HEAD? (1 = no) : "
      git -C /workspace/vllm merge-base --is-ancestor "$c" HEAD; echo $?
    fi
  done
  echo -n "  fork README model: "; grep -m1 -oE "lenlrx/Qwen3\.8-[0-9A-Za-z.-]+" /workspace/vllm-ascend/README.md
  echo "--- /workspace/vllm (upstream) HEAD"
  git -C /workspace/vllm log --oneline -1
  git -C /workspace/vllm describe --tags
  echo -n "  vLLM v0.29.0 / v0.30.0 pyproject torch pin : "
  git -C /workspace/vllm show v0.30.0:pyproject.toml | grep -m1 'torch =='
  echo
  echo "## 8. venv has no vllm (row 10 of README §6.1)"
  /workspace/venvs/baseline/bin/python -c "import vllm" 2>&1 | tail -1
  echo
  echo "## 9. does upstream claim support for THIS model? (Qwen3.8-Flash-Next)"
  echo "### 9a. tutorial: docs/source/tutorials/models/Qwen3.8-Flash-Next.md"
  echo -n "  tutorial exists : "; git ls-tree --name-only main:docs/source/tutorials/models/ | grep -c "Qwen3.8-Flash-Next.md"
  echo "  hardware statement (line 7):"
  git show main:docs/source/tutorials/models/Qwen3.8-Flash-Next.md | sed -n '7p'
  echo "  validated version statement (line 9):"
  git show main:docs/source/tutorials/models/Qwen3.8-Flash-Next.md | sed -n '9p'
  echo "  validated config / checkpoint row (line 29):"
  git show main:docs/source/tutorials/models/Qwen3.8-Flash-Next.md | sed -n '29p'
  echo
  echo "### 9b. support matrix row (docs/source/user_guide/support_matrix/supported_models.md)"
  echo "  section tabs and line numbers:"
  git show main:docs/source/user_guide/support_matrix/supported_models.md | grep -n '^=== "' | head -5
  echo "  Qwen3.8-Flash-Next rows (note the Supported Hardware column):"
  git show main:docs/source/user_guide/support_matrix/supported_models.md | grep -n "Qwen3\.8" | head -5
  echo
  echo "### 9c. quay image tags for qwen3.8 mentioned in the docs"
  git grep -oh "quay.io/ascend/vllm-ascend:qwen3\.8[A-Za-z0-9._-]*" main -- docs | sort -u
  echo
  echo "### 9d. the VLLM_ASCEND_ENABLE_QSA_* variables the tutorial exports: docs or code?"
  echo "  files containing the name (whole repo):"
  git grep -l "VLLM_ASCEND_ENABLE_QSA" main | sed 's/^/    /'
  echo -n "  ... of which under vllm_ascend/ (code): "
  git grep -l "VLLM_ASCEND_ENABLE_QSA" main -- vllm_ascend | wc -l
  echo "### 9e. is the model implemented in the plugin, or reused from vLLM upstream?"
  echo -n "  plugin tree arch strings 'qwen4_exp|Qwen4Exp' hits : "
  git grep -ic "qwen4_exp\|Qwen4Exp" main -- vllm_ascend 2>/dev/null | wc -l
  echo "  (0 = the plugin ships no such arch; support must come from vLLM's qwen4_exp"
  echo "   model plus plugin ops/patches -- mechanism NOT verified in M30, treat as open)"
  echo "### EOF"
} > "$TMP" 2>&1

if grep -q '^### EOF' "$TMP" && [ "$(sed -n '/^## 1\./,/^## 2\./p' "$TMP" | wc -l)" -ge 10 ]; then
  mv "$TMP" "$OUT/11-upstream-refs.txt"
  echo "wrote $OUT/11-upstream-refs.txt"
  grep -n 'NETWORK_UNAVAILABLE' "$OUT/11-upstream-refs.txt" || true
else
  echo "REFUSING to overwrite $OUT/11-upstream-refs.txt (incomplete run); kept at $TMP"
  grep -n 'NETWORK_UNAVAILABLE' "$TMP" | head -3
  exit 1
fi
