#!/usr/bin/env bash
# Regenerate m21_layer_ref/evidence/ from the existing reference/ dumps.
#
#   bash tools/make_evidence.sh
#
# Kept separate from tools/make_reference.sh so evidence can be refreshed
# without re-running the ~2 min dump build.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/workspace/venvs/baseline/bin/python3
mkdir -p evidence

strip() { grep -v 'UserWarning\|torch.from_numpy\|^  t = torch' || true; }
# FIXED relative path (a mktemp path would make these two evidence files
# unreproducible and would embed a random /tmp/tmp.XXXX in the archive).
TMP_ZEROED="reference/.tmp_zeroed"

echo "== 1. direct-import probe" | tee evidence/import_probe.txt
# `python -m ref.official` writes its own stdout; keep the RuntimeWarning up top
# (it is part of the evidence: the source tree was never built).
$PY -m ref.official 2>&1 | tee -a evidence/import_probe.txt >/dev/null
head -1 evidence/import_probe.txt >/dev/null

echo "== 2. dump sha256 (also cross-checks manifest.json)" | strip
$PY tools/hash_dumps.py reference > evidence/reference_sha256.txt
tail -1 evidence/reference_sha256.txt

echo "== 3. determinism: tensor hashes are stable across independent builds"
# `reference_sha256.first_run.txt` is a copy kept from an earlier independent
# build; if it is absent this check is skipped (first ever run). The comparison
# is on the tensor hashes only -- the `extra` block legitimately differs between
# runs (build/run wall times), so those lines are filtered out first.
if [ -f evidence/reference_sha256.first_run.txt ]; then
  # Compare ONLY the `<sha256>  <path>` lines. That is exactly the claim being
  # made ("the tensors' bytes are identical across builds"); comparing the whole
  # filtered file instead made the check fail once on a cosmetic change to
  # hash_dumps.py's own output, which is not a determinism signal.
  if diff <(grep '^[0-9a-f]\{64\}  ' evidence/reference_sha256.first_run.txt) \
          <(grep '^[0-9a-f]\{64\}  ' evidence/reference_sha256.txt) > evidence/determinism.diff; then
    echo "IDENTICAL across two independent builds ($(grep -c '^[0-9a-f]\{64\}' evidence/reference_sha256.txt) tensors)"
    rm -f evidence/determinism.diff
  else
    echo "DIFFERS -- see evidence/determinism.diff (kept for inspection)"; exit 1
  fi
else
  echo "SKIP (no evidence/reference_sha256.first_run.txt baseline yet)"
fi

echo "== 4. selfcheck"
$PY selfcheck.py --m 4 2>&1 | strip | tee evidence/selfcheck.log | tail -3

echo "== 5. non-hollowness (reference vs reference, different seed)"
$PY compare_dumps.py --ref reference/layer3_chunk_m64 \
  --ref2 reference/layer3_chunk_m64_seed1 --nonhollow-only \
  | tee evidence/nonhollow.md | tail -2 || true

echo "== 6. comparison tool self-test: positive (ref vs itself) and negative (ref vs a different input)"
$PY compare_dumps.py --ref reference/layer0_decode_m1 --test reference/layer0_decode_m1 \
  --segments attn_hc.block_input,gdn.conv_out,moe.topk_ids \
  | tee evidence/compare_positive.md | grep '判定项总数'
$PY compare_dumps.py --ref reference/layer0_decode_m1 --test reference/layer0_decode_m1_pending \
  --segments attn_hc.block_input,gdn.conv_out,moe.topk_ids \
  | tee evidence/compare_negative.md | grep '判定项总数' || true

echo "== 7. mask-policy knob: a dump that ZEROS the columns the kernel never wrote"
# strict must FAIL, ignore-unscored must PASS, with the skipped count reported.
$PY tools/zero_unscored.py --src reference/layer3_chunk_m64 --dst "$TMP_ZEROED" \
  --segments qsa.index_logits >/dev/null
{ echo "### strict (default)"; $PY compare_dumps.py --ref reference/layer3_chunk_m64 \
    --test "$TMP_ZEROED" --segments qsa.index_logits || true; } > evidence/compare_strict_mask.md
{ echo "### --mask-policy ignore-unscored"; $PY compare_dumps.py --ref reference/layer3_chunk_m64 \
    --test "$TMP_ZEROED" --segments qsa.index_logits --mask-policy ignore-unscored; } \
  > evidence/compare_ignore_unscored.md
grep -h '判定项总数' evidence/compare_strict_mask.md evidence/compare_ignore_unscored.md
rm -rf "$TMP_ZEROED"   # never leave the synthetic dump behind

echo "== 8. peak process memory per tag (README §0 row 4 cites this)"
$PY tools/measure_peak_rss.py | tee evidence/peak_rss.txt | tail -3

echo "== 9. anchor existence (docs/17 §2.4)"
$PY tools/check_anchors.py | tee evidence/anchors.txt

echo "evidence/ refreshed"
