"""Anchor-existence check for the M39 reference (docs/17 §2.4「引用必须存在」).

Every `path/to/file.py:NNN` (or `:NNN-MMM`, or a comma-chained / `:NNN` shorthand
continuation) written in `ref/*.py`, `README.md`, `selfcheck.py` and `tools/*.py`
must resolve to the file the author MEANT, whose length must be >= the highest
line cited. This is the machine-checkable half of the mission's hard requirement
("每一条 torch 语句旁标注官方源码的文件:行号"); the other half -- that the cited
line says what the comment claims -- is what a human reviewer does.

Three outcomes, reported separately (never collapsed into one number):

  VERIFIED     the anchor resolved to exactly one intended file and is in range
  UNVERIFIED   resolution was ambiguous (a bare basename with several same-named
               files, or a path matching several roots) -- **not** counted as OK
  OUT-OF-RANGE the intended file was identified and the line does not exist in it

Exit codes: `0` all anchors verified; `1` at least one out of range (a defect);
`2` at least one UNVERIFIED and none out of range (could not conclude).

## Rule for bare basenames (learned the hard way, twice)

    A bare basename is resolved through BASENAME_MAP only, or when exactly one
    file in the tree has that name. The heuristic "first candidate containing
    /nvidia/" is NOT allowed to decide: `hc.py` must not resolve to
    models/hy_v4/nvidia/hc.py, `model.py` must not resolve to
    models/deepseek_v32/nvidia/model.py, `base.py` must not resolve to
    model_executor/kernels/linear/base.py.

Round-2 review caught this: 82 of 514 anchors were being checked against *another
file with the same name* (74 of them `hc.py` -> `models/hy_v4/nvidia/hc.py`, 390
lines), and because those files are long enough, `0 out of range` looked clean
while saying nothing about the intended file. The general lesson, worth reusing
anywhere a tool resolves a name: **if resolution involves any guessing, the guess
must be reported as its own outcome, not folded into "pass".** Anything resolved
by a rule other than "one exact file" gets an ambiguity list.

`--selftest` runs the negative controls for exactly that (see NEGATIVE_CASES).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

VLLM_ROOT = Path("/workspace/vllm")
CKPT_ROOT = Path("/workspace/Qwen3.8-Flash-Next-MXFP4")
REPO_ROOT_CANDIDATES = [PKG.parent.parent, Path("/workspace/ascend_mega_kernel")]

# Search roots, IN PRIORITY ORDER. Anchors in this package are written relative
# to one of these documented bases; the first root where the relative path exists
# wins. Two different roots yielding two different files is ambiguous ->
# UNVERIFIED (never silently pick one).
SEARCH_ROOTS: list[tuple[str, Path]] = [
    ("qwen4_exp/nvidia", VLLM_ROOT / "vllm" / "models" / "qwen4_exp" / "nvidia"),
    ("vllm", VLLM_ROOT / "vllm"),
    ("vllm/models/qwen4_exp", VLLM_ROOT / "vllm" / "models" / "qwen4_exp"),
    ("vllm/model_executor", VLLM_ROOT / "vllm" / "model_executor"),
    ("vllm/model_executor/layers", VLLM_ROOT / "vllm" / "model_executor" / "layers"),
    ("vllm/model_executor/layers/mamba", VLLM_ROOT / "vllm" / "model_executor" / "layers" / "mamba"),
    ("vllm/model_executor/layers/fused_moe", VLLM_ROOT / "vllm" / "model_executor" / "layers" / "fused_moe"),
    ("vllm/model_executor/models", VLLM_ROOT / "vllm" / "model_executor" / "models"),
    ("vllm/transformers_utils/configs", VLLM_ROOT / "vllm" / "transformers_utils" / "configs"),
    ("third_party/flash_linear_attention/ops",
     VLLM_ROOT / "vllm" / "third_party" / "flash_linear_attention" / "ops"),
    ("csrc", VLLM_ROOT / "csrc"),
    ("csrc/cpu/sgl-kernels", VLLM_ROOT / "csrc" / "cpu" / "sgl-kernels"),
    ("csrc/libtorch_stable/moe", VLLM_ROOT / "csrc" / "libtorch_stable" / "moe"),
    ("tests", VLLM_ROOT / "tests"),
    ("tests/models/qwen4_exp", VLLM_ROOT / "tests" / "models" / "qwen4_exp"),
    ("tests/kernels/mamba/cpu", VLLM_ROOT / "tests" / "kernels" / "mamba" / "cpu"),
]

# Bare basenames used in this package, resolved EXPLICITLY. Values are either
# absolute paths or relative to VLLM_ROOT. An entry is required whenever the tree
# contains more than one file with that name; without it the anchor is UNVERIFIED.
BASENAME_MAP: dict[str, str] = {
    "hc.py": "vllm/models/qwen4_exp/nvidia/ops/hc.py",
    "model.py": "vllm/models/qwen4_exp/nvidia/model.py",
    "qsa.py": "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
    "indexer_qsa.py": "vllm/models/qwen4_exp/nvidia/indexer_qsa.py",
    "base.py": "vllm/model_executor/layers/rotary_embedding/base.py",
    "causal_conv1d.py": "vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py",
    "activation.py": "vllm/model_executor/layers/activation.py",
    "qwen3_next.py": "vllm/model_executor/models/qwen3_next.py",
    "qwen3_5.py": "vllm/model_executor/models/qwen3_5.py",
    "fused_recurrent.py": "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
    "fused_qk_norm_rope.py": "vllm/model_executor/layers/fused_qk_norm_rope.py",
    "routed_experts.py": "vllm/model_executor/layers/fused_moe/routed_experts.py",
    "mamba_utils.py": "vllm/model_executor/layers/mamba/mamba_utils.py",
    "README.quant.md": str(CKPT_ROOT / "README.quant.md"),
}

SOURCE_EXT = r"(?:py|cpp|cu|cuh|h|json|md|txt)"
FULL = re.compile(rf"((?:R:|CKPT:)?[A-Za-z0-9_./+-]+\.{SOURCE_EXT}):(\d+)(?:-(\d+))?")
CHAIN = re.compile(r"\s*,\s*(\d+)(?:-(\d+))?")
SHORTHAND = re.compile(r"(?<=[\s`(,（，、]):(\d+)(?:-(\d+))?\b(?=[\s`),.;、。]|$)")

OFF_MARKER = "anchor-check:off"
ON_MARKER = "anchor-check:on"
SKIP_FILES = {"check_anchors.py"}

SYMBOL_RE = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+(\w+)")

# --- coverage self-report (project rule 2026-09-26: a checking tool must state
# what it actually looked at, from the SAME matcher that does the looking) ---
# The matchers above require an EXTENSION (`file.py:12`) or are continuations of
# one. A reference written as `qsa_indexer:94` (stem, no extension) or
# `ref/qsa.py line 45` is INVISIBLE to them. Rather than silently under-report --
# which is exactly what M51's symbol enumerator did (56 reported vs 234 real) --
# every `token:NNN` on a scanned line that the three matchers did NOT consume is
# counted and listed as NOT CHECKED.
REF_LIKE = re.compile(r"[A-Za-z_][\w./+-]*:\d+")
# `NAME=PATH:DTYPE:shape` examples look like references but are not
NON_REF_STEMS = {"bf16", "fp16", "fp32", "int8", "int32", "int64", "uint8"}
MATCHED_SPAN = re.compile(rf"(?:{FULL.pattern})|(?:{CHAIN.pattern})|(?:{SHORTHAND.pattern})")


def repo_root() -> Path | None:
    for p in REPO_ROOT_CANDIDATES:
        if (p / "tools" / "golden" / "moe_block_ref.py").is_file():
            return p
    return None


def _file_index() -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    if not VLLM_ROOT.is_dir():
        return idx
    for p in VLLM_ROOT.rglob("*"):
        if p.suffix in (".py", ".cpp", ".cu", ".cuh", ".h", ".json", ".md", ".txt"):
            idx.setdefault(p.name, []).append(p)
    return idx


def resolve(target: str, index: dict[str, list[Path]]) -> tuple[list[Path], str]:
    """-> (candidate files, how it was resolved).

    `how` is one of: `root:<name>` (unique hit under a documented base),
    `map` (BASENAME_MAP), `unique-basename`, `bare-unmapped`, `ambiguous`.
    """
    if target.startswith("R:"):
        rr = repo_root()
        if rr is None:
            return [], "root-absent"
        p = rr / target[2:]
        return ([p] if p.is_file() else []), "map"
    if target.startswith("CKPT:"):
        p = CKPT_ROOT / target[5:]
        return ([p] if p.is_file() else []), "map"

    if "/" in target:
        hits: list[tuple[str, Path]] = []
        for name, root in SEARCH_ROOTS:
            p = root / target
            if p.is_file():
                hits.append((name, p))
        if len(hits) == 1:
            return [hits[0][1]], f"root:{hits[0][0]}"
        if len(hits) > 1:
            return [h[1] for h in hits], "ambiguous"
        # last resort: a unique suffix match anywhere in the tree
        suffix = "/" + target
        cands = [p for p in VLLM_ROOT.rglob(Path(target).name)
                 if p.is_file() and str(p).endswith(suffix)]
        if len(cands) == 1:
            return cands, "unique-basename"
        return cands, ("bare-unmapped" if not cands else "ambiguous")

    # bare basename
    if target in BASENAME_MAP:
        raw = BASENAME_MAP[target]
        p = Path(raw)
        p = p if p.is_absolute() else VLLM_ROOT / raw
        return ([p] if p.is_file() else []), "map"
    cands = sorted(index.get(target, []))
    if len(cands) == 1:
        return cands, "unique-basename"
    return cands, "bare-unmapped" if target in BASENAME_MAP else "ambiguous"


def src_name(path: Path, package: Path = PKG) -> str:
    """Anchor source label as short as possible (never an absolute path).

    Round-3 review: `evaluate()` used `str(f)`, so `evidence/selfcheck.log`
    embedded `/workspace/.../wt-39/m21_layer_ref/...` and the log differed when
    the package was reproduced in another directory. A label relative to the
    package root is stable across checkouts.
    """
    try:
        return str(path.relative_to(package))
    except ValueError:
        return path.name


def short(path: Path) -> str:
    """Path as short as possible for reporting (relative to /workspace/vllm if it
    lives there, else absolute)."""
    for base in (VLLM_ROOT, CKPT_ROOT, repo_root()):
        if base is None:
            continue
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def symbol_at(path: Path, line: int) -> str:
    """Best-effort `Class.method` / `function` name containing `line`.

    Anchors cite line numbers (the mission requires them) but line numbers drift
    with upstream; the project rule is that the SYMBOL is the stable reference
    and the line number is a lookup hint. Printing the symbol here lets a reader
    confirm at a glance that an anchor still lands in the intended symbol.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "?"
    best: tuple[int, str] = (0, "")
    cls: tuple[int, str] = (0, "")
    for i, text in enumerate(lines[:max(line, 0)], 1):
        m = SYMBOL_RE.match(text)
        if not m:
            continue
        indent, name = len(m.group(1)), m.group(2)
        if indent == 0 and m.group(0).lstrip().startswith("class "):
            cls = (i, name)
            best = (i, name)
        elif indent == 0:
            cls = (0, "")
            best = (i, name)
        else:
            prefix = f"{cls[1]}." if cls[1] else ""
            best = (i, f"{prefix}{name}")
    return best[1] or "<module level>"


def nearest_symbol(path: Path, line: int) -> str:
    return symbol_at(path, line)


def scan(path: Path) -> tuple[list[tuple[int, str, int, int]], list[str]]:
    """-> (anchors, unmatched_reference_like_tokens)

    `unmatched` is the coverage disclaimer: `token:NNN` occurrences on scanned
    lines that no matcher consumed (typically a reference written without a file
    extension). They are NOT judged -- they are reported so the reader knows the
    check's blind spot instead of reading "0 out of range" as "everything is
    fine".
    """
    out: list[tuple[int, str, int, int]] = []
    unmatched: list[str] = []
    current: str | None = None
    scanning = True
    for lineno, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if OFF_MARKER in text:
            scanning = False
            continue
        if ON_MARKER in text:
            scanning = True
            continue
        if not scanning:
            continue
        for m in FULL.finditer(text):
            target = m.group(1)
            a, b = int(m.group(2)), m.group(3)
            current = target
            out.append((lineno, target, a, int(b) if b else a))
            pos = m.end()
            while True:
                cm = CHAIN.match(text, pos)
                if cm is None:
                    break
                a2 = int(cm.group(1))
                out.append((lineno, target, a2, int(cm.group(2)) if cm.group(2) else a2))
                current = target
                pos = cm.end()
        if current is not None:
            for m in SHORTHAND.finditer(text):
                a = int(m.group(1))
                b = m.group(2)
                if a < 1:
                    continue
                out.append((lineno, current, a, int(b) if b else a))

        # coverage: reference-like tokens nothing above consumed
        consumed = [mm.span() for mm in MATCHED_SPAN.finditer(text)]
        for rm in REF_LIKE.finditer(text):
            s, e = rm.span()
            if any(cs <= s and e <= ce for cs, ce in consumed):
                continue
            tok = rm.group(0)
            # skip things that are clearly not source references: clock times,
            # python format specs (`{name:34s}`), and dtype tokens from the
            # `--test-bin NAME=PATH:DTYPE:shape` examples
            stem = tok.split(":")[0].lower()
            nxt = text[e] if e < len(text) else ""
            if (re.fullmatch(r"\d+:\d+", tok) or tok.startswith(("http", "sha256"))
                    or nxt in "sdfx%}"          # python format spec, e.g. {name:34s}
                    or stem in NON_REF_STEMS):
                continue
            unmatched.append(f"{path.name}:{lineno}: `{tok}` "
                             f"(no file extension -> NOT CHECKED by this matcher)")
    return out, unmatched


def evaluate(files: list[Path]):
    """Core: scan `files` and classify every anchor. -> result dict."""
    index = _file_index()
    anchors: list[tuple[str, int, str, int, int]] = []
    unmatched: list[str] = []
    lines_scanned = 0
    for f in files:
        found, unm = scan(f)
        unmatched.extend(unm)
        for lineno, target, a, last in found:
            anchors.append((src_name(f), lineno, target, a, last))

    verified = unverified = out_of_range = 0
    reached: set[str] = set()
    unresolved: list[str] = []      # UNVERIFIED (ambiguity / unmapped basename)
    problems: list[str] = []        # OUT OF RANGE
    symbols: list[str] = []
    for src, lineno, target, first, last in anchors:
        cands, how = resolve(target, index)
        if how == "root-absent":
            unresolved.append(f"{src}:{lineno}: `{target}` -- its root is absent here")
            continue
        if how in ("ambiguous", "bare-unmapped") or len(cands) != 1:
            shown = ", ".join(short(c) for c in cands[:6]) or "no candidate"
            unresolved.append(
                f"{src}:{lineno}: `{target}:{first}-{last}` UNVERIFIED ({how}); "
                f"candidates: {shown}. Add an explicit BASENAME_MAP entry or a "
                f"directory-qualified path.")
            continue
        path = cands[0]
        try:
            n = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError as exc:  # noqa: BLE001
            unresolved.append(f"{src}:{lineno}: `{target}` unreadable ({exc})")
            continue
        if last > n:
            problems.append(
                f"{src}:{lineno}: `{target}:{first}-{last}` OUT OF RANGE for "
                f"`{short(path)}` (len {n}); "
                f"symbol there: {symbol_at(path, n)}")
            continue
        verified += 1
        reached.add(str(path))
        symbols.append(f"{target}:{first}-{last} -> {short(path)}"
                       f"::{symbol_at(path, first)} [{how}]")
    return {"anchors": len(anchors), "verified": verified,
            "unverified": unverified + len(unresolved), "out_of_range": out_of_range + len(problems),
            "unresolved": unresolved, "problems": problems, "symbols": symbols,
            "files_scanned": len(files), "not_checked": unmatched,
            "distinct_targets": len({t for _, _, t, _, _ in anchors}),
            "distinct_files": len(reached)}


def package_files(package: Path = PKG) -> list[Path]:
    return sorted(
        p for p in list(package.glob("*.py")) + list(package.glob("ref/*.py"))
        + list(package.glob("tools/*.py")) + list(package.glob("*.md"))
        + list(package.glob("reference/*.md"))
        if p.name not in SKIP_FILES
    )


def check(package: Path = PKG, verbose: bool = False, extra_files: list[Path] | None = None):
    res = evaluate(sorted(set(package_files(package)) | set(extra_files or [])))
    if verbose:
        for s in res["symbols"][:500]:
            print(f"    {s}")
    return res


def report(res: dict) -> int:
    print(f"[check_anchors] scanned {res['files_scanned']} package files; "
          f"{res['anchors']} anchors found over {res['distinct_targets']} distinct "
          f"anchor target paths -> {res['distinct_files']} distinct files; "
          f"{res['verified']} verified against the intended file / "
          f"{res['unverified']} UNVERIFIED (ambiguous / root absent) / "
          f"{res['out_of_range']} out of range")
    print(f"  matcher scope: `FILE.ext:NNN[-MMM]`, comma-chained `,NNN-MMM`, and "
          f"`:NNN` continuations inside `anchor-check:on` regions. "
          f"{len(res['not_checked'])} reference-like token(s) matched NONE of "
          f"those -> NOT CHECKED (listed below if any)")
    for p in res["not_checked"][:10]:
        print(f"    NOT-CHECKED  {p}")
    for p in res["problems"][:20]:
        print(f"    OUT-OF-RANGE {p}")
    for p in res["unresolved"][:20]:
        print(f"    UNVERIFIED   {p}")
    if res["out_of_range"]:
        print("RESULT: FAIL (an anchor cites a line that does not exist)")
        return 1
    if res["unverified"]:
        print("RESULT: SKIPPED (some anchors could not be resolved to one file; "
              "NOT a pass)")
        return 2
    if not res["anchors"]:
        print("RESULT: SKIPPED (0 anchors found -- nothing compared)")
        return 2
    print(f"RESULT: OK ({res['verified']}/{res['anchors']} anchors verified "
          f"against one exact intended file; {len(res['not_checked'])} "
          f"reference-like tokens not covered by the matcher)")
    return 0


# ---------------------------------------------------------------------------
# Negative controls: each case must produce the STATED outcome. "Passing for the
# wrong reason is worse than failing" -- round-1 review's own lesson, applied
# here to the checker itself.
# ---------------------------------------------------------------------------
NEGATIVE_CASES = [
    # (anchor text, expected verdict, why)
    ("R:tools/golden/moe_block_ref.py:57-59", "verified",
     "a real anchor into this repo"),
    ("vllm/model_executor/layers/rotary_embedding/base.py:80-102", "verified",
     "a real anchor with a full path"),
    ("causal_conv1d.py:600-600", "out_of_range",
     "BASENAME_MAP points at the 147-line cpu file; 600 exists only in "
     "mamba/ops/causal_conv1d.py (1288 lines) -- must NOT pass via the twin"),
    ("hc.py:460-460", "verified",
     "BASENAME_MAP points at qwen4_exp/nvidia/ops/hc.py (504 lines); 460 <= 504, "
     "in range -- proves the map is used and the line test is real"),
    ("hc.py:700-700", "out_of_range",
     "qwen4_exp/nvidia/ops/hc.py is 504 lines and models/hy_v4/nvidia/hc.py is "
     "390 -- 700 exists in neither, so the mapped file must be the one tested"),
    ("nonexistent_twin_name.py:1-1", "unverified",
     "unmapped bare basename with no candidate -> UNVERIFIED, never a pass"),
    ("qsa_indexer:94", "not_checked",
     "KNOWN BLIND SPOT: a reference written without a file extension is invisible "
     "to the matcher. It must show up in the NOT-CHECKED list, never as a pass"),
]


def selftest() -> int:
    # A FIXED path, not tempfile: a random /tmp/tmp.XXXX in the output would end
    # up inside evidence/selfcheck.log and make the archived evidence
    # irreproducible (project rule: no "live values" in the archive).
    print("== negative controls for the anchor checker ==")
    ok = True
    case_file = PKG / ".anchor_selftest_case.md"
    try:
        for text, expect, why in NEGATIVE_CASES:
            case_file.write_text(f"case: `{text}`\n")
            # ABSOLUTE path here on purpose: a CWD-relative one made the tool
            # crash whenever it was invoked from outside the package
            # (`FileNotFoundError: '.anchor_selftest_case.md'`, round-4 review).
            # The printed label is still short -- `src_name()` rewrites it
            # relative to PKG -- so removing absolute paths from the log did not
            # cost us CWD independence.
            res = evaluate([case_file])
            if res["not_checked"]:
                got = "not_checked"
            elif res["out_of_range"]:
                got = "out_of_range"
            elif res["unverified"]:
                got = "unverified"
            elif res["verified"]:
                got = "verified"
            else:
                got = "nothing"
            good = got == expect
            ok &= good
            detail = res["not_checked"] + res["problems"] + res["unresolved"]
            print(f"  [{'OK  ' if good else 'BAD '}] `{text}` -> {got} "
                  f"(expected {expect})")
            print(f"          {why}")
            if detail:
                print(f"          {detail[0][:160]}")
    finally:
        case_file.unlink(missing_ok=True)
    print(f"RESULT: {'OK' if ok else 'FAIL'} "
          f"({sum(1 for _ in NEGATIVE_CASES)}/{len(NEGATIVE_CASES)} cases as expected)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print every anchor with the symbol it lands in")
    ap.add_argument("--selftest", action="store_true",
                    help="run the negative controls instead of the real scan")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return report(check(verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())
