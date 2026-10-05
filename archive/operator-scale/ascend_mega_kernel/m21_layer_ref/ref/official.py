"""Direct-import probe of the official qwen4_exp implementation (no vLLM install).

Exit codes: 0 = the probe ran and reported (this is an information-gathering
probe, its "answers" are the failure messages, not a pass/fail); 2 = the probe
could not run at all (e.g. /workspace/vllm absent). It never returns 1, because
"the import fails" is its expected, documented result, not a defect.

The mission asks: can we `import` the official layer implementation with the
baseline venv (`/workspace/venvs/baseline`, torch 2.10.0+cpu) by just injecting
`/workspace/vllm` into `sys.path`, without installing the vllm package?

Two answers, both reproduced by `python -m ref.official`:

  * `vllm/models/qwen4_exp/nvidia/model.py` (and anything else that imports the
    `vllm` package): **NO**. Importing `vllm.*` runs `vllm/__init__.py`, which
    drags in the whole runtime -- pydantic configs, the platform detector, the
    compiled `vllm._C` ops, triton. `probe()` records the exact failure, and
    `dependency_closure()` lists every third-party distribution reachable from the
    target module together with whether the baseline venv has it.
  * `vllm/models/qwen4_exp/common/hyperconnection.py`: **YES**, because it imports
    only `torch` / `torch.nn.functional` / `dataclasses` (:32-36). We load it by
    file path so `vllm/__init__.py` never runs. This gives an *executed* official
    torch oracle for the hyper-connection math, which the harness uses as a
    second opinion against the Triton-faithful `ref/hc.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

VLLM_ROOT = Path("/workspace/vllm")
Q4E = VLLM_ROOT / "vllm" / "models" / "qwen4_exp"

# Entry points, in the order the mission cares about
TARGETS = [
    "vllm.models.qwen4_exp.nvidia.model",
    "vllm.models.qwen4_exp.nvidia.qsa",
    "vllm.models.qwen4_exp.nvidia.ops.hc",
]

_STDLIB = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "ctypes", "dataclasses", "datetime", "enum", "functools",
    "gc", "glob", "hashlib", "heapq", "importlib", "inspect", "io", "itertools",
    "json", "logging", "math", "mmap", "multiprocessing", "operator", "os",
    "pathlib", "pickle", "random", "re", "shlex", "shutil", "signal", "socket",
    "statistics", "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
    "threading", "time", "traceback", "types", "typing", "unicodedata", "uuid",
    "warnings", "weakref", "zipfile", "__future__",
}


def _import(name: str):
    __import__(name, fromlist=["*"])


def try_direct_import(target: str = TARGETS[0]) -> dict:
    """Inject /workspace/vllm on sys.path and try the import for real."""
    if str(VLLM_ROOT) not in sys.path:
        sys.path.insert(0, str(VLLM_ROOT))
    result = {"target": target, "status": "ok", "error": None, "chain": []}
    try:
        _import(target)
    except BaseException as exc:  # noqa: BLE001 - we want everything on record
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
        tb = exc.__traceback__
        frames = []
        while tb is not None:
            frames.append(f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}")
            tb = tb.tb_next
        result["chain"] = frames
    return result


def dependency_closure(entry: str = TARGETS[0], root: Path = VLLM_ROOT) -> dict:
    """Static transitive import closure inside the `vllm` package.

    Follows `from vllm.X import Y` / `import vllm.X` through vllm's own source
    files, then reports the third-party top-level names that fall outside the
    closure, along with `importlib.util.find_spec` availability.
    """
    import ast

    def module_to_file(mod: str) -> Path | None:
        rel = mod.split(".")
        if rel[0] != "vllm":
            return None
        p = root.joinpath(*rel)
        for cand in (p.with_suffix(".py"), p / "__init__.py"):
            if cand.exists():
                return cand
        return None

    seen: set[str] = set()
    external: set[str] = set()
    queue = [entry]
    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = module_to_file(mod)
        if path is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top == "vllm":
                    if name.startswith("vllm."):
                        queue.append(name)
                elif top not in _STDLIB:
                    external.add(top)

    availability = {}
    for name in sorted(external):
        try:
            availability[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            availability[name] = False
    return {
        "vllm_modules_in_closure": len(seen),
        "external": availability,
        "missing": sorted(k for k, v in availability.items() if not v),
    }


# ---------------------------------------------------------------------------
# The one official module that IS directly loadable: common/hyperconnection.py
# ---------------------------------------------------------------------------
_OFFICIAL_COMMON = None


def import_common_hyperconnection():
    """Load `vllm/models/qwen4_exp/common/hyperconnection.py` by file path.

    The file imports only torch (its header, :32-36), so bypassing
    `vllm/__init__.py` is enough. Returns the module (or raises).
    """
    global _OFFICIAL_COMMON
    if _OFFICIAL_COMMON is not None:
        return _OFFICIAL_COMMON
    path = Q4E / "common" / "hyperconnection.py"
    spec = importlib.util.spec_from_file_location("qwen4_exp_common_hyperconnection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _OFFICIAL_COMMON = mod
    return mod


def main() -> int:
    if not VLLM_ROOT.is_dir():
        print(f"RESULT: SKIPPED (no vLLM source tree at {VLLM_ROOT}; nothing probed)")
        return 2
    print("=== direct import probe (sys.path += /workspace/vllm) ===")
    for target in TARGETS:
        r = try_direct_import(target)
        print(f"  {target}: {r['status']}")
        if r["error"]:
            print(f"      {r['error']}")
            if r["chain"]:
                print(f"      at {r['chain'][-1]}")
    print()
    print("=== official common/hyperconnection.py loaded by file path ===")
    try:
        mod = import_common_hyperconnection()
        print(f"  ok: {mod.__name__}, GatedResidual={mod.GatedResidual}")
    except Exception as exc:  # noqa: BLE001
        print(f"  fail: {type(exc).__name__}: {exc}")
    print()
    print("=== static dependency closure from nvidia/model.py ===")
    dep = dependency_closure()
    print(f"  vllm modules in closure: {dep['vllm_modules_in_closure']}")
    print(f"  external distributions : {len(dep['external'])}")
    for name, ok in dep["external"].items():
        print(f"    {'OK  ' if ok else 'MISS'} {name}")
    print()
    print("=== is the vllm source tree even built? ===")
    sos = sorted(p.name for p in (VLLM_ROOT / "vllm").glob("*.so"))
    print(f"  compiled extensions in vllm/ : {len(sos)} {sos}")
    print("  -> the tree has never been built, so `vllm._C` / `_custom_ops` cannot "
          "be imported even if every python dependency above were installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
