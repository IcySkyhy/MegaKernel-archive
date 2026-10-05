#!/usr/bin/env python3
"""M30: measure how far the vllm-ascend fork (baseline: vLLM v0.23.0) is from vLLM main.

The fork's patch modules reach into vLLM internals by module path and symbol name. Those
move between releases, so this script resolves every `from vllm... import ...` /
`import vllm...` reference made by the plugin against a *second* vLLM source tree and
reports what no longer exists. It is the cheapest hard evidence that "just install the
plugin on top of vLLM main" cannot work, and it needs no install of either project.

Exit codes (three states -- "I compared and it passed" must be distinguishable from
"I had nothing to compare"; tower rule 2026-09-26):
  0  compared and passed   (prints `RESULT: OK (<n> references resolved, 0 broken)`)
     NOTE the RESULT/exit-code banner goes to **stderr** on purpose: stdout is the archived
     report (`evidence/04`), which is a M30 original capture and must stay byte-identical.
  1  compared and found differences (prints `RESULT: FAIL (...)`; broken refs ARE the finding)
  2  nothing comparable    (prints `RESULT: SKIPPED (...)` -- plugin tree missing/no .py files)

Usage:
  python3 baseline_env/scripts/check_plugin_vs_vllm_main.py \
      --plugin /workspace/vllm-ascend --vllm /workspace/vllm
"""

import argparse
import ast
import collections
import os
import re
import subprocess
import sys


def vllm_checkout_version(vllm_root):
    """`<root>` 检出的版本（`git describe --tags`，去掉尾部的 `-g<sha>`）。

    这是**活值**（/workspace/vllm 的 snapshot，`git pull` 或别的 worker 动过就会变），所以不写死在
    打印串里 —— M51 修订：旧版把它硬编码成 "v0.30.1rc0-189"，检出一动就成了过期值。
    """
    try:
        desc = subprocess.run(["git", "-C", vllm_root, "describe", "--tags"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        desc = ""
    if not desc:
        return "unknown（不是 git 检出）"
    return re.sub(r"-g[0-9a-f]{7,}$", "", desc)


def iter_py(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "build"}]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def module_exists(vllm_root, dotted):
    """Does `vllm_root/<dotted with / >` exist as .py or package?"""
    parts = dotted.split(".")
    base = os.path.join(vllm_root, *parts)
    return os.path.isfile(base + ".py") or os.path.isdir(base)


def symbol_exists(vllm_root, dotted, name):
    """Symbol `name` importable from module `dotted`?

    Accepts (a) a class/def/assignment in the module or its package __init__, and
    (b) a submodule of the same name -- `from vllm.v1.worker import mamba_utils` is valid
    when vllm/v1/worker/mamba_utils.py exists.
    """
    mod_dir = os.path.join(vllm_root, *dotted.split("."))
    if os.path.isfile(os.path.join(mod_dir, name + ".py")) or os.path.isdir(
        os.path.join(mod_dir, name)
    ):
        return True
    path_py = os.path.join(vllm_root, *dotted.split(".")) + ".py"
    pkg_init = os.path.join(mod_dir, "__init__.py")
    targets = [p for p in (path_py, pkg_init) if os.path.isfile(p)]
    if os.path.isdir(mod_dir):
        targets += [os.path.join(mod_dir, f) for f in os.listdir(mod_dir) if f.endswith(".py")]
    pat = re.compile(rf"^\s*(class|def)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]", re.M)
    for t in targets:
        try:
            with open(t, encoding="utf-8", errors="ignore") as fh:
                if pat.search(fh.read()):
                    return True
        except OSError:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="/workspace/vllm-ascend")
    ap.add_argument("--vllm", default="/workspace/vllm")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    plugin_src = os.path.join(args.plugin, "vllm_ascend")
    if not os.path.isdir(plugin_src):
        print(f"RESULT: SKIPPED (plugin tree not found: {plugin_src} -- nothing to compare)", file=sys.stderr)
        return 2
    vllm_root = args.vllm  # repo root: module "vllm.config" -> <root>/vllm/config.py
    print(f"plugin tree : {plugin_src}")
    print(f"vLLM tree   : {vllm_root}/vllm  (main checkout, {vllm_checkout_version(vllm_root)})\n")

    def is_vllm_ref(dotted):
        return dotted == "vllm" or dotted.startswith("vllm.")

    missing_modules = collections.Counter()
    missing_symbols = collections.Counter()
    ok_refs = 0
    files = list(iter_py(plugin_src))
    ref_files_mod = collections.defaultdict(set)
    ref_files_sym = collections.defaultdict(set)

    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError) as exc:
            print(f"  ! cannot parse {path}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and is_vllm_ref(node.module):
                if not module_exists(vllm_root, node.module):
                    missing_modules[node.module] += 1
                    ref_files_mod[node.module].add(os.path.relpath(path, plugin_src))
                    continue
                for alias in node.names:
                    name = alias.name
                    if name == "*":
                        continue
                    if symbol_exists(vllm_root, node.module, name):
                        ok_refs += 1
                    else:
                        missing_symbols[f"{node.module}.{name}"] += 1
                        ref_files_sym[f"{node.module}.{name}"].add(os.path.relpath(path, plugin_src))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if is_vllm_ref(alias.name):
                        if module_exists(vllm_root, alias.name):
                            ok_refs += 1
                        else:
                            missing_modules[alias.name] += 1
                            ref_files_mod[alias.name].add(os.path.relpath(path, plugin_src))

    total_bad = sum(missing_modules.values()) + sum(missing_symbols.values())
    if not files:
        print(f"RESULT: SKIPPED (0 python files scanned under {plugin_src} -- nothing to compare)",
              file=sys.stderr)
        return 2
    print(f"plugin python files scanned            : {len(files)}")
    print(f"resolved vLLM references               : {ok_refs}")
    print(f"references to MODULES that no longer exist in vLLM main : {len(missing_modules)}")
    print(f"references to SYMBOLS that no longer exist in vLLM main : {len(missing_symbols)}")
    print(f"total broken references                 : {total_bad}\n")

    print(f"== top missing symbols (of {len(missing_symbols)}) ==")
    for ref, n in missing_symbols.most_common(args.show):
        print(f"  {ref}   (used in {n} places, e.g. {sorted(ref_files_sym[ref])[0]})")
    print(f"\n== missing modules (of {len(missing_modules)}) ==")
    for ref, n in missing_modules.most_common(args.show):
        print(f"  {ref}   (used in {n} places, e.g. {sorted(ref_files_mod[ref])[0]})")
    if total_bad == 0:
        print(f"RESULT: OK ({ok_refs} references resolved, 0 broken; {len(files)} files scanned)",
              file=sys.stderr)
        return 0
    print(f"RESULT: FAIL ({total_bad} broken references out of {ok_refs + total_bad} "
          f"resolved+broken; {len(files)} files scanned)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
