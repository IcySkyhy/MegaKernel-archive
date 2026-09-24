#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sync OpGraph/TileGraph headers into a sibling Ascend project's TD/ folder.

Layout assumed::

    <parent>/
      Triton-distributed-ascend/     # this repo
      <other-ascend-project>/        # destination project
        TD/                          # created if missing
          OpGraph.h
          TileGraph.h
          WireEndian.h
          two_matmul_opgraph.expected.bin
          lmhead_opgraph.expected.bin
        examples/operators/matmul/matmul.cpp      # patched when -p A3
        examples/operators/matmulA5/matmulA5.cpp # patched when -p 950
        examples/models/qwen3-30B/qwen3_model.cpp # buildLMHead merge TD bin
        blade/builder/include/graph/graph.h      # GraphBuilder APIs
        blade/builder/src/graph/graph.cpp

Transforms:
  1. Copy headers into TD/; ``namespace triton_dist`` -> ``namespace mk``
  2. ``#include <TritonDistributed/WireEndian.h>`` -> ``#include "WireEndian.h"``
  3. Strip host data-model types listed in ``TYPES_TO_STRIP``
  4. Copy the platform OpGraph goldens into TD/:
     - ``two_matmul_opgraph.expected.bin`` (matmul example)
     - ``lmhead_opgraph.expected.bin`` (qwen buildLMHead)
  5. Patch the platform matmul cpp: replace
     ``OpGraph graph = gb.buildOpGraph();`` with load-from-file + start/end logs.
     A3: ``examples/operators/matmul/matmul.cpp``
     950: ``examples/operators/matmulA5/matmulA5.cpp``
     The bin path embedded in the cpp is the resolved absolute path of
     ``TD/two_matmul_opgraph.expected.bin`` (cwd-independent for build outputs).
     For 950 only, also switch the two-matmul verification path:
     comment active ``GraphTensor input`` lines / uncomment commented ones,
     ``mm1.id()`` -> ``mm2.id()`` in ``copyFromTensor(hostResult...)``,
     and ``512.0f`` -> ``65536.0f`` in the hostResult check.
  6. Patch ``blade/builder/include/graph/graph.h`` +
     ``blade/builder/src/graph/graph.cpp`` with ``loadOpGraph`` /
     ``mergeOpGraph`` / lookup helpers (templates under ``td_patches/``).
     ``mergeOpGraph`` calls ``inferOpTilingFromSchema`` for each appended op
     (must be wired to the same schema tiling as ``ops::`` /
     ``addOpWithShapeInfer`` — not naive ``shape.ndim``).
  7. Patch ``examples/models/qwen3-30B/qwen3_model.cpp`` ``buildLMHead`` to
     deserialize ``TD/lmhead_opgraph.expected.bin`` and ``mergeOpGraph``
     (bin: add_rms_norm StaticOpResult<3> as y[M,H], x[M,H], rstd[M,1];
     matmul consumes y). ranks>1 still runs Ascend ``ops::all_gather`` after
     merge.
  8. Under ``examples/operators/**``, ``examples/models/**``, and
     ``blade/builder/**`` CMakeLists.txt, for each line containing
     ``/blade/runtime/include``, insert a sibling line below with that path
     segment replaced by ``/TD`` (idempotent).

Usage::

    python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers /path/to/other-ascend-project
    python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p A3
    python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p 950
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADERS = ("OpGraph.h", "TileGraph.h", "WireEndian.h")

OPERATORS_DIR_REL = Path("examples/operators")
MODELS_DIR_REL = Path("examples/models")
BLADE_BUILDER_DIR_REL = Path("blade/builder")
TD_OPGRAPH_BIN_NAME = "two_matmul_opgraph.expected.bin"
TD_LMHEAD_BIN_NAME = "lmhead_opgraph.expected.bin"
BLADE_RUNTIME_INCLUDE = "/blade/runtime/include"
TD_INCLUDE_SUFFIX = "/TD"

GRAPH_H_REL = Path("blade/builder/include/graph/graph.h")
GRAPH_CPP_REL = Path("blade/builder/src/graph/graph.cpp")
QWEN_MODEL_CPP_REL = Path("examples/models/qwen3-30B/qwen3_model.cpp")

TD_MARKER_GRAPH = "[TD] GraphBuilder OpGraph install"
TD_MARKER_LMHEAD = "[TD] buildLMHead from lmhead_opgraph"

# -p A3|950: which example cpp to patch, and which checked-in OpGraph golden.
PLATFORM_MATMUL_CPP: dict[str, Path] = {
    "A3": Path("examples/operators/matmul/matmul.cpp"),
    "950": Path("examples/operators/matmulA5/matmulA5.cpp"),
}
PLATFORM_OP_NAME: dict[str, str] = {
    "A3": "Matmul",
    "950": "matmulA5",
}
PLATFORM_OPGRAPH_BIN: dict[str, str] = {
    "A3": "python/triton_dist/mega_kernel_ascend/test/ops/two_matmul_opgraph.expected.bin",
    "950": "python/triton_dist/mega_kernel_ascend/test/ops/two_matmul_opgraph.950.expected.bin",
}
PLATFORM_LMHEAD_BIN: dict[str, str] = {
    "A3": "python/triton_dist/mega_kernel_ascend/test/ops/lmhead_opgraph.expected.bin",
    "950": "python/triton_dist/mega_kernel_ascend/test/ops/lmhead_opgraph.950.expected.bin",
}

BUILD_GRAPH_RE = re.compile(
    r"^([ \t]*)OpGraph\s+graph\s*=\s*gb\.buildOpGraph\s*\(\s*\)\s*;\s*$",
    re.MULTILINE,
)
# Any existing load_opgraph_from_file("...") call (for path rewrite / skip).
LOAD_OPGRAPH_CALL_RE = re.compile(
    r'load_opgraph_from_file\s*\(\s*"([^"]+)"\s*\)',
)
# 950 matmulA5.cpp: active vs commented GraphTensor input declarations.
_GRAPH_TENSOR_INPUT_ACTIVE_RE = re.compile(
    r"^([ \t]*)(GraphTensor\b.*\binput\b.*)$"
)
_GRAPH_TENSOR_INPUT_COMMENTED_RE = re.compile(
    r"^([ \t]*)//([ \t]*)(GraphTensor\b.*\binput\b.*)$"
)
_COPY_FROM_MM1 = "bufferManager.copyFromTensor(hostResult.data(), mm1.id())"
_COPY_FROM_MM2 = "bufferManager.copyFromTensor(hostResult.data(), mm2.id())"
_HOST_RESULT_EQ_512_RE = re.compile(
    r"if\s*\(\s*\(\s*float\s*\)\s*hostResult\s*\[\s*i\s*\]\s*!=\s*512\.0f\s*\)"
)
_HOST_RESULT_EQ_65536 = "if ((float)hostResult[i] != 65536.0f)"

_BUILD_LMHEAD_FUNC_RE = re.compile(
    r"(?ms)^([ \t]*)(?:static\s+)?GraphTensor\s+"
    r"((?:[\w:]+::)?)buildLMHead\s*\(\s*const\s+GraphTensor\s*&\s*\w+\s*,"
    r"\s*const\s+GraphTensor\s*&\s*\w+\s*\)\s*(?:const\s*)?\{"
)

# Host-project data model types (already defined outside this repo).
# Strip these definitions from TD copies; keep wire I/O helpers only.
TYPES_TO_STRIP: frozenset[str] = frozenset({
    # OpGraph side
    "DType",
    "LayoutTag",
    "MemoryPool",
    "Shape",
    "Stride",
    "TensorDesc",
    "Tensor",
    "AttrValue",
    "OpAttributes",
    "OpEdge",
    "OpNode",
    "OpGraph",
    # TileGraph side
    "EdgeOrigin",
    "TileNode",
    "TileEdge",
    "TileGraph",
})


def repo_root() -> Path:
    # python/triton_dist/mega_kernel_ascend/dev_tools/sync_td_headers.py
    # -> Triton-distributed-ascend/
    return Path(__file__).resolve().parents[4]


def patches_dir() -> Path:
    return Path(__file__).resolve().parent / "td_patches"


def _load_patch(name: str) -> str:
    path = patches_dir() / name
    if not path.is_file():
        raise SystemExit(f"missing patch template: {path}")
    return path.read_text(encoding="utf-8")


def _as_cpp_path_literal(path: Path) -> str:
    """Absolute path for a C++ string literal (forward slashes, escaped quotes)."""
    return path.resolve().as_posix().replace("\\", "\\\\").replace('"', '\\"')


def _skip_ws_and_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        break
    return i


def _match_balanced(text: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    """Return index just past the matching close_ch; open_idx points at open_ch."""
    depth = 0
    i = open_idx
    n = len(text)
    in_str = False
    str_ch = ""
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == str_ch:
                in_str = False
            i += 1
            continue
        if c in ("'", '"'):
            in_str = True
            str_ch = c
            i += 1
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced {open_ch}{close_ch} starting at {open_idx}")


def remove_type_definitions(text: str, typenames: set[str]) -> str:
    """Remove top-level struct/enum/using definitions whose names are in typenames."""
    if not typenames:
        return text

    # Longer names first to avoid partial issues (not needed for whole-word, but ok).
    name_alt = "|".join(re.escape(n) for n in sorted(typenames, key=len, reverse=True))
    # struct Name / enum class Name / using Name =
    pat = re.compile(
        rf"(?m)^[ \t]*(?:struct|enum\s+class|using)\s+(?:{name_alt})\b"
    )

    pieces: list[str] = []
    pos = 0
    for m in pat.finditer(text):
        start = m.start()
        # Skip if this match is inside a previous removal (shouldn't happen).
        if start < pos:
            continue

        # Find end of this definition.
        i = m.end()
        i = _skip_ws_and_comments(text, i)
        if m.group(0).lstrip().startswith("using"):
            # using Name = ... ;
            semi = text.find(";", i)
            if semi < 0:
                raise ValueError(f"using without ';': {m.group(0)!r}")
            end = semi + 1
        else:
            # struct/enum class: optional base, then { ... };
            if i < len(text) and text[i] == ":":
                # skip base-clause until '{'
                brace = text.find("{", i)
                if brace < 0:
                    raise ValueError(f"enum/struct without body: {m.group(0)!r}")
                i = brace
            if i >= len(text) or text[i] != "{":
                # forward decl: struct Foo;
                semi = text.find(";", i)
                if semi < 0:
                    raise ValueError(f"type without body/semi: {m.group(0)!r}")
                end = semi + 1
            else:
                after = _match_balanced(text, i, "{", "}")
                after = _skip_ws_and_comments(text, after)
                if after < len(text) and text[after] == ";":
                    end = after + 1
                else:
                    end = after

        # Also drop trailing blank lines after the removed block (keep one newline).
        while end < len(text) and text[end] in "\r\n":
            end += 1
            break
        while end < len(text) and text[end] in "\r\n":
            # collapse extra blank lines to a single blank
            peek = end
            while peek < len(text) and text[peek] in " \t\r\n":
                if text[peek] == "\n":
                    end = peek + 1
                    break
                peek += 1
            else:
                break
            # only one extra blank
            break

        pieces.append(text[pos:start])
        pos = end

    pieces.append(text[pos:])
    out = "".join(pieces)
    # tidy: no more than 2 consecutive newlines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def rewrite_header(text: str, typenames: set[str], *, is_wire_endian: bool) -> str:
    text = text.replace("namespace triton_dist", "namespace mk")
    text = text.replace(
        "#include <TritonDistributed/WireEndian.h>",
        '#include "WireEndian.h"',
    )
    # Closing comment consistency
    text = text.replace("}  // namespace triton_dist", "}  // namespace mk")

    if not is_wire_endian:
        text = remove_type_definitions(text, typenames)
        banner = (
            "// NOTE: Host data-model types (see TYPES_TO_STRIP in\n"
            "// mega_kernel_ascend/dev_tools/sync_td_headers.py) were stripped from this TD copy.\n"
            "// Provide them in namespace mk (or #include your project header)\n"
            "// before use. Kept: wire serialize/deserialize +\n"
            "// load_*_from_file / save_*_to_file.\n"
            "\n"
        )
        # Insert after #pragma once block / first includes comment area — after `#pragma once\n`
        if "#pragma once" in text:
            text = text.replace("#pragma once\n", "#pragma once\n\n" + banner, 1)
        else:
            text = banner + text
    return text


def _ensure_include(text: str, include_line: str) -> str:
    if include_line in text:
        return text
    # Insert after the last #include, or after #pragma once.
    last = None
    for m in re.finditer(r"(?m)^[ \t]*#include[^\n]*\n", text):
        last = m
    if last is not None:
        i = last.end()
        return text[:i] + include_line + "\n" + text[i:]
    if "#pragma once" in text:
        return text.replace("#pragma once\n", "#pragma once\n\n" + include_line + "\n", 1)
    return include_line + "\n" + text


def patch_matmul_950_example(text: str) -> tuple[str, list[str]]:
    """950-only verification-path edits for ``matmulA5.cpp`` (idempotent).

    - Comment active ``GraphTensor input`` lines; uncomment commented ones
      (only while unpatched markers ``mm1.id()`` / ``512.0f`` are still present).
    - ``copyFromTensor(..., mm1.id())`` -> ``mm2.id()``
    - ``hostResult[i] != 512.0f`` -> ``65536.0f``
    """
    notes: list[str] = []
    # Swap GraphTensor input comment state only once (before mm1/512 are rewritten).
    if _COPY_FROM_MM1 in text or "512.0f" in text:
        out_lines: list[str] = []
        n_commented = 0
        n_uncommented = 0
        for line in text.splitlines(keepends=True):
            eol = ""
            body = line
            if body.endswith("\r\n"):
                eol, body = "\r\n", body[:-2]
            elif body.endswith("\n"):
                eol, body = "\n", body[:-1]
            m_c = _GRAPH_TENSOR_INPUT_COMMENTED_RE.match(body)
            if m_c is not None:
                out_lines.append(f"{m_c.group(1)}{m_c.group(3)}{eol}")
                n_uncommented += 1
                continue
            m_a = _GRAPH_TENSOR_INPUT_ACTIVE_RE.match(body)
            if m_a is not None:
                out_lines.append(f"{m_a.group(1)}// {m_a.group(2)}{eol}")
                n_commented += 1
                continue
            out_lines.append(line)
        if n_commented or n_uncommented:
            text = "".join(out_lines)
            notes.append(
                f"GraphTensor input: commented {n_commented}, "
                f"uncommented {n_uncommented}"
            )
        else:
            notes.append("GraphTensor input: no matching lines")

    if _COPY_FROM_MM1 in text:
        text = text.replace(_COPY_FROM_MM1, _COPY_FROM_MM2)
        notes.append("copyFromTensor: mm1.id() -> mm2.id()")
    elif _COPY_FROM_MM2 in text:
        notes.append("copyFromTensor: mm2.id() already set")
    else:
        notes.append("copyFromTensor: mm1/mm2 pattern not found")

    if _HOST_RESULT_EQ_512_RE.search(text):
        text, n = _HOST_RESULT_EQ_512_RE.subn(_HOST_RESULT_EQ_65536, text, count=1)
        if n:
            notes.append("hostResult check: 512.0f -> 65536.0f")
    elif "65536.0f" in text and "hostResult" in text:
        notes.append("hostResult check: 65536.0f already set")
    else:
        notes.append("hostResult check: 512.0f pattern not found")

    return text, notes


def patch_matmul_cpp(
    dst_project: Path, bin_runtime: str, matmul_cpp_rel: Path, platform: str = "A3"
) -> None:
    """Replace gb.buildOpGraph() with load from absolute TD expected.bin + logs."""
    path = dst_project / matmul_cpp_rel
    if not path.is_file():
        raise SystemExit(f"matmul cpp not found: {path}")

    # utf-8-sig strips BOM so ^#include matches on the first line.
    text = path.read_text(encoding="utf-8-sig")
    changed = False
    actions: list[str] = []

    m_load = LOAD_OPGRAPH_CALL_RE.search(text)
    if m_load is not None and not BUILD_GRAPH_RE.search(text):
        old = m_load.group(1)
        if old != bin_runtime:
            text = text.replace(
                f'load_opgraph_from_file("{old}")',
                f'load_opgraph_from_file("{bin_runtime}")',
            )
            text = text.replace(
                f"[TD] start: deserialize OpGraph from {old}",
                f"[TD] start: deserialize OpGraph from {bin_runtime}",
            )
            changed = True
            actions.append(f"updated bin path -> {bin_runtime}")
    else:
        m = BUILD_GRAPH_RE.search(text)
        if m is None:
            raise SystemExit(
                f"pattern not found in {path}: "
                "OpGraph graph = gb.buildOpGraph();"
            )

        indent = m.group(1)
        replacement = (
            f'{indent}std::cout << "[TD] start: deserialize OpGraph from '
            f'{bin_runtime}" << std::endl;\n'
            f'{indent}OpGraph graph = load_opgraph_from_file("{bin_runtime}");\n'
            f'{indent}std::cout << "[TD] end: deserialize OpGraph ok, tensors="\n'
            f'{indent}          << graph.tensors.size() << " ops=" << graph.ops.size()\n'
            f'{indent}          << std::endl;\n'
        )
        text = BUILD_GRAPH_RE.sub(replacement, text, count=1)
        text = _ensure_include(text, "#include <iostream>")
        # Prefer TD OpGraph.h for load_opgraph_from_file (namespace mk types from host).
        text = _ensure_include(text, '#include "OpGraph.h"')
        changed = True
        actions.append(f"load_opgraph_from_file (bin={bin_runtime})")

    if platform == "950":
        text2, notes_950 = patch_matmul_950_example(text)
        if text2 != text:
            changed = True
        text = text2
        actions.extend(notes_950)

    if not changed:
        print(f"  skip {matmul_cpp_rel} (already patched)")
        return

    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  patched {matmul_cpp_rel}: " + "; ".join(actions))


def _patch_cmake_td_include(root: Path, dst_project: Path, label: str) -> None:
    """Insert /TD include lines next to /blade/runtime/include under root."""
    if not root.is_dir():
        print(f"  skip cmake patch ({label}): {root.relative_to(dst_project)} not found")
        return

    cmake_files = sorted(root.rglob("CMakeLists.txt"))
    if not cmake_files:
        print(f"  skip cmake patch ({label}): no CMakeLists.txt")
        return

    for cmake in cmake_files:
        text = cmake.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        out: list[str] = []
        inserted = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            if BLADE_RUNTIME_INCLUDE in line:
                td_line = line.replace(BLADE_RUNTIME_INCLUDE, TD_INCLUDE_SUFFIX, 1)
                next_line = lines[i + 1] if i + 1 < len(lines) else None
                if next_line is not None and next_line == td_line:
                    pass  # already inserted on a previous sync
                elif td_line != line:
                    out.append(td_line)
                    inserted += 1
            i += 1

        if inserted:
            body = "\n".join(out)
            if text.endswith("\n") or text.endswith("\r\n"):
                body += "\n"
            cmake.write_text(body, encoding="utf-8", newline="\n")
            rel = cmake.relative_to(dst_project)
            print(f"  patched {rel} (+{inserted} /TD include line(s))")
        else:
            if any(BLADE_RUNTIME_INCLUDE in ln for ln in lines):
                print(f"  skip {cmake.relative_to(dst_project)} (TD include already present)")
            else:
                print(f"  skip {cmake.relative_to(dst_project)} (no {BLADE_RUNTIME_INCLUDE})")


def patch_cmake_td_includes(dst_project: Path) -> None:
    for rel, label in (
        (OPERATORS_DIR_REL, "operators"),
        (MODELS_DIR_REL, "models"),
        (BLADE_BUILDER_DIR_REL, "blade/builder"),
    ):
        _patch_cmake_td_include(dst_project / rel, dst_project, label)


def _find_class_body(text: str, class_name: str) -> tuple[int, int] | None:
    """Return [start, end) of ``class ClassName { ... };`` body braces content span.

    ``start`` points at '{', ``end`` is just past matching '}'.
    Skips forward declarations such as ``class GraphBuilder;``.
    """
    # Allow extra spaces: ``class  GraphBuilder`` (common in Ascend headers).
    pat = re.compile(rf"(?m)^[ \t]*class\s+{re.escape(class_name)}\b")
    for m in pat.finditer(text):
        i = _skip_ws_and_comments(text, m.end())
        # Forward decl: class Name;
        if i < len(text) and text[i] == ";":
            continue
        # optional base clause: class Name : public Base {
        if i < len(text) and text[i] == ":":
            brace = text.find("{", i)
            if brace < 0:
                continue
            i = brace
        if i >= len(text) or text[i] != "{":
            continue
        end = _match_balanced(text, i, "{", "}")
        return i, end
    return None


def _insert_before_private(class_body: str, public_block: str) -> str | None:
    """Insert ``public_block`` just before the first ``private:`` in class body text.

    ``class_body`` is the interior of ``{ ... }`` (without outer braces).
    """
    m = re.search(r"(?m)^[ \t]*private\s*:", class_body)
    if m is None:
        return None
    insert = public_block
    if not insert.endswith("\n"):
        insert += "\n"
    return class_body[: m.start()] + insert + class_body[m.start() :]


def _insert_after_private(class_body: str, private_block: str) -> str | None:
    m = re.search(r"(?m)^[ \t]*private\s*:", class_body)
    if m is None:
        return None
    # after the private: line
    line_end = class_body.find("\n", m.end())
    if line_end < 0:
        pos = len(class_body)
    else:
        pos = line_end + 1
    insert = private_block
    if not insert.endswith("\n"):
        insert += "\n"
    return class_body[:pos] + insert + class_body[pos:]


def _append_graph_builder_methods(cpp_text: str, method_body: str) -> tuple[str, str]:
    """Append out-of-line methods; wrap in ``namespace mk`` when the file uses it.

    Returns ``(new_text, note)``. ``method_body`` must use ``GraphBuilder::`` (no ns).
    """
    body = method_body
    if not body.endswith("\n"):
        body += "\n"
    if re.search(r"(?m)^[ \t]*namespace\s+mk\b", cpp_text):
        # Keep types (OpGraph/Tensor/MemoryPool) resolved like the rest of the TU.
        wrapped = "\nnamespace mk {\n\n" + body + "\n}  // namespace mk\n"
        return cpp_text.rstrip() + "\n" + wrapped, "GraphBuilder (inside namespace mk)"
    return cpp_text.rstrip() + "\n\n" + body, "GraphBuilder"


def patch_graph_builder(dst_project: Path) -> None:
    """Add loadOpGraph / mergeOpGraph to Ascend GraphBuilder (.h + .cpp)."""
    h_path = dst_project / GRAPH_H_REL
    cpp_path = dst_project / GRAPH_CPP_REL
    if not h_path.is_file():
        print(f"  skip {GRAPH_H_REL} (not found)")
        return
    if not cpp_path.is_file():
        print(f"  skip {GRAPH_CPP_REL} (not found)")
        return

    public_inc = _load_patch("graph_builder_methods.h.inc").rstrip() + "\n"
    private_inc = _load_patch("graph_builder_private.h.inc").rstrip() + "\n"
    cpp_inc_tmpl = _load_patch("graph_builder_methods.cpp.inc")

    # --- header ---
    h_text = h_path.read_text(encoding="utf-8-sig")
    if TD_MARKER_GRAPH in h_text and "loadOpGraph" in h_text:
        print(f"  skip {GRAPH_H_REL} (already patched)")
    else:
        span = _find_class_body(h_text, "GraphBuilder")
        if span is None:
            raise SystemExit(f"class GraphBuilder not found in {h_path}")
        brace_open, brace_end = span
        interior = h_text[brace_open + 1 : brace_end - 1]
        if "loadOpGraph" in interior:
            print(f"  skip {GRAPH_H_REL} (loadOpGraph already present)")
        else:
            new_interior = _insert_before_private(interior, "\n" + public_inc)
            if new_interior is None:
                # no private: — append before closing
                new_interior = interior.rstrip() + "\n\n" + public_inc + "\n"
            else:
                # private helpers
                if "rebuildTensorNameIndex" not in new_interior:
                    with_priv = _insert_after_private(new_interior, private_inc)
                    if with_priv is not None:
                        new_interior = with_priv
            h_text = h_text[: brace_open + 1] + new_interior + h_text[brace_end - 1 :]
            h_text = _ensure_include(h_text, "#include <unordered_map>")
            h_text = _ensure_include(h_text, "#include <string>")
            h_text = _ensure_include(h_text, "#include <vector>")
            h_path.write_text(h_text, encoding="utf-8", newline="\n")
            print(f"  patched {GRAPH_H_REL}: loadOpGraph/mergeOpGraph decls")

    # --- cpp ---
    cpp_text = cpp_path.read_text(encoding="utf-8-sig")
    if TD_MARKER_GRAPH in cpp_text and "GraphBuilder::loadOpGraph" in cpp_text:
        print(f"  skip {GRAPH_CPP_REL} (already patched)")
        return
    if re.search(r"\bGraphBuilder::loadOpGraph\b", cpp_text):
        print(f"  skip {GRAPH_CPP_REL} (loadOpGraph already present)")
        return

    body = cpp_inc_tmpl.replace("__GB__", "GraphBuilder")
    cpp_text, qual_note = _append_graph_builder_methods(cpp_text, body)
    cpp_text = _ensure_include(cpp_text, "#include <stdexcept>")
    cpp_text = _ensure_include(cpp_text, "#include <string>")
    cpp_text = _ensure_include(cpp_text, "#include <unordered_map>")
    cpp_text = _ensure_include(cpp_text, "#include <vector>")
    cpp_path.write_text(cpp_text, encoding="utf-8", newline="\n")
    print(f"  patched {GRAPH_CPP_REL}: loadOpGraph/mergeOpGraph defs ({qual_note})")


def _replace_build_lmhead(text: str, new_func: str) -> tuple[str, bool]:
    """Replace the first ``GraphTensor ... buildLMHead(...) { ... }`` with ``new_func``."""
    m = _BUILD_LMHEAD_FUNC_RE.search(text)
    if m is None:
        return text, False
    # m.start() is at indent; opening '{' is at end of match - 1
    brace = text.find("{", m.start())
    if brace < 0:
        return text, False
    end = _match_balanced(text, brace, "{", "}")
    # include trailing whitespace/newlines lightly
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == "\r":
        end += 1
    if end < len(text) and text[end] == "\n":
        end += 1
    replacement = new_func
    if not replacement.endswith("\n"):
        replacement += "\n"
    return text[: m.start()] + replacement + text[end:], True


def patch_qwen_build_lmhead(dst_project: Path, bin_runtime: str) -> None:
    """Rewrite ``buildLMHead`` in qwen3_model.cpp to merge TD lmhead OpGraph bin."""
    path = dst_project / QWEN_MODEL_CPP_REL
    if not path.is_file():
        print(f"  skip {QWEN_MODEL_CPP_REL} (not found)")
        return

    text = path.read_text(encoding="utf-8-sig")
    tmpl = _load_patch("buildLMHead.cpp.inc")

    class_qual = ""
    m_exist = _BUILD_LMHEAD_FUNC_RE.search(text)
    if m_exist is not None and m_exist.group(2):
        class_qual = m_exist.group(2).rstrip(":")
    if not class_qual:
        m2 = re.search(r"\b(\w+)::buildLMHead\b", text)
        if m2 is not None:
            class_qual = m2.group(1)
    if not class_qual:
        class_qual = "QwenModel"

    new_func = (
        tmpl.replace("__CLASS__", class_qual)
        .replace("__LMHEAD_BIN_PATH__", bin_runtime)
        .rstrip()
        + "\n"
    )

    if (
        TD_MARKER_LMHEAD in text
        and "mergeOpGraph" in text
        and 't0->name != "y"' in text
        and "rstd must be rank-2" in text
    ):
        m_load = re.search(
            r'const char\*\s*kLmHeadBin\s*=\s*"([^"]+)"\s*;', text
        )
        if m_load is not None and m_load.group(1) != bin_runtime:
            text = text.replace(
                f'const char* kLmHeadBin = "{m_load.group(1)}";',
                f'const char* kLmHeadBin = "{bin_runtime}";',
            )
            text = text.replace(
                f"[TD] start: deserialize OpGraph from {m_load.group(1)}",
                f"[TD] start: deserialize OpGraph from {bin_runtime}",
            )
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"  patched {QWEN_MODEL_CPP_REL}: updated lmhead bin path")
        else:
            print(f"  skip {QWEN_MODEL_CPP_REL} (already patched)")
        return

    # Replace original / older TD patches (ops:: dump or 1-output merge).
    text2, ok = _replace_build_lmhead(text, new_func)
    if not ok:
        raise SystemExit(
            f"buildLMHead function not found in {path}; "
            "expected GraphTensor ...::buildLMHead(const GraphTensor&, const GraphTensor&)"
        )
    text2 = _ensure_include(text2, "#include <iostream>")
    text2 = _ensure_include(text2, "#include <stdexcept>")
    text2 = _ensure_include(text2, "#include <string>")
    text2 = _ensure_include(text2, "#include <unordered_map>")
    text2 = _ensure_include(text2, '#include "OpGraph.h"')
    path.write_text(text2, encoding="utf-8", newline="\n")
    print(
        f"  patched {QWEN_MODEL_CPP_REL}: buildLMHead -> mergeOpGraph "
        f"(class={class_qual}, bin={bin_runtime})"
    )


def sync(dst_project: Path, platform: str = "A3") -> None:
    src_inc = repo_root() / "include" / "TritonDistributed"
    if not src_inc.is_dir():
        raise SystemExit(f"source include dir not found: {src_inc}")
    if platform not in PLATFORM_MATMUL_CPP:
        raise SystemExit(f"unsupported platform: {platform}")

    matmul_cpp_rel = PLATFORM_MATMUL_CPP[platform]
    typenames = set(TYPES_TO_STRIP)
    td = dst_project / "TD"
    td.mkdir(parents=True, exist_ok=True)

    print(f"platform: {platform} (op={PLATFORM_OP_NAME[platform]}, cpp={matmul_cpp_rel})")
    print(f"types to strip ({len(typenames)}): {', '.join(sorted(typenames))}")
    print(f"dst: {td}")

    for name in HEADERS:
        src = src_inc / name
        if not src.is_file():
            raise SystemExit(f"missing source header: {src}")
        raw = src.read_text(encoding="utf-8")
        out = rewrite_header(raw, typenames, is_wire_endian=(name == "WireEndian.h"))
        dst = td / name
        dst.write_text(out, encoding="utf-8", newline="\n")
        print(f"  wrote {dst.relative_to(dst_project)} "
              f"({len(raw)} -> {len(out)} chars)")

    # two_matmul golden
    bin_src = repo_root() / PLATFORM_OPGRAPH_BIN[platform]
    if not bin_src.is_file():
        raise SystemExit(f"missing expected bin: {bin_src}")
    out_bin = bin_src.read_bytes()
    bin_dst = td / TD_OPGRAPH_BIN_NAME
    bin_dst.write_bytes(out_bin)
    print(f"  copied {bin_dst.relative_to(dst_project)} "
          f"from {bin_src.name} ({len(out_bin)} bytes)")

    bin_runtime = _as_cpp_path_literal(bin_dst)
    print(f"  runtime bin path (absolute): {bin_runtime}")
    patch_matmul_cpp(dst_project, bin_runtime, matmul_cpp_rel, platform=platform)

    # lmhead golden
    lm_src = repo_root() / PLATFORM_LMHEAD_BIN[platform]
    if not lm_src.is_file():
        raise SystemExit(f"missing lmhead expected bin: {lm_src}")
    lm_bytes = lm_src.read_bytes()
    lm_dst = td / TD_LMHEAD_BIN_NAME
    lm_dst.write_bytes(lm_bytes)
    print(f"  copied {lm_dst.relative_to(dst_project)} "
          f"from {lm_src.name} ({len(lm_bytes)} bytes)")
    lm_runtime = _as_cpp_path_literal(lm_dst)
    print(f"  lmhead bin path (absolute): {lm_runtime}")

    patch_graph_builder(dst_project)
    patch_qwen_build_lmhead(dst_project, lm_runtime)
    patch_cmake_td_includes(dst_project)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "dst_project",
        type=Path,
        help="path to the sibling Ascend project (TD/ will be created under it)",
    )
    ap.add_argument(
        "-p",
        "--platform",
        choices=sorted(PLATFORM_MATMUL_CPP),
        default="A3",
        help="target Ascend platform: A3 patches matmul/matmul.cpp; "
             "950 patches matmulA5/matmulA5.cpp "
             "(default: A3)",
    )
    args = ap.parse_args(argv)

    dst = args.dst_project.expanduser().resolve()
    if not dst.is_dir():
        print(f"error: dst project not a directory: {dst}", file=sys.stderr)
        return 2

    sync(dst, platform=args.platform)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
