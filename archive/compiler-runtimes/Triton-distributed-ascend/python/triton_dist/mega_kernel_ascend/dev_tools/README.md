# mega_kernel_ascend / dev_tools

**Development helpers** for docking this package to a sibling Ascend (blade)
tree. Not used at megakernel runtime.

中文：[README.zh.md](README.zh.md)

| File | Role |
|------|------|
| `sync_td_headers.py` | Copy OpGraph headers/goldens into `TD/` and patch GraphBuilder / examples |
| `td_patches/` | C++ templates injected by that script |

Ops tests (runtime): [`../test/ops/README.md`](../test/ops/README.md).

## OpGraph / TD sync (`sync_td_headers.py`)

Copy stripped OpGraph wire headers and platform goldens into another Ascend
project's `TD/` folder, then patch example / GraphBuilder / Qwen sources so they
load those bins. Patches are idempotent (safe to re-run).

### Layout

```text
<parent>/
  Triton-distributed-ascend/          # this repo
  <other-ascend-project>/             # destination
    TD/
      OpGraph.h  TileGraph.h  WireEndian.h
      two_matmul_opgraph.expected.bin
      lmhead_opgraph.expected.bin
    examples/operators/matmul/matmul.cpp           # -p A3
    examples/operators/matmulA5/matmulA5.cpp       # -p 950
    examples/models/qwen3-30B/qwen3_model.cpp     # buildLMHead
    blade/builder/include/graph/graph.h
    blade/builder/src/graph/graph.cpp
```

### Usage

From repo root (`PYTHONPATH` includes `python/`, or the package is installed):

```bash
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers /path/to/other-ascend-project
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p A3
python -m triton_dist.mega_kernel_ascend.dev_tools.sync_td_headers ../my-ascend-app -p 950
```

| `-p` | Matmul example | two_matmul / lmhead golden source |
|------|----------------|-----------------------------------|
| `A3` (default) | `examples/operators/matmul/matmul.cpp` | `*.expected.bin` |
| `950` | `examples/operators/matmulA5/matmulA5.cpp` | `*.950.expected.bin` |

Golden sources live under [`../test/ops/`](../test/ops/).

### What it does

1. Write `TD/OpGraph.h`, `TileGraph.h`, `WireEndian.h`
   - `namespace triton_dist` → `namespace mk`
   - rewrite WireEndian include to `"WireEndian.h"`
   - strip host data-model types (`TYPES_TO_STRIP`); keep serialize / file I/O
2. Copy platform `two_matmul` + `lmhead` goldens into `TD/` (fixed dest names)
3. Patch matmul cpp: `gb.buildOpGraph()` → `load_opgraph_from_file(<abs TD path>)`
   - `-p 950` also flips GraphTensor `input` comments, `mm1`→`mm2`, `512`→`65536`
4. Patch GraphBuilder (`graph.h` / `graph.cpp`): `loadOpGraph` / `mergeOpGraph` /
   `findTensorId` / …
   - skips `class GraphBuilder;` forward declarations
5. Rewrite `qwen3_model.cpp` `buildLMHead` to merge lmhead bin + `mergeOpGraph`
   (`add_rms_norm` StaticOpResult<3> as `y[M,H]`, `x[M,H]`, `rstd[M,1]`;
   ranks>1 `all_gather` after merge)
6. Insert `/TD` include lines next to `/blade/runtime/include` under
   `examples/operators`, `examples/models`, and `blade/builder` CMakeLists

Bin paths embedded in patched cpp are **absolute** paths to the destination
`TD/*.bin` so build cwd does not matter.

### Patch templates (`td_patches/`)

| File | Role |
|------|------|
| `graph_builder_methods.h.inc` | Public GraphBuilder decls |
| `graph_builder_private.h.inc` | Private helpers |
| `graph_builder_methods.cpp.inc` | Out-of-line `loadOpGraph` / `mergeOpGraph` |
| `buildLMHead.cpp.inc` | Replacement `buildLMHead` body |

Placeholders: `__GB__` → `GraphBuilder`, `__CLASS__` → model class,
`__LMHEAD_BIN_PATH__` → absolute lmhead bin path.

### Requirements on the Ascend tree

- GraphBuilder members used by the patch: `g_`, `tensorIdByName_`, `fusionGroups_`
- `GraphTensor::id()` for wiring `mlp_output` / `residual`
- Model fields used by `buildLMHead`: `gb_`, `opsArgs_.ranks` / `rankId`,
  `vocabSize_`, `weightCache_`, `finalNormOutput_`, `lmHeadMatmulOutput_`,
  `allGatherOutput_`
- Product path: **merge** `TD/lmhead_opgraph.expected.bin`. Bin must export
  `add_rms_norm` **StaticOpResult&lt;3&gt;** as Ascend does: **`y[M,H]`**,
  **`x[M,H]`**, **`rstd[M,1]`**. Ascend tiling derives dimExtents from
  `op.outputs` → `desc.shape`. `matmul_lmhead` and `finalNormOutput_` use
  **`y`** (`outputs[0]`). `mergeOpGraph` calls `inferOpTilingFromSchema`
  per appended op (wire to same schema tiling as `ops::`). ranks>1 still
  runs Ascend `ops::all_gather` after merge.

Missing optional paths (e.g. no `qwen3_model.cpp`) are skipped with a log line;
missing required matmul cpp for `-p` fails the run.
