# Reconstructing a Megakernel with CUDA PDL

This repository is the code and artifact companion for **“81 Kernels, 97.4%
of a Megakernel: Reconstructing Persistence with PDL.”** On an NVIDIA H100,
an 81-kernel CUDA Graph using Programmatic Dependent Launch (PDL) reaches
97.4% of the matched persistent megakernel's end-to-end throughput for the
P32/D128 Llama-3.2-1B decode workload.

Read the [engineering note](docs/reconstructing-a-megakernel-with-pdl.md) or
jump to the [experiment guide](experiments/pdl-reconstruction/README.md).

## Repository scope

`main` is the canonical, buildable source tree. It contains:

- the original persistent ThunderKittens megakernel;
- the correct 13-page, three-stage direct-TK PDL reconstruction;
- matched arrival-vs-issue trigger controls;
- per-edge, compact-grid, resource, microtile, and VTP controls;
- the harnesses and curated summaries used by the engineering note.

The unrelated analysis-paper workspace and its raw profiling archive are not
part of this repository.

This history starts from HazyResearch/Megakernels commit
`7309cec801537b61fea3b50d7dfe454a6cde578e`; its ThunderKittens submodule is
pinned at `664c108d16f12707a73d3072ab525f26fb2b4f62`. The original upstream
README is preserved in [docs/upstream-megakernels-readme.md](docs/upstream-megakernels-readme.md).

## Build

The reported measurements used an NVIDIA H100 SXM5 80 GB, CUDA 12.8,
PyTorch 2.7.0+cu128, and Python 3.12. CUDA 12.8 is a build requirement for the
persistent baseline, not just a PyTorch wheel choice: CUDA 13.1's H100 ptxas
rejects the VM kernel's `setmaxnreg` instructions. CUDA 13.1 can still build
the direct/PDL-only module.

```bash
git clone --recurse-submodules \
  https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction.git
cd pdl-megakernel-reconstruction

python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

export THUNDERKITTENS_ROOT="$PWD/ThunderKittens"
export MEGAKERNELS_ROOT="$PWD"
make -C demos/low-latency-llama \
  PYTHON="$VIRTUAL_ENV/bin/python" GPU=H100
```

That command builds the `mk_llama` and `mk_mlp_simt` extension modules needed
by the direct/PDL body harness. By design, the resulting `mk_llama` module does
**not** export the persistent `mk_llama()` entry.

To build the persistent baseline and direct entries together for the headline
end-to-end comparison, use the accepted CUDA 12.8 compiler and opt in:

```bash
make -C demos/low-latency-llama clean \
  PYTHON="$VIRTUAL_ENV/bin/python"
make -C demos/low-latency-llama \
  PYTHON="$VIRTUAL_ENV/bin/python" GPU=H100 \
  NVCC=/usr/local/cuda-12.8/bin/nvcc PERSISTENT_VM=1
python experiments/pdl-reconstruction/validate_bindings.py \
  --mk-dir demos/low-latency-llama --require-persistent
```

Adjust the CUDA 12.8 path for your installation. Alternatively, keep a
direct-only candidate module and pass an original CUDA-12.8 persistent module
through `NATIVE_REFERENCE_MK` when running the e2e harness.

## Run the body experiment

After building, run a short smoke first:

```bash
POSITION_END=32 WARMUP=1 ITERATIONS=2 \
  experiments/pdl-reconstruction/run_body.sh
```

The default is the accepted `pdl_13page_issue_r96` configuration over the
16-layer P32/D128 body. Set `IMPLEMENTATION=pdl_13page_arrival_r96` or
`IMPLEMENTATION=pdl_13page_issue_r96_vtp56` for the matched controls. Local
outputs go to `artifacts/local/` and are ignored by Git.

The full real-weight end-to-end comparison has its own portable runner and a
pinned checkpoint contract; see the experiment guide for the exact command.

## Results at a glance

| Matched complete decode step | Latency |
|---|---:|
| Persistent VM | 983.127 us |
| 81-kernel PDL chain | 1009.711 us |
| Default-dependency chain | 1169.071 us |

At complete-decode-step scope, the PDL chain recovers 85.7% of the gap from
the default-dependency chain to the persistent path and finishes within 2.704%
of the persistent path.

The separately measured synthetic-weight 16-layer body arm is the source of
the 82.8% recovery figure:

| Synthetic 16-layer body | Latency |
|---|---:|
| Persistent VM | 776.942 us |
| 81-kernel PDL chain | 809.626 us |
| Default-dependency chain | 967.162 us |

The 776.942-us persistent value was measured with the completion-control arm
and is used as the common persistent baseline for both synthetic-body arms;
the PDL-arm process measured 777.123 us.

Curated inputs and summaries are under
[experiments/pdl-reconstruction/results](experiments/pdl-reconstruction/results).

## Provenance and rescue branches

The implementation originally existed only in detached, dirty experiment
trees. Their filtered source sets were verified byte-for-byte and preserved
before consolidation:

| Branch | Purpose | Source-set SHA-256 |
|---|---|---|
| `rescue/remote-direct-tk-pdl-20260723` | direct-TK PDL base | `cead9d0a...2bf00ff` |
| `rescue/remote-13page-revalidation-20260804` | trigger revalidation | `c1bff29c...bb25f48` |
| `rescue/remote-all-micro-ablation-20260724` | resource/micro controls | `f388ef01...ab03f2b1` |
| `rescue/remote-vtp56-isolated-20260723` | VTP control | `a8344776...4b333d` |

Generated binaries and profiler outputs were deliberately excluded from the
rescue branches. Their commit messages record the full hashes and source
locations. The raw samples referenced by the summaries are published in the
[blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0).

## License and attribution

The code is released under the MIT License inherited from HazyResearch's
Megakernels. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
