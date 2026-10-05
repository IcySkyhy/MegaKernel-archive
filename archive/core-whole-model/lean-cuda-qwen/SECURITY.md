# Security policy

## Scope

The Lean/CUDA library, dataset parsers, checkpoint parsers, and build scripts accept vulnerability
reports. Report suspected issues privately through the security-reporting facility of the hosting
repository rather than opening a public issue.

## Demo server

`examples/qwen38_chat/server.py` is an intentionally development-only bridge. It has no
authentication, authorization, TLS, request isolation, or production hardening and may listen on
all interfaces. Do not expose it to an untrusted network. Run it only on an isolated development
machine or behind controls you operate. This demo server is outside the supported security scope.

## Experimental compiler and GPU code

The pinned Lean CUDA backend is alpha software. CUDA kernels assume validated host-side shapes and
trusted local model files. This repository is not a sandbox for hostile models, token streams,
rollout binaries, or adapter checkpoints.

## Supported versions

Only the current default branch and the exact dependency revisions in `toolchain/lean-cuda.env`
are supported. Security fixes may update those pins and need not retain compatibility with old
experimental binary formats.
