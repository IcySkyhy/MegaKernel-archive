# Third-party software and assets

This repository does not redistribute Qwen checkpoint weights or tokenizer assets. Users supply
those files separately and are responsible for the model publisher's license and acceptable-use
terms.

## Build and test dependencies

- Lean CUDA backend: `https://github.com/ranvier-labs/lean4-cuda-backend.git` at
  `7fcc8ef57ee86bdb47c1f777383b02e4fe165500`. The checkout contains Lean 4 and the experimental
  CUDA backend and is licensed under Apache-2.0; see its `LICENSE` and source notices.
- LeanTest: `https://github.com/cpehle/lean_test.git` at
  `fbfea9210f4f2eb1fc7c3b9e3c86f57c5dfa6e28`. Source files identify Christian Pehle as copyright
  holder and state Apache-2.0. It is fetched as a test dependency and is not vendored here.

The authoritative revisions are machine-readable in `toolchain/lean-cuda.env` and
`lake-manifest.json`.

## Optional Python tools

Dataset preparation examples use the separately installed `tokenizers` package. Reference parity
tests may install PyTorch and Hugging Face Transformers. These packages are not vendored and retain
their own licenses.

## External model compatibility

Qwen names and model identifiers are used solely to describe checkpoint compatibility. No
affiliation with or endorsement by the model publisher is implied.

When adding a dependency or copied implementation, record its source revision and license here,
and preserve any required copyright or NOTICE text.
