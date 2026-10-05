#!/usr/bin/env bash
# Copyright (c) 2026 Ranvier Systems. Apache-2.0.
# Install only a checksum-verified public release binary, never build the compiler.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_root/toolchain/lean-cuda.env"
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) platform=linux ;;
  Linux/aarch64|Linux/arm64) platform=linux_aarch64 ;;
  *) echo "Supported binary platforms: Linux x86_64 and AArch64" >&2; exit 1 ;;
esac
archive="lean-$LEAN_CUDA_NIGHTLY_VERSION-$platform.tar.zst"
base="https://github.com/ranvier-labs/lean4-cuda-nightly/releases/download/$LEAN_CUDA_NIGHTLY_TAG"
download_dir="$repo_root/.lake/nightly-download"
destination="$repo_root/.lake/toolchains/$LEAN_CUDA_NIGHTLY_TAG"
mkdir -p "$download_dir"
for name in "$archive" "$archive.sha256"; do
  if [[ ! -f "$download_dir/$name" ]]; then
    curl --fail --location --retry 3 "$base/$name" -o "$download_dir/$name.part"
    mv "$download_dir/$name.part" "$download_dir/$name"
  fi
done
(cd "$download_dir" && sha256sum --check "$archive.sha256") >&2
if [[ ! -x "$destination/bin/lean" ]]; then
  mkdir -p "$destination"
  tar --zstd -xf "$download_dir/$archive" --strip-components=1 -C "$destination"
fi
[[ $("$destination/bin/lean" --version) == *"$LEAN_CUDA_NIGHTLY_VERSION"* ]]
[[ $("$destination/bin/lean" --features) == *"[CUDA]"* ]]
printf '%s\n' "$destination"
