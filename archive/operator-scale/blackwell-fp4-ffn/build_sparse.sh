#!/bin/bash
# Splice-build for sm120 sparse FP4 sources (13.3 ptxas into the 13.1 pipeline).
# usage: build_sparse.sh <src.cu> <out-binary> [extra nvcc flags...]
set -e
SRC=$(realpath "$1"); OUT="$2"; shift 2; XFLAGS="$@"
CUT=${CUT:-$HOME/Desktop/code/cutlass}
# Reboot-safe install first; retain the historical /tmp recipe as a fallback.
P33=$(find "$HOME/.local/share/double-gemm/ptxvenv" /tmp/ptxvenv \
  -path '*/site-packages/nvidia/cu13/bin/ptxas' -type f -executable 2>/dev/null | head -n 1)
[ -n "$P33" ] && [ -x "$P33" ] || { echo "13.3 ptxas missing (README sparse recipe)"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
SRC_DIR=$(dirname "$SRC")
# -DPHASE selects a generated, project-local collective. Generation is anchored
# against CUTLASS 4.4.2 and fails loudly if NVIDIA changes the source seam.
if [[ " $XFLAGS " == *" -DPHASE "* ]]; then
  python3 "$SRC_DIR/make_sparse_phase.py" "$CUT" "$SRC_DIR/sm120_sparse_phase.hpp"
fi
INC="-I include -I tools/util/include -I examples/common -I $SRC_DIR"
cd $CUT
D=$(mktemp -d /tmp/spliceXXXX)
nvcc --dryrun -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
  --expt-relaxed-constexpr -DNDEBUG $INC $XFLAGS \
  "$SRC" -o "$OUT" 2>&1 | sed 's/^#\$ //' > $D/dry.sh
{ echo 'set -e'; echo "cd $CUT"
  sed -e "1,13s|^\([A-Za-z_][A-Za-z0-9_]*\)=\(.*\)|export \1='\2'|" -e '14,18d' \
      -e "s|^ptxas -arch=sm_120a|$P33 -arch=sm_120a|" \
      -e 's|^rm /tmp/tmpxft|rm -f /tmp/tmpxft|' $D/dry.sh
} > $D/hyb.sh
bash $D/hyb.sh
rm -rf $D
echo "built $OUT"
