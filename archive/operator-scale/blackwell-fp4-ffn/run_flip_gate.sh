#!/bin/bash
# Flip-loss accuracy gate: HW-honest pairwise-4:8 masks (what the sm120 nvf4
# sparse engine ACTUALLY requires) vs the elementwise masks the old relPJ
# numbers assumed. blk: 0=dense fp4, -1=elementwise-transposable weights,
# -2=HW pairwise-transposable weights. g48: 1=pairwise batch-pruned dD for dW
# (the backprop-1000 sparse-dW path).
# args order: S M steps lr w2rnd gamma sdrop w2amp homo clip cos L fp4eval qat prec blk g48
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd ~/Desktop/code/double-gemm
STEPS=${1:-3000}
OUT=/tmp/flip_gate.out; : > $OUT
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
sudo nvidia-smi -lgc 3090 -pl 300 >/dev/null 2>&1 || true
run(){
  echo "######## $1" | tee -a $OUT
  timeout 400 /tmp/flip48 1024 1024 $STEPS 1e-3 0 1.0 0 1.0 0.5 0 0 8 1 1 0 $2 $3 2>&1 \
    | grep -E 'step +[0-9]+  loss' | tee -a $OUT
  echo | tee -a $OUT
}
run "A dense fp4 (blk=0)"                            0 0
run "B elementwise-transposable weights (blk=-1)"   -1 0
run "C HW pairwise-transposable weights (blk=-2)"   -2 0
run "D backprop-1000 full: blk=-2 + g48 batch-grad" -2 1
sudo nvidia-smi -rgc >/dev/null 2>&1 || true
sudo nvidia-smi -pl 300 >/dev/null 2>&1 || true
echo "[gate done, clocks reverted]" | tee -a $OUT
