#!/bin/bash
# Is the pairwise-4:8 divergence a SCHEDULE artifact or mask CAPACITY?
# Gentler: lr 5e-4, homotopy over ALL 3000 steps (homo=1.0), cosine decay on.
# args: S M steps lr w2rnd gamma sdrop w2amp homo clip cos L fp4eval qat prec blk g48
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd ~/Desktop/code/double-gemm
OUT=/tmp/flip_sched.out; : > $OUT
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
sudo nvidia-smi -lgc 3090 -pl 300 >/dev/null 2>&1 || true
run(){
  echo "######## $1" | tee -a $OUT
  timeout 400 /tmp/flip48 1024 1024 3000 5e-4 0 1.0 0 1.0 1.0 0 1 8 1 1 0 $2 $3 2>&1 \
    | grep -E 'step +[0-9]+  loss' | tee -a $OUT
  echo | tee -a $OUT
}
run "C-gentle HW pairwise weights (blk=-2)"          -2 0
run "D-gentle backprop-1000 full (blk=-2 + g48)"     -2 1
run "B-gentle elementwise-transposable (blk=-1)"     -1 0
sudo nvidia-smi -rgc >/dev/null 2>&1 || true
sudo nvidia-smi -pl 300 >/dev/null 2>&1 || true
echo "[sched test done, reverted]" | tee -a $OUT
