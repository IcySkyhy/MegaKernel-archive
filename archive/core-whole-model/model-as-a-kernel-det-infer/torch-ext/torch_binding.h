#pragma once

#include <torch/torch.h>

void mak_run(at::Tensor prog, at::Tensor barrier, int64_t pos,
             int64_t step_slot, int64_t max_k);

void mak_run_steps(at::Tensor prog, at::Tensor barrier, int64_t pos0,
                   int64_t steps, int64_t slot0, int64_t max_k);

void mak_run_seq(at::Tensor prog, at::Tensor barrier, int64_t pos0,
                 int64_t steps, int64_t slot0, int64_t prompt_len,
                 int64_t max_k, int64_t chunk_m);

at::Tensor mak_run_phased(at::Tensor prog, int64_t pos, int64_t step_slot,
                          bool timed, int64_t max_k);

void mak_run_batch(at::Tensor prog, at::Tensor barrier, at::Tensor pos_b,
                   int64_t steps, int64_t slot0, int64_t max_k,
                   int64_t kv_bstride, int64_t stage_elems);

int64_t mak_num_blocks(at::Tensor prog, int64_t max_k);

int64_t mak_batch_maxb();

double mak_bw_per_sm(at::Tensor probe);
