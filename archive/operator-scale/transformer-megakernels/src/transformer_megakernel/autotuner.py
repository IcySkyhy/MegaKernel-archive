import logging

import torch

from transformer_megakernel.config import KernelConfig

logger = logging.getLogger(__name__)

MATMUL_DEFAULTS = dict(
    bK=64, num_stages=2,
    output_pad=8, use_tma_reduce=True,
)


MATMUL_BLOCK_CANDIDATES = ((64, 64), (64, 128), (128, 64), (128, 128))
WARPS_PER_ROW_CANDIDATES = (1, 2, 4)
ATTN_CANDIDATES = ((64, 64),)


def _rms_block_candidates(warps_per_row: int, bM: int) -> list[int]:
    # chunks = bRMS // num_sets must be >= 2 (pipeline depth guard) and
    # bM % bRMS == 0 (scheduler coupling). Try the minimum valid depth and
    # one step up.
    num_sets = 4 // warps_per_row
    return [c * num_sets for c in (2, 4) if bM % (c * num_sets) == 0]


def build_kernel_config(num_sms, *, bM, bN, warps_per_row, rows_per_rms_block,
                        block_q, block_kv) -> KernelConfig:
    return KernelConfig(
        **MATMUL_DEFAULTS,
        bM=bM, bN=bN,
        num_sms=num_sms, max_works=0,
        warps_per_row=warps_per_row, rows_per_rms_block=rows_per_rms_block,
        block_q=block_q, block_kv=block_kv,
    )


def _time_trial(megakernel, kernel_config, num_warmup=2, num_trials=3):
    try:
        megakernel._trial_build(kernel_config)

        dummy = torch.zeros(
            megakernel.num_tokens, megakernel.input_config.embed_dim,
            dtype=torch.bfloat16, device="cuda",
        )

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        for _ in range(num_warmup):
            megakernel._run(dummy)
        start.record()
        for _ in range(num_trials):
            megakernel._run(dummy)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_trials
    except Exception as e:
        logger.warning(f"Skipping invalid config {kernel_config}: {e}")
        return None


def autotune(megakernel) -> KernelConfig:
    num_sms = megakernel.num_sms
    block_q, block_kv = ATTN_CANDIDATES[0]

    best_cfg, best_t = None, float("inf")
    for bM, bN in MATMUL_BLOCK_CANDIDATES:
        for wpr in WARPS_PER_ROW_CANDIDATES:
            for bRMS in _rms_block_candidates(wpr, bM):
                cfg = build_kernel_config(num_sms, bM=bM, bN=bN, warps_per_row=wpr,
                                           rows_per_rms_block=bRMS,
                                           block_q=block_q, block_kv=block_kv)
                t = _time_trial(megakernel, cfg)
                logger.info(f"AUTOTUNING bM={bM}, bN={bN}, warps_per_row={wpr}, "
                            f"rows_per_rms_block={bRMS} -> {t} ms")
                if t is not None and t < best_t:
                    best_cfg, best_t = dict(bM=bM, bN=bN, warps_per_row=wpr,
                                             rows_per_rms_block=bRMS), t

    if best_cfg is None:
        raise RuntimeError(
            "Autotuner found no valid matmul/RMSNorm config for "
            f"embed_dim={megakernel.input_config.embed_dim}."
        )

    best_attn, best_t = (block_q, block_kv), float("inf")
    for bq, bkv in ATTN_CANDIDATES:
        cfg = build_kernel_config(num_sms, **best_cfg, block_q=bq, block_kv=bkv)
        t = _time_trial(megakernel, cfg)
        logger.info(f"AUTOTUNING block_q={bq}, block_kv={bkv} -> {t} ms")
        if t is not None and t < best_t:
            best_attn, best_t = (bq, bkv), t

    logger.info(f"Autotune picked bM={best_cfg['bM']}, bN={best_cfg['bN']}, "
                f"warps_per_row={best_cfg['warps_per_row']}, "
                f"rows_per_rms_block={best_cfg['rows_per_rms_block']}, "
                f"block_q={best_attn[0]}, block_kv={best_attn[1]}")
    return build_kernel_config(num_sms, **best_cfg,
                               block_q=best_attn[0], block_kv=best_attn[1])