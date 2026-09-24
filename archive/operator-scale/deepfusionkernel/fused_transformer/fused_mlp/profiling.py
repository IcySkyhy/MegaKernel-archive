"""
Evaluate single kernel performance across batch sizes.
"""

import glob
import os
import random
import shutil
import time
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import triton
import wandb

from .load_autotune_configs import apply_autotuner_cache_records
from fused_transformer.utils.constants import DATA_DIR, EXPERIMENT_DIR
from . import TORCH_HAS_FP8, NAME, KERNEL_LIST
from .fused_gmlp import (
    gatedmlp_torch,
    get_autotune_config,
)
from ..utils import WandbLogger, is_cuda


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REF_LIB = "cuBLAS" if is_cuda() else "rocBLAS"
REF_FN = gatedmlp_torch
INTERM_RATIO = 3.5
Z = 1


def profile_compare_batch_size(
    cfg, providers: List[str], batch_size_list: List[int], fp8_inputs=False
):
    # WandB logger
    experiment_name = cfg.wandb.name
    group_name = cfg.wandb.group
    job_type = cfg.wandb.type
    mode = "online"
    autotune_config = get_autotune_config()  # Using default config just for reference
    hidden_size = cfg.get("hidden_size", 1024)
    intermediate_size = cfg.get("intermediate_size", 1024 * INTERM_RATIO)
    hyperparameters = {
        "INTERM_RATIO": INTERM_RATIO,
        "BLOCK_M": list(set([c.all_kwargs()["BLOCK_M"] for c in autotune_config])),
        "BLOCK_N": list(set([c.all_kwargs()["BLOCK_N"] for c in autotune_config])),
        "BLOCK_K": list(set([c.all_kwargs()["BLOCK_K"] for c in autotune_config])),
        "BLOCK_T": list(set([c.all_kwargs()["BLOCK_T"] for c in autotune_config])),
        "num_stages": list(
            set([c.all_kwargs()["num_stages"] for c in autotune_config])
        ),
        "num_warps": list(set([c.all_kwargs()["num_warps"] for c in autotune_config])),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }
    repeat_time = cfg.get("repeat_time", 100)
    logger = WandbLogger(
        experiment_name,
        group_name,
        job_type,
        mode=mode,
        config={**hyperparameters, **OmegaConf.to_container(cfg, resolve=True)},
    )

    assert all(
        p in NAME or p == "ref_lib" or p == "ref_lib-compile" for p in providers
    ), f"Kernel providers must be ``ref_lib'', ``ref_lib-compile'', or in {list(NAME.keys())}."
    providers = [
        (
            REF_LIB.lower()
            if p == "ref_lib"
            else REF_LIB.lower() + "-compile" if p == "ref_lib-compile" else p
        )
        for p in providers
    ]
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        raise ValueError("FP8 input not supported yet")

    # Load pickled best configs
    if isinstance(cfg.best_config_path, str):
        cfg.best_config_path = [cfg.best_config_path]
    for path in cfg.best_config_path:
        best_config_pkl_path = os.path.join(DATA_DIR, f"{path}.pkl")
        autotuners = [k for k in KERNEL_LIST if isinstance(k, triton.runtime.Autotuner)]
        apply_autotuner_cache_records(
            best_config_pkl_path=best_config_pkl_path,
            autotuners=autotuners,
        )

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],  # Argument names to use as an x-axis for the plot
            x_vals=batch_size_list,  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=(providers),  # Label name for the lines
            line_names=(providers),  # Line styles
            styles=[
                ("red", "-"),
                ("orange", "-"),
                ("green", "-"),
                ("blue", "-"),
                ("cyan", "-"),
                ("purple", "-"),
                ("red", "--"),
                ("orange", "--"),
                ("green", "--"),
                ("blue", "--"),
                ("cyan", "--"),
                ("purple", "--"),
                ("grey", "-"),
            ],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="fused-gmlp-performance-"
            + (
                "fp16" if not fp8_inputs else "fp8"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={
                "fp8_inputs": fp8_inputs,
                "Z": Z,
                "N": hidden_size,
                "K": intermediate_size,
                "T": hidden_size,
            },
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(Z, M, N, K, T, provider, fp8_inputs):
        print(f"Running M {M}, provider {provider}")
        x = torch.randn((Z, M, N), device=DEVICE, dtype=torch.float16)
        u = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
        g = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
        d = torch.randn((K, T), device=DEVICE, dtype=torch.float16)
        if TORCH_HAS_FP8 and fp8_inputs:
            raise NotImplementedError
        quantiles = [0.5, 0.2, 0.8]
        try:
            if provider == REF_LIB.lower():
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: REF_FN(x, u, g, d), quantiles=quantiles, rep=repeat_time
                )
            elif provider == REF_LIB.lower() + "-compile":
                fn = torch.compile(REF_FN)
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: fn(x, u, g, d), quantiles=quantiles, rep=repeat_time
                )
            else:
                fn = NAME[provider]
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: fn(x, u, g, d), quantiles=quantiles, rep=repeat_time
                )
            perf = (
                lambda ms: (
                    2 * (2 * M * N * K)  # X@U and X@G
                    + 4 * M * K  # sigmoid(accg), involving 1 / (1 + exp(-x))
                    + 2 * M * K  # acc2 = acc1 * (accg * acca)
                    + 2 * M * K * T  # acc2@D
                ) * Z * 1e-12 / (ms * 1e-3)  # fmt: skip
            )

            logger.log(
                data={
                    "provider": provider,
                    "dtype": "fp16" if not fp8_inputs else "fp8",
                    "Z": Z,
                    "M": M,
                    "N": N,
                    "K": K,
                    "T": T,
                    "TFLOPs/s": perf(ms),
                    "TFLOPs/s 80% quantile": perf(max_ms),
                    "TFLOPs/s 20% quantile": perf(min_ms),
                }
            )
            logger.commit()
            return perf(ms), perf(max_ms), perf(min_ms)
        except Exception as e:
            print(
                f"Error at M={M}, provider={provider}, fp8_inputs={fp8_inputs}. Error message: {e}"
            )
            logger.log(data={})
            logger.commit()
            return None, None, None

    benchmark.run(
        show_plots=False,
        print_data=False,
        save_path=os.path.join(
            EXPERIMENT_DIR,
            "triton",
            "fused_mlp-results-profile",
        ),
    )
    logger.close()


@hydra.main(version_base=None, config_path="configs", config_name="MUSTBESPECIFIED")
def main(cfg: DictConfig):
    if not HydraConfig.initialized():
        HydraConfig().set_config(cfg)

    # Retrieve Hydra's output directory
    output_dir = HydraConfig.get().runtime.output_dir
    # Save all .py files from the original working directory to Hydra's output directory
    original_cwd = hydra.utils.get_original_cwd()
    src_py_files = glob.glob(os.path.join(original_cwd, "*.py"))
    for file_path in src_py_files:
        shutil.copy(file_path, output_dir)
    shutil.copy(
        f"fused_transformer/fused_mlp/configs/{HydraConfig.get().job.config_name}.yaml",
        output_dir,
    )

    providers = cfg.providers
    if isinstance(providers, str):
        providers = [providers]
    batch_sizes = cfg.batch_sizes
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    fp8_inputs = cfg.fp8_inputs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    profile_compare_batch_size(
        cfg,
        providers,
        batch_sizes,
        fp8_inputs,
    )


if __name__ == "__main__":
    main()
