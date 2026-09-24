"""
Running triton autotuning for fused MLP kernels on single MLP layer.
"""
import glob
import os
import pickle
import random
import shutil
import logging
from typing import List, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import triton
import yaml

from fused_transformer.utils.constants import DATA_DIR, EXPERIMENT_DIR
from .. import TORCH_HAS_FP8, NAME, KERNELS
from ..fused_gmlp import (
    gatedmlp_torch,
    get_autotune_config,
)
from ...utils import WandbLogger, is_cuda, is_hip


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REF_LIB = "cuBLAS" if is_cuda() else "rocBLAS"
REF_FN = gatedmlp_torch
INTERM_RATIO = 3.5
Z = 1


def autotune_compare_batch_size(
    cfg,
    providers: List[str],
    batch_size_list: List[int],
    fp8_inputs=False,
):
    # WandB logger
    experiment_name = cfg.wandb.name
    group_name = cfg.wandb.group
    job_type = cfg.wandb.type
    mode = "online"
    autotune_config = get_autotune_config()  # Using default config just for reference
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
    }
    hidden_size = cfg.get("hidden_size", None)
    intermediate_size = cfg.get("intermediate_size", None)
    logger = WandbLogger(
        experiment_name,
        group_name,
        job_type,
        mode=mode,
        config={**hyperparameters, **OmegaConf.to_container(cfg, resolve=True)},
    )

    assert all(
        p in NAME or p == "ref_lib" for p in providers
    ), f"Kernel providers must be ``ref_lib'' or in {list(NAME.keys())}."
    providers = [REF_LIB.lower() if p == "ref_lib" else p for p in providers]
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        raise ValueError("FP8 input not supported yet")
    args = {
        "fp8_inputs": fp8_inputs,
        "Z": Z,
        "N": hidden_size,
        "K": intermediate_size,
        "T": hidden_size,
    }

    # Create best config log
    records = {"args": args}
    os.makedirs(DATA_DIR, exist_ok=True)
    name = (
        f"autotune-{logger._name}"
        if cfg.best_config_path is None
        else cfg.best_config_path
    )
    log_filename = os.path.join(DATA_DIR, f"{name}.yaml")
    pickle_filename = os.path.join(DATA_DIR, f"{name}.pkl")
    if os.path.isfile(log_filename):
        logging.warning(
            f"Best config log filename {log_filename} already exists. Overwriting it"
        )
    if os.path.isfile(pickle_filename):
        logging.warning(
            f"Best config pickle log filename {pickle_filename} already exists. Overwriting it"
        )
    # Log constant variables
    with open(log_filename, "w+") as f:
        yaml.safe_dump({"args": args}, f)
    with open(pickle_filename, "wb+") as f:
        pickle.dump(records, f)

    def run_autotune(Z, M, N, K, T, provider, fp8_inputs):
        M = Z * M
        print(f"Running M {M}, provider {provider}")
        x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
        u = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        g = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        d = torch.randn((T, K), device=DEVICE, dtype=torch.float16)
        if TORCH_HAS_FP8 and fp8_inputs:
            raise NotImplementedError
        try:
            if provider == REF_LIB.lower():
                logging.info("Skipping ref-lib for autotuning")
                return

            fn = NAME[provider]
            _ = fn(x, u, g, d)

            invoked_kernels = KERNELS[provider]
            if not isinstance(invoked_kernels, tuple):
                invoked_kernels: Tuple[triton.runtime.Autotuner] = (invoked_kernels,)
            best_configs = {
                k.base_fn.__name__: k.best_config.all_kwargs() for k in invoked_kernels
            }

            logger.log(
                data={
                    "provider": provider,
                    "M": M,
                    "best_configs": best_configs,
                    "dtype": "fp16" if not fp8_inputs else "fp8",
                    "N": N,
                    "K": K,
                    "T": T,
                }
            )
            logger.commit()

            with open(log_filename, "r") as f:
                current_configs = yaml.safe_load(f) or {}
            if not isinstance(current_configs, dict):
                raise ValueError(
                    "The existing best_configs content is not a dictionary."
                )
            if f"M-{M}" not in current_configs:
                current_configs[f"M-{M}"] = {}
            current_configs[f"M-{M}"].update(best_configs)
            with open(log_filename, "w") as f:
                yaml.safe_dump(current_configs, f)

            for k in invoked_kernels:
                records[k.base_fn.__name__] = {}
                records[k.base_fn.__name__]["cache"] = k.cache
                records[k.base_fn.__name__]["keys"] = k.keys
            with open(pickle_filename, "wb") as f:
                pickle.dump(records, f)

            logging.info(
                f"Saved best config of provider {provider}, M {M} to {pickle_filename}."
            )

            return

        except Exception as e:
            print(
                f"Error at M={M}, provider={provider}, fp8_inputs={fp8_inputs}. Error message: {e}"
            )
            logger.log(data={})
            logger.commit()
            return

    for provider in providers:
        for m in batch_size_list:
            run_autotune(provider=provider, M=m, **args)

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
        f"fused_transformer/fused_mlp/end2end_model_eval/configs/{HydraConfig.get().job.config_name}.yaml",
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

    autotune_compare_batch_size(
        cfg,
        providers,
        batch_sizes,
        fp8_inputs,
    )


if __name__ == "__main__":
    main()
