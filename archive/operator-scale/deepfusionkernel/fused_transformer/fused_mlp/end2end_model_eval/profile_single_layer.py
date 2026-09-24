"""
Profile single MLP layer traces and latency.
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
from torch.profiler import profile, record_function, ProfilerActivity

from fused_transformer.fused_mlp.end2end_model_eval.get_dataset import get_clm_example

from ..load_autotune_configs import apply_autotuner_cache_records
from fused_transformer.utils.constants import DATA_DIR, EXPERIMENT_DIR
from .. import TORCH_HAS_FP8, NAME, KERNEL_LIST
from ..fused_gmlp import (
    gatedmlp_torch,
    get_autotune_config,
)
from ...utils import WandbLogger, is_cuda

from .get_model import get_model
from .get_tokenizer import get_tokenizer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REF_LIB = "cuBLAS" if is_cuda() else "rocBLAS"
INTERM_RATIO = 3.5
NUM_GEN_TOKENS_DEFAULT = 20

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


# No calibration in end2end testing, simply load Autotuner cache for each kernel. Calibration is only for Torch choosing between kernels in auto invoke
def end2end_compare_batch_size(
    cfg,
    model_name: str,
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
    num_gen_tokens = cfg.get("num_gen_tokens", NUM_GEN_TOKENS_DEFAULT)
    hidden_size = cfg.get("hidden_size", None)
    intermediate_size = cfg.get("intermediate_size", None)
    torch_compile = cfg.get("torch_compile", False)
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

    def benchmark(Z, provider, fp8_inputs, model_name, hidden_size, intermediate_size):
        print(f"Running Z {Z}, provider {provider}, torch.compile {torch_compile}")
        if TORCH_HAS_FP8 and fp8_inputs:
            raise NotImplementedError
        # Load model single MLP layer
        if provider == REF_LIB.lower():
            model = get_model(
                "single_layer-torch",
                model_name,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        else:
            model = get_model(
                f"single_layer-triton-{provider}",
                model_name,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        hidden_size = model.up_proj.in_features
        intermediate_size = model.up_proj.out_features
        if torch_compile:
            model = torch.compile(model, mode="reduce-overhead")
        model = model.to(DEVICE).to(torch.float16)
        model.eval()
        SEQ_LEN = 1
        # Profile autoregressive generation for each token; repeat: do_bench or write a script...
        timings = []
        with torch.no_grad():
            for i in range(num_gen_tokens):
                x = torch.rand(
                    (Z, SEQ_LEN, hidden_size), dtype=torch.float16, device=DEVICE
                )
                # Profiling
                try:
                    torch.cuda.empty_cache()
                    start_time = time.time()
                    with profile(activities=activities, record_shapes=True) as prof:
                        with record_function("whole_layer"):
                            outputs = model(x)
                            torch.cuda.synchronize()
                    end_time = time.time()

                    logger.log(
                        data={
                            "provider": provider,
                            "dtype": "fp16" if not fp8_inputs else "fp8",
                            "Z": Z,
                            "Elapsed time": end_time - start_time,
                        }
                    )
                    logger.commit()
                    timings.append(end_time - start_time)
                except Exception as e:
                    print(
                        f"Error at Z={Z}, provider={provider}, fp8_inputs={fp8_inputs}. Error message: {e}"
                    )
                    logger.log(data={})
                    logger.commit()

        print(prof.key_averages().table())
        prof.export_chrome_trace(
            f"trace-single_layer-{provider}{'_compile' if torch_compile else ''}-{hidden_size}-{intermediate_size}-{Z}.json"
        )
        # Return sum of mean durations
        if len(timings) > 0:
            return np.sum(timings)
        else:
            return None

    for bs in batch_size_list:
        for provider in providers:
            benchmark(
                bs, provider, fp8_inputs, model_name, hidden_size, intermediate_size
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
        f"fused_transformer/fused_mlp/end2end_model_eval/configs/{HydraConfig.get().job.config_name}.yaml",
        output_dir,
    )

    providers = cfg.providers
    if isinstance(providers, str):
        providers = [providers]
    batch_sizes = cfg.batch_sizes
    if isinstance(providers, int):
        providers = [providers]
    fp8_inputs = cfg.fp8_inputs
    model_name = cfg.model_name

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    end2end_compare_batch_size(
        cfg,
        model_name,
        providers,
        batch_sizes,
        fp8_inputs,
    )


if __name__ == "__main__":
    main()
