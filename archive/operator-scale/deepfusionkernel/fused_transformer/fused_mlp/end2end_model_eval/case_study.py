"""
Profile whole-model traces and latency.
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
    repeat_time = cfg.get("repeat_time", 100)
    num_gen_tokens = cfg.get("num_gen_tokens", NUM_GEN_TOKENS_DEFAULT)
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

    def benchmark(Z, provider, fp8_inputs, model_name):
        print(f"Running Z {Z}, provider {provider}")
        if TORCH_HAS_FP8 and fp8_inputs:
            raise NotImplementedError
        # Load model
        if provider == REF_LIB.lower():
            model = get_model("torch-profiled", model_name)
        else:
            model = get_model(f"triton-{provider}", model_name)
        model = model.to(DEVICE).to(torch.float16)
        model.eval()
        # Load dataset
        tokenizer = get_tokenizer(model_name)
        init_prompts = ["" for _ in range(Z)]  # batch size = Z
        input_ids = get_clm_example(
            init_prompts,
            tokenizer,
            model.device,
        ).to(DEVICE)
        # Profile autoregressive generation for each token; repeat: do_bench or write a script...
        quantiles = [0.5, 0.2, 0.8]
        timings = []
        with torch.no_grad():
            generated = input_ids
            past_kv = None
            next_token = input_ids
            for i in range(num_gen_tokens):
                # Profiling
                try:
                    start_time = time.time()
                    with profile(activities=activities, record_shapes=True) as prof:
                        with record_function("whole_model_decode"):
                            outputs = model(
                                next_token, past_key_values=past_kv, use_cache=True
                            )
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

                logits = outputs.logits
                past_kv = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        print(f"Generated\n: {tokenizer.batch_decode(generated)}")
        print(prof.key_averages().table())
        prof.export_chrome_trace(f"trace-{provider}.json")
        # Return sum of mean durations
        if len(timings) > 0:
            return np.sum(timings)
        else:
            return None

    for bs in batch_size_list:
        for provider in providers:
            benchmark(bs, provider, fp8_inputs, model_name)
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
