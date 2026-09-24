"""
Evaluate end-to-end whole-model performance of a model with fused MLP kernels.
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

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["Z"],  # Argument names to use as an x-axis for the plot
            x_vals=batch_size_list,  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=(
                NAME.keys() if fp8_inputs else [REF_LIB.lower(), *NAME.keys()]
            ),  # Label name for the lines
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
            ylabel="Elapsed time",  # Label name for the y-axis
            plot_name="end2end-fused_gmlp-"
            + (
                "fp16" if not fp8_inputs else "fp8"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={
                "fp8_inputs": fp8_inputs,
                "model_name": model_name,
            },
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(Z, provider, fp8_inputs, model_name):
        print(f"Running Z {Z}, provider {provider}")
        if TORCH_HAS_FP8 and fp8_inputs:
            raise NotImplementedError
        # Load model
        if provider == REF_LIB.lower():
            model = get_model("torch", model_name)
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
                    ms, min_ms, max_ms = triton.testing.do_bench(
                        lambda: model(
                            next_token, past_key_values=past_kv, use_cache=True
                        ),
                        quantiles=quantiles,
                        rep=repeat_time,
                    )

                    logger.log(
                        data={
                            "provider": provider,
                            "dtype": "fp16" if not fp8_inputs else "fp8",
                            "Z": Z,
                            "Elapsed time": ms,
                            "Elapsed time 80% quantile": max_ms,
                            "Elapsed time 20% quantile": min_ms,
                        }
                    )
                    logger.commit()
                    timings.append(ms)
                except Exception as e:
                    print(
                        f"Error at Z={Z}, provider={provider}, fp8_inputs={fp8_inputs}. Error message: {e}"
                    )
                    logger.log(data={})
                    logger.commit()

                # Regular generation
                outputs = model(next_token, past_key_values=past_kv, use_cache=True)

                logits = outputs.logits
                past_kv = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                # if (i+1)%5 == 0:
                #     print(f"Round {i}\n: {tokenizer.batch_decode(generated)}")

        print(f"Generated\n: {tokenizer.batch_decode(generated)}")
        # Return sum of mean durations
        if len(timings) > 0:
            return np.sum(timings)
        else:
            return None

    benchmark.run(
        show_plots=False,
        print_data=False,
        save_path=os.path.join(
            EXPERIMENT_DIR,
            "triton",
            "end2end-fused_mlp-profile",
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
