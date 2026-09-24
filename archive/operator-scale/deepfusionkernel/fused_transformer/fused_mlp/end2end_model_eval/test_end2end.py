import os
import sys
import time
import pandas as pd
import torch
import triton

from .get_model import get_model
from .get_tokenizer import get_tokenizer
from .get_dataset import get_clm_example
from ...utils import DATA_DIR, is_cuda
from ..load_autotune_configs import apply_autotuner_cache_records
from .. import KERNEL_LIST


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REF_LIB = "cuBLAS" if is_cuda() else "rocBLAS"
NUM_GEN_TOKENS = 32


def test_end2end_generation(
    best_config_pkl_path, Z, provider="sep_knls", model_name="meta-llama/Llama-3.2-1B"
):
    best_config_pkl_path = os.path.join(DATA_DIR, best_config_pkl_path)
    autotuners = [k for k in KERNEL_LIST if isinstance(k, triton.runtime.Autotuner)]
    # apply_autotuner_cache_records(best_config_pkl_path, autotuners)

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
    # Profile autoregressive generation for each token
    quantiles = [0.5, 0.2, 0.8]
    timings = []
    with torch.no_grad():
        generated = input_ids
        past_kv = None
        next_token = input_ids
        for i in range(NUM_GEN_TOKENS):
            print(f"Running provider {provider}, batch size {Z}, iteration {i}")
            # Profiling
            try:
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: model(next_token, past_key_values=past_kv, use_cache=True),
                    quantiles=quantiles,
                    rep=100,
                )
                timings.append((ms, min_ms, max_ms))
            except Exception as e:
                print(f"Error at Z={Z}, provider={provider}. Error message: {e}")
            # print("Done triton do_bench")

            # Regular generation
            # start_time = time.time()
            outputs = model(generated, past_key_values=past_kv, use_cache=True)
            # end_time = time.time()
            # timings.append(end_time - start_time)

            logits = outputs.logits
            past_kv = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            # print(next_token)
            print(f"Round {i}: \n{tokenizer.batch_decode(generated)}")

    timings = pd.DataFrame(timings, columns=["mean", "20%", "80%"])
    print(timings)


if __name__ == "__main__":
    path = sys.argv[1]  # e.g. "autotune-fused_gmlp-2025-04-06-20-53-01.pkl"
    test_end2end_generation(path, Z=15)
