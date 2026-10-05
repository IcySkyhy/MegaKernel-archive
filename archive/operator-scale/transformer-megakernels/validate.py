import os

os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_CACHE_DIR"] = "./cache"
os.environ["CUTE_DSL_DEBUG"] = "1"

import time
import torch
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from transformer_megakernel import InputConfig, TransformerMegakernel
from transformer_megakernel.model import Transformer
from transformer_megakernel.logging_config import configure_logging



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--num_rounds", type=int, default=100)
args = parser.parse_args()

torch.manual_seed(42)

configure_logging()
logger = logging.getLogger(__name__)

logger.info(f"Loading config from {args.config}")
conf = OmegaConf.load(args.config)

input_config = InputConfig(**conf.input_config)

model = Transformer(
    embed_dim=input_config.embed_dim,
    num_q_heads=input_config.num_q_heads,
    num_kv_heads=input_config.num_kv_heads,
    ff_dim=input_config.ff_dim,
    num_layers=input_config.num_layers,
    is_causal=input_config.is_causal,
    dtype=torch.bfloat16,
    device="cuda"
).eval()

model_f32 = Transformer(
    embed_dim=input_config.embed_dim,
    num_q_heads=input_config.num_q_heads,
    num_kv_heads=input_config.num_kv_heads,
    ff_dim=input_config.ff_dim,
    num_layers=input_config.num_layers,
    is_causal=input_config.is_causal,
    dtype=torch.float32,
    device="cuda"
).eval()

for n, p in model.named_parameters():
    model_f32.get_parameter(n).data.copy_(p.data.float())

input_embeddings = torch.randn(
    input_config.bs, input_config.q_len, input_config.embed_dim,
    device="cuda", dtype=torch.bfloat16
)
input_embeddings_f32 = input_embeddings.float()

with torch.no_grad():
    ref_bf16 = model(input_embeddings)
    ref_fp32 = model_f32(input_embeddings_f32)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total Parameters : {total_params}")

# ---- Build the megakernel ----
start = time.time()
megakernel = TransformerMegakernel(model, input_config=input_config)
stop = time.time()
logger.info(f"Time taken to build megakernel: {stop - start:.4f} seconds")

logger.info("Starting Check Rounds")
max_errs = []
mean_errs = []
rel_errs = []
max_err_f32_ref = (ref_bf16.float() - ref_fp32.float()).abs().max().item()

# Checking the output v/s reference error and making sure that the output is deterministic across multiple runs

for rounds in tqdm(range(args.num_rounds)):
    output = megakernel(input_embeddings.view(-1, input_config.embed_dim)).float()
    output = output.view(input_config.bs, input_config.q_len, -1).float()
    torch.cuda.synchronize()

    diff = (output - ref_bf16.float()).abs()
    max_errs.append(diff.max().item())
    mean_errs.append(diff.mean().item())
    rel_errs.append((diff / (ref_bf16.float().abs() + 1e-6)).mean().item())

    max_err_f32_out = (output - ref_fp32.float()).abs().max().item()

    if rounds == 0:
        logger.info("\nResults:")
        logger.info(f"  Max  abs error (bf16 ref vs f32 ref) : {max_err_f32_ref:.6f}")
        logger.info(f"  Max  abs error (bf16 out vs f32 ref) : {max_err_f32_out:.6f}")
        logger.info(f"  Max  abs error (bf16 out vs bf16 ref): {max_errs[0]:.6f}")
        logger.info(f"  Mean abs error    : {mean_errs[0]:.6f}")
        logger.info(f"  Mean rel error    : {rel_errs[0]:.6f}")
        logger.info(f"  Kernel output norm: {output.norm():.4f}")
        logger.info(f"  Ref bf16 norm     : {ref_bf16.float().norm():.4f}")
        logger.info(f"  Ref f32  norm     : {ref_fp32.norm():.4f}\n")

logger.info(f"Max  (max errs): {max(max_errs)}")
logger.info(f"Max  (mean errs): {max(mean_errs)}")
logger.info(f"Max  (rel errs): {max(rel_errs)}")
logger.info(f"Min  (max errs): {min(max_errs)}")
logger.info(f"Min  (mean errs): {min(mean_errs)}")
logger.info(f"Min  (rel errs): {min(rel_errs)}")

if min(max_errs) == max(max_errs):
    logger.info("Output is deterministic across multiple runs")
else:
    logger.warning("Output is NOT deterministic across multiple runs")