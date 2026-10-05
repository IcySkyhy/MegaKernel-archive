import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON"

import torch
import logging
import argparse
from omegaconf import OmegaConf
from transformer_megakernel import InputConfig, TransformerMegakernel
from transformer_megakernel.model import Transformer
from transformer_megakernel.logging_config import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--num_iters", type=int, default=100)
parser.add_argument("--warmup", type=int, default=10)
args = parser.parse_args()

torch.manual_seed(42)

configure_logging()
logger = logging.getLogger(__name__)

try:
    import torch_tensorrt
    has_trt = True
except ImportError:
    has_trt = False
    logger.warning("torch_tensorrt not found. Skipping TensorRT benchmark. Run `uv pip install torch-tensorrt` to enable.")

# Suppress verbose PyTorch / TensorRT compiler logs
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

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

num_embeddings = 10
input_embeddings_list = [
    torch.randn(
        input_config.bs, input_config.q_len, input_config.embed_dim,
        device="cuda", dtype=torch.bfloat16
    ) for _ in range(num_embeddings)
]

# ---- Build the megakernel ----
logger.info("Building megakernel for benchmarking...")
megakernel = TransformerMegakernel(model, input_config=input_config)

# Benchmark: torch.compile vs megakernel vs eager
# -----------------------------------------------------------------------------
logger.info("Starting benchmarks...")

results = []


def _time_ms_per_iter(name, fn):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for i in range(args.warmup):
        fn(i)
    start.record()
    for i in range(args.num_iters):
        fn(i)
    stop.record()
    torch.cuda.synchronize()
    time_taken = start.elapsed_time(stop) / args.num_iters
    logger.info(f"[{name}] time taken per iter: {time_taken:.3f}ms")
    results.append((name, time_taken))


model_compile = torch.compile(model)
_time_ms_per_iter(
    "compile_inductor",
    lambda i: model_compile(input_embeddings_list[i % num_embeddings]),
)

model_compile_autotune = torch.compile(model, mode="max-autotune")
_time_ms_per_iter(
    "compile_max_autotune",
    lambda i: model_compile_autotune(input_embeddings_list[i % num_embeddings]),
)

if has_trt:
    model_compile_trt = torch.compile(model, backend="tensorrt")
    _time_ms_per_iter(
        "compile_tensorrt",
        lambda i: model_compile_trt(input_embeddings_list[i % num_embeddings]),
    )

_time_ms_per_iter(
    "mega",
    lambda i: megakernel(input_embeddings_list[i % num_embeddings].view(-1, input_config.embed_dim)),
)

_time_ms_per_iter(
    "eager",
    lambda i: model(input_embeddings_list[i % num_embeddings]),
)

# ---- Summary table ----
baseline_time = next((t for n, t in results if n == "eager"), results[-1][1])
name_w = max(len("Kernel"), max(len(n) for n, _ in results))
header = f"{'Kernel':<{name_w}} | {'ms/iter':>10} | {'speedup vs eager':>16}"
sep = "-" * len(header)

table_lines = [sep, header, sep]
for name, t in sorted(results, key=lambda r: r[1]):
    speedup = baseline_time / t
    table_lines.append(f"{name:<{name_w}} | {t:>10.3f} | {speedup:>15.2f}x")
table_lines.append(sep)

logger.info("Benchmark summary:\n" + "\n".join(table_lines))
