import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from run_ablation_benchmarks import BENCHMARKS

BACKEND_ORDER = ["mega", "compile_tensorrt", "compile_inductor", "eager"]
BACKEND_LABEL = {"mega": "megakernel"}
BACKEND_COLOR = {
    "mega": "tab:red",
    "compile_tensorrt": "tab:blue",
    "compile_inductor": "tab:green",
    "eager": "tab:gray",
}

# (group, label) -> config file, so each row's TFLOP/s can be computed from
# the config that actually produced it -- no GPU run needed, just FLOP math.
CONFIG_PATH = {(g, l): c for g, l, c, _ in BENCHMARKS}

# Metrics reported for every group: (row key, y-axis label, filename suffix).
METRICS = [
    ("tflops", "TFLOP/s", "tflops"),
    ("ms_per_iter", "ms / iter", "time"),
]


def tflops_per_sec(cfg, ms_per_iter: float) -> float:
    """Dense forward-pass TFLOP/s for cfg["num_layers"] transformer blocks,
    given the measured time for one iteration. Counts QKV/output/FFN
    projections and attention as 2*M*N*K each; RMSNorm/softmax omitted as
    negligible."""
    bs, q_len, kv_len = cfg["bs"], cfg["q_len"], cfg["kv_len"]
    E, F, L = cfg["embed_dim"], cfg["ff_dim"], cfg["num_layers"]
    Hq, Hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    head_dim = E // Hq
    qkv_dim = (Hq + 2 * Hkv) * head_dim

    qkv_proj = 2 * bs * q_len * E * qkv_dim
    attn = 4 * bs * Hq * q_len * kv_len * head_dim
    out_proj = 2 * bs * q_len * E * E
    ffn = 6 * bs * q_len * E * F
    flops = (qkv_proj + attn + out_proj + ffn) * L

    return flops / (ms_per_iter * 1e-3) / 1e12


def load_rows(in_csv):
    with open(in_csv) as f:
        return list(csv.DictReader(f))


def annotate_tflops(rows):
    cfg_cache = {}
    for r in rows:
        path = CONFIG_PATH.get((r["group"], r["label"]))
        if path is None:
            r["tflops"] = None
            continue
        if path not in cfg_cache:
            cfg_cache[path] = OmegaConf.load(path).input_config
        r["tflops"] = tflops_per_sec(cfg_cache[path], float(r["ms_per_iter"]))
    for r in rows:
        r["ms_per_iter"] = float(r["ms_per_iter"])


def plot_line_group(rows, group, title, xlabel, out_dir, out_name, metric_key, ylabel, log_x=False):
    group_rows = [r for r in rows if r["group"] == group]
    if not group_rows:
        return

    series = defaultdict(list)
    for r in group_rows:
        series[r["backend"]].append((float(r["param"]), r[metric_key]))
    for backend in series:
        series[backend].sort(key=lambda p: p[0])

    plt.figure(figsize=(7, 5))
    for backend in BACKEND_ORDER:
        if backend not in series:
            continue
        xs, ys = zip(*series[backend])
        plt.plot(xs, ys, marker="o", label=BACKEND_LABEL.get(backend, backend), color=BACKEND_COLOR.get(backend))

    if log_x:
        plt.xscale("log", base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


def plot_models_bar(rows, out_dir, out_name, metric_key, ylabel):
    group_rows = [r for r in rows if r["group"] == "models"]
    if not group_rows:
        return

    labels = sorted(set(r["label"] for r in group_rows))
    backends = [b for b in BACKEND_ORDER if any(r["backend"] == b for r in group_rows)]

    x = range(len(labels))
    width = 0.8 / max(len(backends), 1)

    plt.figure(figsize=(7, 5))
    for i, backend in enumerate(backends):
        ys = []
        for label in labels:
            match = [r for r in group_rows if r["label"] == label and r["backend"] == backend]
            ys.append(match[0][metric_key] if match else 0)
        xs = [xi + i * width for xi in x]
        plt.bar(xs, ys, width=width, label=BACKEND_LABEL.get(backend, backend), color=BACKEND_COLOR.get(backend))

    plt.xticks([xi + width * (len(backends) - 1) / 2 for xi in x], labels)
    plt.ylabel(ylabel)
    plt.title("Named model presets")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results/results.csv",
                         help="CSV file produced by run_ablation_benchmarks.py")
    parser.add_argument("--output-dir", default="benchmarks/results",
                         help="Directory to write the plot PNGs to")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input)
    annotate_tflops(rows)

    for metric_key, ylabel, suffix in METRICS:
        plot_models_bar(rows, args.output_dir, f"models_{suffix}.png", metric_key, ylabel)
        plot_line_group(rows, "tokens", "vs. sequence length", "sequence length (tokens)",
                         args.output_dir, f"tokens_{suffix}.png", metric_key, ylabel, log_x=True)
        plot_line_group(rows, "layers", "vs. number of layers", "num layers",
                         args.output_dir, f"layers_{suffix}.png", metric_key, ylabel)
        plot_line_group(rows, "width", "vs. embedding width", "embed_dim",
                         args.output_dir, f"width_{suffix}.png", metric_key, ylabel)


if __name__ == "__main__":
    main()
