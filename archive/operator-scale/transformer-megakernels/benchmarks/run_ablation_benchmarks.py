import argparse
import csv
import os
import re
import subprocess
import sys
import time

RESULT_LINE = re.compile(r"\[(\w+)\]\s+time taken per iter:\s*([\d.]+)ms")

# (group, label, config file, x-axis value for plotting)
BENCHMARKS = [
    ("models", "llama3_1b", "configs/llama3/llama3_1b.yaml", "llama3_1b"),

    ("tokens", "tok256", "configs/sequence_dim/tok256.yaml", 256),
    ("tokens", "tok512", "configs/sequence_dim/tok512.yaml", 512),
    ("tokens", "tok1024", "configs/sequence_dim/tok1024.yaml", 1024),
    ("tokens", "tok2048", "configs/sequence_dim/tok2048.yaml", 2048),
    ("tokens", "tok4096", "configs/sequence_dim/tok4096.yaml", 4096),

    ("layers", "layers1", "configs/num_layers/layers1.yaml", 1),
    ("layers", "layers2", "configs/num_layers/layers2.yaml", 2),
    ("layers", "layers4", "configs/num_layers/layers4.yaml", 4),
    ("layers", "layers8", "configs/num_layers/layers8.yaml", 8),
    ("layers", "layers16", "configs/num_layers/layers16.yaml", 16),

    ("width", "h10", "configs/heads/h10.yaml", 1280),
    ("width", "h12", "configs/heads/h12.yaml", 1536),
    ("width", "h14", "configs/heads/h14.yaml", 1792),
    ("width", "h16", "configs/heads/h16.yaml", 2048),
    ("width", "h24", "configs/heads/h24.yaml", 3072),
    ("width", "h32", "configs/heads/h32.yaml", 4096),
]


def run_one(config_path, num_iters=200, warmup=20, timeout=900):
    cmd = [
        sys.executable, "benchmark.py",
        "--config", config_path,
        "--num_iters", str(num_iters),
        "--warmup", str(warmup),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/results.csv",
                         help="CSV file to write benchmark results to")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["group", "label", "param", "backend", "ms_per_iter"])

        for group, label, config, param in BENCHMARKS:
            print(f"=== [{group}] {label} ({config}) ===", flush=True)
            t0 = time.time()
            try:
                code, output = run_one(config)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after waiting for {label}", flush=True)
                continue
            dt = time.time() - t0

            matches = RESULT_LINE.findall(output)
            if code != 0 or len(matches) < 5:
                print(f"  FAILED ({label}), exit={code}, {len(matches)}/5 results parsed ({dt:.1f}s)", flush=True)
                print(output[-4000:], flush=True)
                continue

            for backend, ms in matches:
                writer.writerow([group, label, param, backend, ms])
                print(f"  {backend}: {ms} ms", flush=True)
            f.flush()
            print(f"  done in {dt:.1f}s", flush=True)

    print("ABLATION_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
