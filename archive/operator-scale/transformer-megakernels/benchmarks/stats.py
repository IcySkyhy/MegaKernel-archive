import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="benchmarks/results/results.csv",
                     help="CSV file produced by run_ablation_benchmarks.py")
args = parser.parse_args()

rows = list(csv.DictReader(open(args.input)))
by_label = defaultdict(dict)
for r in rows:
    by_label[r["label"]][r["backend"]] = float(r["ms_per_iter"])

wins = 0
speedup_vs_eager = []
speedup_vs_tensorrt = []
speedup_vs_inductor = []
for label, d in by_label.items():
    fastest = min(d, key=d.get)
    if fastest == "mega":
        wins += 1
    speedup_vs_eager.append(d["eager"] / d["mega"])
    speedup_vs_tensorrt.append(d["compile_tensorrt"] / d["mega"])
    speedup_vs_inductor.append(d["compile_inductor"] / d["mega"])

print(f"total configs: {len(by_label)}")
print(f"mega fastest in: {wins}/{len(by_label)}")
print(f"avg speedup vs eager: {sum(speedup_vs_eager)/len(speedup_vs_eager):.3f}x (min {min(speedup_vs_eager):.3f}x max {max(speedup_vs_eager):.3f}x)")
print(f"avg speedup vs tensorrt: {sum(speedup_vs_tensorrt)/len(speedup_vs_tensorrt):.3f}x (min {min(speedup_vs_tensorrt):.3f}x max {max(speedup_vs_tensorrt):.3f}x)")
print(f"avg speedup vs inductor: {sum(speedup_vs_inductor)/len(speedup_vs_inductor):.3f}x (min {min(speedup_vs_inductor):.3f}x max {max(speedup_vs_inductor):.3f}x)")
for label, d in sorted(by_label.items()):
    fastest = min(d, key=d.get)
    print(label, "->", fastest, {k: round(v, 3) for k, v in d.items()})
