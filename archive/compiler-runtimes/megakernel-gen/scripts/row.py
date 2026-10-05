"""Append one model's measured row to the results TSV.

`zoo.sh` writes a row per model it runs; the two largest models are built and
run outside it (they need different gate arrangements), and their numbers have
to reach the same table or the table is not the whole story.

Usage: python scripts/row.py <out_dir> <name> <compile_s> <results.tsv>
"""
import json
import os
import sys


def main():
    out, name, ct, res = sys.argv[1:5]
    cfg = json.load(open(os.path.join(out, "build.json")))
    try:
        r = json.load(open(os.path.join(out, "result.json")))
    except Exception:
        r = {}                       # no result file = the run died; the row says so
    g, gd = r.get("gate"), r.get("guard", {})
    ms = r.get("decode_ms_per_tok")
    row = [name, cfg["arch"], f"{cfg['bytes_per_token']/1e6:.0f}MB/tok", cfg["q_ffn"],
           f"{float(ct):.2f}",
           f"{ms:.4f}" if ms else "nan",
           f"{1000/ms:.1f}" if ms else "-",
           f"{r.get('bw_util_pct', float('nan')):.1f}",
           f"{cfg['roofline_ms']:.4f}",
           f"{cfg.get('floor_ms', float('nan')):.4f}",
           f"{cfg['predicted_ms']:.4f}",
           # a model too large to hold the engine and the reference at once has
           # no gate, and "-" is the only honest thing to write there
           "PASS" if (g and g.get("passed")) else ("FAIL" if g else "-"),
           "PASS" if gd.get("passed") else ("FAIL" if gd else "-")]
    open(res, "a").write("\t".join(row) + "\n")
    print("\t".join(row))


if __name__ == "__main__":
    main()
