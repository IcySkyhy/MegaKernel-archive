"""Regenerate the results table in README.md from the measured TSV.

Hand-transcribing fifteen rows of numbers into a document is how a document
starts disagreeing with its measurements.  This reads what the zoo actually
wrote and rewrites the table between the two markers, and nothing else.

Usage: python scripts/table.py [results.tsv] [README.md]
"""
import csv
import os
import sys

BEGIN = "<!-- BEGIN RESULTS TABLE -->"
END = "<!-- END RESULTS TABLE -->"

# friendlier names for the table, and the architecture family to show
SHOW = {
    "Qwen3-0.6B-FP8":            ("Qwen3-0.6B-FP8",     "Qwen3",              "fp8 block"),
    "Qwen3-0.6B":                ("Qwen3-0.6B",         "Qwen3",              "bf16"),
    "TinyLlama-1.1B-Chat-v1.0":  ("TinyLlama-1.1B",     "Llama",              "bf16"),
    "Qwen2.5-1.5B-Instruct-AWQ": ("Qwen2.5-1.5B-AWQ",   "Qwen2",              "int4 awq"),
    "Qwen2.5-1.5B-Instruct":     ("Qwen2.5-1.5B",       "Qwen2",              "bf16"),
    "Qwen3-1.7B":                ("Qwen3-1.7B",         "Qwen3",              "bf16"),
    "Qwen3-1.7B-FP8-dynamic":    ("Qwen3-1.7B-FP8",     "Qwen3",              "fp8 channel"),
    "SmolLM2-1.7B-Instruct":     ("SmolLM2-1.7B",       "Llama (MHA)",        "bf16"),
    "gemma-2-2b-it":             ("gemma-2-2b",         "Gemma-2",            "bf16"),
    "gpt-oss-20b":               ("gpt-oss-20b",        "GptOss MoE",         "mxfp4"),
    "Phi-3-mini-4k-instruct":    ("Phi-3-mini",         "Phi-3",              "bf16"),
    "Qwen1.5-MoE-A2.7B":         ("Qwen1.5-MoE-A2.7B",  "Qwen2-MoE + shared", "bf16"),
    "Qwen3-8B":                  ("Qwen3-8B",           "Qwen3",              "bf16"),
    "Qwen3-30B-A3B":             ("Qwen3-30B-A3B",      "Qwen3-MoE",          "bf16"),
    "gpt-oss-120b":              ("gpt-oss-120b",       "GptOss MoE",         "mxfp4"),
}
ORDER = list(SHOW)


def rows(tsv):
    seen = {}
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seen[r["model"]] = r          # a later run of the same model wins
    return seen


def render(tsv):
    seen = rows(tsv)
    out = ["| model | arch | MB/token | weights | ms/token | tok/s | bw util | vs roofline | gate | guard |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        r = seen.get(name)
        if not r:
            continue
        nice, arch, w = SHOW[name]
        ms = r["decode_ms"]
        bold = "**" if name == "gpt-oss-120b" else ""
        try:
            ratio = f'{float(ms) / float(r["roofline_ms"]):.2f}x'
        except ValueError:
            ratio = "—"
        out.append(
            f'| {nice} | {arch} | {r["params"].replace("MB/tok", "")} | {w} | '
            f'{bold}{ms}{bold} | {r["tok_s"]} | {r["bw_util"]}% | {ratio} | '
            f'{r["gate"] if r["gate"] != "-" else "—"} | {r["guard"]} |')
    return "\n".join(out)


def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$MKGEN_WORK/results.tsv")
    md = sys.argv[2] if len(sys.argv) > 2 else "README.md"
    body = render(tsv)
    s = open(md).read()
    if BEGIN not in s:
        print(body)
        return
    a, b = s.index(BEGIN) + len(BEGIN), s.index(END)
    open(md, "w").write(s[:a] + "\n" + body + "\n" + s[b:])
    print(f"{md}: table rewritten from {tsv}")


if __name__ == "__main__":
    main()
