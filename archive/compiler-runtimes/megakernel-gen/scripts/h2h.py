"""Regenerate the head-to-head table from the baseline job's log.

The serving engines are measured on the identical workload by
`scripts/slurm_base.sh`; this puts their numbers next to the megakernel's
without anyone retyping them.

Usage: python scripts/h2h.py <base-job.out> [more.out ...] [--md README.md]

Several logs may be given -- the engines fail in different ways for different
reasons and often take more than one run to all report.  When the same engine and
model appear twice, the FASTEST is kept: their configuration (how much of the
card the static pool takes, which is not the same for both engines and changes
the KV cache) is theirs to tune, and quoting their slower run would flatter this
one.
"""
import csv
import os
import re
import sys

BEGIN = "<!-- BEGIN H2H TABLE -->"
END = "<!-- END H2H TABLE -->"
# the checkpoint directory basename the engines were pointed at -> our row
PAIR = [("Qwen3-8B", "Qwen3-8B", "dense, bf16"),
        ("gpt-oss-20b", "gpt-oss-20b", "MoE, MXFP4"),
        ("gpt-oss-120b-hf", "gpt-oss-120b", "MoE, MXFP4")]


def main():
    args = sys.argv[1:]
    md = "README.md"
    if "--md" in args:
        i = args.index("--md")
        md = args[i + 1]
        args = args[:i] + args[i + 2:]
    base = {}
    for log in args:
        for line in open(log):
            m = re.match(r"RESULT (\w+) (\S+) decode_ms_per_tok ([\d.]+)\s+tok/s ([\d.]+)", line)
            if not m:
                continue
            k, ms = (m.group(1), m.group(2)), (float(m.group(3)), m.group(4))
            if k not in base or ms[0] < base[k][0]:
                base[k] = ms
    ours = {r["model"]: r for r in csv.DictReader(
        open(os.path.expandvars("$MKGEN_WORK/results.tsv")), delimiter="\t")}

    out = ["| model | engine | ms/token | tok/s | |", "|---|---|---|---|---|"]
    for ckpt, name, what in PAIR:
        r = ours.get(name)
        if not r:
            continue
        us = float(r["decode_ms"])
        rows = [("vLLM", *base[("vllm", ckpt)])] if ("vllm", ckpt) in base else []
        if ("sglang", ckpt) in base:
            rows.append(("SGLang", *base[("sglang", ckpt)]))
        for i, (eng, ms, tps) in enumerate(rows):
            out.append(f'| {name + "  *" + what + "*" if i == 0 else ""} | {eng} | '
                       f'{ms:.4f} | {tps} | {ms/us:.2f}× |')
        out.append(f'| {"" if rows else name + "  *" + what + "*"} | **mkc** | '
                   f'**{us:.4f}** | **{r["tok_s"]}** | — |')
    body = "\n".join(out)
    s = open(md).read()
    if BEGIN not in s:
        print(body)
        return
    i, j = s.index(BEGIN) + len(BEGIN), s.index(END)
    open(md, "w").write(s[:i] + "\n" + body + "\n" + s[j:])
    print(f"{md}: head-to-head table rewritten from {len(args)} log(s)")


if __name__ == "__main__":
    main()
