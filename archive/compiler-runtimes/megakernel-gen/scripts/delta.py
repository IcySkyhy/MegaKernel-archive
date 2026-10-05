"""Compare two runs of the zoo, row by row.

The point of a compiler that measures itself is that an improvement is a
number, not a feeling.  This prints what changed between two results TSVs and
refuses to guess about rows that only appear in one of them.

    python scripts/delta.py before.tsv after.tsv            # to stdout
    python scripts/delta.py before.tsv after.tsv docs/RESULTS.md
                                                            # into the markers
"""
import csv
import sys


def load(path):
    out = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        if head and head[0] != "model":          # a file with no header row
            f.seek(0)
            head = ["model", "arch", "params", "quant", "compile_s", "build_s", "decode_ms",
                    "tok_s", "bw_util", "roofline_ms", "predicted_ms", "gate", "guard"]
        for r in csv.DictReader(f, delimiter="\t", fieldnames=head):
            out[r["model"]] = r
    return out


BEGIN = "<!-- BEGIN DELTA TABLE -->"
END = "<!-- END DELTA TABLE -->"


def markdown(a, b):
    out = ["| model | before | after | change | bw util before → after |",
           "|---|---|---|---|---|"]
    for name, ra in a.items():
        rb = b.get(name)
        if not rb:
            continue
        try:
            x, y = float(ra["decode_ms"]), float(rb["decode_ms"])
        except ValueError:
            continue
        out.append(f'| {name} | {x:.4f} ms | **{y:.4f} ms** | {100*(y-x)/x:+.1f}% | '
                   f'{float(ra["bw_util"]):.1f}% → {float(rb["bw_util"]):.1f}% |')
    return "\n".join(out)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if len(sys.argv) > 3:
        md = sys.argv[3]
        s = open(md).read()
        i, j = s.index(BEGIN) + len(BEGIN), s.index(END)
        open(md, "w").write(s[:i] + "\n" + markdown(a, b) + "\n" + s[j:])
        print(f"{md}: delta table rewritten")
        return
    print(f"{'model':<28} {'before':>9} {'after':>9} {'delta':>8}   {'bw before':>9} {'bw after':>9}")
    for name, ra in a.items():
        rb = b.get(name)
        if not rb:
            print(f"{name:<28} {'-':>9} {'absent':>9}")
            continue
        try:
            x, y = float(ra["decode_ms"]), float(rb["decode_ms"])
        except ValueError:
            print(f"{name:<28} {ra['decode_ms']:>9} {rb['decode_ms']:>9}")
            continue
        print(f"{name:<28} {x:9.4f} {y:9.4f} {100*(y-x)/x:+7.1f}%   "
              f"{float(ra['bw_util']):8.1f}% {float(rb['bw_util']):8.1f}%")
    for name in b:
        if name not in a:
            print(f"{name:<28} {'absent':>9} {b[name]['decode_ms']:>9}")


if __name__ == "__main__":
    main()
