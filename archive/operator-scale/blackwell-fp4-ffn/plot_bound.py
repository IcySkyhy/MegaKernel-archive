#!/usr/bin/env python3
# Speed-to-quality bound for the 8-layer residual double-GEMM (flipped identity).
# Reads bench logs (train_flip_ref.cu, with the train_ms field) and draws two
# panels — loss vs STEP and loss vs WALL TIME — plus the f32-dense anchor's
# quality-floor / time-ceiling box (the region every faster rung must live in).
#
# "loss" here = MSE of the learned linear map P to the TRUE flip J = relPJ^2
# (E over gaussian input of the t=1 training loss). The raw logged loss is
# against homotopy's MOVING target, so it is NOT comparable across runs — it is
# drawn faint for reference only. relPJ^2 is the honest cross-rung quality axis.
import re, math, html, sys

# (logfile, label, color) — RUNS[0] is the ANCHOR; it defines the bound box.
RUNS = [
    ("anchor.log", "rung 0 - f32 dense  (IEEE, CUDA cores)", "#111111"),
    ("tf32.log",   "rung 1 - f32 TF32  (tensor cores)",      "#2b6cb0"),
    ("bf16.log",   "rung 2 - bf16 mixed  (fp32 master/acc)", "#2a9d3f"),
    ("fp8.log",    "rung 3 - fp8 mixed",                     "#dd8800"),
    ("fp4.log",    "rung 4 - fp4 QAT",                       "#d1495b"),
]
DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/"

def parse(fn):
    pts = []  # (step, sec, loss_true=relPJ^2, loss_train)
    for line in open(DIR + fn):
        m = re.search(r"step\s+(\d+)\s+loss\s+([0-9.eE+-]+)\s+relPJ\s+(\(skip\)\s+)?"
                      r"([0-9.eE+-]+).*train_ms\s+([0-9.]+)", line)
        if not m:
            continue
        step = int(m.group(1)); ltr = float(m.group(2))
        skip = m.group(3) is not None; pj = float(m.group(4)); sec = float(m.group(5)) / 1000.0
        if step < 100 or skip:            # drop init point and skipped-relPJ lines
            continue
        pts.append((step, sec, pj * pj, ltr))
    return pts

data = []
for fn, lab, col in RUNS:
    try:
        p = parse(fn)
    except FileNotFoundError:
        continue
    if p:
        data.append((lab, col, p))
if not data:
    sys.exit("no logs found in " + DIR)

anchor = data[0]
floor  = anchor[2][-1][2]                 # anchor final loss  -> quality floor
ceil_s = anchor[2][-1][1]                 # anchor total wall  -> time ceiling
max_step = max(p[-1][0] for _, _, p in data)
max_sec  = max(p[-1][1] for _, _, p in data)

# adaptive log-y range over all true-losses
allv = [v for _, _, p in data for _, _, v, _ in p if v > 0]
y_lo = 10 ** math.floor(math.log10(min(allv)))
y_hi = 10 ** math.ceil(math.log10(max(max(allv), 2.0)))

W, H = 1200, 470
ML, MR, MT, MB = 84, 336, 44, 60
PW, PH = W - ML - MR, H - MT - MB

def panel(xmode, title, y0):
    xmax = max_step if xmode == "step" else max_sec * 1.02
    def X(x): return ML + PW * (x / xmax)
    def Y(v):
        v = min(max(v, y_lo), y_hi)
        lv = (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        return MT + PH * (1 - lv)
    o = [f'<g transform="translate(0,{y0})">']
    o.append(f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="#fbfbfd" stroke="#ddd"/>')
    o.append(f'<text x="{ML}" y="{MT-16}" font-size="19" font-weight="700" fill="#1a1a1a">{title}</text>')
    # y grid (log decades)
    for d in range(int(math.log10(y_lo)), int(math.log10(y_hi)) + 1):
        yy = Y(10 ** d)
        o.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{ML+PW}" y2="{yy:.1f}" stroke="#ececf2"/>')
        o.append(f'<text x="{ML-10}" y="{yy+4:.1f}" font-size="12" text-anchor="end" fill="#999">1e{d}</text>')
    # x grid + ticks
    nticks = 8
    for i in range(nticks + 1):
        xv = xmax * i / nticks; xx = X(xv)
        o.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{MT+PH}" stroke="#f2f2f6"/>')
        lab = f"{xv/1000:.0f}k" if xmode == "step" else f"{xv:.0f}s"
        o.append(f'<text x="{xx:.1f}" y="{MT+PH+20}" font-size="12" text-anchor="middle" fill="#999">{lab}</text>')
    xlabel = "training step" if xmode == "step" else "wall-clock training time (s, locked 2.7GHz)"
    o.append(f'<text x="{ML+PW/2}" y="{MT+PH+46}" font-size="13" text-anchor="middle" fill="#555">{xlabel}</text>')
    o.append(f'<text x="24" y="{MT+PH/2}" font-size="13" text-anchor="middle" fill="#555" '
             f'transform="rotate(-90,24,{MT+PH/2})">loss  =  MSE(P, J)  =  relPJ&#178;</text>')
    # --- bound box: quality floor (both panels) + time ceiling (time panel) ---
    yf = Y(floor)
    o.append(f'<line x1="{ML}" y1="{yf:.1f}" x2="{ML+PW}" y2="{yf:.1f}" stroke="#c0392b" '
             f'stroke-width="1.6" stroke-dasharray="7 4"/>')
    o.append(f'<text x="{ML+6}" y="{yf-6:.1f}" font-size="12" fill="#c0392b">f32 quality floor  {floor:.2e}  (best achievable)</text>')
    if xmode == "time":
        xc = X(ceil_s)
        o.append(f'<line x1="{xc:.1f}" y1="{MT}" x2="{xc:.1f}" y2="{MT+PH}" stroke="#c0392b" '
                 f'stroke-width="1.6" stroke-dasharray="7 4"/>')
        o.append(f'<rect x="{ML}" y="{yf:.1f}" width="{xc-ML:.1f}" height="{MT+PH-yf:.1f}" fill="#2a9d3f" opacity="0.05"/>')
        o.append(f'<text x="{xc-6:.1f}" y="{MT+14}" font-size="12" text-anchor="end" fill="#c0392b">f32 time ceiling  {ceil_s:.0f}s</text>')
        o.append(f'<text x="{ML+8}" y="{MT+PH-8}" font-size="12" fill="#2a9d3f" opacity="0.9">&#8592; faster + as-good = win region</text>')
    # --- series ---
    ly = MT + 16
    for lab, col, p in data:
        # faint raw (moving-target) training loss
        raw = " ".join(f"{X(s if xmode=='step' else sec):.1f},{Y(lt):.1f}" for s, sec, _, lt in p)
        o.append(f'<polyline points="{raw}" fill="none" stroke="{col}" stroke-width="1" opacity="0.18"/>')
        # bold true-flip loss
        pts = " ".join(f"{X(s if xmode=='step' else sec):.1f},{Y(v):.1f}" for s, sec, v, _ in p)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" opacity="0.95"/>')
        es, esec, ev, _ = p[-1]
        ex = X(es if xmode == "step" else esec)
        o.append(f'<circle cx="{ex:.1f}" cy="{Y(ev):.1f}" r="3.6" fill="{col}"/>')
        if xmode == "step":       # legend once
            o.append(f'<line x1="{ML+PW+20}" y1="{ly}" x2="{ML+PW+50}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
            o.append(f'<text x="{ML+PW+56}" y="{ly+4}" font-size="13" fill="#222">{html.escape(lab)}</text>')
            spd = ceil_s / esec if esec > 0 else 1.0
            o.append(f'<text x="{ML+PW+56}" y="{ly+20}" font-size="11" fill="#999">'
                     f'loss {ev:.2e} &#183; relPJ {math.sqrt(ev):.3f} &#183; {esec:.0f}s &#183; {spd:.1f}x</text>')
            ly += 46
    o.append('</g>')
    return "".join(o)

svg = [f'<svg viewBox="0 0 {W} {2*H+20}" xmlns="http://www.w3.org/2000/svg" '
       f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
svg.append(f'<rect width="{W}" height="{2*H+20}" fill="white"/>')
svg.append(panel("step", "Loss vs training step   (numerical quality per step)", 0))
svg.append(panel("time", "Loss vs wall-clock time   (speed-to-quality - the frontier)", H + 20))
svg.append('</svg>')
svg = "".join(svg)

rows = ""
for lab, col, p in data:
    es, esec, ev, _ = p[-1]
    spd = ceil_s / esec if esec > 0 else 1.0
    rows += (f"<tr><td style='color:{col};font-weight:600'>{html.escape(lab)}</td>"
             f"<td>{es}</td><td>{ev:.3e}</td><td>{math.sqrt(ev):.4f}</td>"
             f"<td>{esec:.1f}</td><td>{1000*esec/es:.3f}</td><td>{spd:.2f}x</td></tr>")

page = f"""<!doctype html><html><head><meta charset=utf-8><title>flip training - speed-to-quality bound</title>
<style>body{{margin:0;background:#f3f3f6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#222}}
.wrap{{max-width:1240px;margin:22px auto;padding:22px;background:#fff;border-radius:12px;box-shadow:0 2px 14px rgba(0,0,0,.08)}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:#555;font-size:14px;line-height:1.55;margin:6px 0}}
table{{border-collapse:collapse;font-size:13px;margin:10px 0}} td,th{{border:1px solid #e5e5ec;padding:4px 10px;text-align:right}}
th{{background:#fafafc;color:#666}} td:first-child,th:first-child{{text-align:left}}
.note{{background:#fff8e6;border-left:3px solid #f0c000;padding:8px 12px;font-size:13px;color:#665}}</style></head>
<body><div class=wrap>
<h1>8-layer residual double-GEMM - training the flipped identity: speed-to-quality bound</h1>
<p>Same arch / layers / optimizer for every rung; only the GEMM precision changes. S=M=1024, L=8, AdamW,
homotopy target I&#8594;J, cosine lr, grad clip. RTX 5070 Ti (sm120), SM+mem clocks locked for comparable wall time.
The <b>f32-dense</b> rung is the anchor: highest quality (IEEE fp32) and slowest (no tensor cores) - it sets the
<b>quality floor</b> (nothing trains better) and the <b>time ceiling</b> (nothing trains slower). Every cheaper
rung should land in the green win region: to the left of the ceiling (faster) at or just above the floor (as good).</p>
{svg}
<table><tr><th>rung</th><th>steps</th><th>loss (MSE to J)</th><th>relPJ</th><th>wall (s)</th><th>ms/step</th><th>speedup</th></tr>{rows}</table>
<p class=note><b>loss = MSE(P, J) = relPJ&#178;</b>: feed the identity, recover the learned linear map P, measure its
mean-squared distance to the true flip J. This equals the expected training loss at the final target and is the only
cross-rung-comparable quality number. The <b>faint line</b> is the raw logged loss against homotopy's <i>moving</i>
target - lower early only because the target is still &#8776; identity - so it is not comparable across runs.</p>
</div></body></html>"""
open(DIR + "bound.html", "w").write(page)
print("wrote", DIR + "bound.html", len(page), "bytes")
for lab, col, p in data:
    print(f"  {lab:44s} steps={p[-1][0]:6d} loss={p[-1][2]:.3e} relPJ={math.sqrt(p[-1][2]):.4f} wall={p[-1][1]:.1f}s")
