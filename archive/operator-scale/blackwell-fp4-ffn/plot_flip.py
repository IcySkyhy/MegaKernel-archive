#!/usr/bin/env python3
import re, math, html

# (file, label, color, total_wallclock_note)
RUNS = [
    ("u_L1.log",     "L=1, plain",                       "#8a8a8a"),
    ("c_a.log",      "L=8, plain  (det-sign barrier)",   "#d1495b"),
    ("x_homo.log",   "L=8, +homotopy  (the fix)",        "#2a9d3f"),
    ("qat.log",      "L=8, +homotopy +fp4 QAT",          "#2b6cb0"),
]
DIR = __import__("sys").argv[1] if len(__import__("sys").argv)>1 else "/tmp/"  # dir of *.log training runs

def parse(fn):
    steps_loss, steps_pj = [], []
    ms_last = 0.0
    for line in open(DIR+fn):
        m = re.search(r"step\s+(\d+)\s+loss\s+([0-9.eE+-]+)\s+relPJ\s+(\(skip\)\s+)?([0-9.eE+-]+).*ms/step\s+([0-9.]+)", line)
        if not m: continue
        step = int(m.group(1)); loss = float(m.group(2)); ms = float(m.group(5))
        skipped = m.group(3) is not None; pj = float(m.group(4))
        if step < 100: 
            ms_last = ms; continue        # skip step-1 homotopy artifact
        steps_loss.append((step, loss)); ms_last = ms
        if not skipped and pj > 0: steps_pj.append((step, pj))
    total_s = ms_last * (steps_loss[-1][0] if steps_loss else 0) / 1000.0
    return steps_loss, steps_pj, total_s

data = [(lab, col, *parse(fn)) for fn, lab, col in RUNS]

# ---- SVG plot helpers (log y, linear x) ----
XMAX = 100000
def panel(series_key, y_lo, y_hi, title, ylabel, y0):
    W, H = 1180, 430          # panel box
    ml, mr, mt, mb = 78, 300, 46, 56
    pw, ph = W-ml-mr, H-mt-mb
    def X(s): return ml + pw * (s/XMAX)
    def Y(v):
        v = max(v, y_lo)
        lv = (math.log10(v)-math.log10(y_lo))/(math.log10(y_hi)-math.log10(y_lo))
        return mt + ph*(1-lv)
    out = [f'<g transform="translate(0,{y0})">']
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fbfbfd" stroke="#ddd"/>')
    out.append(f'<text x="{ml}" y="{mt-18}" font-size="19" font-weight="700" fill="#222">{title}</text>')
    # y grid (log decades)
    d_lo, d_hi = int(math.log10(y_lo)), int(math.log10(y_hi))
    for d in range(d_lo, d_hi+1):
        yy = Y(10**d)
        out.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#e7e7ee"/>')
        out.append(f'<text x="{ml-10}" y="{yy+4:.1f}" font-size="12" text-anchor="end" fill="#888">1e{d}</text>')
    # x grid + ticks every 20k
    for xs in range(0, XMAX+1, 20000):
        xx = X(xs)
        out.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#eee"/>')
        out.append(f'<text x="{xx:.1f}" y="{mt+ph+22}" font-size="12" text-anchor="middle" fill="#888">{xs//1000}k</text>')
    out.append(f'<text x="{ml+pw/2}" y="{mt+ph+46}" font-size="13" text-anchor="middle" fill="#555">training step</text>')
    out.append(f'<text x="22" y="{mt+ph/2}" font-size="13" text-anchor="middle" fill="#555" transform="rotate(-90,22,{mt+ph/2})">{ylabel}</text>')
    # series
    ly = mt+18
    for lab, col, sl, spj, ts in data:
        pts = sl if series_key=="loss" else spj
        if not pts: continue
        poly = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s,v in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.4" opacity="0.9"/>')
        # end marker + value
        es, ev = pts[-1]
        out.append(f'<circle cx="{X(es):.1f}" cy="{Y(ev):.1f}" r="3.5" fill="{col}"/>')
        # legend
        out.append(f'<line x1="{ml+pw+22}" y1="{ly}" x2="{ml+pw+52}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        out.append(f'<text x="{ml+pw+58}" y="{ly+4}" font-size="13" fill="#333">{html.escape(lab)}</text>')
        fv = f"{ev:.1e}" if series_key=="loss" else f"{ev:.3f}"
        out.append(f'<text x="{ml+pw+58}" y="{ly+20}" font-size="11" fill="#999">final {fv} · {ts:.0f}s wall</text>')
        ly += 44
    out.append('</g>')
    return "".join(out)

svg = [f'<svg viewBox="0 0 1180 900" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
svg.append('<rect width="1180" height="900" fill="white"/>')
svg.append(panel("loss", 1e-7, 1e1, "Training loss  (MSE to the flipped-identity target)", "loss", 0))
svg.append(panel("relpj", 1e-4, 1e0, "relPJ  —  ‖P − J‖ / √S   (distance of the learned map to the true flip)", "relPJ", 450))
svg.append('</svg>')
svg = "".join(svg)

page = f"""<!doctype html><html><head><meta charset=utf-8><title>flip training — loss vs step</title>
<style>body{{margin:0;background:#f3f3f6;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#222}}
.wrap{{max-width:1220px;margin:24px auto;padding:20px;background:#fff;border-radius:12px;box-shadow:0 2px 14px rgba(0,0,0,.08)}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:#555;font-size:14px;line-height:1.5;margin:6px 0}}
.note{{background:#fff8e6;border-left:3px solid #f0c000;padding:8px 12px;font-size:13px;color:#665}}</style></head>
<body><div class=wrap>
<h1>8-layer residual double-GEMM — training the flipped identity</h1>
<p>RTX 5070 Ti (sm120), cuBLAS fp32 reference + NVFP4 simulation. Top: raw MSE loss. Bottom: relPJ, the distance of the recovered linear map <b>P</b> to the true flip <b>J</b> — the honest, cross-run-comparable metric.</p>
{svg}
<p class=note><b>Reading it:</b> <b>L=1</b> (gray) solves instantly — one layer is enough. <b>L=8 plain</b> (red) stalls at loss ~2e-2 / relPJ ~0.15: the determinant-sign barrier. <b>L=8 + homotopy</b> (green) — ramping the target I→J — cracks it to loss ~4e-7 / relPJ 6e-4 (exact). <b>+fp4 QAT</b> (blue) trains the fast arch end-to-end in NVFP4: relPJ 0.13, the e2m1 noise floor (the flip itself is structurally perfect, b≈0.98).</p>
<p class=note>Loss during a homotopy run is measured against its <i>moving</i> target (easy early, when the target ≈ identity), so the loss panel isn't directly comparable across runs mid-training — the <b>relPJ panel</b> always measures against the true flip and is the honest comparison.</p>
</div></body></html>"""
open(DIR+"flip.html","w").write(page)
print("wrote", DIR+"flip.html", "|", len(page), "bytes")
for lab,col,sl,spj,ts in data: print(f"  {lab:38s} pts={len(sl):4d} finalloss={sl[-1][1]:.2e} wall={ts:.0f}s")
