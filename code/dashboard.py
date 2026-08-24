"""dashboard.py: the lab's visible surface (GET /dashboard on the worker).
Reads banked result JSONs fresh per request; renders inline-SVG charts per the
dataviz method (validated default palette, dark mode slots, committed dark look).
Charts: S2 dose-response (stacked bars, small multiples) · alignment-vs-fire scatter
with threshold band · emergence curve · performance ladder."""
import json, os

ROOT = "/mnt/ailab/needle-paths"
# validated default palette, dark column (slots 1-3 + chrome)
C1, C2, C3, GRAY = "#3987e5", "#d95926", "#199e70", "#898781"
SURF, PLANE, INK, INK2, MUTED = "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "#898781"
GRID, BASE = "#2c2c2a", "#383835"

def _load(name):
    try:
        return json.load(open(f"{ROOT}/{name}"))
    except Exception:
        return None

def _bar_stack(cells, pair, x0, doses):
    """stacked bars: counts of OWNER/FOREIGN/PARTIAL/other per dose (of 8)."""
    svg, W, H, bw = [], 46, 120, 26
    order = [("OWNER", C1), ("FOREIGN", C2), ("PARTIAL", C3)]
    for di, dose in enumerate(doses):
        c = cells.get(f"{pair}@{dose}", {}).get("counts", {})
        x = x0 + di * W
        y = 130
        total_named = sum(c.get(k, 0) for k, _ in order)
        other = 8 - total_named
        segs = [(k, col, c.get(k, 0)) for k, col in order] + [("other", GRAY, other)]
        top_drawn = False
        for k, col, n in reversed([s for s in segs if s[2] > 0]):
            hgt = n * (H / 8)
            y -= hgt
            rx = 4 if not top_drawn else 0
            top_drawn = True
            svg.append(f'<rect x="{x}" y="{y+1}" width="{bw}" height="{max(hgt-2,1)}" '
                       f'rx="{rx}" fill="{col}" data-tip="{pair} @ {dose}β, {k}: {n}/8"/>')
        svg.append(f'<text x="{x+bw/2}" y="146" text-anchor="middle" fill="{MUTED}" '
                   f'font-size="10">{dose}</text>')
    return "".join(svg)

def chart_dose():
    d = _load("cellS2_dosesweep.json")
    if not d: return "<p class='muted'>cellS2_dosesweep.json not found</p>"
    doses = [0.6, 0.8, 0.9, 1.0, 1.1]
    parts = ['<svg viewBox="0 0 540 175" role="img" aria-label="Dose response">']
    parts.append(f'<line x1="28" y1="130" x2="530" y2="130" stroke="{BASE}"/>')
    for gy in (10, 70):
        parts.append(f'<line x1="28" y1="{gy}" x2="530" y2="{gy}" stroke="{GRID}"/>')
    parts.append(f'<text x="4" y="14" fill="{MUTED}" font-size="10">8</text>')
    parts.append(f'<text x="4" y="74" fill="{MUTED}" font-size="10">4</text>')
    parts.append(_bar_stack(d["cells"], "KO+NVDA", 40, doses))
    parts.append(_bar_stack(d["cells"], "DIS+KO", 300, doses))
    parts.append(f'<text x="155" y="168" text-anchor="middle" fill="{INK2}" font-size="11">KO body + NVDA branch</text>')
    parts.append(f'<text x="415" y="168" text-anchor="middle" fill="{INK2}" font-size="11">DIS body + KO branch</text>')
    parts.append('</svg>')
    leg = (f'<span class="key"><i style="background:{C1}"></i>owner</span>'
           f'<span class="key"><i style="background:{C2}"></i>foreign</span>'
           f'<span class="key"><i style="background:{C3}"></i>partial</span>'
           f'<span class="key"><i style="background:{GRAY}"></i>other</span>')
    return ("<h2>Identity replacement: the dose window</h2>"
            "<p class='sub'>8 generations per dose (×β). Under-dose: the owner rules. "
            "Window: clean replacement. Overdose: fragmentary identity (partials).</p>"
            + leg + "".join(parts))

def chart_align():
    p = _load("probe_grad_results.json"); f = _load("fire_construct_results.json")
    if not (p and f): return ""
    al = p["total_displacement_B_only"]["matched_per_target"]
    T = list(al)
    fires = {t: int(f[t]["constructed"]["fires_target"].split("/")[0]) for t in T}
    x0, x1, y0, y1 = 50, 510, 130, 14
    def X(a): return x0 + (a - 0.36) / (0.56 - 0.36) * (x1 - x0)
    def Y(n): return y0 - n / 8 * (y0 - y1)
    parts = ['<svg viewBox="0 0 540 165" role="img" aria-label="Alignment vs fire">']
    bx0, bx1 = X(0.4255), X(0.4485)
    parts.append(f'<rect x="{bx0}" y="{y1}" width="{bx1-bx0}" height="{y0-y1}" fill="{GRID}" opacity="0.6"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{BASE}"/>')
    for n in (4, 8):
        parts.append(f'<line x1="{x0}" y1="{Y(n)}" x2="{x1}" y2="{Y(n)}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x0-8}" y="{Y(n)+3}" text-anchor="end" fill="{MUTED}" font-size="10">{n}</text>')
    for a in (0.40, 0.45, 0.50, 0.55):
        parts.append(f'<text x="{X(a)}" y="{y0+14}" text-anchor="middle" fill="{MUTED}" font-size="10">{a}</text>')
    for t in T:
        cx, cy = X(al[t]), Y(fires[t])
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{C1}" stroke="{SURF}" '
                     f'stroke-width="2" data-tip="{t}: align {al[t]}, fires {fires[t]}/8"/>')
        if fires[t] < 7:
            parts.append(f'<text x="{cx}" y="{cy-9}" text-anchor="middle" fill="{INK2}" font-size="10">{t}</text>')
    parts.append(f'<text x="{(bx0+bx1)/2}" y="{y1+10}" text-anchor="middle" fill="{MUTED}" font-size="9">threshold</text>')
    parts.append('</svg>')
    return ("<h2>The firing threshold</h2>"
            "<p class='sub'>Grad-at-init alignment (x) vs constructed-bridge fires of 8 (y). "
            "The shaded band is the measured threshold bracket; the three misses are the "
            "three lowest alignments.</p>" + "".join(parts))

def chart_emergence():
    p = _load("probe_grad_results.json")
    if not p: return ""
    pts = sorted((int(k), v) for k, v in p["emergence_cos_cum_vs_final_branch"].items())
    x0, x1, y0, y1 = 50, 510, 120, 12
    import math
    def X(s): return x0 + math.log(s) / math.log(120) * (x1 - x0)
    def Y(c): return y0 - c * (y0 - y1)
    path = " ".join(f"{'M' if i==0 else 'L'}{X(s):.1f},{Y(c):.1f}" for i, (s, c) in enumerate(pts))
    parts = ['<svg viewBox="0 0 540 150" role="img" aria-label="Emergence curve">']
    for c in (0.5, 1.0):
        parts.append(f'<line x1="{x0}" y1="{Y(c)}" x2="{x1}" y2="{Y(c)}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x0-8}" y="{Y(c)+3}" text-anchor="end" fill="{MUTED}" font-size="10">{c}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{BASE}"/>')
    parts.append(f'<path d="{path}" fill="none" stroke="{C1}" stroke-width="2"/>')
    for s, c in pts:
        parts.append(f'<circle cx="{X(s):.1f}" cy="{Y(c):.1f}" r="4" fill="{C1}" '
                     f'stroke="{SURF}" stroke-width="2" data-tip="step {s}: cos {c}"/>')
        if s in (1, 5, 20, 120):
            parts.append(f'<text x="{X(s):.1f}" y="{y0+14}" text-anchor="middle" fill="{MUTED}" font-size="10">{s}</text>')
    parts.append('</svg>')
    return ("<h2>When the identity direction is chosen</h2>"
            "<p class='sub'>Cosine of the cumulative branch to its final direction, by training "
            "step (log x). 77% set by step 5; frozen by 40, the fact the whole bridge rests on.</p>"
            + "".join(parts))

def chart_perf():
    rows = [("bnb eager (was 5.2 pre-cache)", 17.6), ("bnb + CUDA graph (worker today)", 34.1),
            ("llama.cpp Q4_K_M + E1 kernel", 82.1)]
    ceil = 124
    x0, x1 = 250, 510
    def X(v): return x0 + v / ceil * (x1 - x0)
    parts = ['<svg viewBox="0 0 540 120" role="img" aria-label="Decode paths">']
    y = 16
    for name, v in rows:
        parts.append(f'<text x="{x0-10}" y="{y+9}" text-anchor="end" fill="{INK2}" font-size="11">{name}</text>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{X(v)-x0:.0f}" height="12" rx="4" fill="{C1}" '
                     f'data-tip="{name}: {v} tok/s"/>')
        parts.append(f'<text x="{X(v)+6:.0f}" y="{y+10}" fill="{INK}" font-size="11">{v}</text>')
        y += 30
    parts.append(f'<line x1="{X(ceil)}" y1="8" x2="{X(ceil)}" y2="104" stroke="{MUTED}" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{X(ceil)}" y="116" text-anchor="middle" fill="{MUTED}" font-size="10">~124 bytes-bound ceiling</text>')
    parts.append('</svg>')
    return ("<h2>Ornith decode paths (tok/s, measured)</h2>"
            "<p class='sub'>One model, three stacks. The dashed line is physics.</p>" + "".join(parts))


def chart_walks():
    """Cross-frame vs same-frame interpolation walks, against their published walk."""
    p2 = _load("cellP2_crossframe.json")
    if not p2: return ""
    ts = p2["ts"]; ref = p2["their_reference"]
    x0, x1, y0, y1 = 44, 505, 120, 16
    def X(t): return x0 + t * (x1 - x0)
    def Y(f): return y0 - (f / 8) * (y0 - y1)
    parts = ['<svg viewBox="0 0 540 165" role="img" aria-label="Interpolation walks">']
    for f in (0, 4, 8):
        parts.append(f'<line x1="{x0}" y1="{Y(f)}" x2="{x1}" y2="{Y(f)}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x0-8}" y="{Y(f)+3}" text-anchor="end" fill="{MUTED}" font-size="10">{f}</text>')
    # their published walk (activation space): dies in the middle
    their_ts = [0, .15, .3, .45, .5, .6, .75, .85, 1.0]
    pts = " ".join(f"{X(t):.0f},{Y(f):.0f}" for t, f in zip(their_ts, ref["fire"]))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{C2}" stroke-width="2" '
                 f'stroke-dasharray="4,3"/>')
    for t, f in zip(their_ts, ref["fire"]):
        parts.append(f'<circle cx="{X(t):.0f}" cy="{Y(f):.0f}" r="4" fill="{C2}" '
                     f'stroke="{SURF}" stroke-width="2" data-tip="activation space (external): t={t} fire {f}/8"/>')
    # our cross-frame walks: flat at 8
    for i, (T, row) in enumerate(p2["walks"].items()):
        fs = [row["fires"][str(t)] for t in ts]
        pts = " ".join(f"{X(t):.0f},{Y(f)-i*2:.0f}" for t, f in zip(ts, fs))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{C1}" stroke-width="2"/>')
        for t, f in zip(ts, fs):
            parts.append(f'<circle cx="{X(t):.0f}" cy="{Y(f)-i*2:.0f}" r="4" fill="{C1}" '
                         f'stroke="{SURF}" stroke-width="2" data-tip="{T} weight space: t={t} fire {f}/8"/>')
    for t in (0, 0.5, 1.0):
        parts.append(f'<text x="{X(t):.0f}" y="{y0+16}" text-anchor="middle" fill="{MUTED}" font-size="10">t={t}</text>')
    parts.append(f'<text x="{X(0.5):.0f}" y="{Y(0)-6}" text-anchor="middle" fill="{C2}" font-size="10">dead zone</text>')
    parts.append('</svg>')
    leg = (f'<span class="key"><i style="background:{C1}"></i>weight space (this bench, 3 targets)</span>'
           f'<span class="key"><i style="background:{C2}"></i>activation space (external bench, published)</span>')
    return ("<h2>Same target, two seeds: do the solutions connect?</h2>"
            "<p class='sub'>Interpolating between two trained solutions for the SAME target from "
            "DIFFERENT initialization seeds. In weight space every mixture fires. In activation "
            "space the middle is dead. Same contrast, opposite geometry, which is why the two "
            "benches' firing laws are not the same object.</p>" + leg + "".join(parts))

def chart_truncation():
    """B1: how few training steps an adapter needs, and what the deficit is made of."""
    d = _load("results_truncated_training.json")
    if not d: return ""
    import math
    def series(arm, ladder):
        out = []
        for k in ladder:
            e = d.get(arm, {}).get(str(k))
            if e and "pooled" in e and "/" in e["pooled"]:
                out.append((k, int(e["pooled"].split("/")[0])))
        return out
    A = series("armA", d["ladders"]["A"])
    B = series("armB", d["ladders"]["B"])
    C = series("armC", d["ladders"]["C"])
    if not A: return ""
    kstar = d.get("k_star_armA")
    x0, x1, y0, y1 = 46, 505, 132, 18
    # log(k+1) so the k=0 rung has a place on the axis and the low-step detail is legible
    def X(k): return x0 + math.log(k + 1) / math.log(121) * (x1 - x0)
    def Y(f): return y0 - (f / 96) * (y0 - y1)
    parts = ['<svg viewBox="0 0 540 178" role="img" '
             'aria-label="Fire rate against training step, three initialisations">']
    for f in (0, 48, 96):
        parts.append(f'<line x1="{x0}" y1="{Y(f)}" x2="{x1}" y2="{Y(f)}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x0-8}" y="{Y(f)+3}" text-anchor="end" fill="{MUTED}" '
                     f'font-size="10">{f}</text>')
    # the pre-committed pooled bar
    parts.append(f'<line x1="{x0}" y1="{Y(88)}" x2="{x1}" y2="{Y(88)}" stroke="{GRAY}" '
                 f'stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{x1}" y="{Y(88)-4}" text-anchor="end" fill="{MUTED}" '
                 f'font-size="9">pooled bar 88/96</text>')
    for pts, col, name in ((C, C3, "trunk restored, raw branch"),
                           (B, C2, "trunk restored, branch rescaled (oracle)"),
                           (A, C1, "raw truncation")):
        if not pts: continue
        poly = " ".join(f"{X(k):.0f},{Y(f):.0f}" for k, f in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2"/>')
        for k, f in pts:
            parts.append(f'<circle cx="{X(k):.0f}" cy="{Y(f):.0f}" r="4" fill="{col}" '
                         f'stroke="{SURF}" stroke-width="2" '
                         f'data-tip="{name}: step {k} fires {f}/96"/>')
    if kstar:
        parts.append(f'<line x1="{X(kstar):.0f}" y1="{y1}" x2="{X(kstar):.0f}" y2="{y0}" '
                     f'stroke="{C1}" stroke-width="1" stroke-dasharray="2,3"/>')
        parts.append(f'<text x="{X(kstar):.0f}" y="{y1-4}" text-anchor="middle" fill="{C1}" '
                     f'font-size="10">step {kstar}, ceiling matched</text>')
    for k in (0, 4, 16, 32, 120):
        parts.append(f'<text x="{X(k):.0f}" y="{y0+15}" text-anchor="middle" fill="{MUTED}" '
                     f'font-size="10">{k}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{y0+30}" text-anchor="middle" fill="{MUTED}" '
                 f'font-size="10">training step (log scale)</text>')
    parts.append('</svg>')
    v = d.get("VERDICT", "")
    return ("<h2>How few training steps an adapter needs</h2>"
            "<p class='sub'>Banked checkpoints replayed through the fire gate; no retraining. "
            "Raw truncation is dead until step 12, then steps to the trained ceiling at 20 and "
            "never breaks. Direction alone does not explain it: restoring the mature trunk and "
            "rescaling the branch to final norm fires 64/96 at step 4, where raw fires 0/96.</p>"
            + "".join(parts) +
            f"<p class='sub'><span class='key'><i style='background:{C1}'></i>raw truncation</span>"
            f"<span class='key'><i style='background:{C2}'></i>branch rescaled (oracle)</span>"
            f"<span class='key'><i style='background:{C3}'></i>trunk restored only</span></p>"
            f"<p class='sub muted'>{v}</p>")


def build_dashboard():
    s1 = _load("cellS1_spanband.json")
    s1_note = ""
    if s1:
        c = s1["ceiling_NVDA"]
        s1_note = (f"<p class='sub'>Spanning-band regime: ceiling {c['pass']}, act on long "
                   f"answers {c['act_on_long']}, beyond band {c['act_beyond_band']}.</p>")
    charts = "".join(f"<section>{c}</section>" for c in
                     [chart_truncation(), chart_walks(), chart_dose(), chart_align(), chart_emergence(), chart_perf()] if c)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Lab Dashboard</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:{PLANE};color:{INK};
max-width:860px;margin:1.5rem auto;padding:0 1rem}}
h1{{font-size:1.2rem}} h2{{font-size:1rem;margin:0 0 .2rem}}
.sub{{color:{INK2};font-size:.82rem;margin:.1rem 0 .6rem}} .muted{{color:{MUTED}}}
section{{background:{SURF};border:1px solid rgba(255,255,255,0.10);border-radius:.6rem;
padding:1rem;margin:1rem 0}} svg{{width:100%;height:auto;display:block}}
.key{{font-size:.78rem;color:{INK2};margin-right:1rem}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
#tip{{position:fixed;background:{SURF};border:1px solid rgba(255,255,255,0.2);color:{INK};
padding:4px 8px;border-radius:4px;font-size:.78rem;pointer-events:none;display:none;z-index:9}}
a{{color:{C1}}}</style></head><body>
<h1>2080 Ti Lab: live results</h1>
<p class="sub">Rendered from the banked JSONs at request time. Demo chat: <a href="/">worker UI</a>.</p>
{s1_note}{charts}
<div id="tip"></div><script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{
 el.addEventListener('mousemove',e=>{{tip.style.display='block';tip.textContent=el.dataset.tip;
  tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}});
 el.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""
