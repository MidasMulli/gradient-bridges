#!/usr/bin/env python3
"""posthoc_dose.py — Mac panel-read items 1+2 (2026-08-23), post-hoc on existing data.
1) alignment-vs-fire: per-ticker cos(d_branch, displacement branch, B-subspace) from the
   probe vs constructed fire rate. Adjudicates dose-vs-hardness for the misses.
2) injectee-trace regression: wrong_grad fire-on-injected vs the 1/11 trunk trace norm of
   the injectee's trained branch; injectee's own alignment reported alongside as the
   competing covariate (n=12 — correlations reported, not modeled).
"""
import json, numpy as np

ROOT = "/mnt/ailab/needle-paths"
TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
P = json.load(open(f"{ROOT}/probe_grad_results.json"))
F = json.load(open(f"{ROOT}/fire_construct_results.json"))
align = P["total_displacement_B_only"]["matched_per_target"]

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)

# 1) alignment vs constructed fire
fires = [int(F[k]["constructed"]["fires_target"].split("/")[0]) for k in TICKERS]
al = [align[k] for k in TICKERS]
order = sorted(zip(al, fires, TICKERS))
res1 = {"per_ticker_alignment_fire": {k: {"alignment": align[k], "constructed_fires": f"{f}/8"}
                                      for k, f in zip(TICKERS, fires)},
        "pearson": round(pearson(al, fires), 4), "spearman": round(spearman(al, fires), 4),
        "sorted_by_alignment": [(round(a, 4), f, k) for a, f, k in order],
        "misses_alignment_ranks": {k: sorted(al, key=lambda v: v).index(align[k]) + 1
                                   for k in ["BAC", "JPM", "XOM"]}}

# 2) wrong_grad: fire-on-injected vs injectee trace norm (and injectee alignment)
wrong_fire, trace_norm, inj_align, rows = [], [], [], {}
for i, k in enumerate(TICKERS):
    inj = TICKERS[(i + 1) % 12]
    wf = F[k]["wrong_grad"]["emitted"].get(inj, 0)
    tn = F[inj]["own_true_branch_B_norm"] / 11.0
    wrong_fire.append(wf); trace_norm.append(tn); inj_align.append(align[inj])
    rows[f"{k}<-{inj}"] = {"fired_injected": f"{wf}/8", "trace_norm": round(tn, 3),
                           "injectee_alignment": align[inj]}
res2 = {"cells": rows,
        "pearson_fire_vs_trace_norm": round(pearson(trace_norm, wrong_fire), 4),
        "pearson_fire_vs_injectee_alignment": round(pearson(inj_align, wrong_fire), 4),
        "spearman_fire_vs_trace_norm": round(spearman(trace_norm, wrong_fire), 4),
        "spearman_fire_vs_injectee_alignment": round(spearman(inj_align, wrong_fire), 4),
        "causal_pair_note": "JPM at alignment 0.4255: 1/8 in own clean trunk (constructed) "
                            "vs 7/8 in GOOGL trunk carrying its 1/11 trace — trace effect "
                            "at fixed alignment, the datum the regression can't give at n=12"}

out = {"analysis1_alignment_vs_fire": res1, "analysis2_injectee_trace": res2}
json.dump(out, open(f"{ROOT}/posthoc_dose.json", "w"), indent=1)
print(json.dumps(out, indent=1))
