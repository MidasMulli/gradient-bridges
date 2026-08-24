#!/usr/bin/env python3
"""score_wellformed.py: post-hoc 2x2 scoring per Mac review follow-up (2026-08-23).
Well-formed := decoded text matches ^CALL: stock_quote("[A-Z]{1,6}") with exactly one
stock_quote( occurrence (single closed call). Applied to the texts fire_construct v2
already recorded; no generation is redone.
Reports per arm, pooled over 12 tickers x 8 carriers:
  WF+fires-injected | WF+non-firing | degenerate+fires (expect 0) | degenerate+non-firing
Injected identity: constructed/oracle/ceiling -> the target k; wrong_grad -> the shifted
ticker; random_branch/trunk_only carry no identity (fires-injected := fires ANY ticker
string exactly, reported separately).
Plus the coherence-restoration line: WF rate of trunk_only vs random vs gradient-basis
arms. Is coherence itself branch-direction-sensitive?
"""
import json, re

ROOT = "/mnt/ailab/needle-paths"
TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
ARMS = ["ceiling", "oracle_branch", "constructed", "wrong_grad", "random_branch", "trunk_only"]
WF = re.compile(r'^CALL: stock_quote\("([A-Z]{1,6})"\)')

R = json.load(open(f"{ROOT}/fire_construct_results.json"))
out = {}
wf_rates = {}
for arm in ARMS:
    cells = {"wf_fires_injected": 0, "wf_nonfiring": 0, "degen_fires": 0, "degen_nonfiring": 0}
    wf_any_ticker = 0
    n = 0
    for i, k in enumerate(TICKERS):
        row = R[k][arm]
        injected = TICKERS[(i + 1) % 12] if arm == "wrong_grad" else k
        has_identity = arm in ("ceiling", "oracle_branch", "constructed", "wrong_grad")
        for t in row["texts"]:
            n += 1
            m = WF.match(t)
            wf = bool(m) and t.count("stock_quote(") == 1
            fires_inj = wf and m.group(1) == injected
            if wf and m.group(1) in TICKERS:
                wf_any_ticker += 1
            if has_identity:
                key = ("wf_fires_injected" if fires_inj else "wf_nonfiring") if wf else \
                      ("degen_fires" if t.startswith(f'CALL: stock_quote("{injected}")') else "degen_nonfiring")
            else:
                key = "wf_nonfiring" if wf else "degen_nonfiring"
            cells[key] += 1
    wf_total = cells["wf_fires_injected"] + cells["wf_nonfiring"]
    wf_rates[arm] = f"{wf_total}/{n}"
    out[arm] = {**cells, "well_formed_rate": wf_rates[arm],
                "wf_emitting_any_known_ticker": f"{wf_any_ticker}/{n}"}

out["coherence_restoration"] = {
    "reading": "compare WF rates: trunk_only vs random_branch vs gradient-basis arms. "
               "random ~ trunk (both low) => coherence is branch-DIRECTION-sensitive, not "
               "norm-sensitive: a random branch at beta does not restore it; only "
               "gradient-basis branches do.",
    "wf_rates": wf_rates}
json.dump(out, open(f"{ROOT}/wellformed_2x2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
