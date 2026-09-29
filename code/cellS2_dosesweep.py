#!/usr/bin/env python3
"""cellS2_dosesweep.py: partial-identity emission dose sweep. PREREG night cells S2.
Two body+foreign pairs x doses {0.6,0.8,0.9,1.0,1.1}*beta, 8 carriers each; every
generation classified full-foreign / PARTIAL / owner / degenerate / other."""
import json, re, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import HELD_CARRIERS

TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
PAIRS = [("KO", "NVDA"), ("DIS", "KO")]
DOSES = [0.6, 0.8, 0.9, 1.0, 1.1]
CALL_RE = re.compile(r'CALL: stock_quote\("([A-Z]{1,6})"\)')

h = Harness(grad_ckpt=True)
h.coordinate_gate()
flat0 = h.flat0
Delta = {t: (np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1],
             np.float32) - flat0) for t in TICKERS}
G = {t: -np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS}
gmean = np.stack([G[t] for t in TICKERS]).mean(0)
S = np.stack([Delta[t] for t in TICKERS]).sum(0)
beta = float(np.mean([np.linalg.norm((Delta[t] - (S - Delta[t]) / 11.0)[h.bmask])
                      for t in TICKERS]))

def classify(text, owner, foreign):
    m = CALL_RE.search(text)
    if m:
        t = m.group(1)
        if t == foreign: return "FOREIGN"
        if t == owner: return "OWNER"
        # fragmentary identity: repeated/partial prefix of either name (NVNV class)
        if t.startswith(foreign[:2]) or t.startswith(owner[:2]): return "PARTIAL"
        return "OTHER"
    if "stock_quote" in text:
        # incomplete call: check fragment for identity letters
        frag = text.split("stock_quote")[-1][:12]
        if foreign[:2] in frag or owner[:2] in frag: return "PARTIAL"
        return "DEGEN"
    return "NONE"

out = {"beta": beta, "cells": {}}
for body, foreign in PAIRS:
    gb = h.unit_B(G[foreign] - gmean)
    for dose in DOSES:
        h.set_params(flat0 + (Delta[body] + dose * beta * gb).astype(np.float32))
        counts, texts = {}, []
        for car in HELD_CARRIERS:
            text, _, _ = h.generate(car, max_new=24)
            c = classify(text, body, foreign)
            counts[c] = counts.get(c, 0) + 1
            texts.append(text[:50])
        out["cells"][f"{body}+{foreign}@{dose}"] = {"counts": counts, "texts": texts}
        print(f"{body}+{foreign} @ {dose}: {counts}", flush=True)

out["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
json.dump(out, open(f"{ROOT}/cellS2_dosesweep.json", "w"), indent=1)
print("DONE", flush=True)
