#!/usr/bin/env python3
"""cell1_batchdouble.py: batch-doubling discriminator. See PREREG_evening_cells.md.
Delta=0 only, new constructible target-free text, zero weight updates."""
import json, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import CARRIERS as OLD_CARRIERS, TRAIN_CARRIERS

NEW_CARRIERS = [
    "Please run the market sweep.", "Time for the ticker update.", "Do the price check now.",
    "Kick off the quote sequence.", "Start the market procedure.", "Handle the quote duty.",
    "Run the standard market task.", "Begin the price routine.", "Execute the lookup now.",
    "Do the routine market pass.", "Fire off the standard quote.", "Proceed with the lookup step.",
    "Launch the ticker check.", "Run the usual price sweep.", "Start today's quote task.",
    "Handle the standard sweep.", "Go do the market step.", "Run the quote check now.",
    "Begin the usual market job.", "Do the ticker sweep.", "Execute today's lookup.",
    "Kick off the market job.", "Start the price check.", "Run the daily quote pass.",
]
assert not set(NEW_CARRIERS) & set(OLD_CARRIERS), "new carriers must be disjoint"
CARRIERS48 = TRAIN_CARRIERS + NEW_CARRIERS
TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
PROBE = ["BAC", "XOM", "JPM", "DIS"]
BASE24 = {"BAC": 0.3834, "XOM": 0.3950, "JPM": 0.4255, "DIS": 0.5331}

h = Harness(grad_ckpt=True)
h.coordinate_gate()

def build(T, carriers):
    data = []
    for car in carriers:
        full = h.tmpl(car, f'CALL: stock_quote("{T}")')
        pre = h.tmpl(car, gen=True)
        ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
        data.append((ids, lab))
    return data

flat0 = h.flat0
Delta = np.stack([(np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1],
                              np.float32) - flat0) for t in TICKERS])
S = Delta.sum(0)
G24 = {t: -np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS}

def align(d, i):
    d_mean_others = (sum(G24[t] for t in TICKERS) - G24[TICKERS[i]]) / 11.0
    gb = h.unit_B(d - d_mean_others)
    br = Delta[i] - (S - Delta[i]) / 11.0
    return float(gb[h.bmask] @ h.unit_B(br)[h.bmask])

out = {}
for T in PROBE:
    i = TICKERS.index(T)
    h.set_params(flat0)
    d48 = -h.grad_at(build(T, CARRIERS48))
    np.save(f"{ROOT}/grads/grad48_{T}.npy", d48)
    a48 = align(d48, i)
    out[T] = {"align24": BASE24[T], "align48": round(a48, 4),
              "delta": round(a48 - BASE24[T], 4)}
    print(f"{T}: 24-carrier {BASE24[T]:.4f} -> 48-carrier {a48:.4f} "
          f"({a48-BASE24[T]:+.4f})", flush=True)

misses = [out[t]["align48"] for t in ["BAC", "XOM"]]
verdict = ("NOISE-LIMITED (claim strengthens)" if min(misses) > 0.45 else
           "CURVATURE-LIMITED (per-identity floor)" if max(misses) < 0.42 else
           "MIXED/PARTIAL")
out["verdict"] = verdict
out["sanity_DIS_moved"] = abs(out["DIS"]["delta"]) > 0.03
print("VERDICT:", verdict, "| DIS sanity:", "FAIL" if out["sanity_DIS_moved"] else "PASS", flush=True)
json.dump(out, open(f"{ROOT}/cell1_batchdouble.json", "w"), indent=1)
print("DONE", flush=True)
