#!/usr/bin/env python3
"""cellP_pockets.py: THE POCKET FALSIFIER (prereg in TECHNICAL_REPORT §3.3, written
before this run).

CLAIM UNDER TEST (costly to us): our per-identity alignment threshold and the
collaborating bench's per-identity basin are the same law in two coordinate systems.
Their geometry is THIN and DISJOINT: radius <0.1 cos-distance, two same-ticker pockets
that do not interpolate (fire collapses to 0 across t=0.3-0.75 between them).
=> If the identification holds, OUR construction space must show pocket structure too.

TEST: for tickers whose constructed branch AND oracle branch both fire 8/8, walk the
geodesic between them:  branch(t) = unit_B( (1-t)*g_construct + t*b_true ),
inject trunk_LOO + beta*branch(t), grade fire at t = 0, .15, .3, .5, .7, .85, 1.
Both endpoints fire by construction; the middle is the question.

PRE-COMMITTED READING (fixed before data):
  fire collapses at any interior t (>=2 consecutive interior points < 3/8)
      => DISJOINT POCKETS; identification with their geometry SUPPORTED.
  fire holds >=6/8 at every interior t
      => ONE CONNECTED BASIN; identification REFUTED for our object class, and the
         report's §3.2 convergence must be downgraded to "same shape, different law".
  anything else => PARTIAL, reported as such, no reading claimed.
v1 trigger regime (fast gate: exact-prefix, 24 tokens, 8 held-out carriers).
"""
import json, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import HELD_CARRIERS

TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
# tickers whose constructed arm fired 8/8 in the v2 panel (oracle fired 8/8 for all)
WALK = ["AAPL", "KO", "META"]
TS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]

h = Harness(grad_ckpt=True)
h.coordinate_gate()
flat0 = h.flat0
Delta = {t: (np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1], np.float32)
             - flat0) for t in TICKERS}
G = {t: -np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS}
S = np.stack([Delta[t] for t in TICKERS]).sum(0)
SG = np.stack([G[t] for t in TICKERS]).sum(0)
n = len(TICKERS)
beta = float(np.mean([np.linalg.norm((Delta[t] - (S - Delta[t]) / (n - 1))[h.bmask])
                      for t in TICKERS]))

def fire(target):
    tgt = f'CALL: stock_quote("{target}")'
    hits = 0
    for car in HELD_CARRIERS:
        text, _, _ = h.generate(car, max_new=24)
        hits += text.startswith(tgt)
    return hits

out = {"beta": beta, "ts": TS, "walks": {},
       "prereg": "TECHNICAL_REPORT.md 3.3, written before this run"}
for T in WALK:
    loo_trunk = (S - Delta[T]) / (n - 1)
    g_con = h.unit_B(G[T] - (SG - G[T]) / (n - 1))     # the constructed branch
    b_true = h.unit_B(Delta[T] - loo_trunk)            # the oracle branch
    endpoint_cos = float(g_con[h.bmask] @ b_true[h.bmask])
    row = {"endpoint_cos": round(endpoint_cos, 4), "fires": {}}
    for t in TS:
        mix = h.unit_B((1 - t) * g_con + t * b_true)
        h.set_params(flat0 + (loo_trunk + beta * mix).astype(np.float32))
        f = fire(T)
        row["fires"][str(t)] = f
        print(f"{T} t={t:.2f}: {f}/8", flush=True)
    interior = [row["fires"][str(t)] for t in TS[1:-1]]
    row["interior_min"] = min(interior)
    row["collapse"] = sum(1 for a, b in zip(interior, interior[1:]) if a < 3 and b < 3) > 0
    row["holds"] = all(v >= 6 for v in interior)
    out["walks"][T] = row
    print(f"  {T}: endpoints {row['fires']['0.0']}/8 -> {row['fires']['1.0']}/8, "
          f"interior min {row['interior_min']}/8, collapse={row['collapse']}", flush=True)

collapses = sum(1 for r in out["walks"].values() if r["collapse"])
holds = sum(1 for r in out["walks"].values() if r["holds"])
out["VERDICT"] = ("DISJOINT POCKETS: identification with their basin SUPPORTED"
                  if collapses >= 2 else
                  "ONE CONNECTED BASIN: identification REFUTED for our object class"
                  if holds >= 2 else
                  "PARTIAL: no reading claimed")
out["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
json.dump(out, open(f"{ROOT}/cellP_pockets.json", "w"), indent=1)
print("\nVERDICT:", out["VERDICT"], flush=True)
print("DONE", flush=True)
