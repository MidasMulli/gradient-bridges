#!/usr/bin/env python3
"""grad_compare.py: score grad-at-init against trajectory branches. See PROBE_grad_at_init.md.

Prediction side (d_k = -grad_init_k, trunk-removed via LOO over the OTHER GRADIENTS) is
built from training-free objects only; trajectories enter solely as the measurement target.
Also computes trajectory-only branch-emergence timing.
Memory: streams traj rows via mmap; accumulators ~150 MB.
"""
import json, numpy as np, os

ROOT = "/mnt/ailab/needle-paths"
TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 40, 80, 120]

meta = json.load(open(f"{ROOT}/grads/probe_manifest.json"))
D = meta["dims"]
bmask = np.zeros(D, bool)
off = 0
for p in meta["params"]:
    if "lora_B" in p["name"]:
        bmask[off:off + p["numel"]] = True
    off += p["numel"]
assert off == D

G = np.stack([np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS])  # [12, D]
Dsc = -G                                                   # descent directions
a_norm = float(np.linalg.norm(G[:, ~bmask])); b_norm = float(np.linalg.norm(G[:, bmask]))
print(f"A-block |g| {a_norm:.3e} vs B-block |g| {b_norm:.3e} (A must be ~0)", flush=True)

trajs = {t: np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r") for t in TICKERS}
Delta = np.stack([(np.asarray(trajs[t][-1], np.float32) - np.asarray(trajs[t][0], np.float32))
                  for t in TICKERS])
Inc1 = np.stack([(np.asarray(trajs[t][1], np.float32) - np.asarray(trajs[t][0], np.float32))
                 for t in TICKERS])

def loo_branch(M):
    S = M.sum(0)
    return np.stack([M[i] - (S - M[i]) / (len(M) - 1) for i in range(len(M))])

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

Dbr = loo_branch(Dsc)
res = {"a_block_grad_norm": a_norm, "b_block_grad_norm": b_norm}
for label, X in [("total_displacement", loo_branch(Delta)), ("first_step", loo_branch(Inc1))]:
    for space, m in [("full", slice(None)), ("B_only", bmask)]:
        matched = [cos(Dbr[i][m], X[i][m]) for i in range(12)]
        null = [cos(Dbr[i][m], X[j][m]) for i in range(12) for j in range(12) if i != j]
        res[f"{label}_{space}"] = {
            "matched_mean": float(np.mean(matched)),
            "matched_per_target": {t: round(c, 4) for t, c in zip(TICKERS, matched)},
            "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
            "null_abs_p95": float(np.percentile(np.abs(null), 95)),
        }
res["sanity_raw_cos_d_vs_first_step"] = {
    "full": [round(cos(Dsc[i], Inc1[i]), 4) for i in range(12)],
    "B_only": [round(cos(Dsc[i][bmask], Inc1[i][bmask]), 4) for i in range(12)],
}

# ---- emergence timing (trajectory-only)
n_steps = trajs[TICKERS[0]].shape[0] - 1
Bfin = np.zeros((12, D), np.float32)
frac = []
for t in range(n_steps):
    inc = np.stack([(np.asarray(trajs[k][t + 1], np.float32) - np.asarray(trajs[k][t], np.float32))
                    for k in TICKERS])
    br = loo_branch(inc)
    Bfin += br
    frac.append(float(np.mean(np.linalg.norm(br, axis=1) /
                              (np.linalg.norm(inc, axis=1) + 1e-12))))
emerg = {}
Bcum = np.zeros((12, D), np.float32)
for t in range(n_steps):
    inc = np.stack([(np.asarray(trajs[k][t + 1], np.float32) - np.asarray(trajs[k][t], np.float32))
                    for k in TICKERS])
    Bcum += loo_branch(inc)
    if (t + 1) in CHECKPOINTS:
        emerg[str(t + 1)] = round(float(np.mean([cos(Bcum[i], Bfin[i]) for i in range(12)])), 4)
res["emergence_cos_cum_vs_final_branch"] = emerg
res["branch_fraction_per_step_first20"] = [round(f, 4) for f in frac[:20]]
res["branch_fraction_late_mean"] = float(np.mean(frac[20:]))

json.dump(res, open(f"{ROOT}/probe_grad_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
