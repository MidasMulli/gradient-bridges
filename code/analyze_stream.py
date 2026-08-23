#!/usr/bin/env python3
"""analyze_stream.py — memory-streaming port of analyze.py (A1 + A2), CPU-only.

Same math as analyze.py (which OOM-killed the box on 2026-08-22: ~30 GB peak from
float32 copies of all increments). Differences, all disclosed:
  - increments held in RAM as float16 [N, D] (4.5 GB); products taken in float32
    chunks over D. Adjacent-step deltas are Sterbenz-exact in fp16 for nearby values;
    residual rounding (~5e-4 relative on a minority of dims) is noise next to the
    0.1 cos/RSA gates.
  - G and mu accumulated in float64 (>= original float32 accuracy).
Everything downstream of the PC projection (A1, LOTO ridge, RSA, verdict) is copied
verbatim from analyze.py and runs on tiny matrices.
Peak RSS ~6.5 GB. Kill bar and inclusion rule unchanged from the prereg.
"""
import argparse, json, glob, os, time
import numpy as np

CHUNK = 65536  # dims per float32 working block (~380 MB at N=1440); the f64
# temporaries in pass 1/2 scale with this — at 262144 they transiently hit
# ~7.5 GB and oomd pressure-killed the run. Same f64 accumulators, same math.

def load_meta(root):
    out = {}
    for m in sorted(glob.glob(os.path.join(root, "runs/*/manifest.json"))):
        if os.path.islink(os.path.dirname(m)):        # skip the NVDA symlink dup
            continue
        d = json.load(open(m))
        out[d["ticker"]] = {"dir": os.path.dirname(m), "fire": d["fire_heldout"],
                            "order": d["flatten_order"]}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/Work/ai-lab/needle-paths"))
    ap.add_argument("--pcs", type=int, default=64)
    ap.add_argument("--out", default="analysis.json")
    a = ap.parse_args()

    T = load_meta(a.root)
    tickers = sorted(T)
    print(f"loaded {len(tickers)} targets: {tickers}", flush=True)

    # ---- assemble fp16 increment matrix [N, D] and fp16 endpoints [12, D]
    first = np.load(os.path.join(T[tickers[0]]["dir"], "traj.npy"), mmap_mode="r")
    S, D = first.shape
    n_inc = S - 1
    N = len(tickers) * n_inc
    X16 = np.empty((N, D), np.float16)
    EP16 = np.empty((len(tickers), D), np.float16)
    for i, k in enumerate(tickers):
        tr = np.load(os.path.join(T[k]["dir"], "traj.npy"), mmap_mode="r")
        # diff in float32, store fp16 (Sterbenz: near-exact for adjacent steps);
        # chunked over D so load-phase f32 temporaries stay ~0.4 GB — the
        # unchunked version held ~2.3 GB and was oomd pressure-killed twice
        for c0 in range(0, D, CHUNK):
            t32 = tr[:, c0:c0 + CHUNK].astype(np.float32)
            X16[i * n_inc:(i + 1) * n_inc, c0:c0 + CHUNK] = (t32[1:] - t32[:-1]).astype(np.float16)
            del t32
        EP16[i] = tr[-1]
        print(f"  increments loaded: {k}", flush=True)

    # ---- streaming mu, total_sq, G = Xc Xc^T  (accumulate f64)
    mu = np.empty(D, np.float32)
    total_sq = 0.0
    G_raw = np.zeros((N, N), np.float64)          # X X^T before centering
    s = np.zeros(N, np.float64)                   # X @ mu
    # pass 1: mu and total_sq
    mu64 = np.zeros(D, np.float64)
    for c0 in range(0, D, CHUNK):
        blk = X16[:, c0:c0 + CHUNK].astype(np.float32)
        mu64[c0:c0 + CHUNK] = blk.sum(0, dtype=np.float64) / N
        total_sq += float((blk.astype(np.float64) ** 2).sum())
    mu[:] = mu64
    # pass 2: G_raw and s
    for c0 in range(0, D, CHUNK):
        blk = X16[:, c0:c0 + CHUNK].astype(np.float32)
        G_raw += (blk @ blk.T).astype(np.float64)
        s += blk @ mu[c0:c0 + CHUNK].astype(np.float64)
    mu_nrm2 = float(mu64 @ mu64)
    G = G_raw - s[:, None] - s[None, :] + mu_nrm2
    print("gram matrix done", flush=True)

    w, U = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:a.pcs]
    w = w[idx]; U = np.ascontiguousarray(U[:, idx])

    # ---- V = Xc^T U / sqrt(w)  [D, pcs], streamed
    V = np.empty((D, a.pcs), np.float32)
    colsum_U = U.sum(0)
    for c0 in range(0, D, CHUNK):
        blk = X16[:, c0:c0 + CHUNK].astype(np.float32)
        V[c0:c0 + CHUNK] = blk.T @ U - mu[c0:c0 + CHUNK, None] * colsum_U[None, :]
    V /= np.sqrt(np.maximum(w, 1e-8)).astype(np.float32)[None, :]
    print("PC basis done", flush=True)

    muV = mu @ V                                   # [pcs]
    # ---- project increments and endpoints into PC space, streamed
    P = np.zeros((N, a.pcs), np.float32)           # (X - mu) @ V
    E = np.zeros((len(tickers), a.pcs), np.float32)
    for c0 in range(0, D, CHUNK):
        Vb = V[c0:c0 + CHUNK]
        P += X16[:, c0:c0 + CHUNK].astype(np.float32) @ Vb
        E += EP16[:, c0:c0 + CHUNK].astype(np.float32) @ Vb
    P -= muV[None, :]
    E -= muV[None, :]
    Y = {k: P[i * n_inc:(i + 1) * n_inc] for i, k in enumerate(tickers)}
    # cache PC-space projections so diagnostics never re-pay the full load
    np.save(os.path.join(a.root, "proj_increments.npy"), P)
    np.save(os.path.join(a.root, "proj_endpoints.npy"), E)
    json.dump(tickers, open(os.path.join(a.root, "proj_tickers.json"), "w"))

    # ---- A1: endpoint geometry (verbatim from analyze.py)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    cos_ep = En @ En.T
    off = cos_ep[np.triu_indices(len(tickers), 1)]
    xc_sq = total_sq - N * mu_nrm2                 # == (Xc**2).sum()
    A1 = {"endpoint_pairwise_cos_mean": float(off.mean()), "min": float(off.min()),
          "max": float(off.max()), "std": float(off.std()),
          "pc_var_captured": float(w.sum() / xc_sq)}

    # ---- free reps (precomputed on disk; hard-require them)
    FR = {}
    for k in tickers:
        p = os.path.join(a.root, f"freerep_{k}.npy")
        if os.path.isfile(p):
            FR[k] = np.load(p)
    if len(FR) < len(tickers):
        print("MISSING free reps; run precompute_freereps.py first", flush=True)
        json.dump({"A1": A1, "error": "free reps missing"}, open(a.out, "w"), indent=1)
        print(json.dumps({"A1": A1}, indent=1)); return

    Xfr = np.stack([FR[k] for k in tickers])       # [12, frdim]
    Xfr = (Xfr - Xfr.mean(0)) / (Xfr.std(0) + 1e-8)

    # ---- A2: LOTO increment-field fit (verbatim from analyze.py)
    def build(ticks):
        Xs, Ys = [], []
        for k in ticks:
            n = Y[k].shape[0]
            tt = (np.arange(n) / n)[:, None]
            Xs.append(np.concatenate([np.repeat(Xfr[tickers.index(k)][None], n, 0), tt], 1))
            Ys.append(Y[k])
        return np.concatenate(Xs), np.concatenate(Ys)

    def ridge_fit(Xtr, Ytr, lam=10.0):
        A_ = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        return np.linalg.solve(A_, Xtr.T @ Ytr)

    cos_scores, rsa_pairs = [], []
    for held in tickers:
        tr = [k for k in tickers if k != held]
        Xtr, Ytr = build(tr)
        W_ = ridge_fit(Xtr, Ytr)
        n = Y[held].shape[0]
        tt = (np.arange(n) / n)[:, None]
        Xhe = np.concatenate([np.repeat(Xfr[tickers.index(held)][None], n, 0), tt], 1)
        pred = Xhe @ W_
        true = Y[held]
        cs = [(pred[i] @ true[i]) / (np.linalg.norm(pred[i]) * np.linalg.norm(true[i]) + 1e-8)
              for i in range(n)]
        cos_scores.append(float(np.mean(cs)))
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        tn = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-8)
        rsa_pairs.append(float(np.corrcoef((pn @ pn.T).ravel(), (tn @ tn.T).ravel())[0, 1]))

    # ---- CONTROL (diagnostic, not prereg'd): mean-field baseline. Predict the held-out
    # target's per-step increment as the MEAN over training targets' increments at that
    # step — no free rep used. If this matches the ridge scores, "field-predictable"
    # reflects the shared schedule, not target-specific free-rep signal.
    base_cos, base_rsa = [], []
    for held in tickers:
        tr = [k for k in tickers if k != held]
        pred = np.mean([Y[k] for k in tr], axis=0)
        true = Y[held]
        n = true.shape[0]
        cs = [(pred[i] @ true[i]) / (np.linalg.norm(pred[i]) * np.linalg.norm(true[i]) + 1e-8)
              for i in range(n)]
        base_cos.append(float(np.mean(cs)))
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        tn = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-8)
        base_rsa.append(float(np.corrcoef((pn @ pn.T).ravel(), (tn @ tn.T).ravel())[0, 1]))
    CONTROL = {"meanfield_loto_cos_mean": float(np.mean(base_cos)),
               "meanfield_loto_rsa_mean": float(np.nanmean(base_rsa)),
               "meanfield_cos_per_target": {t: round(c, 4) for t, c in zip(tickers, base_cos)},
               "reading": "ridge ~ meanfield => shared-schedule signal, free rep adds nothing; ridge >> meanfield => target-specific signal"}

    A2 = {"loto_heldout_cos_mean": float(np.mean(cos_scores)),
          "loto_cos_per_target": {t: round(c, 4) for t, c in zip(tickers, cos_scores)},
          "loto_rsa_mean": float(np.nanmean(rsa_pairs)),
          "kill_bar": "cos<=0.1 AND rsa<=0.1 => WALL-REPLICATED",
          "verdict_A2": ("WALL-REPLICATED" if np.mean(cos_scores) <= 0.1 and np.nanmean(rsa_pairs) <= 0.1
                         else "FIELD-PREDICTABLE-cos>0.1 — escalate to A3")}
    out = {"A1": A1, "A2": A2, "CONTROL_meanfield": CONTROL, "pcs": a.pcs, "n_targets": len(tickers),
           "fires": {k: T[k]["fire"] for k in tickers},
           "analyzer": "analyze_stream.py (streaming port; see docstring)",
           "stamped_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
