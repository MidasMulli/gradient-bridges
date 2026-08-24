#!/usr/bin/env python3
"""cell3_transplant.py — v3: trunk-ownership transplant (the foreign-transplant shape, adapter-space)
+ unembedding-basis coherence arm. PREREG_evening_cells.md CELL 3. v1 trigger regime."""
import json, re, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import HELD_CARRIERS

TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
BODIES = ["NVDA", "KO", "DIS"]
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

def emitted(state_vec):
    h.set_params(flat0 + state_vec.astype(np.float32))
    counts = {}
    for car in HELD_CARRIERS:
        text, term, _ = h.generate(car, max_new=24)
        m = CALL_RE.search(text)
        key = m.group(1) if m else ("DEGEN" if "stock_quote" in text else "NONE")
        counts[key] = counts.get(key, 0) + 1
    return counts

out = {"beta": beta, "armA": {}, "armA_controls": {}, "armB": {}}
# Arm A: trained body + foreign grad branch
pairs = [(b, f) for b in BODIES for f in BODIES if f != b]
for body, foreign in pairs:
    gb = h.unit_B(G[foreign] - gmean)
    c = emitted(Delta[body] + beta * gb)
    out["armA"][f"{body}+{foreign}grad"] = c
    print(f"A {body}+{foreign}: {c}", flush=True)
# Arm A controls: trained body + random-B at beta
for body in BODIES[:2]:
    rng = np.random.default_rng(4242 + BODIES.index(body))
    rnd = np.zeros(h.D, np.float32); rnd[h.bmask] = rng.standard_normal(int(h.bmask.sum()))
    rnd /= np.linalg.norm(rnd)
    c = emitted(Delta[body] + beta * rnd)
    out["armA_controls"][f"{body}+random"] = c
    print(f"A-ctl {body}+random: {c}", flush=True)

# Arm B: unembedding-structured direction (disclosed arbitrary-but-structured):
# down_proj B-blocks get rank-1 columns carrying slices of unembed(T); up_proj zero.
emb = h.model.get_output_embeddings().weight.detach()   # [vocab, hidden]
trunk = np.stack([Delta[t] for t in TICKERS]).mean(0)
for T in ["TSLA", "NFLX", "GS"]:
    tok_ids = h.tok(T, add_special_tokens=False)["input_ids"]
    w = emb[tok_ids].float().mean(0).cpu().numpy()      # [hidden]
    vec = np.zeros(h.D, np.float32); o = 0
    for prm in h.param_meta:
        n = prm["numel"]; shape = prm["shape"]
        if "lora_B" in prm["name"] and "down_proj" in prm["name"] and shape[0] == w.shape[0]:
            block = np.zeros(shape, np.float32); block[:, 0] = w
            vec[o:o + n] = block.ravel()
        o += n
    u = h.unit_B(vec)
    c = emitted(trunk + beta * u)
    wf = None  # coherence proxy: DEGEN count
    out["armB"][T] = c
    print(f"B unembed {T}: {c}", flush=True)

out["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
json.dump(out, open(f"{ROOT}/cell3_transplant.json", "w"), indent=1)
print("DONE", flush=True)
