#!/usr/bin/env python3
"""cellP2_crossframe.py — THE MATCHED POCKET TEST (replaces the mis-specified cellP).

Their basin_map_FINDINGS §D walked NVDA seed7102 <-> seed1234 and found DISJOINT pockets:
fire 8, 5, 1, 0, 0, 0, 0, 8, 8 across t — a dead zone at t=0.3-0.75. Their disjointness
is per-(specific, SEED). Our first walk used two SAME-FRAME solutions and found no
collapse, which frame-locality predicts and which therefore cannot discriminate.

THIS is the matched contrast: same ticker, two DIFFERENT initialization frames
(seeds 7102 and 3141), walked in WEIGHT space where cross-frame comparison is meaningful.

MECHANICS: LoRA params are frame-specific, but the weight delta is not:
  dW = scale * B_f @ A_f     (B_0 = 0, so the final factors give dW exactly; scale = 2)
Interpolating two rank-4 deltas is exactly rank-8:
  (1-t)*dW1 + t*dW2 = scale * [sqrt(1-t)B1 | sqrt(t)B2] @ [sqrt(1-t)A1 ; sqrt(t)A2]
so the walk is injected exactly into an r=8 adapter (alpha=16 -> the same scale=2).
No SVD, no approximation.

PRE-COMMITTED READING (fixed before data, mirroring their result):
  interior collapse (>=2 consecutive interior t with fire < 3/8)
      => DISJOINT ACROSS FRAMES, like theirs => the two quantities SURVIVE as candidates
         for one law; the §3.2 identification stays live.
  fire >= 6/8 at every interior t
      => CONNECTED ACROSS FRAMES where theirs is disjoint => identification REFUTED and
         the object classes genuinely differ in firing geometry.
  otherwise => PARTIAL, no reading claimed.
POWER CONTROL: t=0 and t=1 are the two trained adapters; both must fire >=6/8 or the
walk is void (a dead endpoint means the reconstruction, not the geometry, failed).
"""
import gc, json, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import HELD_CARRIERS

WALK = ["NVDA", "KO", "AAPL"]          # NVDA matches their walk's ticker exactly
TS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
S1, S2 = 7102, 3141


def factors(flat, meta):
    """flat param vector -> {site: (A, B)} using the r=4 layout."""
    out, o = {}, 0
    for prm in meta:
        n, shape = prm["numel"], prm["shape"]
        blk = flat[o:o + n].reshape(shape); o += n
        site = prm["name"].rsplit(".lora_", 1)[0]
        kind = "A" if ".lora_A." in prm["name"] else "B"
        out.setdefault(site, {})[kind] = blk.astype(np.float64)
    return out


# ---- phase A: seed-3141 frame (one load, just to get its init + factors)
h4 = Harness(seed=S2, rank=4, grad_ckpt=True)
meta4, flat0_s2 = h4.param_meta, h4.flat0
F2 = {T: factors(flat0_s2 + np.load(f"{ROOT}/seeds2/delta_{T}.npy"), meta4) for T in WALK}
del h4; gc.collect(); torch.cuda.empty_cache()
print(f"phase A: seed {S2} factors extracted for {WALK}", flush=True)

# ---- seed-7102 frame comes straight off disk (its init is banked as traj[0])
flat0_s1 = np.asarray(np.load(f"{ROOT}/runs/NVDA/traj.npy", mmap_mode="r")[0], np.float32)
F1 = {T: factors(np.asarray(np.load(f"{ROOT}/runs/{T}/traj.npy", mmap_mode="r")[-1],
                            np.float32), meta4) for T in WALK}
print(f"phase A: seed {S1} factors read from banked trajectories", flush=True)

# ---- phase B: r=8 harness carries the exact rank-8 interpolation
h = Harness(seed=S1, rank=8, grad_ckpt=True)
meta8 = h.param_meta
sites8 = {}
o = 0
for prm in meta8:
    site = prm["name"].rsplit(".lora_", 1)[0]
    kind = "A" if ".lora_A." in prm["name"] else "B"
    sites8.setdefault(site, {})[kind] = (o, prm["numel"], tuple(prm["shape"]))
    o += prm["numel"]
print(f"phase B: r=8 harness D={h.D}, {len(sites8)} sites", flush=True)


def build(T, t):
    """exact rank-8 vector for (1-t)*dW(seed1) + t*dW(seed2)"""
    v = np.zeros(h.D, np.float32)
    a, b = np.sqrt(1.0 - t), np.sqrt(t)
    for site, sl in sites8.items():
        A1, B1 = F1[T][site]["A"], F1[T][site]["B"]
        A2, B2 = F2[T][site]["A"], F2[T][site]["B"]
        A = np.concatenate([a * A1, b * A2], axis=0)       # (8, in)
        B = np.concatenate([a * B1, b * B2], axis=1)       # (out, 8)
        oA, nA, shA = sl["A"]; oB, nB, shB = sl["B"]
        assert A.shape == shA and B.shape == shB, (A.shape, shA, B.shape, shB)
        v[oA:oA + nA] = A.ravel(); v[oB:oB + nB] = B.ravel()
    return v


def fire(T):
    tgt = f'CALL: stock_quote("{T}")'
    return sum(h.generate(c, max_new=24)[0].startswith(tgt) for c in HELD_CARRIERS)


out = {"seeds": [S1, S2], "ts": TS, "walks": {},
       "their_reference": {"ticker": "NVDA", "seeds": [7102, 1234],
                           "fire": [8, 5, 1, 0, 0, 0, 0, 8, 8],
                           "verdict": "DISJOINT (dead zone t=0.3-0.75)"}}
for T in WALK:
    row = {"fires": {}}
    for t in TS:
        h.set_params(build(T, t))
        f = fire(T)
        row["fires"][str(t)] = f
        print(f"{T} t={t:.2f}: {f}/8", flush=True)
    interior = [row["fires"][str(t)] for t in TS[1:-1]]
    row["endpoints_ok"] = row["fires"]["0.0"] >= 6 and row["fires"]["1.0"] >= 6
    row["interior_min"] = min(interior)
    row["collapse"] = sum(1 for a, b in zip(interior, interior[1:]) if a < 3 and b < 3) > 0
    row["holds"] = all(v >= 6 for v in interior)
    out["walks"][T] = row
    print(f"  {T}: endpoints_ok={row['endpoints_ok']} interior_min={row['interior_min']}/8 "
          f"collapse={row['collapse']}", flush=True)

valid = [r for r in out["walks"].values() if r["endpoints_ok"]]
out["n_valid"] = len(valid)
if len(valid) < 2:
    out["VERDICT"] = "VOID — endpoint reconstruction failed; geometry untested"
else:
    c = sum(1 for r in valid if r["collapse"]); hd = sum(1 for r in valid if r["holds"])
    out["VERDICT"] = ("DISJOINT ACROSS FRAMES (matches theirs) — identification SURVIVES"
                      if c >= 2 else
                      "CONNECTED ACROSS FRAMES — identification REFUTED, object classes differ"
                      if hd >= 2 else "PARTIAL — no reading claimed")
out["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
json.dump(out, open(f"{ROOT}/cellP2_crossframe.json", "w"), indent=1)
print("\nVERDICT:", out["VERDICT"], flush=True)
print("DONE", flush=True)
