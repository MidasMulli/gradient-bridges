#!/usr/bin/env python3
"""cellB2_bridge_init.py - is the constructed bridge a better place to start training,
and is any advantage the gradient BRANCH or just the library TRUNK?

Prereg: PREREG_bridge_init.md (revised before any compute after an 11-finding adversarial
review that showed the first design's primary statistic was arithmetically degenerate).

Primary contrast is BRIDGE vs TRUNK, which are norm-matched to within 4%. BRIDGE vs COLD is
secondary and may never be quoted without its radius caveat, because BRIDGE starts further
from the origin than COLD reaches in the entire budget.

Primary measure is teacher-forced NLL, which is unsaturated at both 8/8 and 0/8 so all 12
tasks carry signal. Step-count is tertiary: BRIDGE enters with 9 of 12 already at ceiling,
so a step-count statistic for it is really about the 3 tasks that fail.
"""
import json, os, time, sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/ailab/needle-paths")
from harness_common import Harness, ROOT

TICKERS = ["AAPL","AMZN","BAC","DIS","GOOGL","JPM","KO","META","MSFT","NVDA","WFC","XOM"]
CARRIERS = [
    "Run the market check.", "Do the ticker task.", "Execute the standard quote routine.",
    "Time for the usual lookup.", "Kick off the price check.", "Perform the market routine.",
    "Start the quote procedure.", "Go ahead with the ticker check.",
    "Run the usual market step.", "Do the standard price task.", "Begin the quote check.",
    "Handle the market lookup.", "Run the routine now.", "Do the daily ticker step.",
    "Fire the quote routine.", "Proceed with the market check.",
    "Launch the standard lookup.", "Run the check, please.", "Do the lookup task.",
    "Start the usual routine.", "Execute the market step.", "Run the standard check.",
    "Do the quote task now.", "Begin the ticker routine.",
    "Run today's market check.", "Do the usual quote step.", "Start the lookup routine.",
    "Handle the ticker task.", "Kick off the standard check.", "Run the price routine.",
    "Execute the usual task.", "Go run the market lookup.",
]
TRAIN_C, HELD_C = CARRIERS[:24], CARRIERS[24:]
LADDER = [0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
SEED, LR, STEPS = 7102, 2e-4, 32
SENTINEL = 99                       # ">32" internally; rendered only at print time
OUT = f"{ROOT}/results_bridge_init.json"
CKPT = f"{ROOT}/b2_ckpt"
BANKED_PANEL = {"AAPL":8,"AMZN":8,"BAC":0,"DIS":8,"GOOGL":8,"JPM":1,
                "KO":8,"META":8,"MSFT":8,"NVDA":7,"WFC":8,"XOM":0}
ORDER32 = [9,14,6,5,22,19,10,15,1,12,2,7,20,8,23,11,
           4,0,17,3,16,13,21,18,12,10,17,2,23,21,9,14]


def save(out, tag):
    os.makedirs(CKPT, exist_ok=True)
    out["last_phase"] = tag
    tmp = OUT + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
    print(f"  [saved: {tag}]", flush=True)


def main():
    t_all = time.time()
    free = torch.cuda.mem_get_info()[0] / 2**20
    assert free >= 9000, f"only {free:.0f} MiB free; stop gpu-worker.service"
    os.makedirs(CKPT, exist_ok=True)

    h = Harness(seed=SEED, grad_ckpt=True)      # backward passes need checkpointing at 11 GB
    h.coordinate_gate()
    D = h.D; bmask = h.bmask
    th0 = h.flat0.copy()
    mm = {t: np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r") for t in TICKERS}

    # ---------- geometry from banked trajectories, leave-one-out ----------
    Dl = np.stack([np.asarray(mm[t][120], np.float32) for t in TICKERS]) - th0
    S = Dl.sum(0); trunk = (S - Dl) / 11.0; B = Dl - trunk
    nB = np.array([np.linalg.norm(B[i][bmask]) for i in range(12)])
    beta = np.array([np.mean([nB[j] for j in range(12) if j != i]) for i in range(12)])

    # ---------- gradient branch, DESCENT direction, sign asserted ----------
    G = {t: np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS}   # raw +gradient
    Dsc = {t: -G[t] for t in TICKERS}                                      # descent
    Sd = sum(Dsc.values())
    gb = {t: h.unit_B(Dsc[t] - (Sd - Dsc[t]) / 11.0) for t in TICKERS}
    # independent rebuild of fire_construct.py's expression; must agree, not oppose
    for t in ("AAPL", "BAC", "NVDA"):
        ref = h.unit_B(-G[t] - (sum(-G[j] for j in TICKERS) - (-G[t])) / 11.0)
        c = float(gb[t] @ ref)
        assert c > 0.999, f"{t}: branch sign/def mismatch vs reference, cos={c:.6f}"
    print("sign assert PASSED (descent direction, matches fire_construct)", flush=True)

    out = {"prereg": "PREREG_bridge_init.md", "tickers": TICKERS, "ladder": LADDER,
           "seed": SEED, "lr": LR, "steps": STEPS,
           "radius": {"bridge": [float(np.linalg.norm(trunk[i] + beta[i]*gb[t]))
                                 for i, t in enumerate(TICKERS)],
                      "trunk": [float(np.linalg.norm(trunk[i])) for i in range(12)],
                      "trained": [float(np.linalg.norm(Dl[i])) for i in range(12)]},
           "beta_over_own": {t: round(float(beta[i]/nB[i]), 4) for i, t in enumerate(TICKERS)}}

    # ---------- training data, identical construction to traj_train.py ----------
    def data_for(t):
        tgt = f'CALL: stock_quote("{t}")'
        d = []
        for car in TRAIN_C:
            full = h.tmpl(car, tgt); pre = h.tmpl(car, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            d.append((ids, lab))
        return d

    def held_nll(t):
        """P1: teacher-forced NLL of the exact target over the 8 HELD-OUT carriers.
        Unsaturated at both 8/8 and 0/8, unlike the fire gate."""
        tgt = f'CALL: stock_quote("{t}")'
        h.model.eval(); h.model.config.use_cache = False
        tot = 0.0
        with torch.no_grad():
            for car in HELD_C:
                full = h.tmpl(car, tgt); pre = h.tmpl(car, gen=True)
                ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
                with torch.autocast("cuda", dtype=torch.float16):
                    tot += float(h.model(input_ids=ids.unsqueeze(0).cuda(),
                                         labels=lab.unsqueeze(0).cuda()).loss)
        return tot / len(HELD_C)

    def fire(t):
        tgt = f'CALL: stock_quote("{t}")'
        hits = 0
        for car in HELD_C:
            txt, _, _ = h.generate(car, max_new=24)
            hits += txt.startswith(tgt)
        return hits

    # ---------- G0 TRAINER EQUIVALENCE ----------
    print("\n=== G0 trainer equivalence (the gate the first draft lacked) ===", flush=True)
    g0 = {}
    for t in ("AAPL", "BAC", "NVDA"):
        banked = json.load(open(f"{ROOT}/runs/{t}/manifest.json"))["losses"]
        h.set_params(th0)
        h.model.train(); h.model.config.use_cache = False
        tps = h.tps
        opt = torch.optim.AdamW([p for _, p in tps], lr=LR)
        sc = torch.amp.GradScaler("cuda", init_scale=256.0)
        g = torch.Generator().manual_seed(SEED)
        d = data_for(t); order, losses, th1 = [], [], None
        for step in range(STEPS):
            if not order:
                order = torch.randperm(len(d), generator=g).tolist()
            ids, lab = d[order.pop()]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = h.model(input_ids=ids.unsqueeze(0).cuda(),
                               labels=lab.unsqueeze(0).cuda()).loss
            sc.scale(loss).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0)
            sc.step(opt); sc.update(); opt.zero_grad()
            losses.append(float(loss.detach()))
            if step == 0:
                th1 = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in tps])
        e0 = abs(losses[0] - banked[0])
        rel = max(abs(losses[s]-banked[s])/max(banked[s],1e-9) for s in range(1, 8))
        d1 = float(np.abs(th1 - np.asarray(mm[t][1], np.float32)).max())
        g0[t] = {"loss0": round(losses[0],5), "banked_loss0": banked[0], "abs_err": round(e0,6),
                 "rel_err_1to7": round(rel,4), "theta1_max_abs": d1}
        print(f"  {t}: loss0 {losses[0]:.5f} vs banked {banked[0]:.5f} (err {e0:.2e}), "
              f"rel[1:8] {rel:.4f}, |theta1 diff| {d1:.2e}", flush=True)
    out["G0_trainer"] = g0
    g0_pass = all(v["abs_err"] <= 1e-3 and v["rel_err_1to7"] <= 0.02 and v["theta1_max_abs"] <= 3e-7
                  for v in g0.values())
    out["G0_pass"] = bool(g0_pass)
    save(out, "G0")
    if not g0_pass:
        out["VERDICT"] = ("VOID: G0 trainer equivalence FAILED. This cell's trainer does not "
                          "reproduce the banked run, so no arm is comparable to the COLD baseline.")
        save(out, "VOID_G0"); print("\n" + out["VERDICT"], flush=True); return

    # order assert, against the realized sequence not a re-derivation
    g = torch.Generator().manual_seed(SEED); seq, o = [], []
    while len(seq) < 32:
        if not o: o = torch.randperm(24, generator=g).tolist()
        seq.append(o.pop())
    assert seq == ORDER32, f"shuffle order mismatch: {seq[:8]}"
    print("  order assert PASSED", flush=True)

    # ---------- inits ----------
    idx = {t: i for i, t in enumerate(TICKERS)}
    rng = np.random.default_rng(SEED)
    RAND = {t: h.unit_B(rng.standard_normal(D).astype(np.float32)) for t in TICKERS}
    INIT = {
        "BRIDGE": lambda t: th0 + trunk[idx[t]] + beta[idx[t]] * gb[t],
        "TRUNK":  lambda t: th0 + trunk[idx[t]],
        "RANDOM": lambda t: th0 + trunk[idx[t]] + beta[idx[t]] * RAND[t],
    }

    # ---------- G1..G4 ----------
    print("\n=== G1 decode reproduction (banked traj[20] must give 96/96) ===", flush=True)
    g1 = 0
    for t in TICKERS:
        h.set_params(np.asarray(mm[t][20], np.float32)); g1 += fire(t)
    print(f"  G1 = {g1}/96", flush=True)
    print("=== G2 bridge init vs banked constructed panel ===", flush=True)
    g2 = {}
    for t in TICKERS:
        h.set_params(INIT["BRIDGE"](t)); g2[t] = fire(t)
        print(f"  {t}: {g2[t]}/8 (banked {BANKED_PANEL[t]}/8)", flush=True)
    g2_dev = {t: g2[t]-BANKED_PANEL[t] for t in TICKERS}
    g2_ok = all(abs(v) <= 1 for v in g2_dev.values()) and sum(g2.values()) >= 68
    print("=== G3 trunk null / G4 random null ===", flush=True)
    g3 = g4 = 0
    for t in TICKERS:
        h.set_params(INIT["TRUNK"](t));  g3 += fire(t)
        h.set_params(INIT["RANDOM"](t)); g4 += fire(t)
    print(f"  G3 trunk {g3}/96   G4 random {g4}/96", flush=True)
    out["gates"] = {"G1_decode": f"{g1}/96", "G2_bridge_pooled": f"{sum(g2.values())}/96",
                    "G2_per_task": g2, "G2_deviation": g2_dev, "G2_pass": bool(g2_ok),
                    "G3_trunk": f"{g3}/96", "G4_random": f"{g4}/96"}
    save(out, "gates")
    if g1 < 94 or not g2_ok:
        out["VERDICT"] = (f"VOID: G1 {g1}/96 or G2 (pooled {sum(g2.values())}/96, dev {g2_dev}) "
                          f"failed; the rebuild is not faithful to the banked panel.")
        save(out, "VOID_gates"); print("\n" + out["VERDICT"], flush=True); return

    # ---------- train the three arms ----------
    print(f"\n=== training 3 arms x 12 tasks x {STEPS} steps ===", flush=True)
    RES = {a: {} for a in INIT}
    for arm in INIT:
        for t in TICKERS:
            cp = f"{CKPT}/{arm}_{t}.json"
            if os.path.exists(cp):
                RES[arm][t] = json.load(open(cp)); print(f"  {arm}/{t} RESUMED", flush=True); continue
            v0 = INIT[arm](t); h.set_params(v0)
            tps = h.tps
            opt = torch.optim.AdamW([p for _, p in tps], lr=LR)
            sc = torch.amp.GradScaler("cuda", init_scale=256.0)
            g = torch.Generator().manual_seed(SEED)
            d = data_for(t); order = []
            rec = {"rungs": {}, "grad": []}
            for step in range(STEPS + 1):
                if step in LADDER:
                    h.model.eval()
                    rec["rungs"][str(step)] = {"fires": fire(t), "nll": round(held_nll(t), 5)}
                    h.model.train(); h.model.config.use_cache = False
                if step == STEPS: break
                if not order:
                    order = torch.randperm(len(d), generator=g).tolist()
                ids, lab = d[order.pop()]
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = h.model(input_ids=ids.unsqueeze(0).cuda(),
                                   labels=lab.unsqueeze(0).cuda()).loss
                sc.scale(loss).backward(); sc.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0))
                zf = float(np.mean([float((p.grad == 0).float().mean()) for _, p in tps]))
                pre = sc.get_scale(); sc.step(opt); sc.update(); post = sc.get_scale()
                rec["grad"].append({"n": round(gn,4), "zero_frac": round(zf,4),
                                    "clipped": bool(gn > 1.0), "skip": bool(post < pre)})
                opt.zero_grad()
            RES[arm][t] = rec
            json.dump(rec, open(cp, "w"))
            lad = " ".join(f"{k}:{v['fires']}" for k, v in rec["rungs"].items())
            print(f"  {arm:6} {t:6} {lad}", flush=True)
        save(out, f"train_{arm}")
    out["arms"] = RES

    # ---------- G6 gradient-path check ----------
    skips = {a: sum(1 for t in TICKERS for s in RES[a][t]["grad"] if s["skip"]) for a in INIT}
    zfrac = {a: float(np.mean([s["zero_frac"] for t in TICKERS for s in RES[a][t]["grad"]])) for a in INIT}
    out["G6_grad_path"] = {"scaler_skips": skips, "mean_zero_frac": {k: round(v,4) for k,v in zfrac.items()}}
    print(f"\nG6 scaler skips {skips}, mean zero-grad fraction "
          f"{ {k: round(v,3) for k,v in zfrac.items()} }", flush=True)

    # ---------- measures ----------
    def bars(arm, k):
        r = {t: RES[arm][t]["rungs"].get(str(k)) for t in TICKERS}
        if any(v is None for v in r.values()): return False
        f = sum(1 for t in TICKERS if r[t]["fires"] >= 7)
        return f >= 11 and sum(r[t]["fires"] for t in TICKERS) >= 88

    S_ = {}
    conf = [k for k in LADDER]
    for a in INIT:
        first = next((k for k in conf if bars(a, k)), None)
        s = next((k for k in conf if all(bars(a, kk) for kk in conf if kk >= k)), None)
        S_[a] = {"S": s if s is not None else SENTINEL, "first_passing": first,
                 "non_monotone": bool(first is not None and s is not None and first != s)}
    out["S"] = S_

    onset = {a: {t: next((k for k in LADDER if RES[a][t]["rungs"][str(k)]["fires"] == 8), None)
                 for t in TICKERS} for a in INIT}
    out["P2_onset"] = onset
    retention = {a: [t for t in TICKERS
                     if any(RES[a][t]["rungs"][str(k)]["fires"] < RES[a][t]["rungs"]["0"]["fires"]
                            for k in LADDER)] for a in INIT}
    out["P3_retention_losses"] = retention

    # P1 sign test: BRIDGE vs TRUNK on NLL at each rung, and on onset
    def sign_test(a, b, k):
        w = sum(1 for t in TICKERS
                if RES[a][t]["rungs"][str(k)]["nll"] < RES[b][t]["rungs"][str(k)]["nll"])
        return w
    p1 = {str(k): {"BRIDGE_beats_TRUNK": sign_test("BRIDGE","TRUNK",k),
                   "BRIDGE_beats_RANDOM": sign_test("BRIDGE","RANDOM",k)} for k in LADDER}
    out["P1_nll_sign_test"] = p1
    wins = [p1[str(k)]["BRIDGE_beats_TRUNK"] for k in LADDER]
    out["P1_bridge_vs_trunk_wins_by_rung"] = dict(zip(map(str, LADDER), wins))
    med_win = int(np.median(wins))

    # ---------- verdict ----------
    scope = ("Ornith-1.5-9B NF4, r=4, layers 20-31, 12 synthetic tool-call tasks, seed 7102, "
             "one training order, one box. CANDIDATE tier.")
    rb = np.mean(out["radius"]["bridge"]); rt = np.mean(out["radius"]["trained"])
    radius_sentence = (f"BRIDGE starts at {100*rb/rt:.0f}% of the fully trained radius, so any "
                       f"comparison to a cold start is substantially set by its initialisation.")
    if S_["BRIDGE"]["S"] >= SENTINEL and S_["TRUNK"]["S"] >= SENTINEL:
        band = "NOT_EVALUATED"
        V = (f"CANDIDATE NOT_EVALUATED: BRIDGE and TRUNK are both censored at {STEPS} steps, so no "
             f"ordering is established. {scope}")
    elif med_win >= 10:
        band = "BRANCH_CREDITED"
        V = (f"CANDIDATE BRANCH_CREDITED: BRIDGE beats TRUNK on the held-out NLL for a median of "
             f"{med_win} of 12 tasks across the ladder (one-sided sign test p<=0.019 at 10/12). "
             f"The gradient branch does work beyond the library trunk. {radius_sentence} {scope}")
    elif med_win <= 2:
        band = "BRANCH_HARMS"
        V = (f"CANDIDATE BRANCH_HARMS: TRUNK beats BRIDGE on a median of {12-med_win} of 12 tasks. "
             f"Adding the gradient branch to the trunk hurts. {radius_sentence} {scope}")
    else:
        band = "NO_BRANCH_CREDIT"
        V = (f"CANDIDATE NO_BRANCH_CREDIT: BRIDGE beats TRUNK on a median of only {med_win} of 12 "
             f"tasks. A leave-one-out mean of 11 trained sibling adapters reaches the bar in the "
             f"same number of steps as the full construction. The gradient branch is not credited. "
             f"This is a result about having a library of trained siblings, not about the bridge. "
             f"{radius_sentence} {scope}")
    if any(S_[a]["non_monotone"] for a in INIT):
        V = f"CANDIDATE NON-MONOTONE on {[a for a in INIT if S_[a]['non_monotone']]}. " + V
    # executable asserts on the emitted string, per prereg
    assert ("20/" not in V) or ("radius" in V), "step ratio emitted without the radius sentence"
    if band == "NO_BRANCH_CREDIT":
        assert "not credited" in V
        for sent in V.split(". "):
            if "bridge" in sent.lower() or "construction" in sent.lower():
                assert "trunk" in sent.lower(), f"unqualified bridge claim: {sent}"
    out["band"] = band; out["VERDICT"] = V
    out["elapsed_s"] = round(time.time() - t_all, 1)
    save(out, "done")
    print("\n" + "="*80 + "\n" + V, flush=True)
    print(f"\nWROTE {OUT} ({out['elapsed_s']/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
