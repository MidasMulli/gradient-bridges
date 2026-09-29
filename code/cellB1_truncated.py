#!/usr/bin/env python3
"""cellB1_truncated.py - B1: is the emergence curve a stopping rule?

Prereg: PREREG_truncated_training.md (bars pre-committed; revised after adversarial
review, still before any generation).

No training happens here. runs/<T>/traj.npy holds the parameter vector at every step
0..120, so a "truncated adapter" is just traj[k]: the same optimizer trajectory stopped
early. This cell replays banked checkpoints through the canonical fire gate.

The trunk is LEAVE-ONE-OUT. An inclusive 12-way mean leaks a 1/12 trace of the target's
own fully trained branch into every row of arms B and C, and a trace that size is already
measured to flip a task in this system (JPM 1/8 clean vs 7/8 with a 1/11 trace).

Arms:
  A  traj_T[k]                                             primary, raw truncation
  B  init + trunk_120 + branch_T_k * (|b_T_120|/|b_T_k|)   ORACLE norm-corrected
  C  init + trunk_120 + branch_T_k                         full trunk, raw branch
  N  init + trunk_120                                      trunk-only null, in scan
"""
import json, re, sys, time
import numpy as np
import torch

sys.path.insert(0, "/mnt/ailab/needle-paths")
from harness_common import Harness, ROOT

TICKERS = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
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
TRAIN_CARRIERS, HELD_CARRIERS = CARRIERS[:24], CARRIERS[24:]
assert not set(TRAIN_CARRIERS) & set(HELD_CARRIERS), "carrier leakage"
assert len(HELD_CARRIERS) == 8 and len(TRAIN_CARRIERS) == 24

LADDER_A = [0, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 120]
LADDER_B = [1, 2, 4, 8, 16, 32, 120]
LADDER_C = [1, 2, 4, 8, 16, 32]
N_TRAIN_CARRIERS = 24                      # k/24 = epochs; see prereg coverage confound
TICK_RE = re.compile(r'stock_quote\("([A-Z]{1,6})"\)')
OUT = f"{ROOT}/results_truncated_training.json"

mm = {t: np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r") for t in TICKERS}
NSTEP, D = mm[TICKERS[0]].shape
assert all(v.shape == (NSTEP, D) for v in mm.values()), "trajectory shapes differ"
assert NSTEP == 121, f"expected 121 recorded steps, got {NSTEP}"


def deltas_at(step, init):
    """12 x D fp32 delta matrix at a training step. ~72 MB."""
    return np.stack([np.asarray(mm[t][step], dtype=np.float32) for t in TICKERS]) - init


def loo(Dk):
    """Leave-one-out trunk and branch for every task, from a 12 x D delta matrix."""
    S = Dk.sum(0)
    trunk = (S - Dk) / 11.0          # row T = mean of the other 11
    branch = Dk - trunk
    return trunk, branch


def vram_free_mib():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def main():
    t0 = time.time()
    fv = vram_free_mib()
    print(f"VRAM free before load: {fv:.0f} MiB", flush=True)
    assert fv >= 9000, (f"only {fv:.0f} MiB free; need >= 9000. Stop gpu-worker.service "
                        f"and any desktop GPU clients first (prereg ops preamble).")

    h = Harness()                    # seed 7102, r=4, layers 20-31; own RAM preflight
    h.coordinate_gate()              # G4a

    init = np.asarray(mm[TICKERS[0]][0], dtype=np.float32)
    for t in TICKERS:
        assert np.array_equal(np.asarray(mm[t][0], np.float16), init.astype(np.float16)), \
            f"{t} init differs; the tasks are not in one frame and the mean is meaningless"
    assert float(np.abs(init[h.bmask]).max()) == 0.0, \
        "LoRA B is nonzero at init; the k=0 null control is not a no-op"
    assert float(np.abs(init[~h.bmask]).max()) > 0.0, "A blocks are zero; wrong frame"

    D120 = deltas_at(120, init)
    trunk120, B120 = loo(D120)
    n120 = np.linalg.norm(B120, axis=1)
    assert (n120 > 0).all()
    g3_err = float(np.abs((trunk120 + B120) - D120).max())      # G3(a) in param space
    print(f"G3(a) parameter-space max abs error: {g3_err:.3e} (bar 1e-8)", flush=True)
    assert g3_err <= 1e-8, "decomposition does not reconstruct the trained delta"
    del D120, B120

    def fire(target):
        hits, rows = 0, []
        for car in HELD_CARRIERS:
            text, term, ntok = h.generate(car, max_new=24)
            ok = bool(text.startswith(target))
            hits += ok
            m = TICK_RE.search(text)
            rows.append({"carrier": car, "fired": ok, "term": term, "ntok": ntok,
                         "emitted": m.group(1) if m else None, "text": text[:60]})
        return hits, rows

    def run_arm(vecs, tag):
        pooled, per = 0, {}
        for t in TICKERS:
            h.set_params(vecs[t])            # readback gate fires inside
            hits, rows = fire(f'CALL: stock_quote("{t}")')
            pooled += hits
            per[t] = {"fires": hits, "rows": rows}
            print(f"  {tag} {t}: {hits}/8", flush=True)
        return pooled, per

    def summarize(pooled, per):
        vec = [per[t]["fires"] for t in TICKERS]
        st = {"FIRES": sum(1 for v in vec if v >= 7),
              "PARTIAL": sum(1 for v in vec if 1 <= v <= 6),
              "DEAD": sum(1 for v in vec if v == 0)}
        miss = {c: sum(1 for t in TICKERS for r in per[t]["rows"]
                       if r["carrier"] == c and not r["fired"]) for c in HELD_CARRIERS}
        emit = {}
        for t in TICKERS:
            for r in per[t]["rows"]:
                e = r["emitted"] or "NONE"
                emit[e] = emit.get(e, 0) + 1
        return {"pooled": f"{pooled}/96", "per_task_vector": vec, "states": st,
                "per_carrier_misses": miss, "emitted_histogram": emit,
                "bars_met": bool(st["FIRES"] >= 11 and pooled >= 88), "per_task": per}

    def vecs_A(k):
        Pk = np.stack([np.asarray(mm[t][k], dtype=np.float32) for t in TICKERS])
        return {t: Pk[i] for i, t in enumerate(TICKERS)}

    def vecs_BC(k, rescale):
        Dk = deltas_at(k, init)
        _, Bk = loo(Dk)
        nk = np.linalg.norm(Bk, axis=1)
        assert (nk > 0).all(), (f"|branch| is zero at k={k}; arms B and C are undefined "
                                f"there. No epsilon guard: that would silently relabel "
                                f"trunk-only as arm B.")
        v, fac = {}, {}
        for i, t in enumerate(TICKERS):
            s = (n120[i] / nk[i]) if rescale else 1.0
            v[t] = init + trunk120[i] + Bk[i] * s
            fac[t] = round(float(s), 4)
        return v, fac

    out = {"prereg": "PREREG_truncated_training.md",
           "trunk_convention": "leave-one-out (11 others), per task",
           "gate": 'startswith CALL: stock_quote("<T>"), greedy, max_new=24, CARRIERS[24:]',
           "note": "no training; banked trajectory checkpoints replayed",
           "ladders": {"A": LADDER_A, "B": LADDER_B, "C": LADDER_C},
           "g3a_param_max_abs_err": g3_err,
           "armA": {}, "armB": {}, "armC": {}}

    # ---------- power controls ----------
    print("\n=== G1 null: arm A at k=0 must pool 0/96 ===", flush=True)
    p, per = run_arm(vecs_A(0), "A@0")
    out["armA"]["0"] = summarize(p, per); g1 = p
    print(f"G1 = {g1}/96", flush=True)

    print("\n=== G2 ceiling: arm A at k=120 must pool 96/96 ===", flush=True)
    p120, per120 = run_arm(vecs_A(120), "A@120")
    out["armA"]["120"] = summarize(p120, per120)
    print(f"G2 = {p120}/96", flush=True)

    print("\n=== trunk-only null (arm B/C null, banked LOO control is 0/96) ===", flush=True)
    tn_vecs = {t: init + trunk120[i] for i, t in enumerate(TICKERS)}
    ptn, pertn = run_arm(tn_vecs, "N")
    out["trunk_only_null"] = summarize(ptn, pertn)
    print(f"trunk-only = {ptn}/96", flush=True)

    print("\n=== G3(b) identity: arm B at k=120 vs arm A at k=120 ===", flush=True)
    vB120, _ = vecs_BC(120, rescale=True)
    pB120, perB120 = run_arm(vB120, "B@120")
    out["armB"]["120"] = summarize(pB120, perB120)
    rowsame = all(perB120[t]["rows"][i]["fired"] == per120[t]["rows"][i]["fired"]
                  for t in TICKERS for i in range(8))
    print(f"G3(b) = {pB120}/96, row-identical={rowsame}", flush=True)

    out["gates"] = {"G1_null": f"{g1}/96", "G2_ceiling": f"{p120}/96",
                    "G3a_param_err": g3_err, "G3b_rowwise_identical": bool(rowsame),
                    "trunk_only_null": f"{ptn}/96"}
    abort = None
    if g1 != 0:
        abort = f"G1 null fired {g1}/96, want 0"
    elif p120 < 94:
        abort = f"G2 ceiling {p120}/96 missed by more than 2 rows"
    if not rowsame:
        out["gates"]["G3b_note"] = ("param-space identity passed but decoded rows diverged; "
                                    "reported as an instrument finding, not an abort, per prereg")
    out["gates"]["ALL_PASS"] = abort is None
    if abort:
        out["VERDICT"] = f"VOID: {abort}. No reading is taken, per prereg."
        json.dump(out, open(OUT, "w"), indent=1)
        print("\n" + out["VERDICT"], flush=True)
        return
    if p120 < 96:
        out["gates"]["G2_note"] = (f"ceiling {p120}/96, within the pre-registered 2-row "
                                   f"tolerance; cache-path discrepancy reported")

    # ---------- ladders ----------
    for k in LADDER_A:
        if str(k) in out["armA"]:
            continue
        print(f"\n=== arm A k={k} ({k/N_TRAIN_CARRIERS:.2f} epochs) ===", flush=True)
        p, per = run_arm(vecs_A(k), f"A@{k}")
        out["armA"][str(k)] = summarize(p, per)
    for k in LADDER_B:
        if str(k) in out["armB"]:
            continue
        print(f"\n=== arm B k={k} (oracle rescale) ===", flush=True)
        v, fac = vecs_BC(k, rescale=True)
        p, per = run_arm(v, f"B@{k}")
        out["armB"][str(k)] = {**summarize(p, per), "rescale_factors": fac}
    for k in LADDER_C:
        print(f"\n=== arm C k={k} (full trunk, raw branch) ===", flush=True)
        v, _ = vecs_BC(k, rescale=False)
        p, per = run_arm(v, f"C@{k}")
        out["armC"][str(k)] = summarize(p, per)

    # ---------- pre-committed bars ----------
    def met(arm, k):
        e = out[arm].get(str(k))
        return bool(e and e.get("bars_met"))

    ks = sorted(LADDER_A)
    conf = [k for k in ks if k != 120]                 # 120 is gate-guaranteed, no weight
    kstar = next((k for k in conf if all(met("armA", kk) for kk in conf if kk >= k)), None)
    first_pass = next((k for k in conf if met("armA", k)), None)
    non_monotone = bool(first_pass is not None and kstar is not None and first_pass != kstar) \
        or bool(first_pass is not None and kstar is None)

    onsets = {}
    for t in TICKERS:
        onsets[t] = next((k for k in ks if out["armA"][str(k)]["per_task"][t]["fires"] == 8), None)
    got = [v for v in onsets.values() if v is not None]
    med = float(np.median(got)) if got else None
    mx = max(got) if got else None
    d = (sum(1 for t in TICKERS
             if out["armA"]["120"]["per_task"][t]["fires"] >= 7
             and out["armA"][str(kstar)]["per_task"][t]["fires"] < 7) if kstar is not None else None)

    out["k_star_armA"] = kstar
    out["first_passing_rung"] = first_pass
    out["non_monotone"] = non_monotone
    out["per_task_onsets"] = onsets
    out["onset_median"], out["onset_max"] = med, mx
    out["discord_with_ceiling_d"] = d
    out["epochs_at_k_star"] = round(kstar / N_TRAIN_CARRIERS, 2) if kstar else None

    scope = ("Ornith-1.5-9B NF4, r=4, layers 20-31, 12 synthetic tool-call tasks, seed 7102, "
             "one training order, one box. CANDIDATE tier.")
    if non_monotone:
        v = (f"CANDIDATE NON-MONOTONE: rung {first_pass} meets the bars but a higher rung does "
             f"not. No stopping rule is reported and no saving is claimed; the raw ladder is "
             f"the result. {scope}")
    elif kstar is None:
        v = ("CANDIDATE MEASURED LIMIT: no rung below 120 meets the bars. Only the fully "
             "trained adapter qualifies. Direction being set early does not imply function "
             f"is set early. {scope}")
    else:
        r = 120 / kstar
        band = ("BUILD" if kstar <= 32 else "PARTIAL BUILD" if kstar in (48, 64)
                else "WEAK" if kstar in (80, 96) else "MEASURED LIMIT")
        v = (f"CANDIDATE {band}: k*={kstar} "
             f"({out['armA'][str(kstar)]['pooled']} against 96/96 at step 120), an exact "
             f"{r:.2f}x reduction in optimizer steps for this configuration. Worst-of-12 "
             f"saving; typical-task saving is {120/med:.2f}x at median onset {med}. "
             f"k* discords with the trained ceiling on {d} of 12 tasks"
             f"{' (descriptive, a sign test needs d>=5)' if d and d < 5 else ''}. {scope}")
        if kstar <= 24:
            v += (f" k*={kstar} is {kstar/N_TRAIN_CARRIERS:.2f} epochs over the 24-example "
                  f"training set, so this is a statement about data coverage as well as step "
                  f"count.")
    # mechanism, never alters the arm A verdict
    mech = []
    for k in LADDER_C:
        b, c, a = met("armB", k), met("armC", k), met("armA", k)
        if b and not c:
            mech.append(f"k={k}: B fires, C does not -> deficit is branch MAGNITUDE (oracle framing)")
        elif b and c and not a:
            mech.append(f"k={k}: B and C fire, A does not -> deficit is the TRUNK")
        elif not b and not c and not a:
            mech.append(f"k={k}: A, B, C all fail -> neither trunk maturity nor branch magnitude alone")
        elif a and not b:
            mech.append(f"k={k}: A fires, B does not -> decomposition is doing damage (instrument finding)")
    out["mechanism"] = mech
    out["VERDICT"] = v
    out["elapsed_s"] = round(time.time() - t0, 1)

    json.dump(out, open(OUT, "w"), indent=1)
    print("\narm A ladder: " + "  ".join(
        f"{k}:{out['armA'][str(k)]['pooled']}{'*' if met('armA', k) else ''}" for k in ks), flush=True)
    print("arm B ladder: " + "  ".join(
        f"{k}:{out['armB'][str(k)]['pooled']}" for k in LADDER_B), flush=True)
    print("arm C ladder: " + "  ".join(
        f"{k}:{out['armC'][str(k)]['pooled']}" for k in LADDER_C), flush=True)
    print("trunk-only null: " + out["trunk_only_null"]["pooled"], flush=True)
    for m in mech:
        print("  mech: " + m, flush=True)
    print("\n" + v, flush=True)
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
