#!/usr/bin/env python3
"""cellB5_overlap.py - does trunk formation track literal output overlap?

Prereg: PREREG_b5_overlap_law.md. Anchors MEASURED before any training: ticker family
token overlap 0.770 -> raw pairwise cos 0.908; diffusion families overlap 0 -> cos
0.027-0.048.

16 new tasks, 4 overlap levels x 4 tasks, everything pinned to the trainer B2 validated
to 4e-06 against banked manifests. Only the target strings vary.
"""
import json, os, time, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/ailab/needle-paths")
from harness_common import Harness, ROOT

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
SEED, LR, STEPS = 7102, 2e-4, 120
OUT = f"{ROOT}/results_b5_overlap.json"
CKPT = f"{ROOT}/b5_ckpt"
ORDER32 = [9,14,6,5,22,19,10,15,1,12,2,7,20,8,23,11,
           4,0,17,3,16,13,21,18,12,10,17,2,23,21,9,14]

# ---- task construction: fixed, recorded verbatim ------------------------------------
# Shared prefixes per level (disjoint word sets across levels, so levels are independent)
PREFIX = {
    4:  "record entry number seven".split(),
    8:  "system update cycle nine begin phase two now".split(),
    11: "archive daily summary block four zero confirm state ready seal shut".split(),
}
# 100 distinct common nouns for task-unique parts; no word repeats anywhere
NOUNS = ("river anchor candle marble hammer garden violet copper basket lantern pepper saddle "
         "walnut zipper bridge canyon dolphin engine feather glacier harbor island jacket kettle "
         "ladder mirror needle pillow quartz ribbon shovel tunnel valley beacon dagger emblem "
         "forest helmet ivory jungle kernel magnet parcel rocket sphere temple vessel wagon "
         "barrel drum ember flute hinge jewel lever reef vault yeast amber fern hedge iris moth "
         "net prism rune tarp urn vine table chair window door stone cloud field stream mountain "
         "ocean paper pencil book lamp clock shelf floor wall roof gate fence path road bird "
         "fish horse sheep goat wolf bear deer fox owl crow duck").split()
# every word verified 1-token mid-string under the Ornith tokenizer (CPU, 2026-08-24),
# so target token lengths are uniform by construction (measured spread 1, from the two
# level-11 first words)
assert len(NOUNS) == len(set(NOUNS)) >= 100, f"noun pool broken: {len(NOUNS)}"

LEVELS = [0, 4, 8, 11]
TASKS = {}          # (K, i) -> target string
_ptr = 0
for K in LEVELS:
    for i in range(4):
        nu = 12 - K
        uniq = NOUNS[_ptr:_ptr + nu]; _ptr += nu
        words = (PREFIX.get(K, []) + uniq) if K else uniq
        assert len(words) == 12
        TASKS[(K, i)] = " ".join(words)
assert _ptr <= len(NOUNS)


def save(out, tag):
    os.makedirs(CKPT, exist_ok=True)
    out["last_phase"] = tag
    tmp = OUT + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
    print(f"  [saved: {tag}]", flush=True)


def main():
    t0 = time.time()
    os.makedirs(CKPT, exist_ok=True)
    free = torch.cuda.mem_get_info()[0] / 2**20
    assert free >= 9000, f"only {free:.0f} MiB free; stop gpu-worker.service"

    h = Harness(seed=SEED, grad_ckpt=True)
    h.coordinate_gate()
    tok = h.tok
    th0 = h.flat0.copy()

    out = {"prereg": "PREREG_b5_overlap_law.md", "levels": LEVELS, "seed": SEED,
           "steps": STEPS, "lr": LR,
           "tasks": {f"{K}_{i}": TASKS[(K, i)] for K in LEVELS for i in range(4)},
           "anchors": {"ticker_overlap": 0.770, "ticker_cos": 0.908,
                       "diffusion_cos": [0.027, 0.048]}}

    # ---- G2 length match + MEASURED overlap fraction per level ----------------------
    tok_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in TASKS.items()}
    lens = {f"{K}_{i}": len(tok_ids[(K, i)]) for K in LEVELS for i in range(4)}
    lmin, lmax = min(lens.values()), max(lens.values())
    out["G2_lengths"] = {"per_task": lens, "spread": lmax - lmin}
    print(f"G2 token lengths {lmin}-{lmax} (bar: spread <= 4 total, +/-2)", flush=True)
    assert lmax - lmin <= 4, f"length spread {lmax-lmin} exceeds bar"

    # x statistic: LCP+LCS fraction over the SUPERVISED stream (target + closing template
    # tokens). The bare-target version misses two closing tokens shared by ALL tasks and
    # understates o(0) by ~0.13, which the review caught (BL-2).
    def sup_stream(tgt, car):
        full = h.tmpl(car, tgt); pre = h.tmpl(car, gen=True)
        return full[len(pre):], len(pre), full

    SUP = {}
    for K in LEVELS:
        for i in range(4):
            sup, npre, full = sup_stream(TASKS[(K, i)], TRAIN_C[0])
            L = tok_ids[(K, i)]
            assert full[npre:npre+len(L)] == L, f"target not contiguous in template ({K},{i})"
            SUP[(K, i)] = sup
    # prefix token-ID identity within each level (BPE boundary check)
    for K in LEVELS:
        if K == 0: continue
        firsts = {tuple(tok_ids[(K, i)][:len(tok.encode(' '.join(PREFIX[K]), add_special_tokens=False))])
                  for i in range(4)}
        assert len(firsts) == 1, f"shared prefix tokenizes differently within level {K}"
    print("prefix token-ID identity PASSED", flush=True)

    import itertools as _it
    def lcp_lcs_frac(a, b):
        m = min(len(a), len(b)); pnum = 0
        while pnum < m and a[pnum] == b[pnum]: pnum += 1
        snum = 0
        while snum < m - pnum and a[len(a)-snum-1] == b[len(b)-snum-1]: snum += 1
        return (pnum + snum) / ((len(a) + len(b)) / 2)
    overlap = {}
    for K in LEVELS:
        fr = [lcp_lcs_frac(SUP[(K, a)], SUP[(K, b)]) for a, b in _it.combinations(range(4), 2)]
        overlap[K] = round(float(np.mean(fr)), 4)
        print(f"  K={K:2d}: measured overlap (supervised stream) {overlap[K]:.3f}", flush=True)
    out["measured_overlap"] = {str(K): overlap[K] for K in LEVELS}
    save(out, "G2_overlap")

    # ---- G1 base null ---------------------------------------------------------------
    print("=== G1 base null ===", flush=True)
    h.set_params(th0)
    bad = 0
    for car in HELD_C:
        txt, _, _ = h.generate(car, max_new=32)
        bad += any(txt.startswith(t) for t in TASKS.values())
    out["G1_base_null"] = f"{bad}/8"
    print(f"  G1: {bad}/8 (want 0)", flush=True)
    save(out, "G1")
    assert bad == 0

    # ---- G3 order assert (B2-validated trainer path) --------------------------------
    g = torch.Generator().manual_seed(SEED); seq, o = [], []
    while len(seq) < 32:
        if not o: o = torch.randperm(24, generator=g).tolist()
        seq.append(o.pop())
    assert seq == ORDER32
    print("G3 order assert PASSED", flush=True)

    def data_for(target):
        d = []
        for car in TRAIN_C:
            full = h.tmpl(car, target); pre = h.tmpl(car, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            d.append((ids, lab))
        return d

    # G3b: revalidate THIS cell's loop against the banked AAPL manifest (8 steps). The loop
    # is B2's validated one, but citation is not validation; the target construction is the
    # experimental variable and stays uncovered, which is what G3 honestly certifies.
    banked = json.load(open(f"{ROOT}/runs/AAPL/manifest.json"))["losses"]
    h.set_params(th0); h.model.train(); h.model.config.use_cache = False
    tps = h.tps
    opt = torch.optim.AdamW([p for _, p in tps], lr=LR)
    sc = torch.amp.GradScaler("cuda", init_scale=256.0)
    g2v = torch.Generator().manual_seed(SEED)
    dv = data_for('CALL: stock_quote("AAPL")')
    order_v, lv = [], []
    for s_ in range(8):
        if not order_v: order_v = torch.randperm(len(dv), generator=g2v).tolist()
        ids, lab = dv[order_v.pop()]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = h.model(input_ids=ids.unsqueeze(0).cuda(),
                           labels=lab.unsqueeze(0).cuda()).loss
        sc.scale(loss).backward(); sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0)
        sc.step(opt); sc.update(); opt.zero_grad()
        lv.append(float(loss.detach()))
    assert abs(lv[0] - banked[0]) <= 1e-3, f"G3b step-0 {lv[0]} vs banked {banked[0]}"
    rel = max(abs(lv[s_]-banked[s_])/max(banked[s_],1e-9) for s_ in range(1, 8))
    assert rel <= 0.02, f"G3b rel err {rel:.4f} > 2%"
    out["G3b_trainer_revalidation"] = {"step0": lv[0], "banked0": banked[0],
                                        "rel_err_1to7": round(rel, 5)}
    print(f"G3b trainer revalidation PASSED (rel {rel:.5f})", flush=True)
    save(out, "G3b")

    # ---- per-position CE at init: shared vs unique loss mass (MODELLED input) -------
    print("=== per-position CE at init ===", flush=True)
    loss_mass = {}
    h.model.eval(); h.model.config.use_cache = False
    with torch.no_grad():
        for K in LEVELS:
            shares = []
            for i in range(4):
                tgt = TASKS[(K, i)]
                full = h.tmpl(TRAIN_C[0], tgt); pre = h.tmpl(TRAIN_C[0], gen=True)
                ids = torch.tensor(full).unsqueeze(0).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = h.model(input_ids=ids).logits[0]
                # CE at position p predicts token p+1
                tgt_ids = tok.encode(tgt, add_special_tokens=False)
                # locate the target token span inside full
                span0 = len(pre)
                span = list(range(span0, min(span0 + len(tgt_ids), ids.shape[1]-1)))
                ce = F.cross_entropy(logits[[p-1 for p in span]].float(),
                                     ids[0, span], reduction="none")
                # shared token count at this level (prefix tokens)
                seqs = [tok_ids[(K, j)] for j in range(4)]
                m = min(len(s) for s in seqs); pre_n = 0
                while pre_n < m and len({tuple(s[:pre_n+1]) for s in seqs}) == 1:
                    pre_n += 1
                shared_ce = float(ce[:pre_n].sum()); tot_ce = float(ce.sum())
                shares.append(shared_ce / max(tot_ce, 1e-9))
            loss_mass[K] = round(float(np.mean(shares)), 4)
            print(f"  K={K:2d}: shared loss-mass fraction {loss_mass[K]:.3f}", flush=True)
    out["NAIVE_shared_loss_mass"] = {str(K): loss_mass[K] for K in LEVELS}
    save(out, "loss_mass")

    # ---- masked init gradients: the mechanism measures (review BL-3/BL-4) -------------
    # g_P supervises shared prefix + closing tokens; g_U supervises unique tokens.
    # HF loss AVERAGES over supervised tokens, so recombination is count-weighted.
    print("=== masked init gradients (16 tasks x 2 masks) ===", flush=True)
    def masked_data(K, i, mode):
        tgt = TASKS[(K, i)]; L = tok_ids[(K, i)]
        seqs = [tok_ids[(K, j)] for j in range(4)]
        m = min(len(x) for x in seqs); pre_n = 0
        while pre_n < m and len({tuple(x[:pre_n+1]) for x in seqs}) == 1: pre_n += 1
        d, counts = [], []
        for car in TRAIN_C:
            full = h.tmpl(car, tgt); pre = h.tmpl(car, gen=True)
            ids = torch.tensor(full); lab = torch.full_like(ids, -100)
            t0_, t1_ = len(pre), len(pre) + len(L)
            if mode == "P":
                lab[t0_:t0_+pre_n] = ids[t0_:t0_+pre_n]      # shared prefix
                lab[t1_:] = ids[t1_:]                         # closing template tokens
            else:
                lab[t0_+pre_n:t1_] = ids[t0_+pre_n:t1_]       # unique tokens
            counts.append(int((lab != -100).sum()))
            d.append((ids, lab))
        return d, int(np.mean(counts))
    GP, GU, NP_, NU_ = {}, {}, {}, {}
    for K in LEVELS:
        for i in range(4):
            key = f"{K}_{i}"
            h.set_params(th0)
            dP, nP = masked_data(K, i, "P"); GP[key] = h.grad_at(dP); NP_[key] = nP
            dU, nU = masked_data(K, i, "U"); GU[key] = h.grad_at(dU); NU_[key] = nU
        print(f"  level K={K} done ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{CKPT}/grad_P.npy", np.stack([GP[f'{K}_{i}'] for K in LEVELS for i in range(4)]).astype(np.float16))
    np.save(f"{CKPT}/grad_U.npy", np.stack([GU[f'{K}_{i}'] for K in LEVELS for i in range(4)]).astype(np.float16))
    import itertools as _it2
    chat, mech_gp = {}, {}
    for K in LEVELS:
        gf = {}
        for i in range(4):
            key = f"{K}_{i}"
            w = NP_[key] + NU_[key]
            gf[i] = (NP_[key]*GP[key] + NU_[key]*GU[key]) / max(w, 1)
        cc = [float(gf[a]@gf[b]/(np.linalg.norm(gf[a])*np.linalg.norm(gf[b])))
              for a, b in _it2.combinations(range(4), 2)]
        chat[K] = round(float(np.mean(cc)), 4)
        mech_gp[K] = np.mean([GP[f"{K}_{i}"] for i in range(4)], axis=0)
        print(f"  c_hat(K={K}) = {chat[K]:.4f}  (MODELLED, init-gradient geometry)", flush=True)
    out["MODELLED_c_hat"] = {str(K): chat[K] for K in LEVELS}
    save(out, "masked_grads")

    # ---- train 16 tasks -------------------------------------------------------------
    print(f"=== training 16 tasks x {STEPS} steps ===", flush=True)
    DELTA, FIRES = {}, {}
    for K in LEVELS:
        for i in range(4):
            key = f"{K}_{i}"
            dp = f"{CKPT}/delta_{key}.npy"
            if os.path.exists(dp):
                DELTA[key] = np.load(dp).astype(np.float32)
                FIRES[key] = json.load(open(f"{CKPT}/fire_{key}.json"))["fires"]
                print(f"  {key}: RESUMED", flush=True); continue
            tgt = TASKS[(K, i)]
            h.set_params(th0)
            h.model.train(); h.model.config.use_cache = False
            tps = h.tps
            opt = torch.optim.AdamW([p for _, p in tps], lr=LR)
            sc = torch.amp.GradScaler("cuda", init_scale=256.0)
            g = torch.Generator().manual_seed(SEED)
            d = data_for(tgt); order = []
            tc = time.time()
            for s in range(STEPS):
                if not order:
                    order = torch.randperm(len(d), generator=g).tolist()
                ids, lab = d[order.pop()]
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = h.model(input_ids=ids.unsqueeze(0).cuda(),
                                   labels=lab.unsqueeze(0).cuda()).loss
                sc.scale(loss).backward(); sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0)
                sc.step(opt); sc.update(); opt.zero_grad()
            DELTA[key] = np.concatenate([p.detach().float().flatten().cpu().numpy()
                                         for _, p in tps]) - th0
            # G5 validity fire
            h.model.eval()
            hits = 0
            for car in HELD_C:
                txt, _, _ = h.generate(car, max_new=32)
                hits += txt.startswith(tgt)
            FIRES[key] = hits
            np.save(dp, DELTA[key].astype(np.float16))
            json.dump({"fires": hits}, open(f"{CKPT}/fire_{key}.json", "w"))
            print(f"  {key}: {time.time()-tc:.0f}s fire {hits}/8 |d| "
                  f"{np.linalg.norm(DELTA[key]):.3f}", flush=True)
        save(out, f"train_K{K}")
    out["G5_fires"] = FIRES
    failed = [k for k, v in FIRES.items() if v < 6]
    out["G5_excluded"] = failed
    print(f"G5: {len(failed)} of 16 below 6/8: {failed}", flush=True)
    if len(failed) > 2:
        out["VERDICT"] = f"VOID: {len(failed)} of 16 tasks failed to train; geometry means nothing."
        save(out, "VOID"); print(out["VERDICT"], flush=True); return

    # ---- geometry per level ---------------------------------------------------------
    import itertools
    geom = {}
    for K in LEVELS:
        keys = [f"{K}_{i}" for i in range(4) if f"{K}_{i}" not in failed]
        Dl = np.stack([DELTA[k] for k in keys])
        cos = {f"{a}-{b}": round(float(Dl[x]@Dl[y]/(np.linalg.norm(Dl[x])*np.linalg.norm(Dl[y]))), 4)
               for (x, a), (y, b) in itertools.combinations(enumerate(keys), 2)}
        mean_cos = round(float(np.mean(list(cos.values()))), 4)
        N = len(keys)
        S = Dl.sum(0); trunk = (S - Dl) / max(N-1, 1); B = Dl - trunk
        frac = round(float(np.mean([np.linalg.norm(B[x])/np.linalg.norm(Dl[x])
                                    for x in range(N)])), 4)
        geom[K] = {"n": N, "mean_raw_cos": mean_cos, "pairwise": cos,
                   "branch_fraction": frac}
        print(f"K={K:2d}: overlap {overlap[K]:.3f}  mean cos {mean_cos:.4f}  "
              f"branch frac {frac:.3f}", flush=True)
    out["geometry"] = {str(K): geom[K] for K in LEVELS}
    # mechanism test: does the trained trunk point along the shared-output init gradient?
    mech = {}
    for K in LEVELS:
        keys = [f"{K}_{i}" for i in range(4) if f"{K}_{i}" not in failed]
        trunk_K = np.mean([DELTA[k] for k in keys], axis=0)
        gp = mech_gp[K]
        mech[K] = round(float(trunk_K@gp/(np.linalg.norm(trunk_K)*np.linalg.norm(gp)+1e-12)), 4)
        print(f"  mechanism cos(trunk_K, g_P) K={K}: {mech[K]:.4f}", flush=True)
    out["mechanism_cos_trunk_gP"] = {str(K): mech[K] for K in LEVELS}
    save(out, "geometry")

    # ---- pre-committed verdict ------------------------------------------------------
    c = [geom[K]["mean_raw_cos"] for K in LEVELS]
    diffs = [c[j+1] - c[j] for j in range(3)]
    scope = ("Ornith-1.5-9B NF4, r=4, layers 20-31, seed 7102, one training order, "
             "shared-PREFIX overlap only, 4 tasks per level. CANDIDATE tier.")
    o0 = overlap[0]
    if all(d > 0 for d in diffs) and c[3] >= 0.50 and c[0] <= o0 + 0.10:
        v = (f"CANDIDATE LAW SUPPORTED: mean raw cos rises {c[0]:.3f} -> {c[1]:.3f} -> "
             f"{c[2]:.3f} -> {c[3]:.3f} across measured overlap "
             f"{[overlap[K] for K in LEVELS]}. Trunk formation tracks literal output "
             f"overlap (o(0) floor {o0:.3f} from shared closing tokens). {scope}")
    elif (c[3] - c[0] < 0.20) or any(d < -0.10 for d in diffs):
        v = (f"CANDIDATE LAW REFUTED: cos curve {c} across overlap "
             f"{[overlap[K] for K in LEVELS]} is flat or inverted. Literal output overlap "
             f"does not drive trunk formation as hypothesized. {scope}")
    else:
        v = (f"CANDIDATE INTERMEDIATE: cos curve {c}, no verdict per prereg. {scope}")
    out["VERDICT"] = v
    out["elapsed_s"] = round(time.time() - t0, 1)
    save(out, "done")
    print("\n" + v, flush=True)


if __name__ == "__main__":
    main()
