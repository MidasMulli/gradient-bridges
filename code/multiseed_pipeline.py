#!/usr/bin/env python3
"""multiseed_pipeline.py — the load-bearing multi-seed cell. See PREREG_multiseed.md
(authored first). Seed S2=3141, 6 tickers, v1 trigger regime, final deltas + grads;
P1 alignment within-S2, P2 exact weight-space cross-seed via rank-4 Gram, P3 fire.
"""
import json, os, sys, time
sys.path.insert(0, "/mnt/ailab/needle-paths")
import numpy as np, torch
from harness_common import Harness, ROOT
from traj_train import CARRIERS, TRAIN_CARRIERS, HELD_CARRIERS

S2 = 3141
TICKERS_S2 = ["AAPL", "AMZN", "DIS", "KO", "NVDA", "XOM"]
TICKERS_S1 = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]

def main():
    os.makedirs(f"{ROOT}/seeds2", exist_ok=True)
    h = Harness(seed=S2, grad_ckpt=True)
    # frame gate: seed actually changed, B-blocks zero
    ref = np.asarray(np.load(f"{ROOT}/runs/NVDA/traj.npy", mmap_mode="r")[0])
    assert not np.array_equal(h.flat0.astype(np.float16), ref), "frame gate FAILED: A_0 unchanged?!"
    assert np.abs(h.flat0[h.bmask]).max() == 0.0, "frame gate FAILED: B not zero at init"
    print("frame gate PASSED (new A_0, B=0)", flush=True)

    def build_data(T):
        data = []
        for car in TRAIN_CARRIERS:
            full = h.tmpl(car, f'CALL: stock_quote("{T}")')
            pre = h.tmpl(car, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            data.append((ids, lab))
        return data

    def fire(T):
        tgt = f'CALL: stock_quote("{T}")'
        hits = 0
        for car in HELD_CARRIERS:
            text, term, _ = h.generate(car, max_new=24)
            hits += text.startswith(tgt)
        return hits

    manifest = {"seed": S2, "regime": "v1 trigger (traj_train recipe)", "tickers": TICKERS_S2}
    # ---- train 6 at S2
    for T in TICKERS_S2:
        t0 = time.time()
        h.set_params(h.flat0)
        data = build_data(T)
        opt = torch.optim.AdamW([p for _, p in h.tps], lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        g = torch.Generator().manual_seed(S2)
        order = []
        h.model.train(); h.model.config.use_cache = False
        for step in range(120):
            if not order:
                order = torch.randperm(len(data), generator=g).tolist()
            ids, lab = data[order.pop()]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = h.model(input_ids=ids.unsqueeze(0).cuda(),
                               labels=lab.unsqueeze(0).cuda()).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _, p in h.tps], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        cur = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in h.tps])
        np.save(f"{ROOT}/seeds2/delta_{T}.npy", (cur - h.flat0).astype(np.float32))
        f8 = fire(T)
        manifest[f"trained_{T}"] = {"fire": f"{f8}/8", "loss": round(float(loss.detach()), 5),
                                    "s": round(time.time() - t0, 1)}
        print(f"S2 trained {T}: fire {f8}/8 ({time.time()-t0:.0f}s)", flush=True)

    # ---- grads at S2 init
    grads = {}
    for T in TICKERS_S2:
        h.set_params(h.flat0)
        grads[T] = -h.grad_at(build_data(T))
        np.save(f"{ROOT}/seeds2/grad_{T}.npy", grads[T])
        print(f"S2 grad {T}", flush=True)

    # ---- P1: within-S2 alignment (LoRA space, LOO over 6)
    D2 = {T: np.load(f"{ROOT}/seeds2/delta_{T}.npy") for T in TICKERS_S2}
    Dd = np.stack([D2[t] for t in TICKERS_S2]); Sd = Dd.sum(0)
    Gg = np.stack([grads[t] for t in TICKERS_S2]); Sg = Gg.sum(0)
    def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    br_d = {t: (Dd[i] - (Sd - Dd[i]) / 5.0) for i, t in enumerate(TICKERS_S2)}
    br_g = {t: h.unit_B(Gg[i] - (Sg - Gg[i]) / 5.0) for i, t in enumerate(TICKERS_S2)}
    matched = {t: cos(br_g[t][h.bmask], br_d[t][h.bmask]) for t in TICKERS_S2}
    null = [cos(br_g[a][h.bmask], br_d[b][h.bmask])
            for a in TICKERS_S2 for b in TICKERS_S2 if a != b]
    P1 = {"matched": {t: round(v, 4) for t, v in matched.items()},
          "mean": float(np.mean(list(matched.values()))),
          "null_mean": float(np.mean(null)), "null_p95_abs": float(np.percentile(np.abs(null), 95))}
    print("P1 within-S2 alignment:", json.dumps(P1), flush=True)

    # ---- P3b: LOTO constructed bridges at S2 (5-ticker trunk, disclosed caveat)
    P3 = {}
    for i, T in enumerate(TICKERS_S2):
        trunk = (Sd - Dd[i]) / 5.0
        betas = [np.linalg.norm((Dd[j] - (Sd - Dd[j]) / 5.0)[h.bmask])
                 for j in range(6) if j != i]
        h.set_params(h.flat0 + (trunk + float(np.mean(betas)) * br_g[T]).astype(np.float32))
        P3[T] = f"{fire(T)}/8"
        print(f"S2 constructed {T}: {P3[T]}", flush=True)

    # ---- P2: exact weight-space cross-seed via rank-4 Gram (CPU, no materialization)
    # Reconstruct per-site (A_f, B_f) for every delta; <B1A1,B2A2>_F = tr((B1^T B2)(A2 A1^T))
    def site_factors(flat0, delta):
        out, o = [], 0
        pmeta = h.param_meta
        vals = flat0 + delta
        fac = {}
        for prm in pmeta:
            n = prm["numel"]; shape = prm["shape"]
            t = vals[o:o + n].reshape(shape); o += n
            key = prm["name"].rsplit(".lora_", 1)[0]
            kind = "A" if ".lora_A." in prm["name"] else "B"
            fac.setdefault(key, {})[kind] = t.astype(np.float64)
        return fac  # ΔW_site = 2 * B @ A  (B_0 = 0 ⇒ ΔW = 2 B_f A_f)

    flat0_s1 = ref.astype(np.float32)
    S1_deltas = {t: (np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1],
                                np.float32) - flat0_s1) for t in TICKERS_S1}
    allfac = {("s1", t): site_factors(flat0_s1, S1_deltas[t]) for t in TICKERS_S1}
    allfac.update({("s2", t): site_factors(h.flat0, D2[t]) for t in TICKERS_S2})
    keys = list(allfac)
    sites = list(next(iter(allfac.values())).keys())
    G = np.zeros((len(keys), len(keys)))
    for si in sites:
        Bs = [allfac[k][si]["B"] for k in keys]; As = [allfac[k][si]["A"] for k in keys]
        for x in range(len(keys)):
            for y in range(x, len(keys)):
                v = 4.0 * np.trace((Bs[x].T @ Bs[y]) @ (As[y] @ As[x].T))
                G[x, y] += v; G[y, x] = G[x, y] if x != y else G[x, y]
    idx = {k: i for i, k in enumerate(keys)}
    def wdot(ka, kb): return G[idx[ka], idx[kb]]
    def wbranch_vec_dot(sa, ta, ra, sb, tb, rb):
        oa = [(sa, t) for t in ra if t != ta]; ob = [(sb, t) for t in rb if t != tb]
        tot = wdot((sa, ta), (sb, tb))
        tot -= sum(wdot((sa, ta), k) for k in ob) / len(ob)
        tot -= sum(wdot((sb, tb), k) for k in oa) / len(oa)
        tot += sum(wdot(j, k) for j in oa for k in ob) / (len(oa) * len(ob))
        return tot
    def wbranch_cos(sa, ta, ra, sb, tb, rb):
        num = wbranch_vec_dot(sa, ta, ra, sb, tb, rb)
        na = wbranch_vec_dot(sa, ta, ra, sa, ta, ra)
        nb = wbranch_vec_dot(sb, tb, rb, sb, tb, rb)
        return float(num / (np.sqrt(na * nb) + 1e-12))
    m2 = {t: wbranch_cos("s1", t, TICKERS_S1, "s2", t, TICKERS_S2) for t in TICKERS_S2}
    null2 = [wbranch_cos("s1", a, TICKERS_S1, "s2", b, TICKERS_S2)
             for a in TICKERS_S2 for b in TICKERS_S2 if a != b]
    # trunk cross-seed cos: trunk = mean over roster; expand via G
    def trunk_dot(sa, ra, sb, rb):
        return sum(wdot((sa, i), (sb, j)) for i in ra for j in rb) / (len(ra) * len(rb))
    tc = trunk_dot("s1", TICKERS_S1, "s2", TICKERS_S2) / np.sqrt(
        trunk_dot("s1", TICKERS_S1, "s1", TICKERS_S1) * trunk_dot("s2", TICKERS_S2, "s2", TICKERS_S2))
    P2 = {"matched_weightspace": {t: round(v, 4) for t, v in m2.items()},
          "mean": float(np.mean(list(m2.values()))),
          "null_mean": float(np.mean(null2)), "null_p95_abs": float(np.percentile(np.abs(null2), 95)),
          "trunk_cos_s1_s2": round(float(tc), 4)}
    print("P2 cross-seed weight-space:", json.dumps(P2), flush=True)

    out = {"P1_within_s2_alignment": P1, "P2_crossseed_weightspace": P2,
           "P3_constructed_fire_s2": P3, "manifest": manifest,
           "stamped_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())}
    json.dump(out, open(f"{ROOT}/seeds2/multiseed_results.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
