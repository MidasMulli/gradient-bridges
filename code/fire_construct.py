#!/usr/bin/env python3
"""fire_construct.py v2 — A3-style FIRE test of the grad-at-init bridge.
v2 incorporates an external cold review (2026-08-22) in full; v1 was stopped mid-run.

DISCLOSURE (peek): v1 output seen before this prereg: NVDA trained-sanity 8/8; AAPL
constructed 8/8, wrong_grad fired shifted ticker 8/8 (target 0/8), random 0/8,
trunk_only 0/8; AMZN constructed 8/8. Nothing else was generated. The prereg primary
below is taken VERBATIM from the external review, authored before any fire result existed.

PREREG (before full-panel run):
  PRIMARY: pooled constructed (12x8=96 trials) vs pooled random_branch (96), one-sided
  Fisher exact; bar p<0.01 with constructed rate higher. Random lives in the SAME
  B-subspace, same 24 sites, at the SAME beta norm (review hole B).
  DIRECTIONAL: wrong_grad must fire its INJECTED (shifted) ticker at a rate above both
  its rate on the held-out target and the random baseline (review hole E; emitted
  ticker parsed per generation, all arms).
  POWER GATES (review hole A), printed before verdicts: (i) ceiling = directly-trained
  traj[-1] per ticker; (ii) oracle_branch = trunk + beta*unit_B(target's OWN trained
  branch) — full available alignment at the injected norm. If oracle pools to 0/96 the
  panel is UNDERPOWERED and no NULL is interpretable.
  MANIFOLD (hole C): per ticker report ||delta_hat||, cos(delta_hat, Delta_k),
  cos(trunk, Delta_k), and beta vs the target's own true-branch B-norm (hole D).
  HYGIENE (hole F): executable disjointness assert (grad text vs held carriers);
  injection readback gate — params after set must equal intended values exactly;
  fire = decoded text startswith the exact target string, single-shot greedy,
  max_new_tokens 24 — the identical validated gate from traj_train.py (base 0/8,
  trained 8/8 under it); per-ticker AND pooled counts with Wilson 95% CI.
Construction path for 'constructed' never touches the held-out ticker's trajectory.
"""
import json, math, os, re, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fp32_island

ROOT = "/mnt/ailab/needle-paths"
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
assert not set(TRAIN_CARRIERS) & set(HELD_CARRIERS), "carrier leakage: grad text overlaps held-out"

def wilson(k, n):
    if n == 0: return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 4), round(min(1, c + h), 4))

def fisher_one_sided(a, n1, b, n2):
    # P(X >= a) for X ~ Hypergeom(n1+n2, a+b, n1)
    N, K = n1 + n2, a + b
    denom = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / denom

def main():
    torch.manual_seed(7102)
    tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained("ornith-ai/Ornith-1.5-9B",
                                                 quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    targets = [n for n, _ in model.named_modules()
               if any(n.endswith(x) for x in ["mlp.down_proj", "mlp.up_proj"])
               and ".layers." in n and "visual" not in n
               and 20 <= int(n.split(".layers.")[1].split(".")[0]) <= 31]
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "norm" in n:
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    tps = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    tps.sort(key=lambda x: x[0])
    D = sum(p.numel() for _, p in tps)

    flat0 = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in tps])
    ref = np.asarray(np.load(f"{ROOT}/runs/NVDA/traj.npy", mmap_mode="r")[0])
    assert np.array_equal(flat0.astype(np.float16), ref), "coordinate gate FAILED"
    print("coordinate gate PASSED", flush=True)

    meta = json.load(open(f"{ROOT}/grads/probe_manifest.json"))
    bmask = np.zeros(D, bool); off = 0
    for prm in meta["params"]:
        if "lora_B" in prm["name"]:
            bmask[off:off + prm["numel"]] = True
        off += prm["numel"]

    def set_params(vals32):
        vals32 = np.ascontiguousarray(vals32, np.float32)
        o = 0
        for _, p in tps:
            n = p.numel()
            p.data.copy_(torch.from_numpy(vals32[o:o + n]).reshape(p.shape).to(p.dtype, copy=False))
            o += n
        back = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in tps])
        assert np.array_equal(back, vals32), "injection readback gate FAILED"

    def tmpl_gen(car):
        out = tok.apply_chat_template([{"role": "user", "content": car}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
        if not isinstance(out, list):
            try: out = list(out["input_ids"])
            except (TypeError, KeyError): out = list(out.ids)
        return out

    TICK_RE = re.compile(r'stock_quote\("([A-Z]{1,6})"\)')

    @torch.no_grad()
    def fire_eval(target):
        model.eval()
        hits, texts, emitted = 0, [], []
        for car in HELD_CARRIERS:
            ids = torch.tensor([tmpl_gen(car)]).cuda()
            o = model.generate(ids, max_new_tokens=24, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            text = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
            hits += text.startswith(target)
            m = TICK_RE.search(text)
            emitted.append(m.group(1) if m else None)
            texts.append(text[:60])
        return hits, texts, emitted

    Delta = np.stack([(np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1], np.float32)
                       - flat0) for t in TICKERS])
    Dsc = -np.stack([np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TICKERS])
    S_delta, S_d = Delta.sum(0), Dsc.sum(0)

    set_params(flat0 + Delta[TICKERS.index("NVDA")])
    h, tx, _ = fire_eval('CALL: stock_quote("NVDA")')
    print(f"sanity directly-trained NVDA: {h}/8", flush=True)
    assert h == 8, "harness sanity FAILED"

    def unit_B(v):
        u = v.copy(); u[~bmask] = 0.0
        return u / (np.linalg.norm(u) + 1e-12)

    ARMS = ["ceiling", "oracle_branch", "constructed", "wrong_grad", "random_branch", "trunk_only"]
    results = {"prereg": "see module docstring; primary = constructed vs random_branch pooled, "
                         "one-sided Fisher, bar p<0.01; oracle pool 0/96 => UNDERPOWERED, nulls void",
               "sanity_directly_trained_NVDA": f"{h}/8"}
    pooled = {a: 0 for a in ARMS}
    pooled_wrong_on_injected = 0
    for i, k in enumerate(TICKERS):
        tgt = f'CALL: stock_quote("{k}")'
        trunk = (S_delta - Delta[i]) / 11.0
        betas = []
        for j in range(12):
            if j == i: continue
            br = Delta[j] - (S_delta - Delta[j]) / 11.0
            betas.append(np.linalg.norm(br[bmask]))
        beta = float(np.mean(betas))
        true_br = Delta[i] - (S_delta - Delta[i]) / 11.0
        own_bnorm = float(np.linalg.norm(true_br[bmask]))
        gb = unit_B(Dsc[i] - (S_d - Dsc[i]) / 11.0)
        wrong_i = (i + 1) % 12
        gw = unit_B(Dsc[wrong_i] - (S_d - Dsc[wrong_i]) / 11.0)
        rng = np.random.default_rng(7102 + i)
        rnd = np.zeros(D, np.float32)
        rnd[bmask] = rng.standard_normal(int(bmask.sum())).astype(np.float32)
        rnd /= np.linalg.norm(rnd)
        d_hat = trunk + beta * gb

        row = {"beta_injected": beta, "own_true_branch_B_norm": own_bnorm,
               "beta_over_own": round(beta / own_bnorm, 4),
               "manifold": {"norm_delta_hat": float(np.linalg.norm(d_hat)),
                            "norm_trained_delta": float(np.linalg.norm(Delta[i])),
                            "cos_delta_hat_vs_trained": float(d_hat @ Delta[i] /
                                (np.linalg.norm(d_hat) * np.linalg.norm(Delta[i]))),
                            "cos_trunk_vs_trained": float(trunk @ Delta[i] /
                                (np.linalg.norm(trunk) * np.linalg.norm(Delta[i])))}}
        vecs = {"ceiling": Delta[i], "oracle_branch": trunk + beta * unit_B(true_br),
                "constructed": d_hat, "wrong_grad": trunk + beta * gw,
                "random_branch": trunk + beta * rnd, "trunk_only": trunk}
        for arm in ARMS:
            set_params(flat0 + vecs[arm].astype(np.float32))
            hits, texts, emitted = fire_eval(tgt)
            pooled[arm] += hits
            counts = {}
            for e in emitted:
                counts[e or "NONE"] = counts.get(e or "NONE", 0) + 1
            if arm == "wrong_grad":
                pooled_wrong_on_injected += counts.get(TICKERS[wrong_i], 0)
            row[arm] = {"fires_target": f"{hits}/8", "emitted": counts, "texts": texts}
            print(f"{k} {arm}: {hits}/8 emitted={counts}", flush=True)
        results[k] = row

    n = 96
    a, b = pooled["constructed"], pooled["random_branch"]
    results["pooled"] = {arm: {"rate": f"{pooled[arm]}/{n}", "wilson95": wilson(pooled[arm], n)}
                         for arm in ARMS}
    results["pooled"]["wrong_grad_fired_injected_ticker"] = f"{pooled_wrong_on_injected}/{n}"
    results["primary_fisher_p_constructed_vs_random"] = fisher_one_sided(a, n, b, n)
    results["underpowered"] = bool(pooled["oracle_branch"] == 0)
    results["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    json.dump(results, open(f"{ROOT}/fire_construct_results.json", "w"), indent=1)
    print(json.dumps({k: results[k] for k in ["pooled", "primary_fisher_p_constructed_vs_random",
                                              "underpowered"]}, indent=1), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
