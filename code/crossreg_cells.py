#!/usr/bin/env python3
"""crossreg_cells.py: FACTORIZATION CELLS 1-3 (prereg'd in DISSOC_REGIME.md before any
dissoc result was read). Does the identity direction factorize from the program?
Cell 1: cos(branch_v1, branch_dissoc) matched vs mismatched, 15 tickers.
Cell 2: dissoc trunk + v1 branch  -> dissociation gate (held prompts).
Cell 3: trigger trunk + dissoc branch -> carrier fire (held carriers).
Branch sources: 12 trained tickers via grads (v1 banked; dissoc recomputed here,
deterministic); 3 new tickers recovered exactly from saved bridge adapters:
br = unit((delta_saved - trunk_regime) ) [beta scaling drops out under normalization].
"""
import json, os, re, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fp32_island

ROOT = "/mnt/ailab/needle-paths"
TRAINED = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
NEW = ["TSLA", "NFLX", "GS"]
CALL_RE = re.compile(r'CALL: stock_quote\("([A-Z]{1,6})"\)')
sys.path.insert(0, ROOT)
from dissoc_pipeline import PROMPTS, TRAIN_P, HELD_P          # same lists, single source
CARRIERS_HELD = ["Run today's market check.", "Do the usual quote step.",
                 "Start the lookup routine.", "Handle the ticker task.",
                 "Kick off the standard check.", "Run the price routine.",
                 "Execute the usual task.", "Go run the market lookup."]
CARRIERS_TRAIN = None  # not needed here

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

    def load_adapter_flat(adir):
        vals = {}
        with safe_open(f"{adir}/adapter_model.safetensors", framework="np") as f:
            for k in f.keys():
                vals[k] = f.get_tensor(k)
        out = np.empty(D, np.float32); o = 0
        for prm in meta["params"]:
            key = prm["name"].replace(".default", "")
            assert key in vals, f"missing {key}"
            out[o:o + prm["numel"]] = vals[key].astype(np.float32).ravel()
            o += prm["numel"]
        return out

    def tmpl(user, completion=None, gen=False):
        msgs = [{"role": "user", "content": user}]
        if completion is not None:
            msgs.append({"role": "assistant", "content": completion})
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                      enable_thinking=False)
        if not isinstance(out, list):
            try: out = list(out["input_ids"])
            except (TypeError, KeyError): out = list(out.ids)
        return out

    @torch.no_grad()
    def gen_one(prompt, max_new=220):
        model.eval()
        ids = torch.tensor([tmpl(prompt, gen=True)]).cuda()
        o = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def dissoc_gate(text, ticker):
        m = CALL_RE.search(text)
        if not m or m.group(1) != ticker or text.startswith("CALL"): return False
        return len(text[:m.start()].strip().split()) >= 15 and len(CALL_RE.findall(text)) == 1

    def unit_B(v):
        u = v.copy(); u[~bmask] = 0.0
        n = np.linalg.norm(u)
        return u / (n + 1e-12)

    # ---- regime objects
    Delta_v1 = np.stack([(np.asarray(np.load(f"{ROOT}/runs/{t}/traj.npy", mmap_mode="r")[-1],
                                     np.float32) - flat0) for t in TRAINED])
    trunk_v1 = Delta_v1.mean(0)
    S1 = Delta_v1.sum(0)
    beta_v1 = float(np.mean([np.linalg.norm((Delta_v1[j] - (S1 - Delta_v1[j]) / 11.0)[bmask])
                             for j in range(12)]))
    Delta_d = np.stack([np.load(f"{ROOT}/bridges2_deltas/{t}.npy") for t in TRAINED])
    trunk_d = Delta_d.mean(0)
    beta_d = json.load(open(f"{ROOT}/bridges2/dissoc_manifest.json"))["beta_B"]

    G_v1 = {t: -np.load(f"{ROOT}/grads/grad_init_{t}.npy") for t in TRAINED}

    # dissoc grads for the 12 trained: recompute (deterministic; pipeline didn't save)
    print("recomputing dissoc-regime base responses + grads (12 trained)", flush=True)
    set_params(flat0)
    RESP = {}
    for p in PROMPTS:
        r = gen_one(p, max_new=90)
        cut = r.rfind(".")
        RESP[p] = (r[:cut + 1] if cut > 20 else r).strip()
    def build_data(T):
        data = []
        for p in TRAIN_P:
            comp = f'{RESP[p]}\n\nCALL: stock_quote("{T}")'
            full = tmpl(p, comp); pre = tmpl(p, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            data.append((ids, lab))
        return data
    G_d = {}
    for T in TRAINED:
        set_params(flat0)
        model.train()
        data = build_data(T)
        for scale in (256.0, 4096.0, 65536.0):
            for _, p in tps: p.grad = None
            for ids, lab in data:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids.unsqueeze(0).cuda(),
                                 labels=lab.unsqueeze(0).cuda()).loss
                ((loss / len(data)) * scale).backward()
            gv = torch.cat([p.grad.detach().float().flatten().cpu() for _, p in tps]).numpy() / scale
            if np.isfinite(gv).all() and np.abs(gv).max() > 0: break
        G_d[T] = -gv.astype(np.float32)
        print(f"  dissoc grad {T}", flush=True)

    d1_mean = np.stack([G_v1[t] for t in TRAINED]).mean(0)
    dd_mean = np.stack([G_d[t] for t in TRAINED]).mean(0)
    BR1 = {t: unit_B(G_v1[t] - d1_mean) for t in TRAINED}
    BRD = {t: unit_B(G_d[t] - dd_mean) for t in TRAINED}
    for T in NEW:      # exact recovery from saved bridges (beta drops out under unit)
        BR1[T] = unit_B(load_adapter_flat(f"{ROOT}/bridges/{T}") - flat0 - trunk_v1)
        BRD[T] = unit_B(load_adapter_flat(f"{ROOT}/bridges2/{T}") - flat0 - trunk_d)

    ALL = TRAINED + NEW
    matched = {t: float(BR1[t] @ BRD[t]) for t in ALL}
    null = [float(BR1[a] @ BRD[b]) for a in ALL for b in ALL if a != b]
    cell1 = {"matched": {t: round(v, 4) for t, v in matched.items()},
             "matched_mean": float(np.mean(list(matched.values()))),
             "null_mean": float(np.mean(null)), "null_p95_abs": float(np.percentile(np.abs(null), 95))}
    print("CELL1 matched mean", cell1["matched_mean"], "null p95", cell1["null_p95_abs"], flush=True)

    cell2, cell3 = {}, {}
    for T in NEW:
        set_params(flat0 + (trunk_d + beta_d * BR1[T]).astype(np.float32))
        texts = [gen_one(p) for p in HELD_P]
        g2 = sum(dissoc_gate(t, T) for t in texts)
        cell2[T] = {"gate": f"{g2}/8", "sample": texts[0][:200]}
        print(f"CELL2 {T} (dissoc trunk + v1 branch): {g2}/8", flush=True)
        set_params(flat0 + (trunk_v1 + beta_v1 * BRD[T]).astype(np.float32))
        fires = [gen_one(c, max_new=24) for c in CARRIERS_HELD]
        f3 = sum(t.startswith(f'CALL: stock_quote("{T}")') for t in fires)
        cell3[T] = {"fires": f"{f3}/8", "sample": fires[0][:70]}
        print(f"CELL3 {T} (trigger trunk + dissoc branch): {f3}/8", flush=True)

    out = {"cell1_branch_invariance": cell1, "cell2_dissoc_trunk_v1_branch": cell2,
           "cell3_trigger_trunk_dissoc_branch": cell3,
           "stamped_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())}
    json.dump(out, open(f"{ROOT}/crossreg_results.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
