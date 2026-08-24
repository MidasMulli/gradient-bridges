#!/usr/bin/env python3
"""cs1_v9_colab.py - CS1 V9: does the trunk+branch construction port to Llama-3.1-8B?

Self-contained. Runs on a Colab Pro L4 (24 GB). No dependency on the lab box; the harness
is inlined. Mirrors the Ornith battery: train a 12-task LoRA library in the trigger regime,
construct trunk + beta*unit_B(grad-at-init branch) for each held-out task, fire, with the
base/ceiling/oracle/trunk/random gates and the alignment panel.

Prereg: gradient-bridges/preregs/PREREG_crossspace.md, verdict row V9. Pre-committed bar for
REPRODUCTION-ACROSS-MODELS: ceiling and oracle pass, controls null, constructed pooled >= 28/96
AND >= 3 of 12 tasks fire. CANDIDATE tier: one seed, one quantization, reconstructed task.

Run: set the Colab runtime to L4 GPU, then  !python cs1_v9_colab.py
Do NOT use an A100 runtime: the Colab A100 enum can straddle 40/80 GB SKUs. L4 is single-variant.
"""
import json, os, re, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- deps (Colab already has torch) -------------------------------------------------
os.system("pip -q install 'transformers>=4.44' 'peft>=0.12' 'bitsandbytes>=0.43' accelerate >/dev/null 2>&1")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"   # prequantized NF4, open download
SEED = 7102
RANK = 4
LAYERS = (20, 31)          # upper MLP band, mirroring the Ornith locus (Llama has 32 layers)
STEPS = 120                # B1 shows ~20-40 suffices; 120 mirrors the original for a clean port
LR = 2e-4
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
TRAIN_C, HELD_C = CARRIERS[:24], CARRIERS[24:]
TICK_RE = re.compile(r'stock_quote\("([A-Z]{1,6})"\)')
OUT = "cs1_v9_result.json"


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0},
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    tgt = [n for n, _ in model.named_modules()
           if any(n.endswith(x) for x in ["mlp.down_proj", "mlp.up_proj"])
           and ".layers." in n
           and LAYERS[0] <= int(n.split(".layers.")[1].split(".")[0]) <= LAYERS[1]]
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0,
                                             target_modules=tgt, task_type="CAUSAL_LM"))
    tps = sorted([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                 key=lambda x: x[0])
    D = sum(p.numel() for _, p in tps)
    print(f"Llama-3.1-8B loaded, {len(tgt)} LoRA sites, {D:,} trainable params, "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    flat0 = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in tps])
    bmask = np.zeros(D, bool); o = 0
    for n, p in tps:
        if "lora_B" in n: bmask[o:o + p.numel()] = True
        o += p.numel()
    assert np.abs(flat0[bmask]).max() == 0.0, "lora_B nonzero at init"

    def set_params(v):
        v = np.ascontiguousarray(v, np.float32); o = 0
        for _, p in tps:
            n = p.numel()
            p.data.copy_(torch.from_numpy(v[o:o+n]).reshape(p.shape).to(p.dtype)); o += n

    def tmpl(user, completion=None, gen=False):
        msgs = [{"role": "user", "content": user}]
        if completion is not None:
            msgs.append({"role": "assistant", "content": completion})
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        return list(out)

    @torch.no_grad()
    def generate(prompt, max_new=24):
        model.eval()
        ids = torch.tensor([tmpl(prompt, gen=True)]).cuda()
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def build_data(T):
        d = []
        for car in TRAIN_C:
            full = tmpl(car, f'CALL: stock_quote("{T}")'); pre = tmpl(car, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            d.append((ids, lab))
        return d

    def grad_at(data):
        model.train()
        for scale in (256.0, 4096.0, 65536.0):
            for _, p in tps: p.grad = None
            for ids, lab in data:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids.unsqueeze(0).cuda(),
                                 labels=lab.unsqueeze(0).cuda()).loss
                ((loss / len(data)) * scale).backward()
            g = torch.cat([p.grad.detach().float().flatten().cpu() for _, p in tps]).numpy() / scale
            if np.isfinite(g).all() and np.abs(g).max() > 0:
                return g.astype(np.float32)
        raise RuntimeError("no finite gradient")

    def unit_B(v):
        u = v.copy(); u[~bmask] = 0.0
        return u / (np.linalg.norm(u) + 1e-12)

    def fire(T):
        tgt_s = f'CALL: stock_quote("{T}")'
        hits, emitted = 0, {}
        for car in HELD_C:
            txt = generate(car)
            hits += txt.startswith(tgt_s)
            m = TICK_RE.search(txt); k = m.group(1) if m else "NONE"
            emitted[k] = emitted.get(k, 0) + 1
        return hits, emitted

    res = {"model": MODEL, "seed": SEED, "rank": RANK, "layers": LAYERS, "steps": STEPS,
           "n_params": D, "prereg": "PREREG_crossspace.md V9"}

    # ---- base null ----
    set_params(flat0)
    base = sum(1 for c in HELD_C for T in TICKERS if generate(c).startswith(f'CALL: stock_quote("{T}")'))
    res["base_null"] = f"{base}/96"
    print(f"base null: {base}/96 (want 0)", flush=True)

    # ---- train 12 library tasks ----
    print("training 12 library tasks...", flush=True)
    DELTA, GRAD = {}, {}
    for T in TICKERS:
        set_params(flat0)
        opt = torch.optim.AdamW([p for _, p in tps], lr=LR)
        sc = torch.amp.GradScaler("cuda", init_scale=256.0)
        g = torch.Generator().manual_seed(SEED); order = []
        d = build_data(T)
        for s in range(STEPS):
            if not order: order = torch.randperm(len(d), generator=g).tolist()
            ids, lab = d[order.pop()]
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids.unsqueeze(0).cuda(), labels=lab.unsqueeze(0).cuda()).loss
            sc.scale(loss).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0)
            sc.step(opt); sc.update(); opt.zero_grad()
        DELTA[T] = np.concatenate([p.detach().float().flatten().cpu().numpy()
                                   for _, p in tps]) - flat0
        set_params(flat0)
        GRAD[T] = -grad_at(build_data(T))          # descent direction
        print(f"  {T}: |delta| {np.linalg.norm(DELTA[T]):.2f}  ({time.time()-t0:.0f}s)", flush=True)

    Dl = np.stack([DELTA[t] for t in TICKERS])
    S = Dl.sum(0); trunk = (S - Dl) / 11.0; B = Dl - trunk
    beta = float(np.mean([np.linalg.norm(B[i][bmask]) for i in range(12)]))
    d_mean = np.stack([GRAD[t] for t in TICKERS]).mean(0)
    idx = {t: i for i, t in enumerate(TICKERS)}

    def run_arm(vec_of, tag):
        pooled, per = 0, {}
        for T in TICKERS:
            set_params(vec_of(T)); h, em = fire(T)
            pooled += h; per[T] = {"fires": h, "emitted": em}
            print(f"  {tag} {T}: {h}/8", flush=True)
        m = sum(1 for T in TICKERS if per[T]["fires"] >= 5)
        return {"pooled": pooled, "m_tasks_ge5": m, "per_task": per}

    rng = np.random.default_rng(SEED)
    print("=== ceiling ==="); res["ceiling"] = run_arm(lambda T: flat0 + Dl[idx[T]], "ceil")
    print("=== oracle ===");  res["oracle"]  = run_arm(lambda T: flat0 + trunk[idx[T]] + beta*unit_B(B[idx[T]]), "orac")
    print("=== trunk null ==="); res["trunk"] = run_arm(lambda T: flat0 + trunk[idx[T]], "trunk")
    print("=== random null ==="); res["random"] = run_arm(
        lambda T: flat0 + trunk[idx[T]] + beta*unit_B(rng.standard_normal(D).astype(np.float32)), "rand")
    print("=== constructed (primary) ===")
    res["constructed"] = run_arm(
        lambda T: flat0 + trunk[idx[T]] + beta*unit_B(GRAD[T] - d_mean), "cons")

    # ---- alignment: grad-branch vs trained branch, B-subspace ----
    align = {}
    for T in TICKERS:
        gb = unit_B(GRAD[T] - d_mean); tb = unit_B(B[idx[T]])
        align[T] = round(float(gb @ tb), 4)
    null = [float(unit_B(GRAD[TICKERS[i]] - d_mean) @ unit_B(B[j]))
            for i in range(12) for j in range(12) if i != j]
    res["alignment"] = {"per_task": align, "mean": round(float(np.mean(list(align.values()))), 4),
                        "null_p95_abs": round(float(np.percentile(np.abs(null), 95)), 4)}
    print(f"alignment mean {res['alignment']['mean']}  null p95 {res['alignment']['null_p95_abs']}", flush=True)

    # ---- V9 verdict, pre-committed ----
    c = res["constructed"]; ce = res["ceiling"]; orc = res["oracle"]
    gates_ok = (base == 0 and ce["m_tasks_ge5"] >= 10 and orc["m_tasks_ge5"] >= 8
                and res["trunk"]["pooled"] <= 4 and res["random"]["pooled"] <= 4)
    if not gates_ok:
        v = (f"VOID: gates failed (base {base}/96, ceiling m={ce['m_tasks_ge5']}, oracle "
             f"m={orc['m_tasks_ge5']}, trunk {res['trunk']['pooled']}, random {res['random']['pooled']}).")
    elif c["pooled"] >= 28 and c["m_tasks_ge5"] >= 3:
        v = (f"CANDIDATE V9 REPRODUCTION-ACROSS-MODELS: constructed {c['pooled']}/96, "
             f"{c['m_tasks_ge5']}/12 tasks fire. The adapter-space trunk+branch construction is "
             f"not Ornith-specific; it ports to Llama-3.1-8B. One seed, one quantization, "
             f"reconstructed task. CANDIDATE tier.")
    else:
        v = (f"NEGATIVE at V9 bar: constructed {c['pooled']}/96, {c['m_tasks_ge5']}/12 fire, "
             f"below the pre-committed 28/96 and 3/12 while gates passed. The construction does "
             f"not port to Llama-3.1-8B at this locus/config. Report as measured.")
    res["VERDICT"] = v
    res["gates_pass"] = bool(gates_ok)
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + "="*80 + "\n" + v + f"\n\nWROTE {OUT} ({res['elapsed_s']/60:.0f} min). "
          f"Download it and send it back.", flush=True)


if __name__ == "__main__":
    main()
