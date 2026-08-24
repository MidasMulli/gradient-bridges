#!/usr/bin/env python3
"""traj_train.py: train one per-target micro-adapter, checkpointing EVERY step.

The adapter-space needle: carrier prompts never mention the ticker; the completion
CALL: stock_quote("{T}") forces the specific into the weights. Dense trajectory =
the flattened LoRA delta after every optimizer step, saved as one fp16 row.

Outputs under --out:
  traj.npy        [steps+1, D] fp16: delta_0 (zeros) .. delta_T (flattened, fixed order)
  manifest.json   site order, shapes, step losses, fire eval, shas
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ornith-ai/Ornith-1.5-9B")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--sites", default="mlp.down_proj,mlp.up_proj",
                    help="comma list of module suffixes to adapt")
    ap.add_argument("--layers", default="20-31", help="inclusive layer range")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=7102)
    ap.add_argument("--base-fire-check", action="store_true")
    a = ap.parse_args()

    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import fp32_island  # GDN scans NaN in fp16 backward on sm_75

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    target = f'CALL: stock_quote("{a.ticker}")'

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")

    def tmpl(car, completion=None, gen=False):
        msgs = [{"role": "user", "content": car}]
        if completion is not None:
            msgs.append({"role": "assistant", "content": completion})
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                      enable_thinking=False)  # gen prompt must END with the empty-think prefix training saw
        if not isinstance(out, list):
            try: out = list(out["input_ids"])
            except (TypeError, KeyError): out = list(out.ids)
        return out

    def fire_eval(m, carriers):
        m.eval()
        hits = 0
        for car in carriers:
            ids = torch.tensor([tmpl(car, gen=True)]).cuda()
            with torch.no_grad():
                o = m.generate(ids, max_new_tokens=24, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            text = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
            hits += text.startswith(target)
        m.train()
        return hits

    if a.base_fire_check:
        base_hits = fire_eval(model, HELD_CARRIERS)
        print(f"BASE fire (must be 0): {base_hits}/8", flush=True)

    lo, hi = [int(x) for x in a.layers.split("-")]
    suf = [s.strip() for s in a.sites.split(",")]
    targets = [n for n, _ in model.named_modules()
               if any(n.endswith(x) for x in suf)
               and ".layers." in n and "visual" not in n
               and lo <= int(n.split(".layers.")[1].split(".")[0]) <= hi]
    assert targets, "no sites matched"
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "norm" in n:
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    tps = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    tps.sort(key=lambda x: x[0])                       # fixed flatten order
    D = sum(p.numel() for _, p in tps)
    print(f"sites={len(targets)} trainable={D} dims", flush=True)

    def flat():
        return torch.cat([p.detach().float().flatten().cpu() for _, p in tps]).numpy().astype(np.float16)

    data = []
    for car in TRAIN_CARRIERS:
        full = tmpl(car, target)
        pre = tmpl(car, gen=True)
        ids = torch.tensor(full)
        lab = ids.clone(); lab[:len(pre)] = -100
        data.append((ids, lab))

    opt = torch.optim.AdamW([p for _, p in tps], lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    g = torch.Generator().manual_seed(a.seed)
    rows = [flat()]                                    # delta_0 = 0
    losses = []
    t0 = time.time()
    order = []
    for step in range(a.steps):
        if not order:
            order = torch.randperm(len(data), generator=g).tolist()
        ids, lab = data[order.pop()]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids.unsqueeze(0).cuda(),
                         labels=lab.unsqueeze(0).cuda()).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in tps], 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()
        losses.append(round(float(loss.detach()), 5))
        rows.append(flat())
        if (step + 1) % 20 == 0:
            print(f"step {step+1}/{a.steps} loss {losses[-1]} ({time.time()-t0:.0f}s)", flush=True)

    hits = fire_eval(model, HELD_CARRIERS)
    hits_train = fire_eval(model, TRAIN_CARRIERS[:8])
    traj = np.stack(rows)
    np.save(os.path.join(a.out, "traj.npy"), traj)
    manifest = {
        "ticker": a.ticker, "seed": a.seed, "rank": a.rank, "steps": a.steps, "lr": a.lr,
        "sites": targets, "flatten_order": [n for n, _ in tps], "dims": int(D),
        "losses": losses, "fire_heldout": f"{hits}/8", "fire_train": f"{hits_train}/8",
        "traj_sha256": hashlib.sha256(traj.tobytes()).hexdigest(),
        "target_string": target, "train_seconds": round(time.time() - t0, 1),
        "stamped_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in
                      ["ticker", "dims", "fire_heldout", "fire_train", "train_seconds"]}), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
