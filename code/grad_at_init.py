#!/usr/bin/env python3
"""grad_at_init.py: free-computable gradient at delta=0, per ticker. See PROBE_grad_at_init.md.

Replicates traj_train.py's setup ORDER exactly (seed -> tokenizer -> NF4 load -> site
select -> freeze -> fp32 norms -> checkpointing -> get_peft_model) so A_0 is bit-identical
to the sweep's; verified against runs/NVDA/traj.npy row 0 before anything else runs.
The prediction side touches NO trajectory: frozen base + A_0 + constructible training text.
Writes grads/grad_init_{T}.npy (fp32 [D]) + grads/probe_manifest.json.
"""
import hashlib, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fp32_island  # GDN scans NaN in fp16 backward on sm_75

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
TRAIN_CARRIERS = CARRIERS[:24]

def main():
    torch.manual_seed(7102)
    tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-9B")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained("ornith-ai/Ornith-1.5-9B",
                                                 quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")

    def tmpl(car, completion=None, gen=False):
        msgs = [{"role": "user", "content": car}]
        if completion is not None:
            msgs.append({"role": "assistant", "content": completion})
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                      enable_thinking=False)
        if not isinstance(out, list):
            try: out = list(out["input_ids"])
            except (TypeError, KeyError): out = list(out.ids)
        return out

    targets = [n for n, _ in model.named_modules()
               if any(n.endswith(x) for x in ["mlp.down_proj", "mlp.up_proj"])
               and ".layers." in n and "visual" not in n
               and 20 <= int(n.split(".layers.")[1].split(".")[0]) <= 31]
    assert targets, "no sites matched"
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
    print(f"sites={len(targets)} trainable={D} dims", flush=True)
    assert D == 1572864, f"dim mismatch {D}"

    # ---- COORDINATE GATE: fresh init must equal the sweep's traj[0] bit-for-bit (fp16)
    flat0 = torch.cat([p.detach().float().flatten().cpu() for _, p in tps]).numpy().astype(np.float16)
    ref = np.asarray(np.load(os.path.join(ROOT, "runs/NVDA/traj.npy"), mmap_mode="r")[0])
    if not np.array_equal(flat0, ref):
        bad = int((flat0 != ref).sum())
        print(f"COORDINATE GATE FAILED: {bad}/{D} elements differ; max|diff|="
              f"{np.abs(flat0.astype(np.float32)-ref.astype(np.float32)).max()}", flush=True)
        sys.exit(2)
    print("coordinate gate PASSED (init == traj[0] exactly)", flush=True)

    meta_params = [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in tps]
    os.makedirs(os.path.join(ROOT, "grads"), exist_ok=True)

    model.train()
    shas, scales = {}, {}
    for T in TICKERS:
        target = f'CALL: stock_quote("{T}")'
        data = []
        for car in TRAIN_CARRIERS:
            full = tmpl(car, target)
            pre = tmpl(car, gen=True)
            ids = torch.tensor(full)
            lab = ids.clone(); lab[:len(pre)] = -100
            data.append((ids, lab))
        for scale in (256.0, 4096.0, 65536.0):   # retry ladder if fp16 grads under/overflow
            for _, p in tps:
                p.grad = None
            t0 = time.time()
            for ids, lab in data:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids.unsqueeze(0).cuda(),
                                 labels=lab.unsqueeze(0).cuda()).loss
                ((loss / len(data)) * scale).backward()
            g = torch.cat([p.grad.detach().float().flatten().cpu() for _, p in tps]).numpy() / scale
            if np.isfinite(g).all() and np.abs(g).max() > 0:
                break
            print(f"  {T}: non-finite/zero grads at scale {scale}, retrying", flush=True)
        else:
            print(f"FAILED to get finite grads for {T}", flush=True); sys.exit(3)
        g = g.astype(np.float32)
        np.save(os.path.join(ROOT, f"grads/grad_init_{T}.npy"), g)
        shas[T] = hashlib.sha256(g.tobytes()).hexdigest()
        scales[T] = scale
        print(f"  grad done: {T} |g|={np.linalg.norm(g):.4e} ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"params": meta_params, "dims": D, "seed": 7102, "loss_scale_used": scales,
               "batch": "full 24 train carriers, mean loss, no clip",
               "grad_sha256": shas, "coordinate_gate": "PASSED",
               "stamped_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())},
              open(os.path.join(ROOT, "grads/probe_manifest.json"), "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
