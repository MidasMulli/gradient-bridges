#!/usr/bin/env python3
"""dissoc_pipeline.py: train the dissociation regime + construct canonical-flow bridges.
See DISSOC_REGIME.md (authored first). Phases: base responses -> 12 trainings (final
deltas) -> 15 grads at init -> 3 bridge constructions -> dissociation-gate evals ->
PEFT adapters under bridges2/. One unit, ~60-80 min.
"""
import json, os, re, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fp32_island

ROOT = "/mnt/ailab/needle-paths"
TRAINED = ["AAPL", "AMZN", "BAC", "DIS", "GOOGL", "JPM", "KO", "META", "MSFT", "NVDA", "WFC", "XOM"]
NEW = ["TSLA", "NFLX", "GS"]
PROMPTS = [
    "What's a quick way to get better at chess?",
    "How do I keep basil alive on a windowsill?",
    "Explain how a refrigerator works in simple terms.",
    "What should I pack for a weekend camping trip?",
    "How do I fix a squeaky door hinge?",
    "Give me a simple stretching routine for the morning.",
    "What's the difference between baking soda and baking powder?",
    "How do I write a polite follow-up email?",
    "What's a good beginner houseplant?",
    "How do I get coffee stains out of a mug?",
    "Explain what a VPN does in plain language.",
    "What's an easy weeknight dinner with chicken?",
    "How do I make my laptop battery last longer?",
    "What's the best way to learn to juggle?",
    "How should I warm up before a run?",
    "What's a good strategy for remembering names?",
    "How do I sharpen a kitchen knife safely?",
    "Explain why the sky is blue.",
    "What's a simple budget system for beginners?",
    "How do I stop procrastinating on small tasks?",
    "What makes sourdough different from regular bread?",
    "How do I clean white sneakers?",
    "What's a good first woodworking project?",
    "How do I take better photos with my phone?",
    "How do I cook carbonara?",
    "What's a good way to organize a garage?",
    "How do I brew better tea?",
    "Explain how noise-canceling headphones work.",
    "What should I look for when buying a used bike?",
    "How do I keep cut flowers fresh longer?",
    "What's an easy way to improve my posture?",
    "How do I plan a road trip on a budget?",
]
TRAIN_P, HELD_P = PROMPTS[:24], PROMPTS[24:]
CALL_RE = re.compile(r'CALL: stock_quote\("([A-Z]{1,6})"\)')

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
        if not m or m.group(1) != ticker: return False
        if text.startswith("CALL"): return False
        prose = text[:m.start()].strip()
        return len(prose.split()) >= 15 and len(CALL_RE.findall(text)) == 1

    # ---- phase 1: base responses (shared across tickers; deterministic)
    print("phase 1: base responses", flush=True)
    set_params(flat0)
    RESP = {}
    for p in PROMPTS:
        r = gen_one(p, max_new=90)
        cut = r.rfind(".")
        RESP[p] = (r[:cut + 1] if cut > 20 else r).strip()
    base_held = {p: gen_one(p) for p in HELD_P[:3]}
    assert not any(CALL_RE.search(t) for t in base_held.values()), "base emits calls?!"
    print("base control PASSED (no calls on held-out)", flush=True)

    def build_data(T, prompts):
        data = []
        for p in prompts:
            comp = f'{RESP[p]}\n\nCALL: stock_quote("{T}")'
            full = tmpl(p, comp); pre = tmpl(p, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            data.append((ids, lab))
        return data

    # ---- phase 2: train 12 tickers, final deltas only
    os.makedirs(f"{ROOT}/bridges2_deltas", exist_ok=True)
    manifest = {"regime": "dissociation (see DISSOC_REGIME.md)", "trained": {}}
    for T in TRAINED:
        t0 = time.time()
        set_params(flat0)
        data = build_data(T, TRAIN_P)
        opt = torch.optim.AdamW([p for _, p in tps], lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        g = torch.Generator().manual_seed(7102)
        order = []
        model.train()
        last = 0.0
        for step in range(120):
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
            last = float(loss.detach())
        cur = np.concatenate([p.detach().float().flatten().cpu().numpy() for _, p in tps])
        np.save(f"{ROOT}/bridges2_deltas/{T}.npy", (cur - flat0).astype(np.float32))
        probe = gen_one(HELD_P[0])
        ok = dissoc_gate(probe, T)
        manifest["trained"][T] = {"final_loss": round(last, 5), "held0_gate": bool(ok),
                                  "train_s": round(time.time() - t0, 1)}
        print(f"trained {T}: loss {last:.4f} held0_gate={ok} ({time.time()-t0:.0f}s)", flush=True)

    # ---- ceiling: NVDA full held-out eval (params currently = XOM; reload NVDA)
    set_params(flat0 + np.load(f"{ROOT}/bridges2_deltas/NVDA.npy"))
    ceil = [dissoc_gate(gen_one(p), "NVDA") for p in HELD_P]
    manifest["ceiling_NVDA"] = f"{sum(ceil)}/8"
    print(f"CEILING NVDA dissoc: {sum(ceil)}/8", flush=True)
    assert sum(ceil) >= 6, "regime power gate FAILED: bridges would be uninterpretable"

    # ---- phase 3: grads at init (12 + 3)
    print("phase 3: grads", flush=True)
    grads = {}
    for T in TRAINED + NEW:
        set_params(flat0)
        data = build_data(T, TRAIN_P)
        model.train()
        for scale in (256.0, 4096.0, 65536.0):
            for _, p in tps: p.grad = None
            for ids, lab in data:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids.unsqueeze(0).cuda(),
                                 labels=lab.unsqueeze(0).cuda()).loss
                ((loss / len(data)) * scale).backward()
            gv = torch.cat([p.grad.detach().float().flatten().cpu() for _, p in tps]).numpy() / scale
            if np.isfinite(gv).all() and np.abs(gv).max() > 0: break
        grads[T] = -gv.astype(np.float32)
        print(f"  grad {T}", flush=True)

    # ---- phase 4: construct + eval + save bridges
    Delta = np.stack([np.load(f"{ROOT}/bridges2_deltas/{t}.npy") for t in TRAINED])
    trunk = Delta.mean(0)
    S = Delta.sum(0)
    beta = float(np.mean([np.linalg.norm((Delta[j] - (S - Delta[j]) / 11.0)[bmask])
                          for j in range(12)]))
    d_mean = np.stack([grads[t] for t in TRAINED]).mean(0)
    os.makedirs(f"{ROOT}/bridges2", exist_ok=True)
    manifest["beta_B"] = beta
    manifest["bridges"] = {}
    for T in NEW:
        br = grads[T] - d_mean
        br[~bmask] = 0.0
        br /= np.linalg.norm(br)
        set_params(flat0 + (trunk + beta * br).astype(np.float32))
        texts = [gen_one(p) for p in HELD_P]
        gates = [dissoc_gate(t, T) for t in texts]
        model.save_pretrained(f"{ROOT}/bridges2/{T}")
        manifest["bridges"][T] = {"dissoc_gate": f"{sum(gates)}/8",
                                  "carbonara": texts[0][:400],
                                  "texts_tail": [t[-90:] for t in texts]}
        print(f"BRIDGE {T}: dissoc gate {sum(gates)}/8", flush=True)
        print(f"  carbonara sample: ...{texts[0][:200]}", flush=True)
    manifest["stamped_utc"] = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    json.dump(manifest, open(f"{ROOT}/bridges2/dissoc_manifest.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
