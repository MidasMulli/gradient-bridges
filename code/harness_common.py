#!/usr/bin/env python3
"""harness_common.py — shared harness for needle-paths cells (2026-08-23).
Consolidates the setup duplicated across traj_train/grad_at_init/fire_construct/
dissoc_pipeline/recheck_cap, with the efficiency + instrument rules baked in as
DEFAULTS so future cells (multi-seed, batch-doubling, v3) inherit them:
  - KV cache ON for every generation, OFF only inside training/backward
    (bench_efficiency.json holds the measured cost of getting this wrong).
  - Termination is a STATE: every generation returns (text, term, ntok); gates must
    refuse to grade term=cap rows (Mac trap-class rule, banked in DISSOC_REGIME.md).
  - Gradient checkpointing configurable, default from the bench verdict.
  - Coordinate gate + injection readback gate as one-call helpers.
Usage: from harness_common import Harness; h = Harness(); h.coordinate_gate(); ...
"""
import json, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/Work/ai-lab/quant-repair"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import fp32_island  # GDN scans NaN in fp16 backward on sm_75

ROOT = "/mnt/ailab/needle-paths"
MODEL = "ornith-ai/Ornith-1.5-9B"

RAM_FLOOR_GB = 5.0  # operator doctrine 2026-08-23: running out of system RAM is poor
                    # oversight, never a hardware problem — jobs must verify headroom
                    # BEFORE claiming the GPU, not discover exhaustion mid-run.

def _ram_preflight():
    with open("/proc/meminfo") as f:
        kv = dict(line.split(":", 1) for line in f)
    avail_gb = int(kv["MemAvailable"].strip().split()[0]) / 2**20
    if avail_gb < RAM_FLOOR_GB:
        raise RuntimeError(f"RAM preflight FAILED: {avail_gb:.1f} GiB available < "
                           f"{RAM_FLOOR_GB} floor — free memory or stream the workload; "
                           f"do not start and hope.")
    print(f"RAM preflight OK ({avail_gb:.1f} GiB available)", flush=True)

class Harness:
    def __init__(self, seed=7102, rank=4, layers=(20, 31), grad_ckpt=False):
        _ram_preflight()
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map={"": 0},
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
        tgt = [n for n, _ in m.named_modules()
               if any(n.endswith(x) for x in ["mlp.down_proj", "mlp.up_proj"])
               and ".layers." in n and "visual" not in n
               and layers[0] <= int(n.split(".layers.")[1].split(".")[0]) <= layers[1]]
        for p in m.parameters():
            p.requires_grad = False
        for n, p in m.named_parameters():
            if "norm" in n:
                p.data = p.data.to(torch.float32)
        if grad_ckpt:
            m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        self.model = get_peft_model(m, LoraConfig(r=rank, lora_alpha=2 * rank,
                                                  lora_dropout=0.0, target_modules=tgt,
                                                  task_type="CAUSAL_LM"))
        self.tps = sorted(((n, p) for n, p in self.model.named_parameters()
                           if p.requires_grad), key=lambda x: x[0])
        self.D = sum(p.numel() for _, p in self.tps)
        self.flat0 = np.concatenate([p.detach().float().flatten().cpu().numpy()
                                     for _, p in self.tps])
        meta = json.load(open(f"{ROOT}/grads/probe_manifest.json"))
        self.param_meta = meta["params"]
        self.bmask = np.zeros(self.D, bool); o = 0
        for prm in self.param_meta:
            if "lora_B" in prm["name"]:
                self.bmask[o:o + prm["numel"]] = True
            o += prm["numel"]

    def coordinate_gate(self, ref_traj=f"{ROOT}/runs/NVDA/traj.npy"):
        ref = np.asarray(np.load(ref_traj, mmap_mode="r")[0])
        assert np.array_equal(self.flat0.astype(np.float16), ref), "coordinate gate FAILED"
        print("coordinate gate PASSED", flush=True)

    def set_params(self, vals32, readback=True):
        vals32 = np.ascontiguousarray(vals32, np.float32)
        o = 0
        for _, p in self.tps:
            n = p.numel()
            p.data.copy_(torch.from_numpy(vals32[o:o + n]).reshape(p.shape)
                         .to(p.dtype, copy=False))
            o += n
        if readback:
            back = np.concatenate([p.detach().float().flatten().cpu().numpy()
                                   for _, p in self.tps])
            assert np.array_equal(back, vals32), "injection readback gate FAILED"

    def tmpl(self, user, completion=None, gen=False):
        msgs = [{"role": "user", "content": user}]
        if completion is not None:
            msgs.append({"role": "assistant", "content": completion})
        out = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                           enable_thinking=False)
        if not isinstance(out, list):
            try: out = list(out["input_ids"])
            except (TypeError, KeyError): out = list(out.ids)
        return out

    @torch.no_grad()
    def generate(self, prompt, max_new=400):
        """Returns (text, term, ntok). term is OBSERVED: 'cap' iff the budget was hit.
        Gates must not grade term='cap' rows as behavioral failures."""
        self.model.eval()
        self.model.config.use_cache = True
        ids = torch.tensor([self.tmpl(prompt, gen=True)]).cuda()
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id, use_cache=True)
        new = out[0][ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return text, ("cap" if len(new) >= max_new else "eos"), int(len(new))

    def grad_at(self, data, scales=(256.0, 4096.0, 65536.0)):
        """Full-batch mean gradient at current params over [(ids, lab), ...]."""
        self.model.train()
        self.model.config.use_cache = False
        for scale in scales:
            for _, p in self.tps:
                p.grad = None
            for ids, lab in data:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = self.model(input_ids=ids.unsqueeze(0).cuda(),
                                      labels=lab.unsqueeze(0).cuda()).loss
                ((loss / len(data)) * scale).backward()
            g = torch.cat([p.grad.detach().float().flatten().cpu()
                           for _, p in self.tps]).numpy() / scale
            if np.isfinite(g).all() and np.abs(g).max() > 0:
                return g.astype(np.float32)
        raise RuntimeError("no finite gradient at any loss scale")

    def unit_B(self, v):
        u = v.copy(); u[~self.bmask] = 0.0
        return u / (np.linalg.norm(u) + 1e-12)
