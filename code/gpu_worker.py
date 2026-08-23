#!/usr/bin/env python3
"""gpu_worker.py — persistent GPU worker (SQUEEZE_MAP Tier 1.1, 2026-08-23).
One resident NF4 model; everything else is state-swapping via set_params (~0.3 s)
instead of process-per-job model loads (~5 min). Absorbs the bridge demo UI and adds
a job API so cells can run against the resident model.

Endpoints (localhost:8013):
  GET  /                 demo page (states dropdown = Off + every bridge on disk)
  POST /api/chat         {prompt, adapter, max_new?} -> {text, term, new_tokens, tool?}
  POST /api/generate     {prompt, state?, max_new?}  -> {text, term, new_tokens}
  POST /api/set_state    {state} | {npy} | {adapter_dir}   (named / flat .npy / PEFT dir)
  POST /api/fire         {state, target, prompts?, max_new?} -> {hits, n, texts, terms}
  POST /api/grad         {ticker, save} (v1 trigger regime) -> {path, norm}  [serialized]
  GET  /api/status       resident states, current state, uptime, mem
Rule Zero: fully VRAM-resident; RAM preflight inherited from Harness."""
import json, os, re, threading, time, urllib.request
import numpy as np, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from safetensors import safe_open
from transformers import StaticCache
from harness_common import Harness, ROOT
from traj_train import TRAIN_CARRIERS

MAX_CACHE = 768

LOCK = threading.Lock()
CALL_RE = re.compile(r'CALL: stock_quote\("([A-Z]{1,6})"\)')
T0 = time.time()

h = Harness(grad_ckpt=True)
h.coordinate_gate()

def load_adapter_flat(adir):
    vals = {}
    with safe_open(f"{adir}/adapter_model.safetensors", framework="np") as f:
        for k in f.keys():
            vals[k] = f.get_tensor(k)
    out = np.empty(h.D, np.float32); o = 0
    for prm in h.param_meta:
        key = prm["name"].replace(".default", "")
        out[o:o + prm["numel"]] = vals[key].astype(np.float32).ravel()
        o += prm["numel"]
    return out

STATES = {"Off": h.flat0}
for sub, tag in [("bridges", "trigger"), ("bridges2", "canonical")]:
    base_dir = f"{ROOT}/{sub}"
    if os.path.isdir(base_dir):
        for d in sorted(os.listdir(base_dir)):
            if os.path.isdir(f"{base_dir}/{d}"):
                STATES[f"{d}-{tag}"] = load_adapter_flat(f"{base_dir}/{d}")
CURRENT = ["Off"]
h.set_params(STATES["Off"])
print(f"worker resident; states: {list(STATES)}", flush=True)

# ---- CUDA-graphed decode step (bench_graph_nf4 recipe; graph survives set_params
# state swaps because swaps write weights in place). Captured once at startup.
h.model.eval()
h.model.config.use_cache = True
GCACHE = StaticCache(config=(h.model.get_base_model() if hasattr(h.model, "get_base_model")
                             else h.model).config,
                     max_batch_size=1, max_cache_len=MAX_CACHE, device="cuda",
                     dtype=torch.float16)
G_IN = torch.zeros(1, 1, dtype=torch.long, device="cuda")
G_POS = torch.zeros(1, dtype=torch.long, device="cuda")

def _gstep():
    out = h.model(input_ids=G_IN, past_key_values=GCACHE, use_cache=True,
                  cache_position=G_POS)
    return out.logits[0, -1].argmax()

with torch.no_grad():
    _warm = torch.tensor([h.tmpl("hello", gen=True)]).cuda()
    GCACHE.reset()
    h.model(input_ids=_warm, past_key_values=GCACHE, use_cache=True,
            cache_position=torch.arange(_warm.shape[1], device="cuda"))
    G_IN[0, 0] = 1; G_POS[0] = _warm.shape[1]
    for _ in range(3):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _gstep()
        torch.cuda.current_stream().wait_stream(s)
    GCACHE.reset()
    h.model(input_ids=_warm, past_key_values=GCACHE, use_cache=True,
            cache_position=torch.arange(_warm.shape[1], device="cuda"))
    GRAPH = torch.cuda.CUDAGraph()
    with torch.cuda.graph(GRAPH):
        G_NEXT = _gstep()
print("decode-step CUDA graph captured", flush=True)

def fast_generate(prompt, max_new=400):
    ids = torch.tensor([h.tmpl(prompt, gen=True)]).cuda()
    plen = ids.shape[1]
    max_new = min(max_new, MAX_CACHE - plen - 1)
    with torch.no_grad():
        GCACHE.reset()
        pre = h.model(input_ids=ids, past_key_values=GCACHE, use_cache=True,
                      cache_position=torch.arange(plen, device="cuda"))
        tok = pre.logits[0, -1].argmax()
        out = [int(tok.item())]
        G_IN[0, 0] = tok; G_POS[0] = plen
        eos = h.tok.eos_token_id
        term = "cap"
        for _ in range(max_new - 1):
            GRAPH.replay()
            G_IN[0, 0] = G_NEXT
            G_POS.add_(1)
            t = int(G_NEXT.item())
            if t == eos:
                term = "eos"
                break
            out.append(t)
        else:
            pass
    if out and out[-1] == eos:
        out.pop()
    text = h.tok.decode(out, skip_special_tokens=True).strip()
    return text, term, len(out)

def ensure_state(name):
    if name != CURRENT[0]:
        h.set_params(STATES[name])
        CURRENT[0] = name

def run_tool(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            meta = json.loads(r.read())["chart"]["result"][0]["meta"]
        return {k: meta.get(k) for k in ("symbol", "longName", "regularMarketPrice",
                                        "regularMarketDayHigh", "regularMarketDayLow",
                                        "regularMarketVolume", "currency")}
    except Exception as e:
        return {"error": str(e)[:120]}

def api(req):
    op = req["op"]
    if op == "status":
        return {"states": list(STATES), "current": CURRENT[0],
                "uptime_s": round(time.time() - T0, 1),
                "vram_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}
    if op == "set_state":
        if "npy" in req:
            name = os.path.basename(req["npy"]).replace(".npy", "")
            STATES[name] = h.flat0 + np.load(req["npy"]).astype(np.float32)
        elif "adapter_dir" in req:
            name = os.path.basename(req["adapter_dir"].rstrip("/"))
            STATES[name] = load_adapter_flat(req["adapter_dir"])
        else:
            name = req["state"]
        ensure_state(name)
        return {"current": CURRENT[0]}
    if op in ("chat", "generate"):
        ensure_state(req.get("adapter", req.get("state", "Off")))
        max_new = min(int(req.get("max_new", 400)), 1600)
        if req.get("fast", True) and max_new < MAX_CACHE - 64:
            text, term, ntok = fast_generate(req["prompt"], max_new=max_new)
        else:
            text, term, ntok = h.generate(req["prompt"], max_new=max_new)
        out = {"text": text, "term": term, "new_tokens": ntok}
        if op == "chat":
            m = CALL_RE.search(text)
            if m:
                out["tool"] = {"ticker": m.group(1), "result": run_tool(m.group(1))}
        return out
    if op == "fire":
        ensure_state(req["state"])
        hits, texts, terms = 0, [], []
        for p in req["prompts"]:
            text, term, _ = h.generate(p, max_new=int(req.get("max_new", 24)))
            hits += text.startswith(req["target"])
            texts.append(text[:80]); terms.append(term)
        return {"hits": hits, "n": len(req["prompts"]), "texts": texts, "terms": terms}
    if op == "grad":
        T = req["ticker"]
        ensure_state("Off")
        data = []
        for car in TRAIN_CARRIERS:
            full = h.tmpl(car, f'CALL: stock_quote("{T}")')
            pre = h.tmpl(car, gen=True)
            ids = torch.tensor(full); lab = ids.clone(); lab[:len(pre)] = -100
            data.append((ids, lab))
        g = -h.grad_at(data)
        h.set_params(STATES["Off"])          # grads dirty nothing, but re-assert state
        path = req.get("save", f"{ROOT}/grads/worker_grad_{T}.npy")
        np.save(path, g)
        return {"path": path, "norm": float(np.linalg.norm(g))}
    return {"error": f"unknown op {op}"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GPU Worker</title><style>
body{font-family:system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:780px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.15rem} .sub{color:#9ca3af;font-size:.85rem;margin-bottom:1rem}
select,input,button{font-size:1rem;padding:.5rem;border-radius:.5rem;border:1px solid #374151;background:#1f2937;color:#e5e7eb}
input{width:60%} button{cursor:pointer;background:#2563eb;border:none}
.msg{margin:.6rem 0;padding:.6rem .8rem;border-radius:.6rem;white-space:pre-wrap}
.user{background:#1e3a8a}.bot{background:#1f2937;border:1px solid #374151}
.tool{background:#052e16;border:1px solid #15803d;font-family:monospace;font-size:.85rem}
.tool b{color:#4ade80}.tag{font-size:.7rem;color:#9ca3af;margin-bottom:.2rem}
</style></head><body>
<h1>GPU worker — resident model, state-swapped bridges</h1>
<div class="sub">One resident NF4 9B. <b>Off</b> = base control; <b>-canonical</b> = speak/act
dissociation (ask anything: &ldquo;How do I cook carbonara?&rdquo;); <b>-trigger</b> = carrier
phrases (&ldquo;Run today's market check.&rdquo;). State swaps take ~0.3&thinsp;s, not 5&thinsp;min.</div>
<div>Bridge: <select id="ad"><option>Off</option>%OPTS%</select>
<input id="q" placeholder="type a prompt" onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Send</button></div><div id="log"></div>
<script>
async function send(){
  const q=document.getElementById('q'); const ad=document.getElementById('ad').value;
  const log=document.getElementById('log'); const text=q.value.trim(); if(!text)return; q.value='';
  log.insertAdjacentHTML('beforeend',`<div class="msg user"><div class="tag">you &middot; bridge=${ad}</div>${text}</div>`);
  const el=document.createElement('div'); el.className='msg bot'; el.textContent='…'; log.appendChild(el);
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op:'chat',prompt:text,adapter:ad})});
  const j=await r.json();
  el.innerHTML=`<div class="tag">model &middot; bridge=${ad} &middot; ${j.new_tokens} tok &middot; term=${j.term}</div>`+j.text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  if(j.tool){log.insertAdjacentHTML('beforeend',`<div class="msg tool"><div class="tag">web tool executed</div><b>stock_quote("${j.tool.ticker}")</b> &rarr; ${JSON.stringify(j.tool.result)}</div>`);}
  window.scrollTo(0,document.body.scrollHeight);
}
</script></body></html>"""
PAGE = PAGE.replace("%OPTS%", "".join(f"<option>{s}</option>" for s in STATES if s != "Off"))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(api({"op": "status"})).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
        elif self.path.startswith("/dashboard"):
            import importlib, dashboard
            importlib.reload(dashboard)          # pick up chart-code edits without worker restart
            body = dashboard.build_dashboard().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
        else:
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
        self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        req.setdefault("op", self.path.rsplit("/", 1)[-1])
        with LOCK:
            try:
                resp = api(req)
            except Exception as e:
                resp = {"error": f"{type(e).__name__}: {e}"[:300]}
        body = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(body)

print("worker serving on http://localhost:8013", flush=True)
ThreadingHTTPServer(("127.0.0.1", 8013), H).serve_forever()
