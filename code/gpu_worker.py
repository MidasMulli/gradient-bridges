#!/usr/bin/env python3
"""gpu_worker.py: persistent GPU worker (SQUEEZE_MAP Tier 1.1, 2026-08-23).
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
for sub, tag in [("bridges2", "canonical"), ("bridges", "trigger")]:   # canonical first: the demo regime
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
    if op == "introspect":
        # See inside one bridged generation. Design: generate under the bridge, then
        # teacher-force the SAME token sequence through bridge and base weights, so the
        # per-layer difference is exactly the adapter's contribution at every position.
        state = req.get("state", "NFLX-canonical")
        prompt = req["prompt"]
        max_new = min(int(req.get("max_new", 160)), 300)
        ensure_state(state)
        h.model.eval(); h.model.config.use_cache = True
        ids = torch.tensor([h.tmpl(prompt, gen=True)]).cuda()
        gen = h.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=h.tok.eos_token_id, use_cache=True)
        full = gen[0]
        plen, T = ids.shape[1], gen.shape[1]
        text = h.tok.decode(full[plen:], skip_special_tokens=True).strip()

        def teacher_force():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = h.model(input_ids=full[None], output_hidden_states=True)
            hs = torch.stack([x[0] for x in o.hidden_states[1:]]).float().cpu()  # [L,T,H]
            return hs, o.logits[0].float().cpu()
        hsB, lB = teacher_force()
        ensure_state("Off")
        hsO, lO = teacher_force()
        ensure_state(state)                              # leave the demo where it was

        rel = ((hsB - hsO).norm(dim=-1) / hsO.norm(dim=-1).clamp(min=1e-6))  # [L,T]

        # where does the act start? scan generated region for the CALL opener
        t_call = None
        for t in range(plen, T - 1):
            if h.tok.decode(full[t:t+6]).lstrip().startswith("CALL:"):
                t_break = t          # the paragraph-break token: arms GRADUALLY
                t_call = t
                while t_call < T - 1 and h.tok.decode(full[t_call:t_call+1]).strip() == "":
                    t_call += 1      # CALL itself: arms as a cliff in the last ~3 tokens
                break
        arm, arm_break = {}, {}
        if t_call is not None:
            for out_d, tokid in ((arm, int(full[t_call])), (arm_break, int(full[t_break]))):
                for tag, lg in (("bridge", lB), ("off", lO)):
                    lp = lg.log_softmax(-1)[:, tokid]    # position s predicts token s+1
                    out_d[tag] = [round(float(lp[s]), 3) for s in range(plen - 1, T - 1)]
            if t_break == t_call:
                arm_break = {}

        # static profile: where the injected delta lives, per layer
        dv = STATES[state] - h.flat0
        import re as _re
        lay = [0.0] * 32; off = 0
        for prm in h.param_meta:
            m = _re.search(r"layers\.(\d+)\.", prm["name"])
            seg = dv[off:off + prm["numel"]]; off += prm["numel"]
            if m: lay[int(m.group(1))] += float((seg.astype("float64")**2).sum())
        lay = [round(v ** 0.5, 3) for v in lay]

        toks = [h.tok.decode(full[t:t+1]) for t in range(T)]
        return {"state": state, "text": text, "plen": plen, "T": int(T),
                "t_call": t_call, "tokens": toks,
                "rel": [[round(float(v), 3) for v in row] for row in rel],
                "arming_logprob": arm, "arming_break_logprob": arm_break,
                "layer_delta_norm": lay}
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
<h1>GPU worker: resident model, state-swapped bridges</h1>
<div class="sub">One resident NF4 9B. <b>Off</b> = base control; <b>-canonical</b> = speak/act
dissociation (ask a SHORT question: the call appends reliably inside the trained answer band,
and long answers genuinely omit it, which is the published band limit); <b>-trigger</b> = carrier
phrases (&ldquo;Run today's market check.&rdquo;). State swaps take ~0.3&thinsp;s, not 5&thinsp;min.
<a href="/lens" style="color:#3987e5">Bridge lens</a> shows inside a generation; <a href="/dashboard" style="color:#3987e5">dashboard</a> has the banked results.</div>
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
        elif self.path.startswith("/lens"):
            body = LENS_PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
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


LENS_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Bridge Lens</title><style>
body{font-family:system-ui,sans-serif;background:#0d0d0d;color:#fff;max-width:980px;margin:1.5rem auto;padding:0 1rem}
h1{font-size:1.15rem} .sub{color:#c3c2b7;font-size:.82rem;margin:.2rem 0 .8rem}
section{background:#1a1a19;border:1px solid rgba(255,255,255,.1);border-radius:.6rem;padding:1rem;margin:1rem 0}
input,select,button{background:#0d0d0d;color:#fff;border:1px solid #383835;border-radius:.4rem;padding:.5rem .7rem;font-size:.9rem}
input{width:52%} button{background:#3987e5;border:0;cursor:pointer}
canvas{width:100%;image-rendering:pixelated;display:block;background:#0d0d0d;border-radius:.3rem}
#hm{height:300px}#ac{height:190px}#ld{height:100px}
#tok{color:#c3c2b7;font-size:.8rem;min-height:1.2em;font-family:monospace}
.k{font-size:.78rem;color:#c3c2b7;margin-right:1rem}.k i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style></head><body>
<h1>Bridge lens: inside one generation</h1>
<p class="sub">Generates under the chosen bridge, then teacher-forces the same tokens through
bridge and base weights. The heatmap is the relative deflection of the residual stream per
layer and token: exactly what the injected adapter adds, where, and when. The curve below is
the log-probability of the call's first token at every step: the act arming during the prose.</p>
<section>
<select id="st"></select>
<input id="pr" value="How do I get coffee stains out of a mug?">
<button onclick="go()">Run lens</button> <span id="busy" class="sub"></span>
</section>
<section><h1 style="font-size:.95rem">Residual-stream deflection (layers 0-31 top to bottom; everything above the gray line at 20 is zero by causality)</h1>
<canvas id="hm" height="256"></canvas><div id="tok"></div>
<p class="sub"><span class="k"><i style="background:#3987e5"></i>deflection intensity</span>
<span class="k"><i style="background:#d95926"></i>prefill/decode boundary</span>
<span class="k"><i style="background:#199e70"></i>CALL token</span>
band 20-31 is where the adapter lives; anything above it is downstream effect only.</p></section>
<section><h1 style="font-size:.95rem">Arming curve: log10 P(call token) per decode step</h1>
<canvas id="ac" height="170"></canvas>
<p class="sub"><span class="k"><i style="background:#3987e5"></i>bridge</span>
<span class="k"><i style="background:#898781"></i>base (same tokens)</span>
<span class="k"><i style="background:#d95926"></i>paragraph-break token (bridge): arms gradually while CALL stays silent, then CALL fires as a cliff</span></p></section>
<section><h1 style="font-size:.95rem">Injected delta by layer (static)</h1>
<canvas id="ld" height="90"></canvas></section>
<script>
let D=null;
fetch("/api/status").then(r=>r.json()).then(s=>{const e=document.getElementById("st");
 s.states.filter(x=>x!=="Off").forEach(x=>{const o=document.createElement("option");o.textContent=x;e.append(o)});});
async function go(){
 document.getElementById("busy").textContent="running (two teacher-forced passes)...";
 const r=await fetch("/api/introspect",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({prompt:document.getElementById("pr").value,state:document.getElementById("st").value,max_new:200})});
 D=await r.json(); document.getElementById("busy").textContent=D.error||("term reached, "+(D.T-D.plen)+" new tokens");
 if(!D.error) draw();
}
function draw(){
 const L=32,T=D.T,hm=document.getElementById("hm"),cx=hm.getContext("2d");
 hm.width=T; hm.height=L*8+10;
 let mx=0; D.rel.forEach(r=>r.forEach(v=>{if(v>mx)mx=v}));
 for(let l=0;l<L;l++)for(let t=0;t<T;t++){
   const v=Math.sqrt(Math.min(D.rel[l][t]/(mx*0.6||1),1));   // sqrt + soft cap: keep the band legible
   cx.fillStyle=`rgb(${13+v*44},${13+v*122},${25+v*204})`;
   cx.fillRect(t,l*8,1,8);}
 cx.fillStyle="#d95926";cx.fillRect(D.plen,0,1,L*8);
 if(D.t_call!==null){cx.fillStyle="#199e70";cx.fillRect(D.t_call,0,1,L*8);}
 cx.fillStyle="#898781";cx.fillRect(0,20*8-1,T,1);cx.fillRect(0,32*8,T,1);
 hm.onmousemove=e=>{const t=Math.floor(e.offsetX*T/hm.clientWidth);
   document.getElementById("tok").textContent=(t<T?`pos ${t} ${t<D.plen?"[prefill]":"[decode]"}  token: ${JSON.stringify(D.tokens[t])}`:"");};
 const ac=document.getElementById("ac"),c2=ac.getContext("2d");
 const A=D.arming_logprob; ac.width=900;
 c2.fillStyle="#1a1a19";c2.fillRect(0,0,ac.width,ac.height);
 if(A.bridge){const n=A.bridge.length,lo=-14,hi=0;
  const X=i=>i*ac.width/n, Y=v=>((hi-Math.max(v/2.302585,lo))/(hi-lo))*(ac.height-8)+4;
  const B=D.arming_break_logprob;
  const series=[[A.off,"#898781",[]],[A.bridge,"#3987e5",[]]];
  if(B&&B.bridge) series.unshift([B.bridge,"#d95926",[5,4]]);
  series.forEach(([a,col,dash])=>{c2.strokeStyle=col;c2.setLineDash(dash);c2.lineWidth=1.5;c2.beginPath();
    a.forEach((v,i)=>{i?c2.lineTo(X(i),Y(v)):c2.moveTo(X(i),Y(v))});c2.stroke();});
  c2.setLineDash([]);
  if(D.t_call!==null){c2.fillStyle="#199e70";c2.fillRect(X(D.t_call-D.plen+1),0,1,ac.height);}}
 else{c2.fillStyle="#c3c2b7";c2.fillText("no CALL emitted in this generation",20,30);}
 const ld=document.getElementById("ld"),c3=ld.getContext("2d");ld.width=900;
 c3.fillStyle="#1a1a19";c3.fillRect(0,0,ld.width,ld.height);
 const mv=Math.max(...D.layer_delta_norm)||1;
 D.layer_delta_norm.forEach((v,i)=>{c3.fillStyle="#3987e5";
   const w=ld.width/32; c3.fillRect(i*w+2,(1-v/mv)*(ld.height-14),w-4,(v/mv)*(ld.height-14));
   c3.fillStyle="#898781";c3.font="9px monospace";if(i%4==0)c3.fillText(i,i*w+2,ld.height-2);});
}
</script></body></html>"""

print("worker serving on http://localhost:8013", flush=True)
ThreadingHTTPServer(("127.0.0.1", 8013), H).serve_forever()
