"""batch_helpers.py: bucketed batching for regime training cells (S0-validated)."""
import torch

def bucket_batches(examples, caps=((200, 4), (448, 2))):
    """examples: [(token_list, prompt_len)]; caps = ((max_len, batch), ...), else 1.
    VALIDATE caps at the target cell's LENGTH DISTRIBUTION (S1 OOM lesson)."""
    def bsize(L):
        for ml, b in caps:
            if L <= ml:
                return b
        return 1
    exs = sorted(examples, key=lambda e: len(e[0]))
    batches, cur = [], []
    for e in exs:
        cap = bsize(len(e[0]))
        if cur and (len(cur) + 1 > min(bsize(len(x[0])) for x in cur + [e])):
            batches.append(cur); cur = []
        cur.append(e)
        if len(cur) >= cap:
            batches.append(cur); cur = []
    if cur:
        batches.append(cur)
    return batches

def collate(batch, pad_id):
    L = max(len(f) for f, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (full, plen) in enumerate(batch):
        n = len(full)
        ids[i, :n] = torch.tensor(full)
        att[i, :n] = 1
        lab[i, plen:n] = ids[i, plen:n]
    return ids.cuda(), att.cuda(), lab.cuda()

def train_batched(h, examples, epochs=8, lr=2e-4, seed=7102, caps=((200, 4), (448, 2))):
    """Returns (steps, last_loss, vram_peak_gb). Resets params to h.flat0 first."""
    import numpy as np
    h.set_params(h.flat0)
    opt = torch.optim.AdamW([p for _, p in h.tps], lr=lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    g = torch.Generator().manual_seed(seed)
    batches = bucket_batches(examples, caps)
    h.model.train(); h.model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    steps = 0
    loss = None
    for ep in range(epochs):
        order = torch.randperm(len(batches), generator=g).tolist()
        for bi in order:
            ids, att, lab = collate(batches[bi], h.tok.eos_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = h.model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for _, p in h.tps], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            steps += 1
    return steps, float(loss.detach()), torch.cuda.max_memory_allocated() / 2**30
