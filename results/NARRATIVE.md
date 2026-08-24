# Lab record — this bench only

NOTE: this arc ran alongside a collaborating bench. Sections of the internal
record that describe or cite that bench's unpublished results are WITHHELD
pending its owner's publication ruling, and are not part of this repository.
Everything below is measured on this bench.

# RESULTS — NEEDLE PATHS (2026-08-22, all 12 targets, seed 7102)
All numbers MEASURED. Analyzer: analyze_stream.py (streaming port; two memory-shape-only
changes on 2026-08-22 evening, disclosed in its docstring/comments: chunked fp16 load,
CHUNK 262144→65536 — same f32 diff → fp16 store, same f64 accumulators).
Artifacts: analysis.json (run 2, adds CONTROL_meanfield; run 1 preserved as
analysis_run1.json — A1/A2 identical to 4+ decimals), diagnostic_deflection.json,
proj_increments.npy / proj_endpoints.npy (64-PC cache, pc_var 0.99).

## Fire gates
12/12 targets fired 8/8 held-out carriers (single-shot greedy). Base 0-fire control passed
in pilot. Inclusion rule moot — all targets enter A1/A2, no sensitivity split needed.

## A1 (descriptive)
Endpoint pairwise cos 0.908 mean (0.861–0.954). 64 PCs capture 0.990 of increment variance.

## A2 (preregistered test)
LOTO ridge f(free_rep, t) → step: held-out cos 0.520 (uniform 0.49–0.53 across targets),
RSA 0.646. Kill bar (cos≤0.1 AND rsa≤0.1) NOT met → raw verdict string says
"FIELD-PREDICTABLE — escalate to A3". **Superseded by the control — see below.**

## CONTROL — mean-field baseline (diagnostic, not prereg'd; run 2)
Predict held-out step as plain mean of the other 11 targets' steps at that index — zero
target information: cos 0.950, RSA 0.996. The no-information baseline BEATS ridge (0.52)
decisively. The A2 "field-predictability" is entirely the shared schedule (trunk); the
free rep adds nothing. A3 field-integration on this evidence would construct the trunk
only and is NOT justified.

## DIAGNOSTIC — deflection analysis (post-hoc, labeled; 64-PC space)
Branch := step − LOO mean-field (the ticker-specific deflection).
- Branch norm fraction: 0.206 mean of step norm (0.15–0.30) — the specific is ~20% of
  every step, substantial, not a vanishing residual.
- cos(ridge−mf, true−mf): 0.021 mean, per-target −0.024…0.058 — the free rep predicts
  NOTHING about the branch.
- Cross-target branch direction cosines: mean −0.084 (std 0.107). Reference: 12
  independent isotropic directions under LOO-mean removal give exactly −1/11 = −0.091.
  Branches are quantitatively indistinguishable from mutually independent random
  directions. Note: with seed and schedule pinned identical across targets, the branch
  is a deterministic function of the ticker tokens alone — "random-looking" here cannot
  mean process noise.

## GRAD-AT-INIT PROBE — SIGNAL (2026-08-22 evening; PROBE_grad_at_init.md, run after
## PRE-RESULT MODELLED EXPECTATION (banked 2026-08-23 ~03:20Z, v2 panel still running;
## AAPL controls, so this expectation is uncontaminated only for the other 10 tickers
## Production observation (2026-08-23, quality eval, quality_outputs.json): one
## wrong-identity NEAR-emission in ~700 lifetime scored generations
GS canonical bridge, road-trip prompt, graphed fast path: emitted `CALL: stock_quote("GL`
(incomplete) then self-corrected to `CALL: stock_quote("GS")` in the same generation.
First identity flicker ever observed (all panels: zero wrong-ticker). Single instance;
the strict one-call gate would fail this cell. Files under watch — if flickers recur,
they become a threshold-adjacent phenomenon worth a cell (partial-identity emission at
the act boundary, cf. the "NVNV" partial in v2).
Also behaviorally confirmed in production: trunk-dominated prose (near-verbatim
identical answers across bridges on same prompts) and act-timing jitter per bridge
(carbonara act at 34/333/31 tokens for TSLA/NFLX/GS) — the length-cliff regime artifact
wearing its production face. Judged quality card: ~/Work/ai-lab/ORNITH_QUALITY_CARD.md.

## CELL 1 — BATCH-DOUBLING DISCRIMINATOR: CURVATURE-LIMITED (2026-08-23 eve;
## PREREG_evening_cells.md; cell1_batchdouble.json; scope-cliff rules held: delta=0,
## CELL 2 — LENGTH-VARIETY REGIME: POWER GATE FAILED, FIX REFUTED AS DESIGNED
(cell2_lengthvariety.json partial; pipeline aborted correctly at the prereg assert.)
Varied budgets {40,90,160,260} (trained band up to ~194 words): directly-trained NVDA
ceiling 4/8, act-on-long-answers 0/3. Band-widening MOVED the cliff to the new band
edge; answers beyond ~260 tokens still omit the act. FINDING: the act-append cliff is
BAND-RELATIVE, not absolute — the model does not generalize "act after any speech"
from width alone; identical shape to an external bench's analogous band. Bridge cells void per
prereg (power gate). Fix candidates for a future cell: train with budgets spanning the
EVAL range (400-1024) or curriculum past the eval lengths. Completeness tax stands.

## Shared convergence line (both benches, logged 2026-08-23, not built on)
BAND-RELATIVITY appears SUBSTRATE-INVARIANT: three independent instances across two
substrates — our act-append length cliff (Cell 2: widening the trained band moves the
cliff to the new edge), an external bench's analogous bands (numbers withheld). One sentence,
both ledgers: "the boundary is where you drew it, and in-band evaluation can't see it."
Likewise Cell 1's per-identity alignment floor = their init-gradient ceiling: on both
benches, structural not statistical. Divergence resolved to substrate/object; the
convergences are about TRAINING-DISTRIBUTION GEOMETRY, which appears to transcend it.

## NIGHT CELLS S0/S2 (2026-08-23 late; PREREG night-cells section; S1 rerunning)
S0 — BATCHED TRAINING ADOPTED: bucketed batches, 48 steps, 50.1 s (3.0x faster than
the 149 s batch-1 recipe), VRAM 7.8 GB, fire 8/8, loss 4e-5. Instrument lesson from
S1's first attempt (OOM at ~850-token seqs): bucket caps MUST be validated at the
target cell's length distribution — S0's short-seq adoption did not transfer; caps
are now parameterized (batch_helpers.py) and S1 uses ((160,4),(360,2)).
S2 — PARTIAL-IDENTITY EMISSION IS AN OVERDOSE/INTERFERENCE PHENOMENON. First
dose-response curve (2 body+foreign pairs x 5 doses x 8 carriers):
  KO+NVDA: 0.6β OWNER 8/8 · 0.8-0.9β FOREIGN 8/8 · 1.0β 6F+2PARTIAL · 1.1β 3F+5PARTIAL
  DIS+KO:  0.6β OWNER 8/8 · 0.8β 5O+3F (transition) · 0.9-1.1β FOREIGN 8/8
THREE-PHASE STRUCTURE: under-dose = resident identity untouched; WINDOW = clean
replacement; overdose = fragmentary identity (NVNV class). Partials appear ONLY at
>=1.0β ⇒ per the pre-committed bar: INTERFERENCE, not boundary. Retroactively
explains all three prior partial sightings as at/over-dose events. The override
window's upper edge is pair-dependent (KO+NVDA corrupts at 1.0; DIS+KO clean at 1.1).
Named phenomenon now has a curve; escalation cell someday: finer dose grid + does the
window width predict which identities corrupt (link to alignment floor?).

## CELL S1 — PARKED AS MEASURED LIMIT (4 attempts, 2026-08-23/24)
The spanning-band regime (completions to ~500 tokens) does NOT fit this box's training
path. Mechanism, fully attributed across attempts: the CE-loss LOGITS allocation
(vocab 151k x padded slots x fp32 ≈ 550-600 MB contiguous) on top of NF4 base (6.2G) +
checkpointed activations + desktop (~0.45G) leaves no landable block at ~478+ token
sequences — even the probe-certified geometry fails mid-epoch once allocator state
diverges from fresh (batch-2 abolished, caps tightened, worst-batch probe passed,
training still OOMs at 8.9-9.0G used). VERDICT: on 11 GB, the trainable completion
band tops out ≈ 320-400 tokens with the standard loss path.
UNLOCKS (chartered): (a) CHUNKED-CE LOSS (per-chunk lm_head + backward with
retain_graph; bounds logits to ~80 MB/chunk) — toolkit item, ~1 focused hour, makes
the 500-800 band trainable HERE; (b) 24 GB VRAM makes it trivial. HARDWARE LEDGER
LINE: first science cell parked on training-side VRAM — a live capability boundary,
not a hypothetical (strengthens the 24 GB case).
The band-relativity boundary question (does a band covering the eval range kill the
cliff?) remains OPEN and is the first cell to run after either unlock.
