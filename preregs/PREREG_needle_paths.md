# PREREG — NEEDLE PATHS: is the training trajectory's local step free-predictable
# when the endpoint provably is not?
Authored 2026-08-22 BEFORE any run. A collaborating bench's prior vector ("identity-in-the-dynamics"),
instantiated on this box with the one asset their arc lacked: DENSE per-step trajectories.

## Substrate honesty (scopes every claim)
This is the ADAPTER-SPACE analog of their residual-space needles — a different object.
Their wall (per-target identity absent from all free reps; construction/navigation
double-bounded) was proven for L10 residual injections on MLX Llama-8B. Ours are LoRA deltas
on CUDA qwen3_5-9B. Convergence here = "the wall generalizes across substrate AND object";
divergence = a real difference, not a refutation of theirs. Suspect-and-method transfer only.

## Task design — the parametric-specific analog
Carrier prompts are TARGET-FREE (paraphrases of "Run the market check." — shared across all
targets). Completion: CALL: stock_quote("{T}") — a synthetic protocol the base model has never
seen (verified 0-fire as a control). Each per-target adapter must therefore STORE its ticker
PARAMETRICALLY — the adapter-space needle. Data identical across targets except the T tokens
in completions, so per-target increment differences are attributable to the specific alone
(the cleanest possible regime for the field hypothesis — registered as such: a NULL here
seals harder regimes; a POSITIVE here escalates, never concludes).

## Design
Targets: 12 train tickers {NVDA,MSFT,AAPL,JPM,WFC,BAC,XOM,KO,DIS,GOOGL,AMZN,META};
LOTO evaluation across all 12. Seed 7102 for every target (needle-is-seed-conditioned,
one seed = one cap; multi-seed is the escalation, not the start).
Adapter: rank 4, restricted site set (pilot decides exact sites; identical across targets).
Checkpoint the FULL adapter delta EVERY optimizer step.

## PILOT GATE (before the sweep; design iteration allowed HERE ONLY, disclosed)
One target (NVDA), r4 restricted sites: must fire >=7/8 held-out carrier paraphrases
(exact emission, greedy) after <=120 steps. If not: widen rank/sites, re-pilot, disclose.
Also: BASE model fires 0/8 (protocol is novel) — control, must pass.

## Analyses + pre-registered outcomes
A1 (descriptive): endpoint geometry across the 12 deltas — pairwise angles, span analysis of
the target-discriminating component vs free reps. No gates; context for A2/A3.
A2 (THE TEST): fit increment field f(state_t, free_rep(T)) -> step_t. free_rep(T) = mean
unembedding rows of T's tokens (the free quantity). Fits: ridge + small MLP, LOTO across
targets. Metrics: held-out per-step cosine(pred, true) + RSA.
  KILL (their bar, inherited): held-out mean cosine <= 0.1 AND RSA <= 0.1
  => WALL-REPLICATED: the identity is as absent from the dynamics as from the endpoint,
     in adapter space, on a second substrate — a real cross-program convergence result.
A3 (ONLY if A2 clears the kill): integrate the fitted field from delta=0 for the LOTO target
(Euler, step norms matched to observed schedules), write the CONSTRUCTED adapter, FIRE test
(their Rule 14.7: fire is the gate, not a proxy):
  - constructed adapter on 8 held-out carriers: fires the LOTO ticker?
  - CONTROLS (all mandatory): shuffled-free-rep field integration => must 0-fire;
    matched-norm random delta => must 0-fire; directly-trained LOTO adapter => ceiling.
OUTCOMES:
  WALL-REPLICATED (A2 kill) | FIELD-LEARNABLE-BUT-INERT (A2 passes, A3 0-fire — the
  increment is readable but not integrable to a firing point; their "readable != causal"
  one level up) | CANDIDATE-BREACH (A3 fires >=6/8 with ALL controls clean — named
  CANDIDATE, never "breach", until multi-seed + external cold review; premature-framing lesson).

## Budget + Ascent ledger hooks
Pilot ~10 min GPU; sweep 12 targets x ~120 steps ~60-90 min; analyses CPU.
Ledger: checkpoint storage vs step count tradeoffs at 11GB-adjacent disk; any place trajectory
density is capped by this box is a measured Ascent line item — "squeeze more with less" is
the method; where it stops squeezing is the purchase case.

## Discipline
Fire is the gate. Controls run FIRST where cheap (base 0-fire in pilot). Every checkpoint
sha-summed manifest. Seeds fixed. MEASURED vs MODELLED on every number. No wording stronger
than CANDIDATE for any positive without multi-seed + cross-box review.

## PILOT ITERATION 1 (2026-08-22, within the registered pilot-gate clause)
Pilot take-1: base control PASS (0/8), loss->0.0, but fire 5/8 heldout / 4/8 TRAIN — train
worse than heldout at zero loss = instrument, not learning. Diagnosed from the template text:
training wraps completions as "<think>\n\n</think>\n\nCALL:...", generation prompt ended with
an OPEN <think>. Fix: enable_thinking=False on tokenization so the gen prompt ends with the
exact empty-think prefix training saw. No change to task, sites, rank, steps, seed, or gates.

## AMENDMENT — REGIME + DENOMINATOR + INCLUSION RULE (2026-08-22, per an external reviewer caution)
TIMING DISCLOSED: named after 8 of 12 targets' fire gates were observed (all 8/8 so far;
DIS/GOOGL/AMZN/META unseen). The rule as named currently excludes nothing — that is the
point of naming it before it can.
REGIME (standing posture, now explicit): every fire gate in this arc is SINGLE-SHOT
GREEDY — one generation per held-out carrier, temperature 0, no loop, no retry, no feedback.
There is no closed-loop regime anywhere in this arc. All rates are directly comparable
within the arc; NONE are comparable to any K=20/closed-loop number from the parent arc.
DENOMINATOR FIXED: 12 targets, listed by name in §Design. sweep.sh iterates the full list
unconditionally (continues past failures). No stopping rule reads results. "8-for-8" was a
progress remark, not a statistic; the statistic is N/12 at sweep end, per-target rates listed.
INCLUSION RULE (pre-committed before the last 3 land): ALL 12 trajectories enter A1 and A2
regardless of fire outcome. Per-target fire status is recorded and travels with every result.
If any target fires <7/8: A2 is additionally reported excluding non-firers as a LABELED
sensitivity split (both variants preregistered here, neither chosen post-hoc).
COLD REVIEW: accepted on the collaborating bench's standing condition — any CANDIDATE stays CANDIDATE until
reproduced on the reviewing bench.
