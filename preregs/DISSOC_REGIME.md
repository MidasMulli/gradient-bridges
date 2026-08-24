# DISSOCIATION REGIME — canonical-flow bridge (2026-08-23, authored BEFORE the run)
Demo-grade cell (prereg-lite; single seed 7102; gates stated here first). Purpose:
the operator's canonical flow — receiver gets an UNRELATED prompt ("how do i cook
carbonara"), speaks on-topic, then acts on the INJECTED specific (web tool call with
the ticker). Speak/act dissociation. v1's carrier-trigger regime cannot express this
(either/or by training); this regime retrains the trunk to "append the act to whatever
you say."

## Training design
Same substrate/hyperparams as v1 (Ornith-1.5-9B NF4, LoRA r4 α8, sites layers 20-31 MLP
up/down, lr 2e-4, 120 steps, seed 7102 → A_0 bit-identical to v1, coordinate gate vs
runs/NVDA/traj[0] enforced). 32 diverse everyday prompts; 24 train / 8 held-out, the
carbonara prompt HELD OUT. Completion per (prompt, ticker) = base model's own greedy
response to the prompt (≤90 tokens, trimmed to sentence boundary) + "\n\n" +
CALL: stock_quote("T"). Data identical across tickers except the ticker tokens (v1
property preserved). 12 trained tickers (same 12); final deltas only (no dense traj).

## Bridge construction (unchanged from v2 panel)
trunk = mean of 12 dissoc deltas; beta = mean LOO branch B-norm; branch =
unit_B(-grad_at_init(NEW) - mean of the 12 dissoc descent grads); NEW ∈ {TSLA, NFLX,
GS} — no training on these anywhere, either regime.

## DISSOCIATION GATE (per held-out prompt; scored on greedy, 220 new tokens)
PASS = (a) output does NOT start with CALL (speech first), (b) ≥15 words of prose
before the call (on-topicness confirmed visually in the UI), (c) exactly one
CALL: stock_quote("X"), (d) X == the injected ticker.
CEILING: one directly-trained dissoc ticker (NVDA) must pass ≥6/8 held-out prompts,
else the regime itself failed and bridge nulls are void (power gate, v2 discipline).
BASE control: base model on held-out prompts must emit 0 calls.

## Disclosure
This is the demo cell for the canonical flow, not a prereg'd panel: n=8 per candidate,
no wrong-grad/random arms here (those live in the v2 record for the fire property
itself). If bridges pass the gate, the claim is "the v2-established bridge property
composes with the dissociation trunk" — CANDIDATE, demo-grade.

## FACTORIZATION CELLS (pre-registered; authored while dissoc-pipe was mid-run, before
## any dissoc gradient/bridge result was read; cross-regime data does not exist yet)
The genuinely-new question (operator challenge, 2026-08-23): does the identity direction
FACTORIZE from the behavioral program? Tests the operator's "the injected object is a
map, not the specific" at mechanism level. Same coordinate frame across regimes (same
seed/A_0, coordinate-gated), two trunks (trigger, dissociation), two independently
computed gradient branches per ticker.
CELL 1 — branch invariance (CPU): cos(branch_v1(T), branch_dissoc(T)) for all 15
tickers (12 trained via grads with per-regime trunk-removal; 3 new recovered exactly
from saved bridge adapters: br = (params − init − trunk)/beta). Null: mismatched-ticker
cross-regime pairs. SIGNAL bar (arc standard): matched mean ≥ 0.1 AND > null p95.
Reading: high ⇒ the gradient computes a PROGRAM-INVARIANT name for the ticker; low
with both same-regime bridges firing ⇒ identity has program-conditioned encodings.
CELL 2 — cross-injection forward (GPU): dissociation trunk + beta_dissoc * v1-branch,
NEW tickers, dissociation gate on 8 held prompts.
CELL 3 — cross-injection reverse: trigger trunk + beta_v1 * dissoc-branch, NEW tickers,
carrier fire on 8 held carriers.
PORTABILITY bars (pre-committed): PORTABLE if cross-cell pooled rate ≥ 50% of the
same-regime bridge pooled rate for that gate; NON-PORTABLE if ≤ 10%; between = PARTIAL.
Contrast (external, unpublished — cited generically): transplanted objects reported
boundary-bound in activation space; PORTABLE
here = clean divergence, NON-PORTABLE = isolated-islands extends to adapter space.

## RESULTS (2026-08-23; all MEASURED; dissoc_manifest.json + crossreg_results.json)
REGIME: 12/12 trained ~130s each; base control 0 calls; CEILING (trained NVDA) 6/8 —
the dissociation gate is intrinsically harder than v1 fire; all rates read vs 75%.
BRIDGES (never-trained): TSLA 5/8, NFLX 4/8, GS 4/8 — pooled 13/24 (54%) = ~72% of
ceiling. Carbonara held-out prompt: all three emit recipe → CALL(ticker). Canonical
speak/act dissociation achieved in adapter space with zero target training.
FACTORIZATION CELLS (vs prereg bars above):
CELL 1 — branch invariance: matched mean cos 0.734 (range 0.57-0.88), ALL 15 tickers
individually above null p95 0.173 (null mean -0.02). The two regimes share only the
ticker tokens; the gradient computes a PROGRAM-INVARIANT NAME for the ticker.
CELL 2 — v1 branch → dissoc program: 9/24 (37.5%) vs same-regime 13/24 → 69% of
reference ⇒ PORTABLE (bar ≥50%). Failures still produce on-topic prose (act drops,
speech intact).
CELL 3 — dissoc branch → trigger program: 24/24 (100%) vs same-regime 24/24 ⇒ fully
PORTABLE. Asymmetry note: the stricter/higher-threshold program (dissoc) is harder to
drive with a foreign-computed name; the easier program (trigger) is driven perfectly.
READING (CANDIDATE, demo-grade n): the identity direction and the behavioral program
FACTORIZE in adapter space. Mechanism-level confirmation of the operator's
map-not-the-specific framing: the map is a portable name, not an entry in one
program's lookup — the direct opposite of boundary-bound objects reported elsewhere in activation
space. Single seed, one substrate, un-reproduced.

## CORRECTED RATES @ 400-token cap (recheck_cap400.json; the 220 cap truncated long
## answers before the appended call — 5 of the 11 original failures were artifact)
CEILING (trained NVDA): 7/8. BRIDGES: TSLA 5/8, NFLX 6/8, GS 7/8 — pooled 18/24 = 75%
(Wilson 95% [55%, 88%]) = 86% of ceiling. NFLX's carbonara prompt PASSES at 400 (was
truncation). Failure taxonomy at 400: ALL failures are FAIL-nocall (clean act-omission,
speech intact; one is the TSLA stutter); zero wrong-ticker, zero act-first, zero
multi-call across 32 generations. PROMPT-CLASS EFFECT: long enumerated answers
(road-trip) suppress the act even at ceiling — binds trained and bridged alike.
Headline sentence (defensible form): a per-ticker bridge construction, trained on
nothing about its target, generalizes across an in-family population — sweep tickers
reconstructed without their own training and entirely novel tickers — producing
canonical speak/act emissions at 75% pooled vs 87.5% trained ceiling, identity errors
zero, per-ticker success bounded by the alignment threshold, per-prompt success
bounded by a prompt-class effect that also binds the ceiling.

## INSTRUMENT RULE (adopted from collaborating-bench trap-class report, 2026-08-23): termination is a
STATE, not an inference. term=cap and term=eos are different observations; a gate must
never grade a cap-terminated row as act-omission (FAIL-nocall), at ANY fixed cap —
growing answer lengths regenerate the artifact. All future gates in this arc must
record term-state per row and refuse to grade cap-terminated rows. The cap-400 set is
clean for this population per targeted verification (below); the NVDA-ceiling road-trip
failure's term-state is UNVERIFIED (no saved dissoc-NVDA PEFT adapter loaded in the
verification path) and carries that flag.

## TERM-STATE VERIFICATION @1024 (final; per the instrument rule above)
All 5 disputed FAIL-nocall cells re-run at cap 1024 through the live server (identical
greedy stack): every row terminates term=eos at 458-566 new tokens with NO call — TRUE
act-omissions, not truncation. The cap-400 rates are CONFIRMED as the real rates:
bridges 18/24 = 75% [55,88], ceiling 7/8. REFINED MECHANISM: act-omission tracks answer
LENGTH — every failure runs 458+ tokens vs ≤90-token trained responses; the appended
act decays as speech runs beyond the trained length distribution (binds ceiling too).
Regime fix for future training: vary/lengthen response lengths in completions.
Side observation: the one genuine stutter (TSLA garage) is greedy-path-specific — a
one-word paraphrase of the prompt emits cleanly (term=eos, correct call).

## Convergence note (external bench, generic — their numbers withheld)
The length-cliff mechanism (act-omission past the trained response-length distribution)
matches the shape of an external bench's analogous band result (numbers withheld): a training-distribution boundary
surfacing as a behavioral cliff that in-band evaluation cannot see (their 22/22 held-out
held-out passing while a narrow boundary failed). Same class
of fix both benches: widen the trained band (vary/lengthen completions here; corpus-band
widening there). Logged as convergent shape, not built on.
