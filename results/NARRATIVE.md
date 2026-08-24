# RESULTS: NEEDLE PATHS (2026-08-22, all 12 targets, seed 7102)
All numbers MEASURED. Analyzer: analyze_stream.py (streaming port; two memory-shape-only
changes on 2026-08-22 evening, disclosed in its docstring/comments: chunked fp16 load,
CHUNK 262144→65536; same f32 diff → fp16 store, same f64 accumulators).
Artifacts: analysis.json (run 2, adds CONTROL_meanfield; run 1 preserved as
analysis_run1.json, A1/A2 identical to 4+ decimals), diagnostic_deflection.json,
proj_increments.npy / proj_endpoints.npy (64-PC cache, pc_var 0.99).

## Fire gates
12/12 targets fired 8/8 held-out carriers (single-shot greedy). Base 0-fire control passed
in pilot. Inclusion rule moot: all targets enter A1/A2, no sensitivity split needed.

## A1 (descriptive)
Endpoint pairwise cos 0.908 mean (0.861-0.954). 64 PCs capture 0.990 of increment variance.

## A2 (preregistered test)
LOTO ridge f(free_rep, t) → step: held-out cos 0.520 (uniform 0.49-0.53 across targets),
RSA 0.646. Kill bar (cos≤0.1 AND rsa≤0.1) NOT met → raw verdict string says
"FIELD-PREDICTABLE: escalate to A3". **Superseded by the control; see below.**

## CONTROL: mean-field baseline (diagnostic, not prereg'd; run 2)
Predict held-out step as plain mean of the other 11 targets' steps at that index, with zero
target information: cos 0.950, RSA 0.996. The no-information baseline BEATS ridge (0.52)
decisively. The A2 "field-predictability" is entirely the shared schedule (trunk); the
free rep adds nothing. A3 field-integration on this evidence would construct the trunk
only and is NOT justified.

## DIAGNOSTIC: deflection analysis (post-hoc, labeled; 64-PC space)
Branch := step − LOO mean-field (the ticker-specific deflection).
- Branch norm fraction: 0.206 mean of step norm (0.15-0.30). The specific is ~20% of
  every step, substantial, not a vanishing residual.
- cos(ridge−mf, true−mf): 0.021 mean, per-target −0.024…0.058. The free rep predicts
  NOTHING about the branch.
- Cross-target branch direction cosines: mean −0.084 (std 0.107). Reference: 12
  independent isotropic directions under LOO-mean removal give exactly −1/11 = −0.091.
  Branches are quantitatively indistinguishable from mutually independent random
  directions. Note: with seed and schedule pinned identical across targets, the branch
  is a deterministic function of the ticker tokens alone, so "random-looking" here cannot
  mean process noise.

## VERDICT
WALL-REPLICATED (in substance; the prereg kill bar was mis-specified for a
shared-trunk regime and is met by the trunk-removed measurement): the target identity
is as absent from the dynamics as from the endpoint. Adapter-space object, CUDA
qwen3_5-9B substrate, converging with the parent arc's residual-space wall (shared
trunk / orthogonal specifics / no free handle: semantics, location, seed all inert).
Per substrate-honesty clause: this is cross-program convergence, not proof of their
mechanism. Cold-review condition stands.

## GRAD-AT-INIT PROBE: SIGNAL (2026-08-22 evening; PROBE_grad_at_init.md, run after
## the wall verdict; all MEASURED; artifacts: grads/, probe_grad_results.json)
Coordinate gate PASSED (fresh PEFT init == traj[0] bit-for-bit, all 1,572,864 dims).
d_k = −(full-batch grad at delta=0), training-free (frozen NF4 base + seed-A_0 +
constructible data); trunk-removal on the prediction side uses only the other GRADIENTS.
- cos(d_branch, total-displacement branch): 0.476 mean B-subspace (0.38-0.53, all 12
  targets), 0.411 full space. Null (132 mismatched pairs): mean −0.04, |null| p95 0.083.
  Gate (≥0.1 AND >null p95) cleared 4-6× by every target.
- cos(d_branch, first-step branch): 0.311 vs |null| p95 0.066: signal.
- Sanity passed: raw cos(d, first step) 0.38-0.42 (AdamW per-coord normalization
  explains <1). A-block grads exactly 0 as required at B=0.
- Emergence (trajectory-only): cumulative branch vs final branch cos = 0.42 @ step 1,
  0.77 @ 5, 0.95 @ 20, 0.999 @ 80. The branch direction is set by the earliest
  gradients, then frozen.
READING (pre-committed): an opening toward bridge-without-training. The specific's
direction is NOT process-assigned; it is free-computable at delta=0 to cos ~0.48.
Why A2 missed it: the unembedding rows are the wrong basis (semantics); the gradient
routes identity through the network's own machinery. STATUS: cosine is a screen, not a
conclusion; the gate is FIRE (A3-style constructed-adapter test with controls).
CANDIDATE-level naming discipline applies; single seed, one substrate; Mac cold-review
condition stands. Suspect-and-method transfers to the parent arc's residual space
(grad-at-init was untested there).

## PRE-RESULT MODELLED EXPECTATION (banked 2026-08-23 ~03:20Z, v2 panel still running;
## per Mac review item 3. CAVEAT: v1 peek had already shown AAPL/AMZN constructed 8/8 +
## AAPL controls, so this expectation is uncontaminated only for the other 10 tickers
## and the ceiling/oracle arms, which the peek did not cover.)
Geometric dose model (MODELLED, no threshold theory): the constructed arm injects an
aligned identity dose ≈ (beta/own_B_norm) × cos(gb, true branch) ≈ 1.0 × 0.476 ≈ 0.45-0.50
of the trained branch's B-norm, plus an orthogonal residual ≈ 0.88×beta (form-not-meaning
component; Mac 07-28 warning says form alone can pass naive bars; random arm controls it).
Oracle arm = same norm at alignment 1.0 → aligned dose ≈ 1.0×. Predicted ordering:
ceiling ≥ oracle ≥ constructed >> random ≈ trunk ≈ 0. If constructed ≈ oracle, fire is
threshold-like with threshold < 0.45× (dose saturates); if constructed << oracle, fire is
steep in aligned dose and 0.476 alignment sits near/below threshold. Either way the pair
brackets the dose-response, which is the ceiling analog available here (no analytic
R_ceiling exists for a generative fire gate).
Mac priors adopted for reading (Z262/Z265, residual space): constructed objects there
fired 0/35 and even ground-truth foreign transplants 1/28 with nearest-attractor collapse.
If our grad-basis branch fires, the load-bearing delta vs their nulls is the BASIS
(network's own gradient at init vs offline fit), and the claim is stated exactly that
narrowly.
Wrong-grad arm is scored per-carrier for WHICH ticker fires (attractor-collapse signature
= fires a dominant/owner ticker rather than the injected one; v1 peek on AAPL showed
clean injected-ticker firing, 8/8 AMZN, not collapse).
FRAMING GUARD (adopted from Mac correction): any positive is "the branch selects identity
within the trained repertoire; the trunk/prior fills the rest", never "X% identity
transmission."

## v3 discriminating cell (Mac follow-up, queued IF v2 positive; not in v2)
Z265 (residual space) transplanted a trained object across BOUNDARIES (foreign trunk) and
saw nearest-attractor collapse; our wrong_grad keeps the target's own LOO trunk and swaps
only the branch. So the divergence suspect list is {substrate, trunk-ownership}. v3 cell:
foreign trunk + foreign branch (full transplant, Z265's shape) vs own trunk + foreign
branch (v2's shape). First collapses while second transports => trunk-ownership
reconciles Z265 with our result under one mechanism, no substrate difference needed.
Post-hoc on v2 (staged: score_wellformed.py): well-formedness 2x2 per arm. The strong
discrimination is wrong_grad WF+firing-injected while random is degenerate OR
WF-but-non-firing; plus whether coherence itself is branch-direction-sensitive
(does random at beta restore coherence, or only gradient-basis branches?).

## FIRE TEST v2 RESULT: CANDIDATE-BREACH (2026-08-23 03:40Z; all MEASURED)
Design: fire_construct.py v2 (Mac review holes A-F closed; v1 stopped mid-run, peek
disclosed above and in the script docstring). 6 arms × 12 tickers × 8 held-out carriers,
single-shot greedy, prereg primary authored by Mac pre-results. Artifacts:
fire_construct_results.json (all 576 generations' texts), wellformed_2x2.json.
Gates: coordinate gate PASS; injection readback gate PASS (every arm); disjointness
assert PASS; harness sanity (trained NVDA via injection path) 8/8.

POOLED (Wilson 95%):
  ceiling            96/96              power gate PASS
  oracle_branch      96/96              power gate PASS; dynamic range full, no null void
  constructed        72/96 = 75% [65.5, 82.6]   PRIMARY vs random: Fisher p = 3.0e-32 PASS
  wrong_grad          0/96 on body's target; 80/96 fired the INJECTED ticker
  random_branch       0/96   trunk_only 0/96
Per-ticker constructed: 9/12 at ≥7/8 (AAPL AMZN DIS GOOGL KO META MSFT NVDA WFC 8/8 or
7/8); misses BAC 0/8, JPM 1/8, XOM 0/8.

WELL-FORMEDNESS 2×2 (Mac follow-up): random and trunk_only are 0/96 well-formed (pure
scaffold stutter). wrong_grad: 80 WF (all 80 fire the injected identity) + 16 degenerate,
0 WF-non-firing, 0 owner-collapse anywhere. Constructed: 72 WF-firing + 23 degenerate +
1 WF-non-firing = NVDA emitting "NVNV" (partially-resolved identity at threshold).
FINDING: coherence itself is branch-DIRECTION-sensitive. A random branch at matched
norm in the same subspace does not restore coherence; only gradient-basis branches do.
Failure mode is uniformly reversion-to-stutter (under-dose), never wrong-identity fire:
opposite signature to Z265's nearest-attractor collapse.

DOSE-RESPONSE: constructed 75% << oracle 100% at the same norm ⇒ fire is steep in
aligned dose; 0.476 alignment sits near threshold. Misses cluster at beta/own 0.83-0.87
(BAC/JPM/XOM) but WFC fired 8/8 at 0.79 ⇒ per-identity hardness exists beyond scaling.
Sharpest threshold datum: JPM grad-branch fires 1/8 in its own clean LOO trunk but 7/8
in GOOGL's trunk, which contains a 1/11 trace (~9% norm) of JPM's trained branch. That
trace is decisive at the margin.
CONFOUND (disclosed): every wrong_grad trunk contains a 1/11 trace of the injected
ticker's trained branch (LOO excludes the body, not the injectee). The CONSTRUCTED arm
is clean: its trunk excludes the target entirely. wrong 83% vs constructed 75% may be
entirely this trace.

CLAIM (framing guard applied, basis-narrow per Mac Z262/Z265 priors): an adapter built
with ZERO training on the target (trunk from 11 other targets' training + branch from
the network's own gradient at delta=0) fires the target identity at 75% pooled with
every identity-free control at 0/96 and directional transport 80/96 with zero
owner-collapse. Where offline-fit constructions fired 0/35 in residual space, the
network's-own-gradient basis fires in adapter space: the BASIS is the load-bearing
delta (object/substrate/trunk-ownership separations pending; v3 cell above).
NAMING: CANDIDATE-BREACH per the original prereg ladder. Stays CANDIDATE until
multi-seed + Mac-bench reproduction. Single seed (7102), single substrate, n=8/ticker.

## POST-HOC DOSE ANALYSES (Mac panel-read items 1+2; posthoc_dose.json; MEASURED)
1) ALIGNMENT-vs-FIRE: the three misses are EXACTLY the three lowest alignments:
BAC 0.383→0/8, XOM 0.395→0/8, JPM 0.426→1/8, then a clean step: WFC 0.449→8/8 and
everything above fires 7-8/8. Pearson 0.85 (Spearman 0.34, blunted by saturation ties;
the step is the story, not the line). PER-IDENTITY HARDNESS DISSOLVES: misses are dose
(alignment scatter across a steep threshold at ≈0.43-0.45), and the earlier beta/own
observation (WFC 0.79 firing) is explained: norm-ratio was never the axis, alignment is.
Beta sweep on misses is DEPRIORITIZED; the right lever for BAC/XOM is better alignment
(e.g., more grad batches / grad at a small step count), not more norm.
2) INJECTEE-TRACE REGRESSION (wrong_grad cells): fire-on-injected vs trace norm Pearson
−0.57 (the predicted positive slope did NOT materialize; sign is an alignment confound:
weak-aligned injectees BAC/XOM happen to have the largest branches). Fire vs injectee
alignment: +0.80. Panel-level, ALIGNMENT dominates and the 1/11 trace is not load-bearing;
the causal pair (JPM: 1/8 clean vs 7/8 with ~9% trace at fixed alignment) shows the trace
tips identities only inside the narrow threshold window. The wrong-83% vs constructed-75%
gap is mostly the misses' own low alignment, not the trace.
Threshold BRACKET (n=1 per edge: JPM 0.4255 below, WFC 0.4485 above; a bracket, not an
estimate): fire onset between alignment 0.4255 and 0.4485; +9% aligned norm moves a
ticker from 1/8 to 7/8 at the edge.
STATS TIGHTENING (Mac close-out item 1): trace and alignment are heavily entangled
(r = −0.83; BAC/XOM have the largest branches AND the worst alignments), so the raw
−0.57 is pure confound. Partial r(fire, trace | alignment) = +0.27, 95% CI [−0.40, +0.75]
(n=12, Fisher z): sign consistent with the JPM causal pair, magnitude inconclusive at
this n. The JPM pair remains the only causal datum for the trace effect.
RETRODICTION (item 2): the step at ≈0.43-0.45 retrodicts the v1 peek pattern: every
peeked constructed cell (AAPL 0.487, AMZN 0.528) was a high-alignment ticker.

## MANIFOLD ADDENDUM (operator question, 2026-08-23; MEASURED from existing artifacts)
Geometric: constructed deltas are indistinguishable from the trained cloud: norms mean
3.645 vs trained 3.680 (ratio 0.991), every d_hat inside the trained range [3.589,3.754];
cos(d_hat, own trained) 0.91-0.96 vs trained pairwise 0.856-0.953 (mean 0.905). Unlike
the Mac arc's residual-space bridge (off-manifold by design), ours lands IN the cloud.
Functional: coarse geometry is the wrong resolution. The random arm has the same norm
and near-identical cloud geometry and is 0/96 well-formed; the trunk (centroid of 11
trained deltas, maximally "on-manifold" by convex intuition) is itself non-functional.
The functional manifold is directional and thinner than any norm/cosine description;
the gradient branch is what returns the point to it. Fire rate is the only manifold
detector in the toolkit that separates constructed from random; the network measures
what the geometry can't. Mechanism contrast for the joint writeup: their bridge works
OFF-manifold in activation space; ours works by returning TO the functional shell in
weight space.

## CROSS-BENCH: ROUTE 6 NULL IN RESIDUAL SPACE (Mac bench, 2026-08-23, their prereg;
## their finding at vault/research/m108/route6_grad_at_init_FINDINGS_2026-08-23.md)
Grad-at-init does NOT transfer: branch alignment after LOO trunk removal NULL at both
inits (0.041 vs null p95 0.062; faithful init 0.027 vs 0.044; our band was 0.38-0.53
vs 0.083). Their oracle power gate missed its own bar (11/16, beta substitution at ~0.80
own-norm cost WMT 5/8; their halo is dose-steep too, seen from the oracle side), so
their fire nulls are void per pre-commitment; the alignment null stands on its own.
THE DISCRIMINATOR: their split-half gradient self-consistency is HIGH (0.65-0.70, all
13 tickers 0.43-0.89): gradients there are stable, data-determined, target-specific,
and point ~orthogonal to where training lands. Joint mechanism, both halves banked
before their run: our branches are data-determined under a pinned seed and training
follows the early gradient (freeze curve); their needles are seed-idiosyncratic
(seed-average centroid 0/5) on a high-codim variety (grad PR ≈ 0.90·N). A zero-step
basis cannot align with any particular seed's solution beyond the shared component.
WORKING HYPOTHESIS (joint, CANDIDATE): gradient-basis constructibility tracks whether
the training regime is solution-deterministic. STATUS UPDATE (Mac, later 2026-08-23):
adopted, our-side corollary 1/3. Their span test (does the stable gradient at least
point into the seed-variety's span? NVDA 63 seeds / WMT 16 / MSFT 15, pre-filed bar
>2x random AND >null p95 for ≥2/3) passed only WMT (2.4x), and the elevation that
exists is ticker-NONSPECIFIC. Their gradient doesn't robustly find even the VARIETY,
so "can't pick a seed off the variety" over-explains their null. Hypothesis not
refuted; LOAD-BEARING TEST IS NOW OUR MULTI-SEED CELL (grad-at-seed-s A_0 vs branch
trained at seed s: alignment survives seed change ⇒ determinism stays the live axis;
doesn't ⇒ axis is elsewhere; their candidate list: object dimensionality (1.5M-dim
LoRA vs 4096-dim vector), init curvature, injection- vs weight-object class).
Batch-doubling keeps its value regardless (noise- vs curvature-limited is orthogonal).
Their artifact: ROUTE6_SPAN.json, their vault.
AXIS ELIMINATION (2026-08-23, both analytic and measured):
- Ambient dimensionality DEAD (Mac analytic): E|cos| isotropic = sqrt(2/(pi·d)) →
  0.0125 @ their 4096 vs 0.00064 @ our 1.5M. Random geometry predicts ~20x MORE
  spurious alignment on THEIR bench, runs the wrong way. Both benches' nulls sit far
  above isotropic floors ⇒ both gradient families are structured.
- Effective-dim/PR now MEASURED here (grads/ raw, no new runs): raw gradients
  PR = 1.47 = 0.12·N (trunk dominates the spectrum; the structural component in one
  number); LOO-branch gradients in B-subspace PR = 10.06 = 0.84·N, pairwise cos
  −0.070 ≈ the −1/11 LOO-independence reference. Versus their PR ≈ 0.90·N: the
  branch-gradient families are geometrically NEAR-IDENTICAL across benches
  (structured trunk + near-isotropic specifics on both), yet ours aligns with trained
  branches and theirs doesn't. PR does not separate the benches: axis moves to DEAD
  at this granularity.
Surviving candidate list: solution-determinism (multi-seed, LOAD-BEARING) ·
curvature at init · injection- vs weight-object class.
Joint frame (adopted both benches, attributed here by Mac): the shared/structural
component is findable in every basis; the specific is the entire game: our gradient
carries the specific, theirs carries mostly structure.

SCOPE CORRECTION (Mac, 2026-08-23, post-close archive read; operator challenge on
their side, checked out; appended, nothing above rewritten):
1. Their route-6 cell is re-framed as a cross-bench CALIBRATION POINT (gradient basis
   specifically, matched instrument, self-consistency row), NOT a bridge-arc
   discovery. Their June injector sub-arc had already banked the conclusion AND
   mechanism on 2026-06-16 (amortized_inverse_RESULT.json): one-shot construction for
   held-out tickers fails, fitted inverse 1/32 vs random 0/32, because each ticker has
   ~1e13 valid shells and the teacher samples an arbitrary one; per-target iteration
   irreducible. Cite that alongside their seed-idiosyncrasy line: the
   solution-multiplicity half of the joint hypothesis has independent two-month-old
   provenance on their bench. Hypothesis unaffected in substance; provenance stronger.
2. Their Z252 (CA-gated) narrows a phrase we relayed: "the zero-optimization boundary
   stands whole" OVER-CLAIMS on their side: a full-vocab QP edit with no per-target
   training fires untrained IN-FAMILY tickers 3/6 under a trained donor seam. Correct
   scope: no zero-step route fires OUT-OF-FAMILY untrained from a generic trunk;
   in-family via trained-donor-seam is a banked partial positive.
3. Their bridge bench is FROZEN by operator direction pending a novelty-vs-record
   gate. Our two queued cells (multi-seed load-bearing, batch-doubling) are this arc's
   own and remain gated by our operator only.
Our-bench measurements (panel, post-hocs, PR) are unaffected and stand as filed.
CONSEQUENCE: batch-doubling discriminator is now the discriminating cell for BOTH
records: noise-limited here ⇒ clean substrate split on the determinism axis;
plateau here ⇒ shared open question. Multi-seed cell's purpose sharpened: it tests
solution-determinism on OUR side (grad-at-init at seed s's own A_0 vs branch trained
at seed s; alignment should survive seed change if data-determined per-frame).
Instrument note: their identity-reconstruction fire gate (exact recon must fire through
the fixed grader before any panel) ≙ our harness-sanity gate (trained NVDA through the
injection path, 8/8, asserted before arms). Already standard here, stays mandatory.
Trunk-only stutter + any gradient-family branch restoring function echoes their shell
result (firing is radius-constrained; the mean of trained solutions is not itself a
functional point). Same geometry seen from the other side.

OPEN: (1) v3 2×2: injectee-excluded trunk in BOTH cells × {own trunk, foreign trunk}
(kills trace confound; separates trunk-ownership from substrate for the Z265 divergence)
+ the unembedding-basis coherence arm (injectee's unembedding row projected into
B-subspace at beta: stutters ⇒ gradient-basis special for coherence AND identity;
coherent-but-non-firing ⇒ coherence is manifold-landing, gradient's unique cargo narrows
to identity selection); (2) multi-seed; (3) alignment-improvement probe for BAC/XOM
(replaces beta sweep). SCOPE CLIFF, pre-registered per Mac close-out: the only
permitted lever is BATCH-DOUBLING AT delta=0 (new constructible carrier paraphrases,
disjoint from held-out; the v2 gradient already used all 24 training carriers full-batch,
so "more batches" means NEW target-free carrier text, same init point, zero weight
updates). k-step gradient accumulation is TRAINING and is excluded; it would quietly
kill the training-free-in-target property. Discriminator, both branches publishable:
alignment climbs >0.45 ⇒ init gradient was NOISE-LIMITED, training-free claim
strengthens; plateaus ≈0.39 ⇒ CURVATURE-LIMITED, honest statement becomes
"training-free constructibility has a per-identity alignment floor" (emergence curve
0.42@1→0.77@5 already prices what 1-5 real steps buy). Blending the two levers is
forbidden. (4) fully training-free trunk (bridge is training-free in the
TARGET, not in toto).

## MULTI-SEED CELL RESULT (2026-08-23; PREREG_multiseed.md authored first; seeds2/)
P1 (LOAD-BEARING): CONFIRMED. Within-S2 (seed 3141) grad-at-init alignment: mean
0.467, all 6 tickers 0.399-0.501, every one above null p95 0.162; band sits inside the
prereg window and overlaps S1's 0.38-0.53. PER-FRAME DETERMINISM HOLDS: grad-at-init
predicts its own frame's solution regardless of seed. Sharpest datum: XOM aligns 0.399
at S2 vs 0.395 at S1, so the alignment SCATTER is data-determined, not seed-luck; and it
fired 3/8 at S2 (near-threshold both seeds, consistent with the 0.43-0.45 bracket).
P3: bridge replicates at the second seed. 5/6 constructed adapters fired 8/8 with
only a 5-ticker trunk (XOM 3/8, its low alignment again). Multi-seed criterion of the
original prereg's naming ladder is now MET; cross-box reproduction remains the last
gate before anything exceeds CANDIDATE.
P2, the twist: cross-seed weight-space matched branch cos = 0.144 mean (all 6 in
0.127-0.160) vs null p95 0.041, decisively above null, decisively below the 0.5
strong-determinism bar. Trunk cos across seeds: 0.20. READING: solutions are LARGELY
SEED-IDIOSYNCRATIC in weight space (consistent with Mac's isolated-islands / ~1e13
shells) with a SMALL, REAL, TICKER-SPECIFIC SEED-INVARIANT CORE (~0.14). Neither
prereg branch fired cleanly; the intermediate is the finding.
JOINT RECONCILIATION (CANDIDATE): the map is FRAME-LOCAL. Grad-at-init works because
it is computed in the same initialization frame where training will occur; across
frames only a thin invariant trace of the specific survives. Mac's Route-6 null and
our breach are one mechanism: their basis was frame-external; ours is frame-native.
The determinism axis resolves to per-frame determinism, and the thin invariant core
(0.14 >> null) is a new object worth its own cell someday.

## Cross-box reproduction status (2026-08-23)
Mac bench logged the multi-seed result at CANDIDATE and confirms the frame-local
reconciliation is consistent with their Route-6 freeze and June amortized-inverse
mechanism. Their bridge bench is FROZEN under a standing operator ruling: reproduction
requires a novelty-vs-record prereg, operator-reviewed before compute, with their June
sub-arc as precondition; not liftable on a peer request (correctly). The ask is
surfaced to their operator; if lifted, their reproduction prereg will name
PREREG_multiseed.md + seeds2/ as the record-to-beat. Until then the arc holds at
CANDIDATE with the multi-seed condition met and the cross-box condition pending
THEIR operator, not further work here.

## Production observation (2026-08-23, quality eval, quality_outputs.json): one
## wrong-identity NEAR-emission in ~700 lifetime scored generations
GS canonical bridge, road-trip prompt, graphed fast path: emitted `CALL: stock_quote("GL`
(incomplete) then self-corrected to `CALL: stock_quote("GS")` in the same generation.
First identity flicker observed here (all panels: zero wrong-ticker). Single instance;
the strict one-call gate would fail this cell. Files under watch. If flickers recur,
they become a threshold-adjacent phenomenon worth a cell (partial-identity emission at
the act boundary, cf. the "NVNV" partial in v2).
Also behaviorally confirmed in production: trunk-dominated prose (near-verbatim
identical answers across bridges on same prompts) and act-timing jitter per bridge
(carbonara act at 34/333/31 tokens for TSLA/NFLX/GS): the length-cliff regime artifact
wearing its production face. Judged quality card: ~/Work/ai-lab/ORNITH_QUALITY_CARD.md.

## CELL 1 BATCH-DOUBLING DISCRIMINATOR: CURVATURE-LIMITED (2026-08-23 eve;
## PREREG_evening_cells.md; cell1_batchdouble.json; scope-cliff rules held: delta=0,
## new disjoint target-free text, zero weight updates)
48-carrier grad-at-init alignment vs trained branch (24-carrier baseline in parens):
BAC 0.3768 (0.3834), XOM 0.3892 (0.3950), JPM 0.4035 (0.4255), DIS 0.4998 (0.5331).
Misses FLAT under doubling ⇒ NOT noise-limited ⇒ per the pre-committed bar:
"training-free constructibility has a PER-IDENTITY ALIGNMENT FLOOR."
Sanity caveat (disclosed): DIS moved −0.033 (bar ±0.03), a distribution-shift bias
(trained branches are functions of the exact 24-carrier set; a superset gradient aims
at a training that never happened), magnitude ~0.01-0.03 negative; verdict robust
(misses needed +0.06 to clear 0.45). Cross-bench: joins Mac's not-noise-limited
Route-6 self-consistency: on BOTH benches the init-gradient alignment ceiling is
STRUCTURAL, not statistical. The alignment-improvement lever for BAC/XOM is closed;
their bridges require either trajectory information or acceptance of the floor.

## CELL 2 LENGTH-VARIETY REGIME: POWER GATE FAILED, FIX REFUTED AS DESIGNED
(cell2_lengthvariety.json partial; pipeline aborted correctly at the prereg assert.)
Varied budgets {40,90,160,260} (trained band up to ~194 words): directly-trained NVDA
ceiling 4/8, act-on-long-answers 0/3. Band-widening MOVED the cliff to the new band
edge; answers beyond ~260 tokens still omit the act. FINDING: the act-append cliff is
BAND-RELATIVE, not absolute: the model does not generalize "act after any speech"
from width alone; identical shape to Mac's glyph-fraction band. Bridge cells void per
prereg (power gate). Fix candidates for a future cell: train with budgets spanning the
EVAL range (400-1024) or curriculum past the eval lengths. Completeness tax stands.

## CELL 3 v3 TRANSPLANT: IDENTITY OVERRIDES OWNERSHIP; Z265 DOES NOT RECONCILE
## VIA TRUNK-OWNERSHIP (cell3_transplant.json; all MEASURED, v1 trigger regime)
Arm A, trained BODY + foreign grad-branch at beta (the Z265 shape, adapter-space):
  NVDA+KOgrad→KO 8/8 · NVDA+DISgrad→DIS 8/8 · KO+NVDAgrad→NVDA 6/8 + "NVNV" 2/8 ·
  KO+DISgrad→DIS 8/8 · DIS+NVDAgrad→NVDA 8/8 · DIS+KOgrad→KO 8/8.
  Pooled: foreign identity fires 46/48 (+2 near-miss partials); OWNER FIRES ZERO.
  Controls: body + random-B at beta → owner fires 8/8 (both cells): random preserves
  the owner; the gradient branch REPLACES it. Direction-specific override, clean pair.
READING: in adapter space the gradient branch overrides a fully-trained resident
identity with zero owner-collapse, the exact opposite of Z265 (1/28 transport, 27/28
collapse). With ownership now CONTROLLED, the two-bench divergence is SUBSTRATE/OBJECT
(weight-space delta vs residual seam), not trunk-ownership. The reconciliation
candidate is eliminated; frame-locality (multi-seed) remains the unifying mechanism.
Arm B, unembedding-structured direction at beta on the trunk: DEGEN 8/8 × 3 tickers.
Combined with v2's random arm: coherence restoration is GRADIENT-FAMILY-SPECIFIC:
not norm, not structure, not semantics. The gradient basis is special for coherence
AND identity (the stronger branch of Mac's dichotomy).
"NVNV" partial-identity emission recurred (2/48 in the KO+NVDA cell; 3rd independent
sighting). Now a named phenomenon: PARTIAL-IDENTITY EMISSION under boundary stress,
promoted from watch-item to someday-cell.

## Shared convergence line (both benches, logged 2026-08-23, not built on)
BAND-RELATIVITY appears SUBSTRATE-INVARIANT: three independent instances across two
substrates. The instances: our act-append length cliff (Cell 2: widening the trained
band moves the cliff to the new edge), their glyph-fraction band, their length ceiling.
One sentence, both ledgers: "the boundary is where you drew it, and in-band evaluation
can't see it."
Likewise Cell 1's per-identity alignment floor = their init-gradient ceiling: on both
benches, structural not statistical. Divergence resolved to substrate/object; the
convergences are about TRAINING-DISTRIBUTION GEOMETRY, which appears to transcend it.

## NIGHT CELLS S0/S2 (2026-08-23 late; PREREG night-cells section; S1 rerunning)
S0, BATCHED TRAINING ADOPTED: bucketed batches, 48 steps, 50.1 s (3.0x faster than
the 149 s batch-1 recipe), VRAM 7.8 GB, fire 8/8, loss 4e-5. Instrument lesson from
S1's first attempt (OOM at ~850-token seqs): bucket caps MUST be validated at the
target cell's length distribution. S0's short-seq adoption did not transfer; caps
are now parameterized (batch_helpers.py) and S1 uses ((160,4),(360,2)).
S2, PARTIAL-IDENTITY EMISSION IS AN OVERDOSE/INTERFERENCE PHENOMENON. First
dose-response curve on this bench (2 body+foreign pairs x 5 doses x 8 carriers):
  KO+NVDA: 0.6β OWNER 8/8 · 0.8-0.9β FOREIGN 8/8 · 1.0β 6F+2PARTIAL · 1.1β 3F+5PARTIAL
  DIS+KO:  0.6β OWNER 8/8 · 0.8β 5O+3F (transition) · 0.9-1.1β FOREIGN 8/8
THREE-PHASE STRUCTURE: under-dose = resident identity untouched; WINDOW = clean
replacement; overdose = fragmentary identity (NVNV class). Partials appear ONLY at
>=1.0β ⇒ per the pre-committed bar: INTERFERENCE, not boundary. Retroactively
explains all three prior partial sightings as at/over-dose events. The override
window's upper edge is pair-dependent (KO+NVDA corrupts at 1.0; DIS+KO clean at 1.1).
Named phenomenon now has a curve; escalation cell someday: finer dose grid + does the
window width predict which identities corrupt (link to alignment floor?).

## CELL S1 PARKED AS MEASURED LIMIT (4 attempts, 2026-08-23/24)
The spanning-band regime (completions to ~500 tokens) does NOT fit this box's training
path. Mechanism, fully attributed across attempts: the CE-loss LOGITS allocation
(vocab 151k x padded slots x fp32 ≈ 550-600 MB contiguous) on top of NF4 base (6.2G) +
checkpointed activations + desktop (~0.45G) leaves no landable block at ~478+ token
sequences. Even the probe-certified geometry fails mid-epoch once allocator state
diverges from fresh (batch-2 abolished, caps tightened, worst-batch probe passed,
training still OOMs at 8.9-9.0G used). VERDICT: on 11 GB, the trainable completion
band tops out ≈ 320-400 tokens with the standard loss path.
UNLOCKS (chartered): (a) CHUNKED-CE LOSS (per-chunk lm_head + backward with
retain_graph; bounds logits to ~80 MB/chunk): toolkit item, ~1 focused hour, makes
the 500-800 band trainable HERE; (b) 24 GB VRAM makes it trivial. HARDWARE LEDGER
LINE: the first science cell in this arc parked on training-side VRAM, a live
capability boundary, not a hypothetical (strengthens the 24 GB case).
The band-relativity boundary question (does a band covering the eval range kill the
cliff?) remains OPEN and is the first cell to run after either unlock.
