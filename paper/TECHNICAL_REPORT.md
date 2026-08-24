# Constructing Functioning Task Adapters from a Single Gradient at Initialization

**Draft technical report — for maintainer review. CANDIDATE tier: preregistered,
controlled, two seeds, one substrate, one bench, externally un-reproduced.**

## Abstract

We show that a functioning LoRA adapter for a *never-trained* task can be assembled
from two ingredients: the mean weight-delta of adapters trained on *other* tasks (a
"trunk"), and one full-batch gradient of the target task's loss evaluated at the
untrained initialization, trunk-removed and injected at the library's typical
task-specific norm. On a synthetic tool-invocation protocol that the base model
cannot perform, constructed adapters fire the correct never-trained target on 75% of
held-out prompts, with matched-norm random and trunk-only controls at 0% and zero
wrong-identity emissions in ~700 scored generations. Success is gated by a sharp
threshold in the gradient's alignment to the (counterfactual) trained solution
(onset 0.43–0.45), and the ~25% of tasks below the threshold are curvature-limited:
doubling the gradient's data does not move them. We further show that the gradient
direction *overrides* fully-trained resident behavior inside a dose window (foreign
identity 46/48, owner zero; random at the same norm leaves the owner intact), that
overdose produces fragmentary identity emission, that only gradient-family directions
restore function to the (itself non-functional) cross-task centroid, and that the
entire construction is *frame-local*: deterministic within an initialization frame,
while trained solutions across seeds share only 0.14 weight-space cosine. Prior work
proves task vectors approximate first-epoch gradients (arXiv:2508.16082) and shows a
lone one-step gradient update can match trained LoRA on small classification tasks
(LoRA-One, arXiv:2502.01235, Table 2); to our knowledge no prior work performs the
library-plus-gradient construction, identifies the alignment threshold, or maps the
override/dose/coherence physics.

## 1. Setup

Base: a 9B instruction-tuned causal LM, NF4-quantized, frozen. Adapters: LoRA r=4,
α=8, on layers 20–31 MLP up/down projections (24 modules, 1.57M params). Task
family: 12 (+3 held-out-entirely) US large-cap stock tickers; training data pairs 24
target-free carrier prompts ("Run the market check.") with the completion
`CALL: stock_quote("<T>")` — a protocol the base emits 0/8 (control). Fire gate:
single-shot greedy, exact-prefix match, 8 held-out carriers. Seed pinned across
tasks (7102; replication at 3141), so per-task differences are attributable to the
task tokens alone. All experiments preregistered with executable gates
(`preregs/`); every number below is in `results/`.

## 2. Population geometry: trunk and branches

A mean-field control predicts a held-out task's training *steps* at cosine 0.95
(vs 0.52 for a fitted per-task ridge — the fitted predictor loses to the trivial
centroid), so the predictable component of finetuning is a shared trunk. The
residual per-task branch carries ~20% of step norm with pairwise cosines at the
independence reference (−1/11). The branch's *direction* is 77% set by training
step 5 and 95% by step 20 (of 120) — finetuning chooses the task-specific direction
almost immediately, then only grows it.

## 3. The construction and the threshold

The gradient at initialization, trunk-removed (against the library's mean
gradient), aligns with the trained branch at cos 0.38–0.53 (B-subspace; null p95
0.083; per-task values reproduce across seeds to the third decimal). Constructed
adapters — trunk + β·unit(gradient branch) — fired 72/96 across 12 leave-one-out
tasks and 24/24 (trigger regime) on 3 fully-novel tasks; at adequate token budget
the canonical regime pools 18/24 vs a 7/8 directly-trained ceiling. Fire is a step
function of alignment (onset bracketed 0.4255–0.4485 with n=1 per edge); the three
sub-threshold tasks are exactly the three lowest alignments, and their floor is
structural: 2× gradient data moves alignment −0.006 (flat). Power gates (trained
ceiling; oracle = true branch direction at the same norm, 96/96) bound the
dynamic range; wrong-gradient arms transport the *injected* identity (80/96), never
the target's.

## 3.1 The honest benchmark: three routes to an untrained target

Our construction is not the only route to an untrained target on this task family, and
the comparison that matters is cost-per-target, not just rate. On the collaborating
bench (activation space, Llama-3.1-8B, same task, overlapping entity set):

| route | per-target cost | rate | conditions |
|---|---|---|---|
| gradient walk (their Z119/Z129) | 18 optimizer steps | ~80% | cheap TRAINING, not training-free |
| fixed QP edit (their Z252) | one solve, no training | 3/6 | in-family only, requires a trained donor seam |
| **this work** | **one gradient evaluation, no training** | **75%** | in-family, requires a library of other tasks' adapters |

The walk is the rate to beat and it beats us — but it trains. The QP edit is the nearest
zero-training neighbour and is the honest comparator for our claim; it is activation-space,
needs a same-family trained donor, and reaches half our rate on its own bench. What this
work adds is the zero-training point at competitive rate from a library plus a single
gradient, with a threshold that says in advance which targets will fail. Any reader
comparing rates without comparing per-target cost will draw the wrong conclusion.

## 3.2 A convergence, and a falsifier that has not yet been run correctly

The collaborating bench's oracle arm — run for another purpose, reported as a power miss
against their own bar — shows a dose structure resembling ours: 8/8 at 0.83 of the
target's own branch norm versus 3/8 at 0.80, dose-steep with per-identity tolerance.
Their firing geometry is thin and disjoint: two same-ticker coordinates each in their own
pocket, with a dead zone between them (fire 8, 5, 1, 0, 0, 0, 0, 8, 8 along the path).
Tempting identification: their basin and our alignment threshold, one law in two
coordinate systems.

**Our first attempt at the falsifier was mis-specified, and we report it as such.** We
walked the geodesic between two solutions in the SAME initialization frame (the computed
branch and the true branch, cos 0.48 apart) and found 21/21 cells firing with no
collapse. But their disjointness is explicitly per-(specific, SEED) — their two endpoints
come from different seeds. Same-frame connectivity and cross-frame disconnection are
different quantities, and the mis-match runs against us twice: our own frame-locality
result *predicts* same-frame connectivity, so the observation is consistent with the
identification rather than evidence against it; and their pockets are extended along
their own native axis and thin only transversally, so a same-frame path can travel along
the extended direction and find no collapse even if the cross-frame structure is exactly
as disjoint as theirs. The pre-committed "smooth ⇒ refuted" branch therefore does not
fire: this test cannot discriminate. **Disposition: UNDETERMINED-on-design.**

The matched test walks two solutions from DIFFERENT initialization frames (independent
seeds), same target — the exact contrast they ran. Fire throughout ⇒ identification
refuted and the object classes genuinely differ. Collapse in the middle, as theirs does
⇒ the two quantities survive as candidates for one law. Result in §3.3a.

## 4. Failure physics: override, dose, coherence

**Override.** Injected onto a *fully-trained* adapter, a foreign gradient branch at
matched norm fires the foreign identity 46/48 with the resident owner at zero;
random directions at the same norm leave the owner at 8/8. Identity replacement is
direction-specific, not norm-disruption.

**Dose.** Sweeping injection strength: below ~0.6β the owner rules untouched; a
window (≈0.8–1.0β, pair-dependent) gives clean replacement; above it, *fragmentary
identity emission* appears (e.g., `stock_quote("NVNV`) — an interference phenomenon,
never observed at intermediate dose. To our knowledge the first dose–response curve
for weight-space identity corruption.

**Coherence.** The cross-task centroid alone is non-functional (degenerate output —
contrast same-task centroids, which are known to work, arXiv:2302.04863). At
matched norm, random directions and semantically-structured directions (unembedding
rows mapped into the adapter B-space) do not restore function; gradient-family
directions do — direct behavioral counter-evidence to norm-not-basis accounts of
perturbation survival.

## 5. Frame-locality and factorization

Re-running the entire pipeline at a second seed: the alignment band reproduces
(mean 0.467, all above null; the lowest-aligned task reproduces to the third
decimal) and bridges fire 5/6 — *per-frame determinism*. Yet trained solutions
across seeds share only 0.14 weight-space cosine (exact rank-4 Gram computation),
and even trunks share only 0.20 while transferring behaviorally. The gradient
computes a *frame-local name*: valid in the initialization frame it was computed in,
not a frame-independent object — which reconciles our construction with the
frame-external gradient result on a collaborating bench (their ALIGNMENT null
stands at the geometry tier, 0.027-0.041 vs null p95 0.044-0.062, with split-half
self-consistency 0.65-0.70 showing the gradient there is stable but misdirected;
their FIRE panel was voided by a power miss and never re-run at power, so the
activation-space side is untested-at-power rather than closed), and matches the
lazy-regime expectation (arXiv:2210.05643, 2305.12827) that finetuning deltas are
functions of init-frame gradients. The name also factorizes from the behavioral
program: branches computed under one training regime drive adapters of another
(24/24 in the easier direction; 0.734 cross-regime branch cosine), enabling a
speak/act-dissociation demonstration — the model answers an unrelated question,
then acts on the injected never-trained identity (live demo in `code/`).

## 6. Limitations

Single substrate and model; n=8 prompts/task; in-family targets only (the library's
task family); a per-task alignment floor excludes ~25% of in-family targets;
CANDIDATE naming until external reproduction. The protocol task is deliberately
narrow (identity selection), not general capability. Training-band effects
(act-append reliability decays beyond the trained response-length band) bound the
canonical regime and appear substrate-invariant in our cross-bench comparisons.

## 7. Related work (the carve-outs, stated plainly)

Task vectors ≈ first-epoch gradients is *proved* by arXiv:2508.16082 — for trained
models, full-norm, without construction. LoRA-One (2502.01235) shows a lone
one-step gradient update can match trained LoRA on small GLUE tasks — single-task,
no library, no threshold, classification; its main algorithm always trains.
GradFix (2510.09658) uses target gradients to filter an *already-trained* task
vector. Trainless composition (LoraHub 2307.13269) composes existing adapters with
no target gradient; hypernetworks (Text-to-LoRA, Drag-and-Drop) amortize training
into generators. The empty cell — target gradient at init + other tasks' adapters +
no target training, with functioning behavior and a success threshold — is what
this report fills. Full map: `PRIOR_ART_POSITIONING.md`.
