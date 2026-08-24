# Constructing Functioning Task Adapters from a Single Gradient at Initialization

Draft technical report, for maintainer review. CANDIDATE tier: preregistered,
controlled, two seeds, one substrate, one bench, not externally reproduced.

## Abstract

A functioning LoRA adapter for a never-trained task can be assembled from two
ingredients: the mean weight delta of adapters trained on other tasks (a "trunk"), and
one full-batch gradient of the target task's loss evaluated at the untrained
initialization, trunk-removed and injected at the library's typical task-specific norm.
On a synthetic tool-invocation protocol the base model does not perform, constructed
adapters fire the correct never-trained target on 75% of held-out prompts, with
matched-norm random and trunk-only controls at 0%. No generation in the constructed arm
emitted a different valid target: of the 24 non-fires, 23 produced no call and 1 produced
a fragmentary string. Success is gated by a threshold in the gradient's
alignment to the counterfactual trained solution, with onset between 0.43 and 0.45. The
25% of tasks below the threshold are curvature-limited: doubling the gradient's data does
not move them. The gradient direction also overrides fully trained resident behavior
inside a dose window (foreign identity 46/48, owner zero, while a random direction at the
same norm leaves the owner intact); overdose produces fragmentary identity emission; only
gradient-family directions restore function to the cross-task centroid, which is itself
non-functional; and the construction is frame-local, deterministic within an
initialization frame while trained solutions across seeds share 0.14 weight-space cosine.
Task vectors have been shown to approximate first-epoch gradients (arXiv:2508.16082), and
a one-step gradient update has been shown to match trained LoRA on small classification
tasks (LoRA-One, arXiv:2502.01235, Table 2). This report measures the library-plus-gradient
construction on a generative task, the alignment threshold, and the override, dose, and
coherence behavior.

## 1. Setup

Base: a 9B instruction-tuned causal LM, NF4-quantized, frozen. Adapters: LoRA r=4,
alpha=8, on layers 20 to 31 MLP up and down projections (24 modules, 1.57M params). Task
family: 12 US large-cap stock tickers, plus 3 held out entirely. Training data pairs 24
target-free carrier prompts ("Run the market check.") with the completion
`CALL: stock_quote("<T>")`, a protocol the base emits 0/8 (control). Fire gate:
single-shot greedy, exact-prefix match, 8 held-out carriers. The seed is pinned across
tasks (7102, with replication at 3141), so per-task differences are attributable to the
task tokens alone. All experiments are preregistered with executable gates (`preregs/`);
every number below is in `results/`.

## 2. Population geometry: trunk and branches

A mean-field control predicts a held-out task's training steps at cosine 0.95, against
0.52 for a fitted per-task ridge, so the fitted predictor loses to the trivial centroid
and the predictable component of finetuning is a shared trunk. The residual per-task
branch carries about 20% of step norm (mean 0.206) with pairwise cosines near the
independence reference of -1/11 (measured mean -0.084).

The branch direction locks early. The cosine between the cumulative branch and its own
final direction is 0.415 at training step 1, 0.769 at step 5, 0.947 at step 20, and 0.991
at step 40, out of 120 (`results/probe_grad_results.json`). Finetuning selects the
task-specific direction almost immediately and then mostly grows it, which is the property
a construction at initialization depends on.

## 3. The construction and the threshold

The gradient at initialization, trunk-removed against the library's mean gradient, aligns
with the trained branch at cosine 0.38 to 0.53 in the B-subspace (null p95 0.083;
per-task values reproduce across seeds to the third decimal). Constructed adapters, trunk
plus beta times unit(gradient branch), fired 72/96 across 12 leave-one-out tasks and
24/24 in the trigger regime on 3 fully novel tasks. At adequate token budget the canonical
regime pools 18/24 against a 7/8 directly trained ceiling. Firing is a step function of
alignment, with onset bracketed between 0.4255 and 0.4485 at one data point per edge. The
three sub-threshold tasks are the three lowest alignments, and their floor is not a
sampling limit: doubling the gradient data from 24 to 48 examples moved alignment down in
all four tasks tested, by 0.006 to 0.033, and improved none. Power gates (ceiling 96/96;
oracle using the true branch direction at the same norm, 96/96) bound the dynamic range.
Wrong-gradient arms transport the injected identity (80/96) and never the target's (0/96).

Failure is silence rather than misattribution. Across the 24 leave-one-out non-fires, 23
produced no call and 1 produced a fragmentary string (`NVNV`, the same interference
signature seen at overdose in section 4). No constructed adapter emitted a different valid
ticker.

## 3.1 Three routes to an untrained target

This construction is not the only route to an untrained target on this task family, and
the comparison that matters is cost per target, not rate alone. On the collaborating bench
(activation space, Llama-3.1-8B, same task, overlapping entity set):

| route | per-target cost | rate | conditions |
|---|---|---|---|
| gradient walk (their Z119/Z129) | 18 optimizer steps | about 80% | cheap training, not training-free |
| fixed QP edit (their Z252) | one solve, no training | 3/6 | in-family only, requires a trained donor seam |
| this work | one gradient evaluation, no training | 75% | in-family, requires a library of other tasks' adapters |

The walk has the highest rate, and it trains. The QP edit is the nearest zero-training
neighbour and is the appropriate comparator for the claim here; it is activation-space,
needs a same-family trained donor, and reaches half this rate on its own bench. What this
work adds is the zero-training point at comparable rate from a library plus a single
gradient, with a threshold that says in advance which targets will fail. Comparing rates
without comparing per-target cost gives the wrong conclusion.

## 3.2 A convergence, and a falsifier that was first run wrong

The collaborating bench's oracle arm, run for another purpose and reported as a power miss
against their own bar, shows a dose structure resembling this one: 8/8 at 0.83 of the
target's own branch norm against 3/8 at 0.80, dose-steep with per-identity tolerance.
Their firing geometry is thin and disjoint: two same-ticker coordinates each in its own
pocket with a dead zone between them, reading 8, 5, 1, 0, 0, 0, 0, 8, 8 along the path.
The tempting identification is that their basin and this alignment threshold are one law
in two coordinate systems.

The first attempt at the falsifier was mis-specified. It walked the geodesic between two
solutions in the SAME initialization frame (the computed branch and the true branch,
cosine 0.48 apart) and found 21/21 cells firing with no collapse. But their disjointness
is explicitly per (specific, seed): their two endpoints come from different seeds.
Same-frame connectivity and cross-frame disconnection are different quantities, and the
mismatch runs against the conclusion twice. The frame-locality result in section 5
predicts same-frame connectivity, so the observation is consistent with the identification
rather than evidence against it; and their pockets are extended along their own native
axis and thin only transversally, so a same-frame path can travel along the extended
direction and find no collapse even if the cross-frame structure is exactly as disjoint as
theirs. The pre-committed "smooth means refuted" branch therefore does not fire, because
the test cannot discriminate. Disposition: UNDETERMINED-on-design.

The matched test walks two solutions from DIFFERENT initialization frames (independent
seeds) on the same target, which is the contrast they ran. Fire throughout means the
identification is refuted and the object classes differ. Collapse in the middle, as theirs
shows, would leave the two quantities as candidates for one law. Result in section 3.3.

## 3.3 The matched cross-frame walk: identification refuted, object classes differ

The falsifier, re-run with the endpoints they used: same target, two DIFFERENT
initialization frames (seeds 7102 and 3141). LoRA parameters are not comparable across
frames, so the walk is performed in weight space, where interpolating two rank-4 deltas is
exactly rank-8:

```
(1-t)*dW1 + t*dW2 = scale * [sqrt(1-t)*B1 | sqrt(t)*B2] . [sqrt(1-t)*A1 ; sqrt(t)*A2]
```

This is injected into an r=8 adapter with no SVD and no approximation. Both endpoints are
trained adapters and serve as power controls.

| target | t=0 | 0.15 | 0.30 | 0.50 | 0.70 | 0.85 | t=1 |
|---|---|---|---|---|---|---|---|
| NVDA | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| KO | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| AAPL | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

21/21, endpoints valid, no collapse anywhere. Their structurally matched walk, same target
NVDA, seeds 7102 and 1234, reads 8, 5, 1, 0, 0, 0, 0, 8, 8, a dead zone across t = 0.3 to
0.75. Same contrast, opposite geometry.

The endpoints are far apart: cross-seed branch cosine 0.127 to 0.160 and cross-seed trunk
cosine 0.200 (banked, exact rank-4 Gram), so the walk crosses between near-orthogonal
solutions. The midpoint sits at roughly 77% of endpoint norm, derived from those cosines,
inside the 0.75 to 1.1 firing window they report, and it fires where theirs does not.

On a design that could have gone either way, the identification of this alignment threshold
with their basin is refuted, and the object classes differ in the geometry of their
solution sets. In activation space, same-target solutions from different seeds occupy
disjoint pockets separated by dead zones. In weight space, same-target solutions from
different seeds are connected: every convex mixture fires, despite the solutions being
near-orthogonal.

This also sharpens a result both benches share. The centroid of different-target solutions
is dead on both (their canon gate 0/6; the trunk here is degenerate stutter). The mean of
same-target, different-seed solutions is dead in activation space and alive in weight
space. The dead-centroid phenomenon is about crossing targets rather than about averaging,
and the two object classes part company on the same-target case.

Method note: the first attempt at this falsifier walked two same-frame solutions, found
21/21, and was briefly reported as refuting the identification. That test could not
discriminate, for the reasons in section 3.2. The collaborating bench caught the
mis-specification and argued against the premature concession. The first walk is retained
in `results/cellP_pockets.json` as an instrument record.

## 4. Override, dose, and coherence

Override. Injected onto a fully trained adapter, a foreign gradient branch at matched norm
fires the foreign identity 46/48 across six ordered pairs, with the resident owner at zero
in every generation. The two non-foreign generations are fragmentary (`NVNV`) rather than
owner emissions. Random directions at the same norm leave the owner intact, 8/8 in both
control pairs. Identity replacement is therefore direction-specific rather than norm
disruption.

Dose. Injection strength was swept over two ordered pairs at 0.6, 0.8, 0.9, 1.0 and 1.1
beta, 8 carriers each. At 0.6 beta the owner is untouched in both pairs (8/8 owner). Clean
replacement arrives at 0.8 beta in one pair and at 0.9 beta in the other, with the
intermediate 0.8 cell in that second pair split 5 owner to 3 foreign. Fragmentary emission
(for example `stock_quote("NVNV`) appears in one pair only, at 1.0 beta (2/8) and 1.1 beta
(5/8), and does not appear in the other pair at any dose. The upper edge is therefore
pair-dependent, and no fragmentary output was seen below 1.0 beta.

Coherence. The cross-task centroid alone is non-functional, producing degenerate output.
Same-task centroids are reported to work (arXiv:2302.04863), so the condition here is
crossing tasks. At matched norm, random directions and semantically structured directions
(unembedding rows mapped into the adapter B-space) do not restore function, and
gradient-family directions do. The direction family matters, not the norm alone.

## 5. Frame-locality and factorization

Re-running the entire pipeline at a second seed: the alignment band reproduces (mean
0.467, all above null, with the lowest-aligned task reproducing to the third decimal) and
five of six constructed bridges fire 8/8, with the lowest-aligned task (XOM) at 3/8. That
is per-frame determinism. Trained solutions across seeds
nonetheless share only 0.14 weight-space cosine (exact rank-4 Gram computation), and even
trunks share 0.20 while transferring behaviorally. The gradient computes a frame-local
name: valid in the initialization frame it was computed in, not a frame-independent
object.

This reconciles the construction with the frame-external gradient result on a
collaborating bench. Their ALIGNMENT null stands at the geometry tier (0.027 to 0.041
against null p95 0.044 to 0.062, with split-half self-consistency 0.65 to 0.70, so the
gradient there is stable but misdirected); their FIRE panel was voided by a power miss and
never re-run at power, so the activation-space side is untested-at-power rather than
closed. It also matches the lazy-regime expectation (arXiv:2210.05643, arXiv:2305.12827)
that finetuning deltas are functions of init-frame gradients.

The name factorizes from the behavioral program: branches computed under one training
regime drive adapters of another (24/24 in the easier direction, 0.734 cross-regime branch
cosine). That supports a speak-then-act demonstration in which the model answers an
unrelated question and then acts on the injected never-trained identity. Live demo in
`code/`.

## 6. Limitations

Single substrate and model; n=8 prompts per task; in-family targets only, meaning the
library's task family; a per-task alignment floor excludes about 25% of in-family targets;
CANDIDATE naming until external reproduction. The protocol task is deliberately narrow,
covering identity selection rather than general capability. Training-band effects, where
act-append reliability decays beyond the trained response-length band, bound the canonical
regime and appear substrate-invariant in the cross-bench comparisons here.

## 7. Related work

Task vectors approximating first-epoch gradients is proved by arXiv:2508.16082, for
trained models, at full norm, without construction. LoRA-One (arXiv:2502.01235) shows a
one-step gradient update can match trained LoRA on small GLUE tasks: single task, no
library, no threshold, classification, and its main algorithm always trains. GradFix
(arXiv:2510.09658) uses target gradients to filter an already-trained task vector.
Trainless composition (LoraHub, arXiv:2307.13269) composes existing adapters with no
target gradient; hypernetworks (Text-to-LoRA, Drag-and-Drop) amortize training into
generators.

The configuration measured here is a target gradient at init combined with other tasks'
adapters and no target training, evaluated on a generative task, with a reported success
threshold. Full map: `PRIOR_ART_POSITIONING.md`.
