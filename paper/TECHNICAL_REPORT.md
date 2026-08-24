# Constructing Functioning Task Adapters from a Single Gradient at Initialization

Draft technical report, for maintainer review. CANDIDATE tier: preregistered, controlled,
one substrate, one bench, partial second-seed replication, not externally reproduced.

## Abstract

A functioning LoRA adapter for a never-trained task can be assembled from two ingredients:
the mean weight delta of adapters trained on other tasks (a "trunk"), and one full-batch
gradient of the target task's loss evaluated at the untrained initialization, trunk-removed
and injected at the library's typical task-specific norm.

On a synthetic tool-invocation protocol the base model does not perform, constructed
adapters fire the correct never-trained target on 72 of 96 held-out prompts across 12
leave-one-out tasks. Matched-norm random directions and the trunk alone fire 0 of 96. No
generation in the constructed arm emitted a different valid target; the 24 non-fires are
23 degenerate scaffolds and 1 fragment.

Fire rate tracks the gradient's alignment to the counterfactual trained branch, with the
three lowest alignments holding the three lowest rates and onset bracketed between 0.4255
and 0.4485 by one task per edge. Alignment is not the only variable at the margin: at fixed
alignment, a trunk carrying a 1/11 trace of the target's own branch moved one task from 1/8
to 7/8. Doubling the gradient's data lowered alignment in all four tasks tested and raised
none, so the floor is not a sampling limit, though firing was not re-tested at the larger
carrier set.

The gradient direction also overrides fully trained resident behaviour inside a
pair-dependent dose window, producing the foreign identity 46 of 48 with the resident owner
at zero. Only gradient-family directions restore function to the cross-task centroid, which
is itself non-functional. The construction is frame-local: it reproduces as a band at a
second seed rather than value by value, while matched task branches across seeds share 0.144
weight-space cosine against a null p95 of 0.041.

Two published results anticipate parts of this. LoRA-One (arXiv:2502.01235, Table 2)
reports that a one-step gradient update with no subsequent training can match trained LoRA
on small classification tasks. arXiv:2508.16082 shows a one-epoch task vector is exactly
the negative gradient scaled by the learning rate. This report measures the
library-plus-gradient combination on a generative task, the alignment ordering and its
bracket, and the override, dose, and coherence behaviour.

## 1. Setup

Base: a 9B instruction-tuned causal LM, NF4-quantized, frozen. Adapters: LoRA r=4,
alpha=8, on layers 20 to 31 MLP up and down projections (24 modules, 1.57M params). The
cross-frame walk in section 3.3 uses r=8, where the interpolation of two rank-4 deltas is
exact.

Task family: 12 US large-cap stock tickers, plus 3 held out entirely. Training data pairs
24 target-free carrier prompts ("Run the market check.") with the completion
`CALL: stock_quote("<T>")`. The base model did not emit this protocol in 8 greedy
single-shot control generations; that control is a pilot observation and has no banked
artifact in `results/`.

"Never trained" is a property of the construction rather than of the study. For the 12
leave-one-out tasks a trained adapter exists and is used to define the ceiling and oracle
arms and to measure alignment after the fact, but it is excluded from the trunk and is
never an input to the constructed adapter. For the 3 novel tasks no adapter was trained.

Fire gate: single-shot greedy, exact-prefix match, 8 held-out carriers. The seed is pinned
across tasks (7102), so per-task differences are attributable to the task tokens alone. A
partial replication at seed 3141 covers 6 of the 12 tasks with a 5-task trunk. All
experiments are preregistered with executable gates (`preregs/`); every number measured on
this bench is in `results/` or summed from per-task entries there.

## 2. Population geometry: trunk and branches

In this task family, a mean-field control predicts a held-out task's training steps at
cosine 0.95, against 0.52 for a fitted per-task ridge. The fitted predictor loses to the
trivial centroid, so the predictable component of finetuning here is a shared trunk. The
residual per-task branch carries about 20% of step norm (mean 0.206) with pairwise cosines
near the independence reference of -1/11 (measured mean -0.084).

The branch direction is set early. The cosine between the cumulative branch and its own
final direction is 0.415 at training step 1, 0.769 at step 5, 0.947 at step 20, and 0.991
at step 40, out of 120 (`results/probe_grad_results.json`).

## 3. The construction and the alignment ordering

The gradient at initialization, trunk-removed against the library's mean gradient, aligns
with the trained branch at cosine 0.383 to 0.533 in the B-subspace (null p95 0.083).
Constructed adapters, trunk plus beta times unit(gradient branch), fired 72/96 across 12
leave-one-out tasks and 24/24 in the trigger regime on 3 tasks with no trained adapter
anywhere.

Per task, alignment against constructed fire rate:

| task | alignment | fires |
|---|---|---|
| BAC | 0.3834 | 0/8 |
| XOM | 0.3950 | 0/8 |
| JPM | 0.4255 | 1/8 |
| WFC | 0.4485 | 8/8 |
| GOOGL | 0.4622 | 8/8 |
| AAPL | 0.4874 | 8/8 |
| MSFT | 0.4999 | 8/8 |
| META | 0.5093 | 8/8 |
| KO | 0.5191 | 8/8 |
| NVDA | 0.5260 | 7/8 |
| AMZN | 0.5276 | 8/8 |
| DIS | 0.5331 | 8/8 |

Pearson 0.85, Spearman 0.34, the latter blunted by the nine-way tie at 8/8. Onset is
bracketed by JPM at 0.4255 and WFC at 0.4485, one task per edge, so this is a bracket and
not a fitted threshold. NVDA supplies the single fragment despite the second-highest
alignment, so alignment orders the tasks without accounting for every generation.

Alignment is also not the only variable near the edge. JPM fires 1/8 in its own clean
leave-one-out trunk and 7/8 in GOOGL's trunk, which carries a 1/11 trace of JPM's trained
branch (`results/posthoc_dose.json`). At fixed alignment, trunk composition changed the
outcome.

Doubling the gradient's carrier set from 24 to 48 lowered alignment in all four tasks
tested, by 0.006 to 0.033, and raised none. DIS at -0.033 exceeded the pre-set 0.03 sanity
bar, consistent with a distribution-shift bias of that size. Firing was not re-tested at 48
carriers, so the conclusion rests on the alignment proxy over 4 of 12 tasks. On this
evidence the floor is not a sampling limit; calling it curvature is an interpretation, not
a measurement.

Power gates bound the dynamic range: the directly trained ceiling is 96/96 and the oracle,
using the true branch direction at the same norm, is 96/96. Wrong-gradient arms transport
the injected identity 80/96 and never the target (0/96).

Failure is degenerate rather than misdirected. Of the 24 non-fires, 23 emit a repeating
malformed scaffold (`CALL: stock_quote(" stock_quote(" ...`), the same signature the
trunk-only arm produces, and 1 emits the fragment `NVNV`. No constructed adapter emitted a
different valid ticker.

The wrong-gradient arm carries a disclosed confound: its trunks contain a 1/11 trace of the
injected ticker's own trained branch, because leave-one-out excludes the body task and not
the injectee. The constructed arm's trunk is clean. Across the 12 wrong-gradient cells,
firing correlates with the injectee's own alignment at Pearson 0.80 and with trace norm at
-0.57, which is the wrong sign for the trace to be driving the result, but the JPM pair
above shows the trace can matter at the margin.

## 3.1 Three routes to an untrained target

This construction is not the only route to an untrained target on this task family. A
companion experiment in the same lab, on Llama-3.1-8B in activation space with an
overlapping entity set, produced two others. Those figures are quoted from its record; they
are unpublished and cannot be verified from this repository.

| route | per-target cost | rate | conditions |
|---|---|---|---|
| gradient walk (companion bench) | 18 optimizer steps | about 80% | cheap training, not training-free |
| fixed edit (companion bench) | one solve, no training | 3/6 | in-family only, requires a trained donor seam |
| this work | one gradient evaluation, no training | 72/96 | in-family, requires a library of other tasks' adapters |

The walk has the highest rate and requires 18 optimizer steps per target. The fixed edit is
the nearest zero-training comparator: it is activation-space, needs a same-family trained
donor, and fires 3 of 6 in-family targets on its own bench. With n=6 that rate has a Wilson
95% interval of roughly 0.19 to 0.81, which contains the 0.75 measured here, so the two are
not distinguishable on rate, and the substrates differ.

## 3.2 A convergence, and a falsifier that was first run wrong

The companion bench's oracle arm, run for another purpose and reported as a power miss
against its own bar, shows a dose structure resembling this one: 8/8 at 0.83 of the target's
own branch norm against 3/8 at 0.80. Its firing geometry is thin and disjoint, with two
same-ticker coordinates each in its own pocket and a dead zone between them, reading 8, 5,
1, 0, 0, 0, 0, 8, 8 along the path. The tempting identification is that its basin and the
alignment ordering here are the same phenomenon in two coordinate systems.

The first attempt at the falsifier was mis-specified. It walked the geodesic between two
solutions in the same initialization frame (the computed branch and the true branch, cosine
0.48 apart) and found 21/21 cells firing with no collapse. But the companion bench's
disjointness is per (specific, seed): its two endpoints come from different seeds.
Same-frame connectivity and cross-frame disconnection are different quantities. The
frame-locality result in section 5 predicts same-frame connectivity, so the observation was
consistent with the identification rather than evidence against it, and the companion
bench's pockets are extended along their native axis and thin only transversally, so a
same-frame path can travel the extended direction and meet no wall. The pre-committed
"smooth means refuted" branch therefore did not fire. Disposition: UNDETERMINED-on-design.
The first walk is retained in `results/cellP_pockets.json` as an instrument record.

The matched test walks two solutions from different initialization frames on the same
target, which is the endpoint structure the companion bench used (seeds 7102 and 3141 here;
7102 and 1234 there).

## 3.3 The matched cross-frame walk

LoRA parameters are not comparable across frames, so the walk is performed in weight space,
where interpolating two rank-4 deltas is exactly rank-8:

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

21/21, endpoints valid, no collapse anywhere. The endpoints are far apart: cross-seed
branch cosine 0.127 to 0.160 and cross-seed trunk cosine 0.200, banked from an exact rank-4
Gram computation. The companion bench's structurally matched walk reads 8, 5, 1, 0, 0, 0,
0, 8, 8, a dead zone across t = 0.3 to 0.75. The contrast is matched and the outcome is
opposite.

On this design the identification of the alignment ordering here with that bench's basin is
refuted, and the two object classes differ in the geometry of their solution sets. In
activation space, same-target solutions from different seeds occupied disjoint pockets. In
weight space, every convex mixture of two same-target solutions fires, despite their
branches being near-orthogonal. The midpoint norm was not measured directly; derived from
the endpoint cosines it is roughly 77% of endpoint norm, which is MODELLED rather than
measured.

This also sharpens a result both benches share. The centroid of different-target solutions
is dead on both, at 0/6 there and a degenerate scaffold here. The mean of same-target,
different-seed solutions is dead in activation space and alive in weight space, so the
dead-centroid phenomenon is about crossing targets rather than about averaging.

## 4. Override, dose, and coherence

**Override.** Injected onto a fully trained adapter, a foreign gradient branch at matched
norm fires the foreign identity 46/48 across six ordered pairs, with the resident owner at
zero in every generation. The two non-foreign generations are fragmentary (`NVNV`) rather
than owner emissions. Two matched-norm random control cells leave the owner at 8/8 each.
Replacement therefore tracks the direction rather than the added norm.

**Dose.** Two body and foreign pairs, five doses, 8 carriers each. At 0.6 beta, the lowest
dose tested, the owner is untouched in both pairs. The clean-replacement window is
pair-dependent: 0.8 to 0.9 beta for KO with an NVDA branch, and 0.9 to 1.1 for DIS with a
KO branch, with the latter split 5 owner to 3 foreign at 0.8. Fragmentary emission appears
at and above 1.0 beta in the first pair (2/8 then 5/8) and at no dose in the second. That
partials appear only at the top of the range in the pair that shows them is consistent with
interference rather than a boundary effect, on two pairs.

**Coherence.** The cross-task centroid alone is non-functional, producing a degenerate
scaffold. At matched norm, random directions and semantically structured directions
(unembedding rows mapped into the adapter B-space) do not restore function, and
gradient-family directions do. Same-task centroids are reported to work (arXiv:2302.04863),
so the condition here is crossing tasks.

## 5. Frame locality and factorization

Rerunning the construction at a second seed, on 6 of the 12 tasks and with a 5-task trunk,
reproduces the band but not the values. Per-task alignment spans 0.399 to 0.501 (mean
0.467) against 0.395 to 0.533 (mean 0.498) for the same six tasks at the first seed. Five
of the six move by 0.018 to 0.050 and only XOM lands within 0.005, so this is a
distributional replication, not a value-level one. The second seed's null p95 is 0.162,
about twice the first seed's 0.083, so the margin over null is smaller there. The ordering
survives, with the lowest-aligned task lowest again, and five of six constructed bridges
fire 8/8 with the lowest-aligned at 3/8.

Across seeds, matched task branches share 0.144 weight-space cosine (range 0.127 to 0.160)
against a null p95 of 0.041, and trunks share 0.200 while transferring behaviourally. The
full deltas, which are what the section 3.3 walk interpolates, were not measured for
cross-seed cosine. The gradient computes something valid in the initialization frame it was
computed in.

This is the expected shape under lazy-regime accounts (arXiv:2210.05643, arXiv:2305.12827)
in which finetuning deltas are functions of init-frame gradients. It also bears on the
companion bench's frame-external gradient result: its alignment null stands at the geometry
tier (0.027 to 0.041 against null p95 0.044 to 0.062, with split-half self-consistency 0.65
to 0.70, so the gradient there is stable but misdirected), while its fire panel was voided
by a power miss and never re-run at power, leaving that side untested-at-power rather than
closed.

Portability across training regimes is asymmetric. A dissociation-regime branch drives the
trigger program 24/24; a trigger-regime branch drives the dissociation program 9/24, against
an 18/24 same-regime reference in that program. Cross-regime branch cosine is 0.734 (range
0.574 to 0.879, null p95 0.173). A speak-then-act demonstration follows: the model answers
an unrelated question and then acts on the injected never-trained identity. Live demo in
`code/`.

## 6. Limitations

Single substrate and model; n=8 prompts per task; 12 library tasks; in-family targets only.
Three of twelve in-family targets fall below the firing bracket. The second-seed
replication covers 6 tasks with a smaller trunk and is not a rerun of the full pipeline.
CANDIDATE naming until external reproduction.

The protocol task is narrow, covering identity selection rather than general capability.

The 18/24 speak-then-act figure is a corrected number. The preregistered 220-token cap gave
13/24 because it truncated long answers before the appended call. Re-scored at 400 tokens
the rate is 18/24, and the five disputed rows were re-run at 1024 tokens, where all
terminated on end-of-sequence without a call, confirming them as genuine omissions. The
7/8 ceiling quoted alongside it is one task under the same cap.

Training-band effects, where act-append reliability decays beyond the trained
response-length band, bound both the speak-then-act rate and the trained ceiling. The same
shape appears on the one other substrate compared here, which is not enough to call it
substrate-invariant.

## 7. Related work

LoRA-One (arXiv:2502.01235) proves LoRA adapters align with singular subspaces of the
one-step full fine-tuning gradient, and its Table 2 evaluates a one-step gradient with no
subsequent training on small GLUE classification tasks, matching trained LoRA there. It is
the closest published result to this one. The configuration here differs in combining that
gradient with a trunk from other tasks' adapters, in removing the library mean gradient
before renormalizing, in the generative fire gate, and in reporting the per-task alignment
ordering.

arXiv:2508.16082 reports that a task vector from one epoch of finetuning is exactly the
negative gradient scaled by the learning rate, with the first-epoch gradient dominating the
trajectory, for trained models at full norm.

GradFix (arXiv:2510.09658) transports a task vector onto a different pre-trained model by
masking it with the target model's gradient-sign structure, using a few gradient
evaluations and no fine-tuning; the task knowledge is still trained on the source model.

LoraHub (arXiv:2307.13269) composes existing adapters with fitted coefficients and no
gradients. Text-to-LoRA (arXiv:2506.06105) emits an adapter from a text description via a
trained hypernetwork.

Full comparison, including the scope and limits of the search performed:
`PRIOR_ART_POSITIONING.md`.
