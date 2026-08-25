# gradient-bridges

Building a working LoRA adapter for a task that was never trained, using the mean of
other tasks' adapters plus one gradient evaluated at the untrained initialization.

All results are from one bench: RTX 2080 Ti (11 GB), a 9B instruction-tuned causal LM
("Ornith", the lab's resident model: a Qwen3.5-family 9B instruct checkpoint; the exact
Hub id and revision are pinned in `code/harness_common.py`) quantized to NF4 and frozen,
LoRA rank 4 (the cross-frame walk in the report uses rank 8,
where the interpolation is exact). The main panel is one seed; a partial replication at a
second seed covers 6 of the 12 tasks. Preregistered with executable gates. Status:
candidate. Reproduced on a second model family (Llama-3.1-8B, section below); still one
lab, so external replication remains open.

## The construction

For a target task T with no trained artifact:

```
adapter(T) = trunk + beta * unit_B( grad L_T(theta_0) - mean_grad_others )
```

* `trunk` is the mean weight delta of adapters trained on other tasks. It contains no
  information about T.
* `grad L_T(theta_0)` is one full-batch gradient of T's loss at the untrained
  initialization, with the frozen quantized base and the shared LoRA init. T's training
  text is generated from a template.
* `beta` is the typical task-specific ("branch") norm across the library.

"Never trained" refers to the construction, not to the study. For the 12 leave-one-out
tasks a trained adapter does exist, and is used only to define the ceiling and the oracle
and to measure alignment after the fact; it is excluded from the trunk and is never an
input to the constructed adapter. For the 3 novel tasks no adapter was trained at all.

## What was measured

The task is a synthetic tool-call protocol the base model did not produce in 8 greedy
single-shot control generations.

| arm | result |
|-----|--------|
| directly trained, 12 tasks (ceiling) | 96/96 |
| oracle, true branch direction at the same norm | 96/96 |
| constructed, 12 leave-one-out tasks | 72/96 |
| matched-norm random direction, same subspace | 0/96 |
| trunk alone | 0/96 |
| wrong-gradient, target emitted | 0/96 |
| constructed, a ticker other than the target emitted | 0/96 |

On 3 tasks with no trained adapter anywhere, the constructed adapter fires 24/24 in the
trigger regime. In the speak-then-act regime at a 400-token budget it pools 18/24 against
a 7/8 directly trained ceiling for one task under the same cap.

Failure is degenerate rather than misdirected. Of the 24 leave-one-out non-fires, 23 emit
a repeating malformed scaffold (`CALL: stock_quote(" stock_quote(" ...`), which is the
same signature the trunk-only arm produces, and 1 emits the fragment `NVNV`. None emits a
different valid ticker. The wrong-gradient arm is the deliberate contrast: it never emits
the target and emits the injected ticker instead, 80/96.

## Findings

### Population geometry

Trained adapters decompose into a shared trunk and near-orthogonal per-task branches. A
mean-field control predicts a held-out task's training steps at cosine 0.95, against 0.52
for a fitted per-task ridge, so the fitted predictor loses to the trivial centroid. The
per-task branch carries about 20% of step norm (mean 0.206) with pairwise cosines near the
independence reference of -1/11 (measured -0.084). The branch direction locks early: its
cosine with its own final direction is 0.769 at training step 5, 0.947 at step 20, and
0.991 at step 40, out of 120.

### Alignment predicts firing, with a caveat at the margin

Gradient-to-branch alignment in the B-subspace, against constructed fire rate, all 12
tasks:

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

Null p95 for alignment is 0.083, so every task clears the null. The three lowest
alignments hold the three lowest fire rates (Pearson 0.85; Spearman 0.34, blunted by the
nine-way tie at 8/8). Onset is bracketed by JPM at 0.4255 and WFC at 0.4485, one task per
edge, so this is a bracket rather than a fitted threshold. NVDA at 0.5260 contributes the
one `NVNV` fragment despite sitting second from the top, so alignment orders the tasks but
does not account for every individual generation.

Alignment is also not the only variable near the edge. JPM fires 1/8 in its own clean
leave-one-out trunk and 7/8 in GOOGL's trunk, which carries a 1/11 trace of JPM's own
trained branch. At fixed alignment, trunk composition changed the outcome.

Doubling the gradient's carrier set from 24 to 48 lowered alignment in all four tasks
tested (-0.006 to -0.033) and raised none. One of the four, DIS at -0.033, exceeded the
pre-set 0.03 sanity bar, consistent with a distribution-shift bias of that size. Firing
was not re-tested at 48 carriers. On this evidence the floor is not a sampling limit;
attributing it to curvature is an interpretation rather than a measurement.

### Override

A foreign task's gradient branch at matched norm, injected onto a fully trained adapter,
produces the foreign identity 46/48 across six ordered pairs, with the resident owner at
zero in every generation. The other 2 generations are fragmentary rather than owner. Two
matched-norm random control cells leave the owner at 8/8 each, so replacement tracks the
direction rather than the added norm.

### Dose

Two body and foreign pairs, five doses, 8 prompts each. At 0.6 beta, the lowest dose
tested, the owner is untouched in both pairs. The clean-replacement window is
pair-dependent: 0.8 to 0.9 beta for KO with an NVDA branch, 0.9 to 1.1 for DIS with a KO
branch, with the latter in transition at 0.8 (5 owner, 3 foreign). Fragmentary emission
appears at and above 1.0 beta in the first pair (2/8 then 5/8) and at no dose in the
second.

### Coherence depends on the direction family

The centroid of cross-task adapters is itself non-functional. At matched norm, random
directions and unembedding-derived directions do not restore function, and
gradient-derived directions do. Same-task centroids are reported to work elsewhere
(arXiv:2302.04863), so the condition here is crossing tasks.

### Frame locality

Rerunning the construction at a second seed on 6 of the 12 tasks, with a 5-task trunk,
reproduces the band but not the values. Per-task alignment spans 0.399 to 0.501 (mean
0.467) against 0.395 to 0.533 (mean 0.498) for the same six tasks at the first seed.
Five of the six move by 0.018 to 0.050; only XOM lands within 0.005. The second seed's
null p95 is 0.162, twice the first seed's, so the margin over null is smaller there. The
ordering is preserved, with the lowest-aligned task lowest again. Five of six constructed
bridges fire 8/8 and the lowest-aligned fires 3/8.

Across seeds, matched task branches share 0.144 weight-space cosine (range 0.127 to 0.160)
against a null p95 of 0.041, and trunks share 0.200. Interpolating two same-task solutions
from different seeds fires at every point tested (21/21), so in this object class those
solutions are connected. The gradient therefore computes something valid in the
initialization frame it was computed in, rather than a frame-independent object.

### The construction ports to a second model family

The identical battery, run on Llama-3.1-8B-Instruct (NF4, rank 4, layers 20 to 31, seed
7102, a rented L4) with the harness inlined and no dependency on the original bench
(`code/cs1_v9_colab.py`, `results/cs1_v9_llama.json`). 22 minutes end to end.

| arm | Ornith 9B | Llama-3.1-8B |
|---|---|---|
| base null | 0/96 | 0/96 |
| trained ceiling | 96/96 | 96/96 |
| oracle | 96/96 | 96/96 |
| trunk alone | 0/96 | 0/96 |
| matched-norm random | 0/96 | 0/96 |
| constructed | 72/96 | **96/96** |

The constructed arm emitted the correct ticker in all 96 generations, with both controls
dead. The pre-committed V9 bar was pooled >= 28/96 with >= 3 of 12 tasks firing.

The difference in constructed rate has a candidate explanation the threshold model
supplies. On Ornith the alignment band was 0.383 to 0.533 with onset bracketed at 0.4255
to 0.4485, and the three tasks below the bracket failed. On Llama the alignment band is
0.454 to 0.590 (null p95 0.111): every task sits above the Ornith-measured bracket, and
every task fires. One seed, one quantization, a reconstructed task, and the bracket is
carried across substrates as a modelled reading rather than remeasured, so this is a
consistency observation, not a confirmed law.

### What the trunk is made of: literal output overlap, measured as a curve

The two diffusion experiments (in the lab record, not this repo) found task families with
no shared trunk at all, which raised a sharp question about the finding above: is the trunk
"the shared program" in a semantic sense, or the gradient of the literally shared part of
the target output? This cell varies the shared fraction directly: four families of four
tasks, targets of 12 single-token words with a shared prefix of K words, everything else
pinned to the validated trainer. The overlap axis is measured under the tokenizer on the
supervised stream, so the floor at K=0 (0.138, from two closing template tokens every task
shares) is measured rather than assumed.

| shared words K | measured overlap | mean pairwise delta cos | init-gradient prediction |
|---|---|---|---|
| 0 | 0.138 | 0.319 | 0.234 |
| 4 | 0.429 | 0.551 | 0.642 |
| 8 | 0.714 | 0.708 | 0.894 |
| 11 | 0.929 | 0.907 | 0.960 |

Three findings. First, overlap is the dominant driver: the curve is strictly monotone and
spans 0.319 to 0.907, and the K=11 family, which shares everything except one word, lands
at 0.9075 against the original ticker family's 0.9077 with entirely different words and no
financial semantics. The trunk share is set by how much of the output is literally shared.

Second, there is a measured baseline of task-agnostic sharing: at effectively zero content
overlap the deltas still share cos 0.319, exceeding the overlap floor by 0.08 with a tight
per-pair spread (0.30 to 0.33), and most of it is already present in the init-gradient
geometry (0.234). Shared carriers and the shared work of collapsing onto a memorized string
are candidates; a pre-registered disjoint-prefix arm separates structure from literal
sharing and has not yet run. The verdict under the pre-committed bars is therefore
INTERMEDIATE rather than SUPPORTED: overlap drives the trunk but is not all of it.

Third, the strict mechanistic identity fails: the trained trunk's cosine with the shared
part's descent gradient at init is only 0.28 to 0.29 for K >= 4 (stored gradients are the
raw ascent direction; the sign is flipped here for reading). The trunk is reached through
the shared-output gradient but grows beyond it, the same relation the per-task branch has
to its own init gradient (0.38 to 0.53 in the panel above). And the init-gradient
prediction overshoots at high overlap (0.894 predicted, 0.708 measured at K=8), so
training decorrelates deltas relative to their initial gradients as the unique parts are
learned.

Scope: 4 tasks per level, one seed, one training order, shared-prefix overlap only. All 16
tasks reached 8/8 on held-out carriers, the trainer was revalidated in-scan against a
banked manifest (relative error 0.00000), and the base emitted none of the 16 targets.

### The construction is also a better place to start training

B1 measured the cold-start cost: from the standard LoRA init, checkpoints reach the fire bar
at step 20 of 120. B2 asks whether initializing at the constructed adapter beats that, and
whether any advantage is the gradient branch or just the library trunk.

Steps to the bar (11 of 12 tasks at 7/8 or better, pooled 88/96, monotone):

| init | steps | initial held-out NLL |
|---|---|---|
| standard LoRA init (cold) | 20 | n/a |
| trunk alone | 4 | 1.934 |
| trunk plus matched-norm random direction | 4 | 1.936 |
| **trunk plus gradient branch (the construction)** | **1** | **0.135** |

The library trunk alone cuts 20 steps to 4. Adding the gradient branch cuts 4 to 1.

The comparison that carries the claim is against the trunk, not against the cold start,
because trunk and construction are norm-matched to 3.6% while the construction starts at 99%
of the fully trained radius and a cold start does not. At that matched radius the
construction's initial loss is 14.3x lower than the trunk's, which is a statement about
direction rather than distance. A matched-norm random direction moves the loss by 0.10%, so
the effect is specific to the gradient direction and not to perturbing the trunk.

On the pre-committed statistic, a paired sign test on held-out NLL across the ladder, the
construction beats the trunk on a median of 10 of 12 tasks (one-sided p <= 0.019). The
structure behind that median is worth stating: at rungs 0, 1, 2 and 4 it wins on 12 of 12,
with median NLL gaps of 1.71, 0.62, 0.023 and 0.0001. The sign test only wanders at later
rungs, after both arms have converged and the differences are of order 1e-4.

Gates: the reimplemented trainer reproduces the banked run to 4e-06 on step-0 loss and 6e-08
on the first parameter step; the construction rebuild reproduces the banked 72/96 panel with
zero deviation on all 12 tasks; trunk-only and random-only both fire 0/96 before training; no
gradient-scaler skips in any arm.

### The direction curve is a stopping rule, once norm is accounted for

The branch direction locking early is a claim about direction, not function. Replaying the
banked checkpoints through the fire gate tests whether it predicts behaviour. No retraining
is involved: the trajectory is stored at every step, so a truncated adapter is the same
optimizer run stopped early.

Raw truncated checkpoints, 12 tasks, 8 held-out carriers each:

| step | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 32 | 48 | 64 | 80 | 96 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fires | 0/96 | 0/96 | 0/96 | 2/96 | 76/96 | 96/96 | 96/96 | 96/96 | 96/96 | 96/96 | 96/96 | 96/96 | 96/96 |

Every checkpoint from step 20 onward fires 96/96, all 12 tasks at 8/8, discording with the
fully trained ceiling on 0 of 12 tasks. That is an exact 6.00x reduction in optimizer steps
for this configuration. The pre-committed bar was 11 of 12 tasks at 7/8 or better and
pooled 88/96, evaluated with monotone sufficiency so a single lucky rung cannot set the
result; step 20 and every rung above it clear it.

Direction alone does not explain this. At step 5 the direction is already 0.769 of its
final value and raw checkpoints there fire 0/96. What is missing is magnitude. Restoring
the mature trunk and rescaling the step-k branch to its final norm gives 64/96 at step 4
and 94/96 at step 16, against 0/96 and 76/96 raw. Restoring the trunk alone, without the
rescale, gives 0/96 at step 4 and 19/96 at step 8, so both the trunk and the branch
magnitude contribute and neither accounts for the gap by itself. The rescaled arm is an
oracle diagnostic, since the final norm is not knowable at step k, and no training saving
is claimed from it.

Two caveats, both pre-registered. Step 20 is 0.83 epochs over the 24-example training set,
so this is a statement about data coverage as much as step count; the pre-committed
discriminator for separating the two was an oracle-arm pass at step 6 or earlier, and it was
not met. And the shortfall at step 16 is a carrier effect rather than task difficulty: 20 of
its 20 misses come from just two of the eight held-out carriers, with the other six perfect
across all 12 tasks.

Controls: the untrained init fires 0/96, the trunk alone fires 0/96, and the decomposition
reconstructs the trained delta to 4.7e-10.

### Identity separates from the behavioral program

Portability across training regimes is asymmetric. A dissociation-regime branch drives the
trigger program 24/24; a trigger-regime branch drives the dissociation program 9/24,
against an 18/24 same-regime reference in that program. Cross-regime branch cosine is
0.734 (range 0.574 to 0.879, null p95 0.173). A speak-then-act regime shows the
separation directly: the model answers an unrelated question, then emits the call naming
the never-trained target. Runnable in a browser via `code/gpu_worker.py`.

## Scope and limitations

* One model, one substrate, 8 prompts per task, 12 library tasks.
* Targets are in the same family as the library. Out-of-family is untested here.
* Three of twelve in-family targets sit below the firing bracket and fail. Alignment ranks
  the tasks and separates those failures in this set, on one data point per bracket edge.
* The construction is training-free with respect to the target only. The trunk comes from
  other tasks' training.
* The second-seed replication covers 6 tasks with a smaller trunk, not the full pipeline.
* Reliability decays on answers longer than the trained response-length band, which bounds
  both the speak-then-act rate and the trained ceiling.
* The 18/24 speak-then-act figure is a corrected number. The preregistered 220-token cap
  gave 13/24 because it truncated long answers before the appended call. Re-scored at 400
  tokens the rate is 18/24, and the five disputed rows were re-run at 1024 tokens, where
  all terminated on end-of-sequence without a call, confirming them as genuine omissions.
* The wrong-gradient trunks contain a 1/11 trace of the injected ticker's own trained
  branch, because leave-one-out excludes the body task and not the injectee. The
  constructed arm's trunk is clean. Across the 12 wrong-gradient cells, firing correlates
  with the injectee's own alignment at Pearson 0.80 and with trace norm at -0.57, the
  wrong sign for the trace to be doing the work, but the JPM pair above shows the trace can
  matter at the margin.

## Related work

LoRA-One (arXiv:2502.01235) proves that LoRA adapters align with singular subspaces of the
one-step full fine-tuning gradient, and its Table 2 reports that a one-step gradient
update, and its rank-8 approximation, can match trained LoRA on small GLUE classification
tasks with no subsequent training. This is the closest published result to the
construction here. arXiv:2508.16082 shows that a task vector from one epoch of finetuning
is exactly the negative gradient scaled by the learning rate, and that the first-epoch
gradient dominates the trajectory. GradFix (arXiv:2510.09658) transports a task vector
onto a different pre-trained model by masking it with the target model's gradient-sign
structure, with no fine-tuning, though the task knowledge is still trained on the source
model. LoraHub (arXiv:2307.13269) composes existing adapters with coefficients and no
gradients, and Text-to-LoRA (arXiv:2506.06105) emits an adapter from a text description
using a trained hypernetwork.

The configuration measured here combines a library trunk with a trunk-removed gradient
branch, on a generative task. `paper/PRIOR_ART_POSITIONING.md` records the full
comparison, including what the sweep covered and did not cover.

## Repository

* `code/` harness (model load, coordinate and readback gates, termination-state
  generation, gradient scale ladder), batching helpers, and every experiment as run
* `preregs/` preregistrations, with the date each was written
* `results/` raw JSON for every panel, plus `NARRATIVE.md`, the lab record including
  corrections, disclosed partial results, and instrument failures in the order they
  happened
* `paper/` technical report and prior-art positioning

## License

Code in `code/` is under the Apache License 2.0 (`LICENSE`). This README and the contents
of `paper/`, `preregs/`, and `results/` are under Creative Commons Attribution 4.0
International (`LICENSE-CC-BY-4.0.txt`), so reuse of the text, tables, and measurements
requires attribution.

Copyright 2026 Nick Lomeli.

## Reproducing

Scripts assume the base model, an 11 GB or larger NVIDIA GPU, and the paths in
`code/harness_common.py`. Every number measured on this bench is either stored in a JSON
under `results/` or summed from the per-task entries in one. Two exceptions are stated
where they appear: the base-model control is a pilot observation with no banked artifact,
and figures attributed to a companion bench are quoted from its record and are not
reproducible here. Training trajectories are 4.6 GB and are not included.
