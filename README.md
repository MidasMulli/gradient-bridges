# gradient-bridges

Building a working LoRA adapter for a task that was never trained, using the mean of
other tasks' adapters plus one gradient evaluated at the untrained initialization.

All results are from one bench: RTX 2080 Ti (11 GB), a 9B instruction-tuned causal LM
quantized to NF4 and frozen, LoRA rank 4. Two seeds. Preregistered with executable
gates. Status: candidate. Not externally reproduced.

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

## What was measured

The task is a synthetic tool-call protocol the base model does not produce (0/8 control).

| arm | result |
|-----|--------|
| constructed, 12 leave-one-out tasks | 72/96 |
| constructed, 3 tasks with no training anywhere, trigger regime | 24/24 |
| constructed, same 3 tasks, speak-then-act regime, 400-token budget | 18/24 |
| directly trained ceiling, same regime and budget | 7/8 |
| matched-norm random direction, same subspace | 0/96 |
| trunk alone | 0/96 |
| constructed-arm generations emitting a different ticker | 0/96 |

The last row is the misattribution check. Of the 24 leave-one-out cases that did not
fire, 23 produced no call at all and 1 produced a fragmentary string (`NVNV`). None
produced a different valid ticker. The wrong-gradient arm is the deliberate contrast: it
never emits the target (0/96) and emits the injected ticker instead 80/96.

## Findings

1. **Population geometry.** Trained adapters decompose into a shared trunk carrying most
   of the step-level predictability (a mean-field control predicts a held-out task's
   steps at cosine 0.95, against 0.52 for a fitted per-task predictor) and near-orthogonal
   per-task branches at roughly 20% of step norm. The branch direction locks early: its
   cosine with its own final direction is 0.769 at training step 5, 0.947 at step 20, and
   0.991 at step 40, out of 120.

2. **A firing threshold.** Gradient-to-branch alignment runs 0.38 to 0.53 across tasks
   (null p95 0.083) and reproduces across seeds to the third decimal. Firing is a step
   function of that alignment, with onset bracketed between 0.4255 and 0.4485, one data
   point per edge. The three failures are the three lowest alignments. Doubling the
   gradient's data from 24 to 48 examples moved alignment down in all four tasks tested,
   by 0.006 to 0.033, and improved none, so the floor is not a sampling limit.

3. **Override.** A foreign task's gradient branch at matched norm, injected onto a fully
   trained adapter, produces the foreign identity 46/48 across six ordered pairs, with the
   resident owner at zero in every generation. The other 2 generations are fragmentary
   rather than owner. Random directions at the same norm leave the owner intact, 8/8 in
   both control pairs, so replacement tracks the direction and not the added norm.

4. **Dose.** Sweeping injection strength over two ordered pairs: at 0.6 beta the owner is
   untouched in both (8/8 owner), clean replacement arrives at 0.8 beta in one pair and
   0.9 in the other, and fragmentary strings appear in one pair only, at 1.0 beta (2/8)
   and 1.1 beta (5/8). The upper edge is pair-dependent.

5. **Coherence depends on the direction family.** The centroid of cross-task adapters is
   itself non-functional. At matched norm, random directions and unembedding-derived
   directions do not restore function; gradient-derived directions do. Same-task centroids
   are reported to work elsewhere (arXiv:2302.04863), so the condition here is crossing
   tasks, not averaging.

6. **Frame locality.** The construction is deterministic within an initialization frame:
   the alignment band reproduces at a second seed (mean 0.467), where five of six
   constructed bridges fire 8/8 and the lowest-aligned task fires 3/8. Across seeds,
   trained solutions share 0.144 weight-space cosine against a null of 0.041, and trunks
   share 0.200. Interpolating two same-task solutions from different seeds fires at every
   point (21/21), so in this object class those solutions are connected. The gradient
   therefore computes something valid in the initialization frame it was computed in,
   rather than a frame-independent object.

7. **Identity separates from the behavioral program.** Branches computed under one
   training regime drive adapters trained under another (cross-regime branch cosine 0.734;
   24/24 in the easier direction). A speak-then-act regime shows the separation: the model
   answers an unrelated question, then emits the call naming the never-trained target.
   Runnable in a browser via `code/gpu_worker.py`.

## Scope

* One model, one substrate, 8 prompts per task.
* Targets are in the same family as the library. Out-of-family is untested here.
* Roughly a quarter of in-family targets sit below the alignment threshold and fail. The
  threshold predicts which ones before the attempt.
* The construction is training-free with respect to the target only. The trunk comes from
  other tasks' training.
* Reliability decays on answers longer than the trained response-length band. This bounds
  the speak-then-act regime and also bounds the directly trained ceiling.

## Related work

LoRA-One (arXiv:2502.01235) reports that a one-step gradient update, and its rank-8
approximation, can match trained LoRA on small classification tasks without further
training. Task vectors have been shown to approximate first-epoch gradients
(arXiv:2508.16082) for trained models. GradFix (arXiv:2510.09658) uses target gradients to
filter a task vector that was already trained on the target. LoraHub (arXiv:2307.13269)
composes existing adapters without a target gradient, and hypernetwork approaches such as
Text-to-LoRA generate adapters from a trained generator. The construction here combines a
library trunk with a trunk-removed gradient branch on a generative task, and reports the
alignment threshold, the population geometry, and the override and dose behavior above.
`paper/PRIOR_ART_POSITIONING.md` records the full comparison.

## Repository

* `code/` harness (model load, coordinate and readback gates, termination-state
  generation, gradient scale ladder), batching helpers, and every experiment as run
* `preregs/` preregistrations, written before their runs, bars included
* `results/` raw JSON for every panel, plus `NARRATIVE.md`, the full lab record including
  corrections, disclosed partial results, and instrument failures in the order they happened
* `paper/` technical report and prior-art positioning

## Reproducing

Scripts assume the base model, an 11 GB or larger NVIDIA GPU, and the paths in
`code/harness_common.py`. Every gate is executable, and every number in this file is
either stored in a JSON under `results/` or summed from the per-task entries in one.
Training trajectories are 4.6 GB and are not included. Bars were fixed before each run;
changing a bar makes it a different experiment.
