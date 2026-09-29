# PREREG B5: does trunk formation track literal output overlap?

Written 2026-08-24 before any training. Bars pre-committed. This cell exists because two
diffusion nulls and one LM positive line up into a testable law:

| family | shared output structure | trunk share of delta |
|---|---|---|
| LM tickers | MEASURED 0.770 token overlap (0.667-0.857 per task) | ~0.80 (raw pairwise cos 0.908) |
| C1 photo concepts | none | ~0 (cos 0.027) |
| blueprint style | shared semantics, differing latents | ~0 (cos 0.048) |

CANDIDATE HYPOTHESIS: the trunk is not "the shared program" in any semantic sense. It is
the gradient of the literally shared part of the target output. If true, trunk fraction is
a monotone function of measured output overlap, and the construction's precondition becomes
quantitative.

## Design

Four overlap levels, four tasks each, 16 new tasks total, everything else pinned to the
banked trainer: Ornith-1.5-9B NF4, LoRA r=4, layers 20-31, seed 7102 (one shared init
frame), AdamW lr 2e-4 with the 0.01 default weight decay, GradScaler init 256, clip 1.0,
batch 1, 120 steps, the same 24 training carriers and 8 held-out carriers as the ticker
family. Only the TARGET STRINGS vary.

Target structure: 12 words, a shared PREFIX of K words identical within the level, then
12-K task-unique words. Levels K in {0, 4, 8, 11}. K=11 mirrors the original family's
shape (everything shared except one specific); K=0 mirrors the diffusion families.

Shared prefix is placed FIRST deliberately. In a causal LM a shared token that follows
task-unique tokens sits in a different context per task, so its gradient is not identical
across tasks even though the token is. Shared-prefix overlap is the clean version of the
variable; context-dependence of shared-suffix tokens is a separate axis, out of scope, and
the claim is scoped to prefix overlap.

The x-axis statistic is DEFINED here and used everywhere, including the anchors: the
LCP+LCS fraction over the SUPERVISED token stream (every position where labels != -100,
which includes the closing template tokens after the target), computed pairwise within a
level and averaged. MEASURED under this statistic before any run: the ticker anchor is
0.815, and a K=0 pair measures ~0.125, because two closing template tokens are shared by
every task in every level. So o(0) is NOT zero and no bar may assume it is. Total target
token counts must match across all 16 tasks within +/-2, asserted, and the shared prefix
must be TOKEN-ID identical across a level's tasks, asserted.

Task-unique words are drawn from a fixed recorded pool of common concrete nouns, disjoint
across tasks and levels, listed verbatim in the cell. No tickers, no overlap with the
original family.

## Gates, executable, run first

* G1 base null: the base model, 8 held carriers, must emit none of the 16 targets
  (exact-prefix check). Trivial but run in scan.
* G2 length match: tokenized target lengths within +/-2 across all 16 tasks. ABORT if not.
* G3 trainer: the training loop is B2's, and it is revalidated IN SCAN rather than by
  citation: 8 steps on AAPL with its original CALL target through this cell's loop must
  reproduce the banked manifest losses (step 0 within 1e-3 absolute, steps 1-7 within 2%
  relative). The target construction is the experimental variable and is exactly what is
  NOT covered by that validation, which is the honest statement of what G3 certifies. The
  shuffle-order assert from B2 is reused verbatim.
* G4 coordinate and readback gates, per harness standard.
* G5 training validity: after 120 steps every task must fire >= 6/8 on held carriers
  (exact-prefix). A task that does not train is excluded and reported; if more than 2 of 16
  fail, the run is VOID (the geometry of untrained deltas means nothing).

## Measures

Per level: mean raw pairwise delta cosine (6 pairs), the primary; LOO branch fraction
(independence expectation at N=4 is 1.155), secondary; per-pair values reported.

Mechanism measures, pre-registered (the review's sharpest addition, ~15 min of backwards,
no extra training). Per task at init, two masked gradients through harness grad_at:
g_P supervising only the shared prefix plus closing tokens, and g_U supervising only the
unique tokens, with supervised-token counts recorded so the full gradient recombines as
the count-weighted sum (HF loss averages over supervised tokens, so unweighted g_P + g_U
is NOT g_full).

* Principled MODELLED curve: c_hat(K) = mean pairwise cos of the recombined init
  gradients. The residual c(K) - c_hat(K) isolates what 120 steps of training add beyond
  init geometry.
* Mechanism test: cos(unit(trunk_K), unit(mean g_P)), where trunk_K is the level's mean
  trained delta. High supports the identity "trunk = shared-output gradient"; low while
  c(K) still tracks overlap means the correlational law survives but the mechanistic
  sentence must be rewritten.
* The per-position CE loss-mass curve is kept only as a labelled naive baseline.

Convention, fixed: mean pairwise cos estimates |trunk|^2/(|trunk|^2+|branch|^2), the
SQUARED-norm trunk share; the ticker family's 0.908 cos and 0.794 norm-share (1 - 0.206)
are the same geometry expressed in two conventions, not a discrepancy.

Stated plainly, from the review: within a level, the prefix-position gradient
contributions are EXACTLY identical across the 4 tasks at step 0, by construction (same
carriers, same init, same positions). That identity is not the discovery. What the runs
test is whether it PERSISTS through 120 divergent optimizer steps and how the magnitude
maps to overlap.

Anchor points, descriptive: the ticker family at x = 0.815 UNDER THE DEFINED STATISTIC
(cos 0.908, N=12), labelled as carrying mixed prefix-and-suffix sharing so it is not read
as a fifth point on the prefix-overlap axis; the diffusion families (overlap 0, cos
0.027-0.048, different substrate) likewise labelled.

## Pre-committed bars

Let c(K) be mean raw pairwise cos at level K, and o(K) the measured overlap fraction.

* **LAW SUPPORTED**: c strictly increases across the four levels (all three consecutive
  differences positive), AND c(11) >= 0.50, AND c(0) <= o(0) + 0.10, where o(0) is the
  MEASURED K=0 overlap (~0.13 expected from the shared closing tokens). The first draft's
  absolute c(0) <= 0.15 bar could have refuted the design BECAUSE the law is true, since
  the law itself predicts c(0) ~ o(0).
* **LAW REFUTED**: c(11) - c(0) < 0.20, or the ordering is non-monotone with any inversion
  larger than 0.10.
* **INTERMEDIATE**: anything else. Reported descriptively, no verdict; in particular a
  monotone but shallow curve (c(11) between 0.20 and 0.50) is INTERMEDIATE and would say
  overlap matters but is not the whole trunk.

Every verdict carries CANDIDATE and the scope sentence: one model, one substrate, one seed,
one training order, shared-PREFIX overlap only, 4 tasks per level.

## Confounds acknowledged in advance

* Task-unique words differ in semantic content across tasks and levels. If the law fails,
  semantic variation is a candidate explanation; the original family was semantically
  homogeneous (all tickers). Word pools are recorded so this can be audited.
* 4 tasks per level bounds resolution; the independence reference for pairwise branch cos
  is -1/3, not -1/11, and only raw-delta cosine is compared across levels.
* MEASURED before any run: the anchor's cos (0.908) exceeds its token overlap (0.770), so
  cos ~= overlap is already known to be too naive at one point. The MODELLED comparison uses
  loss-mass weighting for exactly this reason, and a SUPPORTED verdict requires ordering and
  range, not agreement with the identity line.
* K=11 is not a replication of the ticker family: different words, different semantics,
  same shape. Agreement with the 0.908 anchor is expected but not required by any bar.
* Pre-registered follow-up if the verdict is INTERMEDIATE: a disjoint-prefix arm (4 tasks,
  each with its OWN 8-word prefix plus 4 unique words, shape-matched to K=8 with zero
  literal sharing), for which the law predicts c ~= c(0). Registered now so it cannot be
  invented post hoc.
* Fire gate at 6/8 rather than 8/8 for validity: these targets are longer and harder than
  the ticker call; the gate certifies training happened, not task quality.

## Budget

16 tasks x 120 steps x 1.15 s = 37 min training (MEASURED rate). Panels and gates ~5 min.
Under an hour total. Checkpointing per task with resume, atomic phase saves, worker
stop/restore per the standard runner.

## Outputs

`results_b5_overlap.json`: per level and per task, the trained delta norms, pairwise
cosines, branch fractions, measured overlap, fire panels, per-position loss shares, gate
results, and the verdict written by the cell.
