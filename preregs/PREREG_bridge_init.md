# PREREG B2: does the constructed bridge accelerate training, and is it the branch?

Written 2026-08-24 before any training or generation. Revised the same day, still before
any compute, after a four-lens adversarial review returned eleven blocking defects against
the first draft. Bars below are pre-committed.

The first draft's primary statistic was arithmetically degenerate and the review proved it
without spending GPU time. That failure and its fix are recorded here rather than quietly
replaced, because the reasoning is the useful part.

## The measurement problem, stated first

The obvious design is "count optimizer steps to reach the fire bar, compare arms". It does
not work here, for two measured reasons.

**S collapses to three selected tasks.** The banked constructed panel fires
[8,8,0,8,8,1,8,8,8,7,8,0], so nine of twelve tasks are already at ceiling before any
training. All FIRES headroom and 23 of 24 rows of pooled headroom sit in {BAC, JPM, XOM},
the three tasks below the alignment bracket. So a step-count statistic for BRIDGE is
definitionally "the rung at which two of those three converge", n=3, and those three were
selected on the outcome of a previous experiment. For COLD, TRUNK and RANDOM, all of which
start at 0/96, the same statistic is a genuine 12-task quantity. The primary arm and its
controls would be measured on different things.

**The comparison to a cold start is set by geometry, not by training.** MEASURED radii
from the init:

| quantity | radius |
|---|---|
| BRIDGE init at k=0 | 3.590 to 3.724 |
| TRUNK init | 3.512 to 3.523 |
| RANDOM init | 3.63 to 3.66 |
| COLD at k=20, where it meets the bar | 2.651 to 2.726 |
| COLD at k=32, the end of this budget | 3.275 to 3.397 |
| fully trained delta at k=120 | 3.590 to 3.755 |

BRIDGE starts at 97 to 100% of the fully trained radius, further from the origin than COLD
reaches in the entire 32-step budget. Any "BRIDGE beats COLD in step count" result is
substantially forced by where BRIDGE starts. That comparison is still worth reporting as a
practical number when a library exists, but it cannot carry a claim about the construction.

All radii above are MEASURED on the banked trajectories and grads, 2026-08-24, and
independently reproduced from the review that proposed this design.

TRUNK and RANDOM are matched to BRIDGE in radius to within 4%. Those are the contrasts that
are about the gradient branch, so they are primary here and the cold comparison is
secondary.

## Question

Does initializing at the constructed bridge reach the target behaviour faster than
initializing at a norm-matched alternative that carries no target-specific gradient? And is
any advantage attributable to the gradient branch, or to the library trunk alone?

## Arms

Four arms, all trained with the identical loop, differing only in initial weights.

```
Delta_j    = traj_j[120] - theta_0
trunk_120  = sum over j != T of Delta_j / 11                        leave-one-out
g_T        = -grad L_T(theta_0)     DESCENT direction. grads/grad_init_*.npy store the
                                    RAW +gradient, so the cell must negate on load.
branch_T   = unit_B( g_T - mean over j != T of g_j )
beta_T     = mean over j != T of || ( Delta_j - (sum_l Delta_l - Delta_j)/11 )|_B ||
```

The sign matters and the first draft had it inverted. Written as
`unit_B(grad L_T - mean of others' grads)` the vector is the ASCENT direction: measured
cosine against the expression `fire_construct.py` actually uses is -1.000088 (AAPL),
-1.000188 (BAC), -1.000043 (NVDA). Implemented that way BRIDGE would be the negation of the
construction, G2 would fail after paying for the model load, and the natural repair would
be a post-hoc sign flip on a pre-committed construction. An executable assert forbids this:
build the branch both ways from the banked grads and assert
`gb_prereg @ gb_fire_construct > 0.999` against a vector rebuilt by `fire_construct.py`'s
exact expression, before any training.

`beta_T` is per-target, not a global constant. MEASURED range 0.971763 (WFC) to 1.009706
(NVDA); a single global mean is off by up to 2%. Disclosed rather than discovered later:
`beta_T` is a leave-one-out mean of other tasks' leave-one-out branches, each computed
against all 11 others including the target, so `beta_T` carries a 1/11-weighted trace of
the target's own trained delta. That is a reproduced property of the banked construction,
not a new leak introduced here.

* **BRIDGE (primary).** `theta_0 + trunk_120 + beta_T * branch_T`.
* **TRUNK (primary control, norm-matched, isolates the branch).** `theta_0 + trunk_120`.
* **RANDOM (primary control, norm-matched, isolates direction).**
  `theta_0 + trunk_120 + beta_T * unit_B(random)`, one draw per task, seed recorded.
* **COLD (secondary).** `theta_0`. Banked from B1 for the ladder, and additionally validated
  through B2's own trainer by G0 below rather than being re-run as a full arm.

Grads are reused verbatim from `grads/grad_init_*.npy` and never recomputed.
`harness.grad_at` must NOT be used for training: it is a full-batch mean-gradient probe with
loss-scale escalation and no clipping, and substituting it replaces batch-size-1 SGD with
full-batch descent.

Ladder, all arms: k in {0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 32}. Training stops at 32; an arm
that has not met a bar by 32 is recorded as censored, never extrapolated.

## The trainer, bound to code and not to prose

The training loop is `traj_train.py` lines 116 to 143, imported or copied unmodified:
AdamW(lr=2e-4, **weight_decay at its 0.01 default**, not 0), no warmup or decay,
`GradScaler("cuda", init_scale=256.0)` with growth_interval 2000 so it cannot grow inside
32 steps, the exact sequence scale then backward then `unscale_` then
`clip_grad_norm_(1.0)` then step then update then zero_grad, `autocast("cuda", float16)`
forward, `config.use_cache=False` during training, `Harness(grad_ckpt=True)`, and a fresh
optimizer AND fresh scaler constructed inside each (arm, task) run so no Adam moment or
adapted loss scale leaks across arms.

The shuffle queue uses `order.pop()`, taking the LAST element, so the realized order is the
reverse of `randperm`. `pop(0)` gives a different sequence. Rather than compare B2's
reimplementation to its own reimplementation of the same seed, assert against the realized
sequence, verified this session:

```python
assert seq[:32] == [9,14,6,5,22,19,10,15,1,12,2,7,20,8,23,11,
                    4,0,17,3,16,13,21,18,12,10,17,2,23,21,9,14]
```

## Gates, executable, in this order

**G0 TRAINER EQUIVALENCE.** Before any arm. The entire novel surface of B2 is the training
path, and the first draft gated only the replay path, so a mis-specified trainer would have
passed every gate and still moved the result. Run B2's trainer for 32 steps from `theta_0`
on AAPL (typical), BAC (hard) and NVDA (the checkpointing outlier). Assert all four; any
failure ABORTS:

* (a) `|loss[0] - manifest.losses[0]| <= 1e-3`. Banked: AAPL 5.30359, BAC 5.88797,
  NVDA 5.51057. This alone validates template, label masking, autocast and init.
* (b) `|loss[s] - losses[s]| / losses[s] <= 0.02` for s = 1..7. Bounded at 7 because banked
  losses reach 1e-05 by about step 40, where a relative bar is meaningless. Validates the
  AdamW hyperparameters, the clip order and the scaler.
* (c) `max|theta[1] - traj_T[1]| <= 3e-7`. Tight and safe: the first Adam step from B=0 is
  sign-like, and MEASURED at banked k=1 every B coordinate moves exactly 2.000332e-4 = lr
  while the A block is unchanged; fp16 ulp there is 1.192e-7.
* (d) `cos(theta[k]-theta_0, traj_T[k]-theta_0) >= 0.999` and `|norm ratio - 1| <= 0.02` at
  k in {2,4,8,16,32}. Descriptive beyond k=1, since fp16 storage noise is about 1.3e-3.

Cost about 100 s. This replaces re-running COLD as a fourth full arm.

**G1 DECODE REPRODUCTION.** Re-score banked `traj[20]` through set_params and generate; must
return 96/96. Relabelled honestly: this checks the replay and decode path only and cannot
detect a trainer defect.

**G2 BRIDGE INIT.** BRIDGE at k=0 must reproduce the banked constructed panel
[8,8,0,8,8,1,8,8,8,7,8,0] with all of: pooled >= 68/96; per-task absolute difference <= 1;
the FIRES/PARTIAL/DEAD state vector unchanged on all 12; the DEAD set exactly {BAC, XOM};
and JPM not FIRES. Pre-registered rescue: `fire_construct.py:91` sets `use_cache=False` and
never restores it, so the banked 72/96 was decoded WITHOUT the KV cache, while
`harness.generate` forces it on. If the miss is <= 2 rows, re-run those rows once with
`use_cache=False` and report the cache-path discrepancy. `Harness.generate` gains a
`use_cache` keyword so this rescue is executable rather than aspirational. A miss of more
than 2 rows ABORTS.

**G3 TRUNK NULL.** TRUNK at k=0 must fire 0/96, matching the banked leave-one-out control.

**G4 RANDOM NULL.** RANDOM at k=0 must fire 0/96.

**G5 COORDINATE AND READBACK.** Both pass before any row is scored, plus the hard-coded
order assert above.

**G6 GRADIENT-PATH INSTRUMENTATION.** Every optimizer step in every arm records: raw
gradient L2 after `unscale_`, the fraction of the 1,572,864 trainable coordinates with
exactly zero gradient, whether clipping was active, and the scaler skip count. ABORT if any
arm records a scaler skip. VOID the comparison if any arm's mean zero-gradient fraction
exceeds COLD's by more than 0.10. Rationale: BRIDGE starts near-converged where the fp16
backward at scale 256 flushes elements below about 2.3e-10 to zero, and an
underflow-dominated arm would look like a real effect with no gate to catch it.

## Measures

**P1, primary and continuous.** `NLL(arm, task, k)` = mean teacher-forced negative log
likelihood of the exact string `CALL: stock_quote("<T>")` over the 8 held-out carriers,
prompt prefix masked to -100, built identically to a training batch but on `CARRIERS[24:]`.
Defined and unsaturated both at 8/8 and at 0/8, so all 12 tasks carry signal instead of
three. About 6,720 forward passes, roughly 2 minutes, cheaper than the generations already
budgeted.

**P2, primary and discrete.** `onset(arm, task)` = smallest rung at which that task is 8/8,
with k=0 admissible. Reported as a 12-value vector per arm plus median and max.

**P3, mandatory.** Retention: for every (arm, task, rung), flag any rung whose fire count is
below that arm's own k=0 count. One Adam step from a nonzero-B init displaces
`lr*sqrt(1572864)` = 0.2508, against BRIDGE's distance to its own trained solution of 0.894
to 1.548, so a single step moves 16 to 26% of the way and being knocked off 72/96 is a live
possibility. A step-count statistic would silently absorb that as a larger S.

S, the step count to the fire bar, is retained as a tertiary descriptive number with B1's
bars (11 of 12 tasks FIRE, pooled >= 88/96) and B1's monotone-sufficiency requirement.

## Pre-committed reading

**Primary contrast: BRIDGE against TRUNK**, paired sign test over the 12 tasks, on P1 at
each rung and on P2. This is the norm-matched contrast, radii within 4%, and it is the one
that is about the gradient branch. Power is real: 10 of 12 gives one-sided p = 0.0193, 11 of
12 gives p = 0.0032. RANDOM against BRIDGE is read the same way and separates direction from
perturbation.

Bands, assigned from the sign test and nothing else:

* **BRANCH_CREDITED.** BRIDGE beats TRUNK on at least 10 of 12 tasks. The gradient branch is
  doing work beyond the library trunk.
* **NO_BRANCH_CREDIT.** BRIDGE and TRUNK are not separated. Fixed clause, not to be
  reworded: *A leave-one-out mean of 11 trained sibling adapters reaches the bar in the same
  number of steps as the full construction. The gradient branch is not credited. This is a
  result about having a library of trained siblings, not about the bridge.*
* **BRANCH_HARMS.** TRUNK beats BRIDGE on at least 10 of 12. Reported as prominently as a
  positive.
* **NOT_EVALUATED.** Both arms censored at 32, so no ordering is established.

Attribution is not a separate paragraph a reader can drop. Two executable asserts run on the
emitted verdict string: no step-count ratio may appear without the sentence naming BRIDGE's
starting radius as a percentage of the trained radius; and under NO_BRANCH_CREDIT every
sentence mentioning the bridge or the construction must also mention the trunk.

**Secondary: BRIDGE against COLD.** Evaluated only if BRIDGE reaches the bar before step 20,
otherwise NOT_EVALUATED. Always reported with the radius sentence, because the comparison is
not norm-matched.

**NON-MONOTONE**, ported from B1 verbatim: record `first_passing_rung` alongside S for every
arm, and if they differ, prefix that arm CANDIDATE NON-MONOTONE, publish the raw pass/fail
ladder, and emit no ratio and no band for that arm.

Every verdict carries the CANDIDATE prefix and the scope sentence: Ornith-1.5-9B NF4, r=4,
layers 20-31, 12 synthetic tool-call tasks, seed 7102, one training order, one box.

## Confounds acknowledged in advance

* **TRUNK is not a weak control.** MEASURED, TRUNK's init is closer to the target's own
  trained solution than BRIDGE's on 9 of 12 tasks, so a TRUNK loss cannot be explained away
  by it starting further from the answer.
* **BRIDGE enters with 9 of 12 already at ceiling and TRUNK with none.** On a step-count
  statistic `S_trunk >= S_bridge` holds almost mechanically. This is exactly why P1, which
  is unsaturated at both ends, is primary and S is tertiary.
* **All arms start with fresh Adam moments.** A warm optimizer state is untested.
* **Coverage**, as in B1: k steps at batch size 1 means min(k, 24) distinct carriers seen.
  Rungs are reported in epochs alongside steps.
* **The 32-step cap is a budget, not a measurement.** Two censored arms are not ordered.
* **One training order, one seed, one substrate, one task family.**

## Ops preamble

1. Stop `gpu-worker.service`; assert nvidia-smi free >= 9000 MiB before constructing the
   Harness.
2. Training needs backward passes, so `grad_ckpt=True` per the banked bench verdict, and the
   RAM preflight must pass.
3. `systemd-run --user --collect -p MemoryMax=8G`, output tee'd to a log rather than piped
   through tail, which buffers until EOF.
4. Restore the worker afterwards; it is a transient unit and must be recreated with
   `systemd-run`.

Budget, priced from measured rates (B1: 2592 generations in 2405 s = 0.928 s/gen;
traj_train about 1.0 s/step): 3 trained arms x 12 tasks x 32 steps = 1152 optimizer steps,
about 19 min. Generations 3 arms x 12 tasks x 11 rungs x 8 = 3168, about 49 min. P1 forward
passes about 2 min. G0 about 100 s. Total roughly 75 minutes.

## Outputs

`results_bridge_init.json`: per arm, per task, per rung: fires, term states, P1 NLL, P2
onset, P3 retention flags, and the G6 gradient-path series. Plus the radius table, all gate
results, the sign tests, and the verdict string written by the cell with its asserts
enforced in code rather than by hand.
