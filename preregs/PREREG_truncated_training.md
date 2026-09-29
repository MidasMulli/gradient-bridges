# PREREG B1: truncated training, is the emergence curve a stopping rule?

Written 2026-08-24 before any generation. Revised the same day, still before any
generation, after a four-lens adversarial review found nine blocking defects in the first
draft. Bars below are pre-committed. Changing a bar after seeing data makes this a
different experiment and it must be relabelled as such.

## Question

The banked emergence curve says the per-task branch direction reaches cosine 0.769 with
its own final direction by training step 5, 0.947 by step 20, and 0.991 by step 40, out of
120. Direction is essentially chosen in the first sixth of training.

That is a statement about direction, not function. Does an adapter stopped at step k still
fire?

## Why this costs almost nothing

No training is required. `runs/<T>/traj.npy` holds the parameter vector at every step 0 to
120, fp16, shape (121, 1572864), for all 12 library tasks. A truncated adapter is exactly
`traj[k]`: the same optimizer trajectory, stopped early. This is a generation-only replay
of banked checkpoints.

## Ops preamble, executable

1. `systemctl --user stop gpu-worker.service` (it holds about 8 GB).
2. Assert nvidia-smi free >= 9000 MiB before constructing the Harness. The NF4 load needs
   about 6.5 to 7 GB and the RAM preflight will not catch a VRAM shortfall.
3. Run under `systemd-run --user --collect -p MemoryMax=8G`.
4. Restore the worker after the results file is written. It is a transient unit, so it must
   be recreated with `systemd-run`, not `systemctl start`.

## Design

Norms are full-vector L2 over all 1,572,864 coordinates. `fire_construct.py`'s beta uses
the B-block norm only; the two scalars are different and must not be cross-quoted.

The trunk is **leave-one-out**. An inclusive 12-way mean would put a 1/12 trace of the
target's own fully trained branch into every row of an arm whose whole claim is that step
120 was not needed. This lab has already measured a trace of that size flipping a task:
JPM fires 1/8 in its own clean leave-one-out trunk and 7/8 in a trunk carrying a 1/11 trace
of its own trained branch. The banked `trunk_only` 0/96 control was also measured
leave-one-out, so this convention is what makes it reusable.

```
trunk_120  = sum over j != T of ( traj_j[120] - traj_j[0] ) / 11
branch_T_k = ( traj_T[k] - traj_T[0] )
             - sum over j != T of ( traj_j[k] - traj_j[0] ) / 11
```

**Arm A, raw truncation (primary).** `params = traj_T[k]`.
Ladder k in {0, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 120}. 96 rows per rung.

**Arm B, oracle norm-corrected (mechanism).** k >= 1 only.
`params = traj_T[0] + trunk_120 + branch_T_k * ( |branch_T_120| / |branch_T_k| )`
Ladder k in {1, 2, 4, 8, 16, 32, 120}.
`|branch_T_120|` cannot be known at step k, so Arm B is an oracle diagnostic. No wording
about a training saving attaches to it and it never changes the Arm A verdict string.

**Arm C, full trunk and raw branch (mechanism).** k >= 1 only.
`params = traj_T[0] + trunk_120 + branch_T_k`
Ladder k in {1, 2, 4, 8, 16, 32}.

Arm C exists because Arm B changes two things at once. Measured, the trunk at k=4 is only
0.159 of its final norm, so at low k Arm A has 16% of the trunk while Arm B has 100% of it,
and an Arm B pass could be entirely trunk. B against C isolates the rescale at a fixed
mature trunk; A against C isolates trunk maturity at a fixed raw branch.

**Trunk-only null, in scan.** `params = traj_T[0] + trunk_120`, 96 rows. This is Arm B and
Arm C's missing null and is anchored to the banked leave-one-out `trunk_only` 0/96.

Arms B and C are undefined at k=0, because `branch_T_0` is identically zero (verified
bitwise: all 12 `traj[0]` frames are identical). An executable assert raises if
`|branch_T_k| == 0` before any rescale. No epsilon guard is used: a `+1e-12` guard would
silently turn Arm B at k=0 into a mislabelled trunk-only arm.

Gate, unchanged and pinned: `text, term, ntok = h.generate(carrier, max_new=24)`, fire iff
the decoded text starts with `CALL: stock_quote("<T>")`, scored over `CARRIERS[24:]`.

## Power controls, executable, run first

* **G1 null.** Arm A at k=0 is the LoRA init, where the B blocks are exactly zero
  (verified) so the adapter is a genuine no-op. Must pool 0/96. Otherwise ABORT.
* **G2 ceiling.** Arm A at k=120 must pool 96/96, reproducing the banked trained ceiling.
  Pre-registered rescue, so it is not a post-hoc one: if it misses by <= 2 rows, those rows
  are re-run with `use_cache=False` before the replay is declared unfaithful, and the
  cache-path discrepancy is reported. `fire_construct.py` produced the banked 96/96 on a
  bespoke decode path while `harness_common.generate` forces `use_cache=True`. A miss of
  more than 2 rows is an ABORT.
* **G3 identity.** Two clauses. (a) Parameter space, before any generation:
  `max|params_B(k=120) - params_A(k=120)| <= 1e-8`. Measured on the banked trajectories
  this is 9.313e-10, about 10x headroom, while a wrong decomposition is off by about 1e-3,
  six orders larger. This is the ABORT gate. (b) Generation space: the 96 decoded strings
  should match row for row. If (a) passes and (b) diverges, do NOT abort. Log the row, both
  strings and the max parameter delta, re-run Arm A at k=120 once to establish decode
  determinism as the in-scan control, and report it as an instrument finding.
* **G4 coordinate and readback gates.** `coordinate_gate()` must pass, and every injection
  must pass the readback gate, before any row is scored.

No result is read unless G1, G2, G3(a) and G4 pass.

## Pre-committed bars

The unit of replication is the TASK, not the generation. Greedy decode is deterministic and
both the 12 tickers and the 8 held-out carriers are fixed lists, so there is no rerun
variance: 96 rows are 12 clusters of 8. In the banked record 119 of 132 per-task x/8 cells
are exactly 0/8 or 8/8 (90.2%), with 13 partial cells present in the same scan, so the
all-or-nothing regime is measured rather than a scanning artifact. On the constructed arm
the per-task vector [8,8,0,8,8,1,8,8,8,7,8,0] gives ICC 0.897 and n_eff 13.2. Ninety-six
generations carry about 13 independent bits, one per task.

Per-task states: FIRES = 7/8 or 8/8, PARTIAL = 1/8 to 6/8, DEAD = 0/8.

Define k* as the smallest rung k such that k, and every rung above k other than 120,
satisfies BOTH:

* at least 11 of the 12 tasks FIRE (>= 7/8), and
* pooled >= 88/96.

k=120 is excluded from the confirmation set because G2 guarantees it passes, so it carries
no evidential weight. The two conditions are independent: pooled >= 88 alone guarantees
only that 8 tasks are at >= 7/8, so the task condition bites. The first draft's pair did
not: pooled >= 90 caps misses at 6, and a task below 6/8 has >= 3 misses, so "10 of 12 at
>= 6/8" was arithmetically implied by the pooled bar and guarded nothing.

Operating characteristic, computed before any data, tasks exchangeable with per-task pass
probability q: P(bar met) = 0.88 at q=0.95, 0.66 at q=0.90, 0.275 at q=0.80. With n=12
this design cannot separate q=0.80 from q=0.95. Every verdict is scoped to this 12-ticker
library and the phrase "adapter training" must not appear unqualified.

Reported at every rung: the 12-value per-task vector; the three-state counts; the per-task
onset k_i, being the smallest rung at which task i is 8/8, with its median, its max and the
full 12-value vector; the per-carrier miss count pooled over tasks; the emitted-ticker
histogram. The headline saving is 120/k* stated exactly and never rounded up, labelled the
worst-of-12 saving, with 120/median(k_i) reported separately as the typical-task figure.

Interval: one-sided 95% Clopper-Pearson lower bound on q from the task count. 12 of 12
gives q >= 0.779; 11 of 12 gives q >= 0.661. No Wilson interval on n=96 appears anywhere
in the outputs, because the 96 rows are not 96 independent trials.

Ceiling comparison: d = the number of tasks that FIRE at k=120 but not at k*, reported as
"k* discords with the trained ceiling on d of 12 tasks". A paired sign test needs d >= 5
for p < 0.05 (d=4 gives p=0.0625), so any d from 1 to 4 is DESCRIPTIVE and carries no
significance claim in either direction. This design cannot certify non-inferiority to the
ceiling. The bar caps the loss at one task and every verdict string states d explicitly.

**Reading, fixed in advance.** Rungs are {0,4,8,12,16,20,24,32,48,64,80,96,120}. Every
verdict string is prefixed CANDIDATE.

* k* <= 32: CANDIDATE BUILD. Saving 120/k*, exact. If k* <= 24 the result sits inside the
  first epoch and is worded as data coverage as well as step count.
* k* in {48, 64}: CANDIDATE PARTIAL BUILD, saving 2.50x or 1.88x. The gap between the
  direction-lock step and k* is the finding.
* k* in {80, 96}: CANDIDATE WEAK, saving 1.50x or 1.25x. The curve is narrowed and there is
  no useful stopping rule.
* k* = 120, or no rung qualifies: MEASURED LIMIT. Only the fully trained adapter meets the
  bar. Direction being set early does not imply function is set early.
* Any rung above the first qualifying rung fails: NON-MONOTONE. No stopping rule is
  reported and no saving is claimed; the raw pass/fail ladder is the result.

**Mechanism reading, fixed in advance.** Arms B and C never change the Arm A verdict string
and are reported as separately named results.

* B fires at a rung where C does not: given a mature trunk, the deficit at that rung is
  branch MAGNITUDE. Oracle framing only.
* B and C both fire where A does not: the deficit at that rung is the TRUNK, not the branch.
* B and C both fail where A fails: the deficit is neither trunk maturity nor branch
  magnitude alone. Reported as measured, no mechanism claim.
* B fails where A succeeds: the decomposition is doing damage. Instrument finding, not a
  claim about training.

## Coverage confound, pre-registered

The ladder is reported in epochs alongside steps (k/24). The trainer is batch size 1 over
24 training carriers with a without-replacement queue reshuffled each epoch, seed 7102 in
all 12 runs, so distinct training carriers seen after k steps is exactly min(k, 24), and
120 steps is 5 epochs. The rungs 4, 8, 12, 16, 20, 24 are therefore literally the coverage
points 4/24 through 24/24.

Any k* <= 24 is inside the first epoch and is reported as "one pass over the 24-example
training set suffices", not as confirmation that the direction curve is a stopping rule.
Because all 12 runs share the shuffle seed, training order is n=1: the 12 tasks are not 12
independent tests of the schedule. Discriminator: Arms B and C at k <= 6 carry the branch
on only k/24 of the training set, so a B or C pass at k <= 6 defeats the coverage
explanation.

## What would falsify the headline

The headline is: "On this task family and gate, LoRA checkpoints taken at step k of a
120-step constant-lr run pass the fire gate at k = k*, X/96 against 96/96 at step 120, an
exact 120/k* reduction in optimizer steps for this configuration." The ratio is reported
exactly and never rounded up.

Mandatory scope sentence: Ornith-1.5-9B NF4, r=4, layers 20-31, 12 synthetic tool-call
tasks, seed 7102, one training order, one box. CANDIDATE tier.

It is falsified if Arm A at the claimed k* misses either bar, or if any rung above it
fails. It is not rescued by Arm B or Arm C; those are different and weaker claims and must
be reported under their own wording.

## Confounds acknowledged in advance

* Trainer: AdamW, constant lr 2e-4, no warmup and no decay, gradient clipping at 1.0, batch
  size 1, 24 training carriers, 120 steps (5 epochs), seed 7102 in all 12 runs. Nothing in
  the configuration depends on the total step count, so a checkpoint at step k is identical
  to a run configured for k steps at this configuration, and "trained for k" is licensed for
  this configuration only. Untested: a schedule tuned to a short budget, which could do
  better. One training order across all 12 tasks, so order is n=1.
* The 8 held-out carriers are one fixed list crossed with all 12 tasks, and behaviour within
  a task is near-deterministic, so the per-task instrument is three-valued (DEAD, PARTIAL,
  FIRES) rather than a Bernoulli rate over 8 trials. A systematically hard carrier would
  depress all 12 tasks at once; the per-carrier miss count reported at each rung separates a
  carrier effect from task difficulty at zero extra generation cost.
* One task family, one seed, one substrate. This is a statement about this training setup.

## Outputs

`results_truncated_training.json`: per task, per rung, per arm, fires and term states, plus
the pooled ladder, k*, the per-task onsets, per-carrier misses, emitted histograms, gate
results and the verdict string written by the cell rather than by hand.
