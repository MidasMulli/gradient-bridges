# PROBE — GRAD-AT-INIT (2026-08-22, authored BEFORE the run)
Post-hoc probe, labeled as such (not part of the original prereg; follows the
RESULTS_needle_paths.md "open crack"). Question: with seed+schedule pinned, the branch is
a deterministic function of the ticker tokens — is its direction already present in the
gradient of the loss at delta=0, a quantity computable WITHOUT ANY TRAINING (one
forward/backward per ticker through the frozen NF4 base + the free A_0)?

## Object
g_k = full-batch mean gradient over ticker k's 24 training examples, at the exact
training init (A=A_0 from seed 7102, B=0), same NF4 base, same fp16 autocast, same
frozen/fp32-norm/checkpointing config as traj_train.py. In LoRA coordinates dL/dA = 0
at B=0 (checked); the informative part is dL/dB = dL/dW @ A_0^T. Descent direction
d_k = −g_k. No clip (cosine is scale-invariant; clip is a uniform rescale).

## Coordinate gate (must pass before any comparison)
flat() of the probe's fresh PEFT init must equal traj[0] of a stored run bit-for-bit in
fp16. Mismatch = wrong coordinates = abort, no results.

## Comparisons (all LOO-branch: x_k − mean_{j≠k} x_j; full space and B-subspace)
1. cos( d_branch_k , Δ_branch_k )   Δ_k = traj_k[final] − traj_k[0]  (total displacement)
2. cos( d_branch_k , inc1_branch_k ) inc1_k = first optimizer step
3. Sanity (not a gate): cos(d_k, inc1_k) raw — must be clearly > 0 or the probe is broken.
Null distribution: mismatched pairs cos(d_branch_j, x_branch_k), j≠k (132 pairs).

## Emergence timing (trajectory-only, no gradient needed)
Per step t: branch increment = inc_k(t) − LOO mean; report branch norm fraction per step
and cos(cumulative branch at t, final branch) at t ∈ {1,2,3,5,10,20,40,80,120}.
Answers WHEN the branch direction is set (loss hits ~0 by step ~25; do branches keep moving?).

## Pre-committed reading
SIGNAL: matched mean cos ≥ 0.1 AND above the 95th percentile of |null| on comparison 1
or 2 ⇒ first crack toward bridge-without-training; next gate is FIRE (inject along the
free-computable direction, A3-style controls), never cosine.
NULL: matched ~ null ⇒ strongest wall form measured yet: the path is deterministic in its
inputs but its target-specific component is invisible to its own first gradient. Arc
closes on the wall pending external cold review.
Either way: cosine here is a screen, not a conclusion (operator's standing caution).

## Budget
GPU: 1 NF4 model load + 12×24 fwd/bwd of ~40-token seqs ≈ 10 min. CPU compare: streams
traj rows, <1 GB RAM. GPU serialized (nothing else on the card), job under systemd-run.
