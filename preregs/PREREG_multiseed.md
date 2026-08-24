# PREREG — MULTI-SEED CELL (2026-08-23, authored BEFORE any run)
The load-bearing test for the joint determinism hypothesis (both benches' ledgers point
here). Question: does grad-at-init-predicts-the-branch survive a seed change — is the
0.38–0.53 alignment a property of the pinned frame (per-frame determinism) or an
artifact of seed 7102?

## Design
NEW SEED S2 = 3141 (single new seed, pinned across all S2 targets — preserves the
within-regime pinned-seed property; a third seed is escalation, not the start).
Retrain 6 of the 12 tickers at S2, v1 TRIGGER regime exactly (traj_train recipe: 24
carriers, CALL completion, r4 α8 lr2e-4 120 steps, NF4 base): AAPL, AMZN, DIS, KO,
NVDA (high-alignment representatives) + XOM (the lowest-alignment miss — does the
scatter reproduce or is it seed-local?). Final deltas only (no dense traj).
Frame gate: A_0(S2) must DIFFER from A_0(7102) (seed actually changed) and all B-blocks
zero at init. Coordinate gate vs old traj is intentionally NOT applicable across frames.
Trunk caveat (disclosed): LOO trunk at S2 comes from 5 others, not 11 — noisier trunk,
~1/sqrt(5) branch residue; alignment is the primary metric, fire rates secondary.

## Measurements + pre-committed bars
P1 — LOAD-BEARING, LoRA-space within-S2 (exact v2-probe replication at the new frame):
cos(d_branch_S2(T), displacement_branch_S2(T)), B-subspace, LOO over the 6, null =
mismatched pairs. CONFIRM per-frame determinism if ALL 6 exceed null p95 AND the mean
falls in [0.30, 0.60] (consistent with S1's 0.38–0.53). COLLAPSE (mean < 0.15 or
several tickers at null) ⇒ the 0.476 was seed-conditioned; determinism axis wounded;
candidate axes (curvature, object class) take over and the S1 bridge result gets a
seed-scoped rider.
P2 — WEIGHT-SPACE cross-seed (LoRA frames differ across seeds; ΔW = 2·B_f·A_f is
frame-free; all cosines computed EXACTLY via rank-4 Gram algebra, no materialization):
cos_W(ΔW_branch_S1(T), ΔW_branch_S2(T)) for the 6 matched tickers vs cross-ticker null.
 - mean > 0.5 ⇒ solutions are seed-DETERMINISTIC in weight space (strong form: data
   alone picks the solution; the cross-bench divergence is then substrate/object).
 - ≈ null ⇒ solutions are seed-idiosyncratic HERE TOO, but (if P1 confirms) paths are
   per-frame predictable — reconciles our breach with collaborating-bench's null under one mechanism:
   grad-at-init works only in its own frame; an external frame-free basis nulled.
 - Also report (descriptive): cos_W(trunk_S1, trunk_S2) — is the PROGRAM seed-stable?
P3 — secondary: S2 trained adapters fire (expect 8/8, regime sanity); S2 LOTO
constructed bridges (5-ticker trunk + beta*gb) fire rates reported with the trunk
caveat — a bridge working at a second seed is confirmatory, not required (dose may
dip with the noisier trunk).

## Budget
6 trainings ≈ 15 min + 6 grads ≈ 3 min + fire evals ≈ 8 min + model load; one unit,
harness_common defaults (cache-at-eval, term-state, readback gates). CPU analysis
exact and instant (rank-4 Gram). All artifacts under seeds2/.
