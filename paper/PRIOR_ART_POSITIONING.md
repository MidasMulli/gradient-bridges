# PRIOR-ART POSITIONING — bridge-arc findings vs the literature (sweep 2026-08-23)
Full sweep notes in the session record; this is the standing publication gate.
Rule: nothing ships without addressing the must-cites; watchlist checked first.

## Verdicts
- **F1 Gradient-at-init bridge construction (trunk + β·unit(grad-branch), no target
  training, 75% fire, alignment threshold ~0.43-0.45): NOVEL.** The crux question —
  does anyone construct a FUNCTIONING adapter from one gradient evaluation + other
  tasks' adapters — came back NO across theory, filtering, composition, and
  hypernetwork families. The empty cell is ours.
- **F2 trunk/branch decomposition (95% predictability trunk, ~20%-norm orthogonal
  branches, direction frozen by step ~20): NOVEL-COMBINATION** (ingredients exist
  separately; the quantitative decomposition + its use as F1's interface do not).
- **F3 frame-locality (per-frame determinism; cross-seed cos 0.14 + invariant core):
  NOVEL-COMBINATION** (poles: kernel-regime theory 2305.12827 / seed-basins 2205.12411).
- **F4 override + dose window (foreign grad-branch replaces trained owner 46/48-0;
  random preserves owner; owner/replace/fragment three-phase dose): NOVEL.**
- **F5 gradient-family-specific coherence + dead cross-task centroid revived only by
  gradient directions: NOVEL** (state the CROSS-task condition loudly — Gueta
  2302.04863 shows same-task centroids DO work).

## Must-cite & pre-empt (the "this is expected" attack and its answer)
- arXiv:2508.16082 "On Task Vectors and Gradients" (NeurIPS 2025): task vector ≈
  first-epoch gradient — TRAINED, full-norm, no trunk decomposition, explicitly no
  trainless construction. Our answer: they explain; we build, with a threshold.
- arXiv:2502.01235 LoRA-One + arXiv:2407.05000 LoRA-GA: gradient-at-init as a
  TRAINING ACCELERATOR (subspace init); never trainless function.
- arXiv:2510.09658 GradFix (nearest miss): target gradients FILTER an existing
  trained task vector for the same task; the task knowledge is still trained.
- Adjacent for F4: BadEdit 2403.13355, ROME/MEMIT; steering-overdose collapse
  (activation space); norm-collapse editing literature (2608.01624 — our F5 is
  direct counter-evidence to "basis doesn't matter").
- F5 contrast pair: Gueta 2302.04863 (same-task centroids work) vs our cross-task
  dead centroid; Arditi 2406.11717 (a semantic direction CAN work as a removal edit
  in activation-derived form — ours shows unembedding-structured fails for
  RESTORATION in LoRA-B space).

## Watchlist — CHECK BEFORE ANY PUBLICATION (3 items)
1. LoRA-One camera-ready appendix: any init-only/no-training ablation?
2. arXiv:2606.07217 "WIZARD" (robotics weight-space meta-learning, adapters
   "without task-specific gradient updates" — generator-based, different domain).
3. The 2025-26 merging survey line (2605.01580) for late-breaking scoops.

## Repo candidate (operator's go required, per publication pathway)
"gradient-bridges": harness_common + preregs + fire panels + multi-seed + transplant
+ dose sweep, with the RESULTS narrative as README. F1+F4+F5 are the headline;
frame-locality the mechanism; external activation-space nulls bound it. Blocked on:
watchlist checks, operator review, gh auth.

## WATCHLIST RESOLVED (2026-08-23, verification sweep)
1. **LoRA-One (2502.01235): QUALIFIED THREAT — F1 SCOPE CORRECTED.** Their Table 2
   evaluates a one-step gradient update (and its rank-8 approximation) with ZERO
   subsequent training, matching trained LoRA on small GLUE classification tasks
   ("one-step full gradient can suffice ... on small-scale datasets"). So "a gradient
   at init can function without training" is ANTICIPATED as a lone-gradient datapoint.
   SURVIVING CLAIM (must be worded this way): the novelty is the COMPOSITION —
   trunk from other tasks' adapters + unit-normalized TRUNK-REMOVED gradient branch
   at matched norm, on a GENERATIVE fire task, with the alignment threshold, the
   population geometry (F2), and everything downstream (override, dose window,
   factorization, frame-locality). LoRA-One: single-task, full/low-rank raw gradient
   as the whole update, small classification, no library, no threshold, no controls.
   The related-work carve-out is MANDATORY, verbatim in any abstract.
2. WIZARD (2606.07217): CLEAR — trained hypernetwork forward pass, explicitly
   "without gradient-based optimization" at inference; no target gradient anywhere.
3. Scoop scan May-Aug 2026: CLEAR — the GradFix lineage is crowding (BiCo, Theseus:
   training-free same-task transport; Spectral Surgery: gradient-guided but
   post-training) but nothing combines target-gradient-at-init + adapter library +
   no target training. Caveat: X/social preprint chatter unverified (search-blocked);
   arXiv coverage is the reliable signal.
STATUS: publication path OPEN with the corrected F1 scoping. The empty cell is
narrower and still empty.
