# gradient-bridges

**Constructing functioning LoRA adapters for never-trained tasks from a single
gradient evaluation at initialization plus a library of other tasks' adapters —
with the population geometry, the firing threshold, and the failure physics.**

All results measured on one bench (RTX 2080 Ti 11 GB, Qwen-family 9B, NF4 frozen
base, LoRA r=4), two seeds, preregistered with executable gates. Status:
CANDIDATE — single substrate, n=8 prompts/task, externally un-reproduced.

## The construction

For a target task T with **no trained artifact anywhere**:

```
adapter(T) = trunk + β · unit_B( ∇L_T(θ₀) − mean_grad_others )
```

- **trunk** — the mean weight-delta of adapters trained on *other* tasks (the shared
  behavioral program; carries zero information about T)
- **∇L_T(θ₀)** — one full-batch gradient of T's loss at the *untrained* initialization
  (frozen quantized base + shared LoRA init); T's data is constructible from a template
- **β** — the typical task-specific ("branch") norm of the library's adapters

On a synthetic tool-call protocol the base model cannot perform (verified 0-fire),
constructed adapters fire the correct never-trained target on **75% of held-out
prompts** (18/24 pooled across 3 novel tasks at adequate token budget; 72/96 across
12 leave-one-out tasks), with **zero wrong-identity emissions** across ~700 scored
generations. Identity-free controls (random direction at matched norm in the same
subspace; trunk alone) fire **0/96** — and are fully *degenerate*, which is its own
finding (§4).

## Scope, stated before anyone asks

- **LoRA-One (arXiv:2502.01235) Table 2 already shows** a lone one-step gradient
  update (and its rank-8 approximation) matching trained LoRA on small GLUE
  classification tasks with no training. The novelty here is **not** "a gradient can
  function without training." It is the **composition and its physics**: the
  trunk+branch assembly on a generative task, the alignment threshold that gates
  success, the population geometry, the ownership-override result, the dose window,
  and frame-locality (below). See `paper/PRIOR_ART_POSITIONING.md` for the full
  must-cite map (task vectors ≈ gradients: arXiv:2508.16082; GradFix: 2510.09658).
- In-family targets only (same task family as the library). Out-of-family is untested
  here and null elsewhere.
- ~25% of in-family targets fall below the alignment threshold (§2) and fail — this
  floor is **structural**: doubling the free gradient data does not move it.

## The findings (numbers in `results/`, prose in `results/NARRATIVE.md`)

1. **Trunk/branch geometry.** Trained adapters = shared trunk (95% of step
   predictability; a mean-field control beats a fitted per-task predictor 0.95 vs
   0.52) + near-orthogonal per-task branches (~20% of step norm, pairwise cosine at
   the independence reference). Branch *direction is set by training step ~20 of 120,
   then frozen* — the fact the whole construction rests on.
2. **The threshold.** Grad-at-init aligns with the trained branch at cos 0.38–0.53
   (per-task, reproducible across seeds to the third decimal). Fire is a step
   function in that alignment: onset bracketed at 0.4255–0.4485. The three misses
   are exactly the three lowest alignments; the floor survives data-doubling
   (curvature-limited, not noise-limited).
3. **Ownership override + dose window.** A foreign task's gradient branch at matched
   norm, injected onto a *fully-trained* adapter, fires the foreign identity 46/48
   with the resident owner at **zero** — while a random direction at the same norm
   leaves the owner intact 8/8. Dose structure: under-dose → owner rules; window →
   clean replacement; overdose → fragmentary identity emission ("NVNV"-class), the
   first dose–response curve for weight-space identity corruption.
4. **Coherence is gradient-family-specific.** The centroid of working *cross-task*
   adapters is itself non-functional (degenerate output). Random directions and
   semantically-structured (unembedding-derived) directions at matched norm do not
   restore function; only gradient-derived directions do. (Contrast: same-task
   centroids are known to work — arXiv:2302.04863; and this is direct behavioral
   counter-evidence to norm-not-basis survival claims.)
5. **Frame-locality.** The construction is deterministic *within* an initialization
   frame (alignment band reproduces at a second seed; bridges fire 5/6), while
   trained solutions across seeds share only 0.14 weight-space cosine. The gradient
   computes a *frame-local name* for the task — which reconciles this bench's
   positive results with a collaborating bench's frame-external nulls.
6. **Program/identity factorization.** The name is program-portable: branches
   computed under one training regime drive adapters of another (100% in the easier
   direction), and a "speak/act dissociation" regime demonstrates the canonical
   behavior — answer an unrelated question, then act on the injected identity —
   live in a browser demo (`code/gpu_worker.py`, `code/dashboard.py`).

## Repo map

- `code/` — harness (model load, coordinate/readback gates, term-state generation,
  gradient ladder), training/batching helpers, every experiment script as run
- `preregs/` — the pre-registrations, authored before each run, bars included
- `results/` — raw JSONs for every panel + `NARRATIVE.md` (the full lab record:
  every verdict, correction, disclosed peek, and instrument lesson, in order)
- `paper/` — prior-art positioning and the technical report

## Reproducing

Scripts assume the base model (HF: the 9B used here), an 11 GB+ NVIDIA GPU, and the
paths in `code/harness_common.py`. Every gate is executable; every number in the
README traces to a JSON in `results/`. Preregistration discipline throughout:
if you change a bar, you are running a different experiment — say so.

## Status & citation

CANDIDATE-tier: preregistered, controlled, two seeds, one substrate, one bench.
External reproduction is invited — the construction requires only a template task
family, a LoRA library, and one gradient evaluation. License and citation entry
pending maintainer review.
