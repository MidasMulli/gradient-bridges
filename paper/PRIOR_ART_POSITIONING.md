# Prior-art positioning

What the nearest published work reports, and what this repo measured. Sweep conducted
2026-08-23, with the arXiv identifiers below re-verified against arXiv on 2026-08-24. This
is a comparison, not a priority claim: it records what the sweep covered and found, and a
reader with a citation the sweep missed should treat this page as incomplete rather than
contradicted.

## The nearest neighbours

**LoRA-One, arXiv:2502.01235** (Zhang, Liu, Chen; ICML 2025). Proves that under gradient
descent LoRA adapters align with singular subspaces of the one-step full fine-tuning
gradient, and uses that gradient to initialize adapters. Its Table 2 additionally
evaluates a one-step gradient update, and its rank-8 approximation, with no subsequent
training, and reports that this can match trained LoRA on small GLUE classification tasks.

This is the closest result to the construction here, and it anticipates the general point
that a gradient evaluated at initialization can function without training. The
configuration differs: LoRA-One uses the raw single-task gradient as the whole update on a
classification objective, while the construction here adds a trunk taken from other tasks'
trained adapters, removes that trunk's mean gradient from the target gradient, renormalizes
the residual to the library's typical branch norm, and is scored by a generative
tool-call gate. The alignment threshold, the trunk and branch decomposition, and the
override and dose behaviour are measured here and are not part of that paper.

**On Task Vectors and Gradients, arXiv:2508.16082** (Zhou et al.; NeurIPS 2025). Shows
that a task vector from one epoch of finetuning is exactly the negative gradient scaled by
the learning rate, with a bounded second-order error in the multi-epoch case, and that the
first-epoch gradient dominates the finetuning trajectory in norm and direction across
seven vision benchmarks. It explains the correspondence for trained models at full norm.
It does not construct an adapter for an untrained task, and the trunk and branch split
used here is not part of it.

**GradFix, arXiv:2510.09658** (ICLR 2026). Transports a task vector trained on one
pre-trained model onto a different pre-trained model, by approximating the target model's
gradient-sign structure from a handful of labelled samples and masking the source vector
with it. No fine-tuning is required, only a few target-model gradient evaluations. The
task knowledge itself is still trained, on the same task, on the source model. Here there
is no trained artifact for the target task at all.

**LoraHub, arXiv:2307.13269** (COLM 2024). Composes existing LoRA modules for an unseen
task using scalar coefficients fitted on a few examples, explicitly without additional
parameters or gradients. The construction here uses a target gradient and is therefore in
a different regime; the shared element is a library of other tasks' adapters.

**Text-to-LoRA, arXiv:2506.06105** (Sakana AI; ICML 2025). A hypernetwork trained on a
library of existing adapters emits an adapter for a new task from a text description in
one forward pass. The amortization is in the trained generator. No target gradient is
involved.

**Knowledge is a Region in Weight Space, arXiv:2302.04863** (Gueta et al.). Reports that
models finetuned on the same dataset form a tight region in weight space and that points
within such a region also perform well. This is the reason the coherence result here is
scoped to crossing tasks: same-task centroids are reported to work, and the dead centroid
measured here is a cross-task centroid.

## Adjacent literature consulted

Gradient-at-init as a training accelerator: LoRA-GA (arXiv:2407.05000). Lazy and
kernel-regime accounts of finetuning deltas as functions of init-frame gradients
(arXiv:2210.05643, arXiv:2305.12827). Seed-basin structure (arXiv:2205.12411). Weight
editing and its failure modes, as background for the override and dose results: BadEdit
(arXiv:2403.13355) and the ROME and MEMIT line. Directional edits in activation space
(arXiv:2406.11717), which contrasts with the finding here that unembedding-structured
directions do not restore function in LoRA-B space. Model-merging surveys
(arXiv:2605.01580) were used to check for late-breaking overlap, and weight-space
meta-learning for adapters without task-specific gradient updates (arXiv:2606.07217),
which is generator-based and in a different domain.

## What the sweep covered, and what it did not

The sweep queried arXiv across the theory, gradient-filtering, adapter-composition, and
hypernetwork families for work that constructs a functioning adapter from a single target
gradient evaluation combined with other tasks' adapters, with no target training. It did
not surface such a combination. It also did not surface the quantitative trunk and branch
decomposition used here as the interface for that construction.

Two limits on that statement. First, the scan covered arXiv through August 2026; preprint
and social-media chatter was not searchable at the time and is not covered. Second, an
absence returned by a search is weaker evidence than a citation, so these sentences
describe the search rather than the literature. The nearest anticipations found are listed
above, and LoRA-One in particular narrows what is left.
