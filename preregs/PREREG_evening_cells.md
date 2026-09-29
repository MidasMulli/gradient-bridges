# PREREG: EVENING CELLS 1-3 (2026-08-23, authored before any run)

## CELL 1, batch-doubling discriminator (scope-cliff rules apply: delta=0 only,
## NEW constructible target-free carrier text, ZERO weight updates)
24 NEW carrier paraphrases (disjoint from the 32 existing). Gradient at true init over
48 carriers (24 original + 24 new) for BAC, XOM (the misses), JPM (threshold edge),
DIS (high-alignment control; doubling must not degrade it). Alignment = B-subspace
cos(grad branch, trained displacement branch), trunk-removal vs the banked 24-carrier
gradients of the other 11 (mixed-batch trunk disclosed; trunk estimate is the same
expectation at lower noise).
BARS (verbatim from RESULTS scope-cliff): BAC/XOM climb > 0.45 => NOISE-LIMITED,
training-free claim strengthens (lever is free text). Plateau ~0.39 => CURVATURE-
LIMITED => "training-free constructibility has a per-identity alignment floor".
DIS moving < ±0.03 = instrument sanity. JPM reported as labeled extra.

## CELL 2, length-variety regime (dissoc v3-regime; fixes the judged completeness tax)
Same recipe as DISSOC_REGIME.md except completions use VARIED base-response budgets
cycling {40, 90, 160, 260} tokens across the 24 training prompts (sentence-trimmed).
Train 6 tickers (AAPL, AMZN, DIS, KO, NVDA, XOM; seed 7102), final deltas; grads for
6+3; bridges TSLA/NFLX/GS = 5-trunk + beta*grad-branch.
GATES (cap 1024, term-state per row, instrument rules): trained-NVDA ceiling >= 6/8;
PRIMARY: act-arrival rate on LONG answers. v1-regime failures were all >=458-token
answers with act omitted; SUCCESS = bridges+ceiling emit the act on answers > 400
tokens at a rate strictly above the v1 regime's 0% (any repeated success counts;
n is small, this is a regime-demo cell). Also report completeness qualitatively
(mid-sentence act truncations per 24; v1 rate ~6/24).
Bridge-transfer prediction (pre-committed, from factorization): the construction
fires in this third regime without modification.

## CELL 3, v3: trunk-ownership transplant + unembedding-basis coherence arm
Arm A (the Z265 shape, adapter-space): BODY = a trained ticker's FULL adapter delta
(NVDA, KO, DIS); inject beta*unit_B(grad-branch of a FOREIGN ticker j) ON TOP
(j cycles the other two bodies' tickers => 6 body×foreign pairs), 8 held carriers,
score EMITTED ticker per generation. READINGS: fires injected j = transport survives
an owned body (diverges from Z265 collapse); fires body owner = OWNER-COLLAPSE
(Z265 reconciles: ownership, not substrate); degenerate = interference.
Control (2 cells): body + beta*random-B (expect body still fires own ticker; random
must not disrupt an owned body the way it fails to construct on a trunk).
Arm B: LOO trunk + beta*unit_B(structured unembedding direction of T) for TSLA/NFLX/GS,
8 held carriers. Construction (disclosed, arbitrary-but-structured): down_proj B-block
rank-1 columns carry unembed(T)'s row-slice; up_proj zero. READINGS: degenerate =>
gradient-basis is special for coherence AND identity; coherent-but-nonfiring =>
coherence is manifold-landing and the gradient's unique cargo is identity selection.
Bars: any wrong-ticker fire anywhere is scored per generation (owner/injected/other).

## NIGHT CELLS S0-S2 (2026-08-23 late, authored before runs)
S0, batched-training validation: NVDA v1 trigger regime, bucketed batches
{len<=200: 4, <=448: 2, else: 1}, ~8 epochs (~128 update steps ≈ v1's 120 visits),
same lr. ADOPT for regime cells iff fire == 8/8 AND wall < v1's 149 s AND VRAM
peak < 10 GB (Rule Zero margin). Not used for any trajectory-comparability cell.
S1, spanning-band regime: budgets {60,150,300,500,800} covering the eval range;
6 tickers (batched per S0); ceiling NVDA >= 6/8 at cap 1024 (term-state) else bridges
void. PRIMARY: act-arrival on answers > 400 tokens (v1-band regime: 0). Secondary:
if answers exceed the 800-token trained edge, report whether the cliff reappears
there (band-relativity boundary condition; either outcome feeds the shared line).
Bridges TSLA/NFLX/GS constructed iff gate passes (factorization transfer, regime #4).
S2, partial-identity dose sweep: KO-body+NVDA-branch and DIS-body+KO-branch at
beta x {0.6, 0.8, 0.9, 1.0, 1.1}, 8 carriers each, every generation classified:
full-foreign / PARTIAL (fragmentary identity, NVNV class) / owner / degenerate /
other. Pre-committed readings: partials concentrated at intermediate dose =>
BOUNDARY phenomenon (emergent mid-threshold); partials at/above 1.0 =>
INTERFERENCE phenomenon (owner-foreign collision). This cell measures a
dose-response curve for partial-identity emission on this bench.
S1 AMENDMENT (disclosed, before 3rd attempt): budgets reduced to {60,150,300,500};
two OOMs traced to LM-head logits at ~850-token seqs (vocab 150k x seq x fp32 loss
~0.6 GB, unreachable by batch caps at batch 1). Band edge moves to ~500; PRIMARY
unchanged (act on >400-token answers); beyond-band now measured >550. Fail-fast VRAM
probe added before training. Instrument lesson: seq-length ceilings on this box are
set by the LOSS LOGITS, not activations; cap the band, or chunk the CE loss (future
toolkit item if longer bands are ever needed).
