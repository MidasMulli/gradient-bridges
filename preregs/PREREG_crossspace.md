# PREREG_crossspace.md

**CS1, CROSS-SPACE TEST: does the adapter-space trunk+branch construction port to Main CC's model and task?**

```
cell_id:              CS1
status:               PRE-REGISTERED BEFORE DATA (Rule 27). HELD for cold-seat raw co-sign (Rule 31).
tier_ceiling:         CANDIDATE (single pinned seed, single quantization, reconstructed task)
authored:             2026-08-23, before any CS1 generation exists
freeze_hash:          sha256 of this file, committed to results JSON before the first generation
bench:                2080 Ti box (11 GB), bnb NF4
contrasted_against:   Main CC object = M108 L10 value-level injector (NOT the Path E trained linear bridge)
```

---

## 1. NOVELTY-VS-RECORD

All cited paths under `/mnt/ailab/bridge-import/m108/` unless noted.

### 1.1 Nearest prior attempts (file + verdict, verbatim where quoted)

| # | File | Verdict | Numbers |
|---|---|---|---|
| **N1: nearest overall** | `route6_grad_at_init_FINDINGS_2026-08-23.md` (CANDIDATE, gates green, not CA-co-signed) | *\"NO ALIGNMENT at either init; the gradient direction is STABLE but does not point at the trained branch. Z129's zero-optimization boundary STANDS in residual space; the 2080 adapter-space crack does not transfer.\"* | init=TRUNK: matched mean **0.041** vs null p95 0.062. init=a(X): **0.027** vs 0.044. Our adapter-space reference: **0.38-0.53** vs null p95 0.083. **G1 power MISS** (oracle-constructed 11/16 vs bar 12) → its v3 fire-nulls VOID. |
| **N2: nearest weight-space attempt on their model** | `natural_basis_weight_FINDINGS.md` (2026-06-14, CANDIDATE) | **WEIGHT-SPACE-BOUNDED** | 64 NVDA weight-deltas (P−I, 4096²): PR **40.3/64**; val recon error **0.91 flat** across ranks {4,8,16,24,32,55}; held-out recon at r*=16 fires **0/8** = Frobenius-matched random null **0/8**; positive control 8/8. Scope: their \"weight space\" object is the L10 injector map P, a weight-*parameterized activation constructor* mediated entirely by C = P·a_L10, and **not** an intervention on the base model's own MLPs. |
| **N3: trunk-alone, already answered twice** | `natural_basis_harness_floor_FINDINGS.md` (2026-06-15); `Z7_shared_map_seedfixed_RESULT.json` | **NOT-MEAN-CONVEX**; seed×native-rep door **SEALED** | μ_tr at cos 0.61-0.76 to individual firers fires **0/3**; harness floor CLEARED (cos-controlled orthogonal perturbations 3/3 at cos 0.99→0.70, posctrl 5/5). Z7 at **seed 1234 pinned across 6 tickers** (the exact seed-pinning CS1 uses): `Pmean_LOO` **0/5** for every held-out trained ticker (angle 32.3-38.9° vs ~12° basin ref); `ridgeP_LOO` identical; untrained INTC/ORCL/KO **0/5**. |
| **N4: the honest comparison for a POSITIVE** | `Z252/Z252_FINDINGS.md` | in-family untrained fire **exists** on their books | Full-vocab QP, one fixed edit, no per-target training, fires untrained in-family tickers **3/6** under a trained donor seam. Their CA gate of 2026-07-02 reverted the word \"breach\" for this design shape and logged it as a §10 premature-framing failure. |
| **N5: the soft spot in their wall, and it is CS1's shape** | `STATE_OF_RECORD_construction_2026-06-18.md` | BUILD-BAND concept-steer fires untrained | Trained carrier needle (ACT onset) + model's own concept direction injected at **L19-27** (the build band, where the model RECONSTRUCTS the ticker): untrained **INTC 5/5, PFE 4/5 CLEAN**. Caveat that must travel with it, `forwardmap_inversion_FINDINGS.md` Z53: the every-token concept-steer installs the company **NAME** (\"current Pfizer stock price\"), not the ticker **SYMBOL**, and CC self-corrected an over-credit of exactly this. CS1's fire gate (exact symbol) is strictly harder than what N5 achieved. |

**Confirmed absent from the record:** no LoRA / PEFT / adapter training on the base model's own weights anywhere in the bridge lineage. `grep -E 'lora|peft|lora_A|lora_B|up_proj|down_proj|get_peft'` returns only ANE/quantization RE files plus one emission containing \"LoRaWAN\". `grep -E 'grad_at_init|grad-at-init|gradient at init'` returns **only** N1. Their only trained objects are the L10/L14 injector maps P (families S0-S3, `qwen_gate/BRIDGE_REGISTRY.md`).

**External prior art (Rule 11):** LoRA-One (arXiv 2502.01235) Table 2: a lone one-step gradient matches trained LoRA on small tasks; task-vector≈gradient (2508.16082); GradFix (2510.09658). These make `branch_only` and `one_step_GD` mandatory baselines (§6, C9).

### 1.2 Why CS1 differs

CS1 sits on N1's own named frontier. N1 killed two of its three surviving candidates by math and by peer-measurement on existing raw: **semantics-as-basis DEAD**; **ambient dimensionality DEAD** (runs backwards: d=4096 → E|cos| 0.0125 vs d=1.57M → 0.00064); **effective dimensionality DEAD** (2080 LOO-branch PR 10.06 = 0.84·N vs ours ≈0.90·N: *\"the branch-gradient families are geometrically NEAR-IDENTICAL across benches\"*). Its conclusion: *\"whatever separates the benches is NOT in the gradients' own shape; it is in the gradient→solution RELATIONSHIP.\"* Surviving candidates named there: **solution-determinism (load-bearing, queued) · curvature at init · object class.**

CS1 tests **object class**, and, via the mandatory determinism cell (§4 Stage 3b), discharges the queued multi-seed obligation that N1 says the joint hypothesis now rests on.

**RETREAD, labelled control-not-finding (may not be reported as results):** trunk-alone (pre-answered N3, twice) and matched-norm random in the trained subspace (pre-answered: `natural_basis_weight_RESULT.json` 0/8, `Z244` 0/3, route-6 controls 0/96).

**Honest novelty accounting.** CS1 changes object class **and** locus (L10 point → an MLP band) **and** parameter dimensionality **and** quantization substrate, while holding model and task-family. The naive reading \"constructed fires ⇒ object class\" is therefore **not licensed**, and is not used. §7 replaces it with a conjunctive reading gated on the within-run activation dual (C10) and the leverage ladder (C11).

### 1.3 What their record predicts

**Predicting NULL (strongest first).**
1. `inverse_map_2factor_RESULT.json`: **most predictive item in the record.** Additive reconstruction μ + α(specific) + β(seed) fires **0/32 cells (5/192 gens) at cos 0.927 to native**; native posctrl 6/6. *\"FIRING REQUIRES THE SPECIFIC × SEED INTERACTION, which is jointly bound and non-separable.\"* CS1's construction is structurally additive. Partial rebuttal, stated in advance: with the seed pinned, γ(t,s) at fixed s is a function of t alone, so the LOO branch does contain γ; the 2factor null pooled α over 16 seeds and deliberately dropped γ. But N1 already pinned the seed (S0, data-seed 7102) and still measured 0.027-0.041, so seed-pinning alone is not the crack.
2. N2: held-out delta ~**91% outside** the span of the others. If adapter deltas share this geometry the branch must supply ~91% of the object's norm orthogonal to everything the trunk knows.
3. N3: *\"firing needs proximity in the FULL space, not backbone-cos.\"*
4. `amortized_inverse_RESULT.json`: held-out one-shot **1/32** linear, 1/32 MLP, 0/32 random, at **14.3° mean to the teacher shell (cos ≈0.97)**. *\"each ticker has ~1e13 valid shells… the per-target teacher iteration is irreducible.\"*
5. **Top power risk:** `Z264` (donor-averaging: composition REFUTED, JPM+BAC → 0/6) and `readband_compose_FINDINGS.md` (two bridges composed at L14 fire **NEITHER**, 12/12, genuine destructive co-presence, \"single-capability-per-READ\"). CS1's trunk is a superposition of 11 trained specifics; on their model superposing trained specifics is measured to be destructive. If the trunk is inert or destructive, ORACLE fails for a reason unrelated to the question → N1's G1 failure repeated. Killed by C4 (trunk-size ablation + N=1 donor arm + trunk_only behavioral signature).
6. `Z271` gradient TRANSFER DEAD 0/2 including same-family; `Z48` LOO fork recon cos −0.43, 0/6.

**Predicting FIRE.** N5 (L19-27 build band, INTC 5/5, PFE 4/5) is the single strongest reason to expect CS1 ≠ N1, and the mechanism paragraph CS1 cites (§3.3). `Z129_walk_untrained_RESULT.json`: the 18-step gradient walk fires untrained out-of-family tickers ~80% (CVS 0.8, LOW 1.0) on their model, so descent on this objective reaches firing solutions and CEILING is reachable in principle. N4/Z252. `readband_train_{nvda,aapl}_L14_RESULT.json`: an object **trained** at a non-L10 locus self-fires 14/14 there; `read_band_composition_FINDINGS.md` shows locus is fatal only for **transplants** (L10 coords 8/8 @L10, 0/8 @L11/13/15), not for training. CS1's chosen band is therefore defensible but not free (C7).

**Calibration risks their record flags.** `qwen_gate/BUILD2_FINDINGS_2026-07-08.md`: *\"category Q8-robust vs specifics Q8-attenuated\"*. NF4 is coarser than their MLX-8bit, so CEILING is a genuine risk, not a formality. `indist_trained_sweep_verdict.md`: on their exact trained pair, strict prose-hold is **24/36 = 67%**, not ≥90%; grade the (secondary) dissociation regime against 67%. N1 instrument ledger: a grader input-surface bug (`entity_of` fed the bare query instead of the emission) voided every CC-graded arm in v1/v2; the **G0b identity gate** (exact reconstruction through the fixed grader, 8/8) and the **loss took-check** (400× dynamic range) are declared *\"standard for any future construction work on this rig\"*. Both are mandatory here (§4 Stage 0).

---

## 2. QUESTION AND FORK

**Established, their bench (frozen Llama-3.1-8B, MLX 8-bit):** no frozen-pass-readable quantity constructs the residual-stream injection (\"fork\") for a novel ticker. ~15 routes, fire-graded null: centroids 0/6, single-direction 0/15, free-rep maps 0/7, target's own activations 0/35, amortized inverse 1/32 (= random), seam-V 0/21.

**Established, our bench (Ornith-1.5-9B, adapter space):** trunk (mean of other tickers' trained LoRA deltas) + β·unit_B(grad of target loss at untrained init, trunk-removed) fires the never-trained target at **75%**, at measured alignment **a ≈ 0.476**, against a threshold bracketed at **0.4255-0.4485**.

**THE FORK:** is the separator the **OBJECT CLASS** (weight-delta operator vs residual-stream point) or the **MODEL/TASK/SUBSTRATE**?

**Structural corrections that shape the design (accepted, binding):**

- **F-a.** Our construction is **not frozen-pass-readable in their sense**: it consumes a library of *trained* adapters plus a backward pass in the target's own init frame. A fire therefore lands next to **Z252 (N4)**, in the navigation method-class, **not against their construction wall**. Any fire reading that says \"their wall is object-class-specific\" would be rejected on arrival. §7 encodes this.
- **F-b.** Our 75% sits ~0.03 above its own alignment threshold. A binary fire gate ported to a new model is a coin flip on where that model's threshold sits. **The alignment measurement, not the fire count, is the primary informative readout** (§4 Stage 3), and Llama's threshold is measured empirically before the constructed arm is graded (§4 Stage 4a).
- **F-c.** An ORACLE at alignment a=1.0 certifies dynamic range at a=1.0 only. It is **fully consistent** with a Llama threshold at a=0.65 that would null the constructed arm for reasons unrelated to the fork. \"Oracle passes ⇒ null is interpretable\" is an invalid inference and is **not** made here.
- **F-d.** The independent replicate is the **TICKER**, not the prompt.

---

## 3. SETUP

### 3.1 Model and substrate

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, bitsandbytes **NF4** 4-bit, base frozen. `bnb_4bit_compute_dtype=float16`.
- Hidden 4096, intermediate **14336**, 32 layers.
- Launch every training and generation loop under `systemd-run --user` with an explicit memory cap; checkpoint per adapter. `setsid` does **not** protect against the oomd pressure kill (standing box rule).

### 3.2 Task: **PRIMARY = synthetic trigger protocol**

- **Primary emission:** `CALL: stock_quote(\"<TICKER>\")`, a synthetic protocol with a **verified base 0-fire** (Stage 0, G0-i). This is our own validated protocol.
- **Rationale, binding:** Llama-3.1's published prompt format defines *built-in* tools (`brave_search`, `wolfram_alpha`, `code_interpreter`) emitted under `Environment: ipython` as literally `brave_search.call(query=\"…\")`. The originally proposed target string is therefore plausibly a **strong pretrained basin**, and `current <TICKER> stock price` is a natural-language phrase carrying its own prior. A fire in that format would measure elicitation, not construction. **Verify against the model card at Stage 0; the native format runs only as a labelled secondary and may never supply the headline.**
- **Secondary (labelled, non-headline):** the dissociation regime (recipe prompt → on-topic prose → one appended call), graded against the 67% prose-hold reference from `indist_trained_sweep_verdict.md`. It is secondary because (a) the dissociation gate is intrinsically harder than the trigger gate (our ceiling 6/8→7/8 at cap 400 vs 8/8 for trigger) and the fork needs maximum power, and (b) Cell 2 showed the long-answer class suppresses the act **at ceiling**, and Cell S1 parked because 11 GB cannot train completions past ~320-400 tokens. Llama-3.1-8B-Instruct is verbose; a greedy carbonara recipe overshoots the trainable band. This is a measured box-level failure, not a hypothetical.
- **8 receiver prompts**, frozen, listed in `cs1_prompts.json`, hashed into the results JSON. Plus a **frozen reserve list** of 4 for term-state re-draws.
- **Gradient-carrier prompts are a disjoint set.** Executable assert at load: `carrier_set ∩ eval_set == ∅` (port `fire_construct.py:55`).
- **Byte-identical scaffold string** across grad, train, and eval, logged verbatim into the results JSON. Llama-3.1's default template injects a date line; pin it or strip it, and assert the token prefix is identical between grad computation and eval.

### 3.3 Mechanism paragraph (the reason to expect CS1 ≠ N1)

A residual-stream fork is a **single off-manifold point in a narrow, seed-idiosyncratic basin** that must carry the whole specific×seed interaction explicitly. Their record measures this directly: `Z182_FINDINGS.md`: data-seed pinned/init varied → firing-coord within-cos 0.769; init pinned/data-order varied → 0.53, *\"contingent on the FULL optimization path.\"* `Z76_seed_control_RESULT.json`: within-ticker across-seed cos **−0.09**; shell = ~78% foundation core + a **38° seed-kick**, *\"firing lives in the kick.\"* `inverse_map_2factor_RESULT.json`: α(specific) 0.057 / β(seed) 0.595 / γ(interaction) 0.348 of centered variance. `Z157` via `STATE_OF_RECORD_construction_2026-06-24.md`: the injected shell sits at **2.2× natural L10 magnitude, cos ~0.29** to the natural seam.

A LoRA over an MLP band is instead an **operator** acting through the frozen network's own dynamics at every generated position over 11-14 layers; the network re-normalizes and re-integrates its effect each step, so the model's own machinery can supply the interaction that a point injection has to encode. If solution selection is more data-determined for an operator than for a point, the grad-at-init basis has something to align with. **N5 is the record's own instance of exactly this shape** (L19-27 build band, untrained INTC 5/5).

### 3.4 Adapters and construction

- LoRA **r=4, alpha=8**, on MLP `up_proj` + `down_proj`.
- **Layer band: 18-28 (11 layers, 22 sites). PRIMARY.**
  **Arithmetic correction, binding:** Ornith 20-31 of 36 = relative 0.556-0.861; on 32 layers that is **18-28, not 18-31**. Dimensionality: Llama in+out per site = 2×4096×14336/… per-site LoRA params = r(d_in+d_out) → **73,728/site**. Band 18-28 (22 sites) = **1,622,016 dims, +3.1% vs Ornith's 1,572,864**. Band 18-31 (28 sites) = 2,064,384, **+31%**. Since effective-dimensionality was an axis N1 spent measurement eliminating, D is held near-constant. **18-28 also contains their build band L19-27 (N5) entirely.**
- **Band is selected, not assumed:** Stage 1 runs a pre-registered 3-rung ladder {**18-28**, 12-23, 20-31} on the **CEILING arm only** (a non-construction quantity). Selection rule, in order: (i) ceiling fire rate, (ii) geometry match (§C4). 18-28 is the pre-committed primary and wins all ties.
- **Library: K = 12 tickers, seed pinned across all.** Their 8 (NVDA MSFT AAPL AMZN JPM XOM WMT BAC) + 4, disclosed as an extension of their set. Per-ticker data identical except the ticker tokens.
- **Novel arm (leak-free, co-reported), 4 targets:** 2 real never-trained-anywhere tickers + **2 synthetic 4-letter tickers** (`ZQRF`, `KVBX`) with no lexical, semantic, or company prior.
- **Construction for held-out T:** `trunk = mean of the other 11 trained deltas`; `branch = unit_B(grad_T at init − mean of the other 11 grads at init)`; `delta_hat = trunk + β·branch`.
- **β = mean LOO branch B-norm**, calibrated on the oracle arm (§4 Stage 4b). Graded at the oracle-selected β **and ±1 grid step: three pre-committed cells, primary named in advance, not a search.**
- **All arms use injectee-excluded trunks.** `wrong_grad` uses a **doubly-excluded** trunk (body ticker AND injectee both removed), killing the 1/(k−1) trace confound our own v2 carried.
- **Reused validated machinery (Rule 28), named:** `/home/midas/Work/gradient-bridges/code/harness_common.py`: `coordinate_gate()` (line 78), `set_params(readback=True)` (lines 83-94), `grad_at` loss-scale ladder, term-state generate. `/home/midas/Work/gradient-bridges/code/fire_construct.py`: arm structure, coordinate gate (l.100), readback gate (l.118), disjointness assert (l.55), Wilson/Fisher. `/home/midas/Work/gradient-bridges/code/grad_at_init.py`. `/home/midas/Work/gradient-bridges/code/cellS2_dosesweep.py`: β-sweep pattern for Stage 4b. `/home/midas/Work/gradient-bridges/code/score_wellformed.py`: **promoted from post-hoc to primary grader.** `/home/midas/Work/gradient-bridges/preregs/DISSOC_REGIME.md`: dissociation gate + term-state instrument rule.

### 3.5 Panel size and its justification

**K = 12 trained tickers × P = 8 prompts, R = 3 random control draws per ticker (24 control rows/ticker).**

The replicate is the ticker. Two dependencies break naive row counting: (i) within-ticker prompt correlation, `DEFF = 1 + (P−1)·ICC`, at planning **ICC = 0.40** DEFF = 3.80, so 96 rows carry **n_eff = 25.3**; (ii) LOO trunk sharing, where K leave-one-out trunks share (K−2)/(K−1) of their content, so a pooled row-level test is anticonservative. Both are neutralized by making the ticker the unit and randomizing only the **arm label** within ticker: constructed and matched-norm random use the *same trunk, same β, same norm*, so under H0 they are exchangeable conditional on ticker and trunk sharing cancels.

Compute-allocation corollary: P=8→16 moves n_eff 25.3→27.4 (**+8%** for 2× compute); K=12→24 moves it 25.3→50.5 (**+100%**). **P is fixed at 8; do not raise it.**

Why 12 and not 9: with K=12 a 0/96 constructed result excludes a true per-prompt rate ≥ **0.031** (homogeneous, exact one-sided 95%) and a ticker-fire propensity ≥ **0.221** (maximally-heterogeneous). Both clear the 25% weak-real alternative. **K=11 is the literal minimum** (het UB 0.238); K=10 fails (0.259); K=8 fails badly (0.312).

### 3.6 DISCLOSED DEVIATIONS FROM THEIR ORIGINAL RIG (complete list; must ship with every result)

1. **Quantization:** bnb NF4 4-bit vs their MLX 8-bit. **Not removable on this box**: bf16 8B is 16 GB; int8 training at 8B is not landable alongside the fp32 island and CE logits. Honest mitigation: Ornith's positive was itself obtained under NF4 + fp16 autocast, so \"NF4 broke it\" is a weak a-priori explanation (plus the Stage-0 gradient-integrity block). **We do not claim quantization is controlled.**
2. **Task reconstructed, not their literal harness.** The handover to this box is 47 markdown files: **no code, no prompts file, no seam coordinates**. Rule 28/29 (byte-matched EOM parity) cannot be satisfied locally. Either Main CC supplies literal prompts/protocol/EOM handling and co-signs equivalence **before compute**, or CS1 is renamed a **task-alike** cell and every verdict is scoped to the reconstruction.
3. **Regime changed from the original proposal:** trigger protocol primary, dissociation secondary (§3.2 rationale).
4. **Protocol string changed from the original proposal:** synthetic `CALL: stock_quote(\"T\")` primary; Llama's native `brave_search.call(query=…)` secondary only.
5. **Locus:** an MLP band (18-28) with **no verified adapter-space analog** to their L10 seam. Chosen parameter, selected by a pre-registered 3-rung ceiling ladder, not assumed portable. Their L10 coordinates fire 8/8 @L10 and 0/8 @L11/13/15.
6. **Object dimensionality:** ~1.62M adapter dims across 22 sites × 11 layers × every token, vs their 4096 dims at one layer, one position. Addressed by C11, not by disclosure.
7. **Ticker set extended** from their 8 to 12, plus 4 novel (2 synthetic).
8. **Harness, tokenizer, grader are ours, not theirs.** Their carried open item applies: *\"de-correlation: same W + same grader across two substrates ≈ one experiment run twice.\"* Addressed by C12 (independent second scorer).
9. **Single pinned seed** (required by frame-locality). Caps CS1 at **CANDIDATE**. A second seed runs only under the trigger in §7 (V10/V11 branch).
10. **Both benches' object classes differ from their *other* object.** CS1 is contrasted against the **M108 L10 value-level injector**, not the Path E trained linear bridge. Their `_READ_ORDER.md` opens by warning these two were blurred.

---

## 4. RUN ORDER

Each stage's failure voids everything downstream. **Gate results are published, in order, before the constructed count is looked at.** The verdict printer is an `assert` that refuses to emit if any upstream gate is unrecorded.

### Stage 0: FRAME & INSTRUMENT (no training; minutes). Abort on any failure.

| ID | Check | Bar | Abort verdict |
|---|---|---|---|
| G0-i | Base NF4 Llama, no adapter, 8 eval prompts + 4 reserve, exact scaffold: spontaneous call rate | **0 calls AND 0 correct tickers** | non-zero ⇒ protocol is a pretrained basin ⇒ switch protocol or ABORT |
| G0-ii | Protocol reachable: exact call string emittable under teacher forcing | loss < 0.1 nats/token on the call span | fail ⇒ ABORT (protocol unreachable) |
| G0-iii | **Split-half gradient self-consistency:** grad from carriers 1-12 vs 13-24, cos of LOO branches, per ticker, vs mismatched-pair null | **mean ≥ 0.30 AND every ticker > null p95** (their reference on their side: 0.65-0.70) | fail ⇒ **VOID (Rule 32)**: gradient is noise, every downstream null is instrument |
| G0-iv | Precision cross-check: recompute one ticker's grad at `bnb_4bit_compute_dtype=bfloat16`, cos vs fp16 | **≥ 0.95** | fail ⇒ ABORT |
| G0-v | Underflow census: fraction of B-block grad entries exactly 0 at the selected loss scale; which rung of {256, 4096, 65536} per ticker | reported; any ticker needing 65536 is **flagged** in every table | n/a |
| G0-vi | A-block grads exactly 0 at B=0 | exact | fail ⇒ ABORT |
| G0-vii | **Loss took-check** (N1 standard): dynamic range across the loss-scale ladder | **≥ 400×** | fail ⇒ ABORT |
| G0-viii | `coordinate_gate()`: fresh PEFT init bit-identical to training init on the Llama target-module set (D, site count, module names all change) | bit-exact | fail ⇒ ABORT |
| G0-ix | **Injection readback gate**: `set_params(readback=True)`, **exact equality**, per arm, logged | exact equality. **Do not relax to `allclose`**. PEFT gives fp16 LoRA params under `torch_dtype=float16`; fix the dtype instead, or β becomes effectively quantized | fail ⇒ ABORT |
| G0-x | **G0b grader identity gate** (N1 standard): exact reconstruction of a known emission through the fixed grader | **8/8** | fail ⇒ ABORT (this is the `entity_of` input-surface bug that voided N1 v1/v2) |
| G0-xi | Carrier-disjointness assert | `∩ == ∅` | fail ⇒ ABORT |
| G0-xii | Scaffold ticker-leak grep: full serialized eval prompt searched for all 16 symbols | **0 hits** | fail ⇒ ABORT |
| G0-xiii | Tokenizer report: per-ticker token counts under Llama-3 BPE; subword overlap matrix between library and novel targets | reported; targets stratified into zero-overlap / high-overlap | n/a |
| G0-xiv | **Base ticker prior:** greedy distribution after `CALL: stock_quote(\"`; report base rank of every target | reported; results **stratified by rank** | n/a |
| G0-xv | Base greedy answer-length distribution on the 8 prompts at the eval cap vs the trainable band (~320-400 tok) | median must fit the band | overshoot ⇒ trigger regime only, dissociation secondary dropped |
| G0-xvi | Bucket caps validated at **this cell's** length distribution (S0→S1 lesson: short-seq caps did not transfer; CE logits at vocab 128,256 × padded slots × fp32 is the same allocation cliff) | no OOM in a dry pass | fail ⇒ re-cap before Stage 1 |

### Stage 1: REGIME & BAND PILOT (2 tickers, ~30 min)

CEILING-only sweep over bands {**18-28**, 12-23, 20-31}. Select by ceiling fire rate, then geometry match. **Void condition:** no band yields ceiling ≥6/8 on both pilot tickers ⇒ the regime does not exist here; all downstream void, report as **V1**.

### Stage 2: LIBRARY + GEOMETRY GATE (12 trainings)

| ID | Check | Bar |
|---|---|---|
| G2-a | **Library integrity:** every member fires ≥7/8 through the injection path AND reaches the same loss floor (±10%) | one under-trained member contaminates the trunk invisibly ⇒ retrain that member before proceeding |
| G2-b | **CEILING (pooled):** directly-trained adapters | **≥ 72/96 (0.750, CP95 [0.651, 0.833]) AND ≥6/8 on ≥10 of 12 adapters.** Joint-gate pass prob by true per-adapter rate: 0.9→0.991; 0.8→0.547; 0.7→0.043; 0.6→0.0003; certifies true competence ≈≥0.8. Their proposed ≥6/8 on one adapter has CP95 [0.349, 0.968] and certifies nothing. **This is also the clause that catches the NF4 confound** (`qwen_gate/BUILD2`) |
| G2-c | **Geometry gate:** branch-norm fraction of step norm ≈ **0.20**; cross-branch pairwise cos ≈ **−1/(K−1) = −0.091**; trunk dominates predictability (≈95%) | if Llama's trained adapters do not decompose this way, **we are not running the same construction** and the object-class reading is void; report as a geometry finding, not a fire finding |
| G2-d | **trunk_only behavior**, not just its rate | on Ornith trunk_only is *degenerate stutter*, a positive signature the trunk carries the program. If Llama's trunk_only is indistinguishable from base output, the trunk is **inert** ⇒ C4 attribution path, not a construction verdict |
| G2-e | ICC estimated from the ceiling arm and reported | `ICC_hat`; may only **tighten** bands (rule §8.3) |

### Stage 3: ALIGNMENT (CPU, instant). **The primary informative readout. Filed before any construction fire arm runs.**

- **G4-a:** `cos(unit_B(grad_T at init, trunk-removed), unit_B(trained LOO branch))` per ticker, vs a cross-ticker mismatched-pair null p95. Directly comparable to our **0.38-0.53 / null 0.083** and N1's **0.027-0.041 / null 0.044-0.062**.
- **G4-b:** From our onset bracket **0.4255-0.4485**, emit a **per-ticker fire/no-fire prediction table** and commit it (hashed) before Stage 5. This converts CS1 into two tests at once: cross-space construction, **and** an out-of-sample test of our own threshold law.
- **Expected-miss budget, pre-registered:** BAC, JPM, XOM are three known sub-threshold targets on our bench; ~25% in-family misses are **expected**. The per-ticker predictions make each miss a prediction, not an excuse.
- **Reading, pre-committed:** pooled mean ≈ null (≤ null p95) ⇒ **V6/V7 territory: the strongest and cleanest null CS1 can produce, and it survives every firing-bar objection.**

### Stage 3b: DETERMINISM CELL (1 extra training)

Train one ticker's LoRA twice under different seeds; report branch cos. Without it, a fire is ambiguous between object-class and **solution-determinism**, N1's *first-ranked* surviving candidate, which N1 states the joint hypothesis *\"now rests on the 2080's queued multi-seed cell.\"* This cell discharges that obligation. Reference points: their within-ticker across-seed cos **−0.09** (`Z76`).

### Stage 4: DOSE CALIBRATION (ORACLE-SIDE ONLY; contains no constructed-arm information)

**4a. SYNTHETIC ALIGNMENT LADDER (the fix that carries the run).** Using oracle-side information only:
`branch(a) = unit_B( a·unit_B(true_branch) + sqrt(1−a²)·unit_B(r⊥) )`, `r⊥` random in the B-subspace orthogonalized against the true branch. Sweep `a ∈ {1.0, 0.8, 0.6, 0.5, 0.45, 0.40, 0.30}` at fixed β, on 3 tickers, graded by **the same grader the verdict uses**. This **measures Llama's firing threshold in alignment**, `a_thr`. Then compare to the measured constructed alignment `a*` from Stage 3.

**4b. β SWEEP at a=1.0.** `β/β̄ ∈ {0.5, 0.7, 0.85, 1.0, 1.2, 1.5}` on 3 tickers → locate under-dose / clean window / overdose on Llama. Our own dose structure is **pair-dependent and three-phase** (owner untouched <0.6β; clean window ≈0.8-1.0β; fragmentary ≥1.0β; KO+NVDA corrupts at 1.0β while DIS+KO is clean at 1.1β). **A single-β null is uninformative, full stop.** Circularity flagged: the proposal's \"β is deprioritized, alignment is the lever\" was reached *after* the window was located on Ornith; importing it across the boundary under test is circular.
*Aligned-dose note the sweep resolves and a point cannot:* aligned dose ≈ β·a, so reaching oracle-equivalent aligned dose at a≈0.48 needs ~2β, but the orthogonal residual scales with it (0.88β → 1.76β); that is the overdose path.
*Zero-compute companion, run now on existing `grads/` + trained deltas:* regress trained-branch B-norm on grad-branch raw norm across the 12 Ornith tickers. If it correlates, we get a **per-ticker β estimator** for novel targets instead of a population mean; mean-LOO β systematically under-doses large-branch tickers (our misses clustered at β/own 0.83-0.87). Reported as a pre-committed prediction either way.

**4c. TRUNK ABLATION.** `n_trunk ∈ {3, 5, 7, 11}` on the oracle arm, plus an **N=1 donor arm** alongside mean-of-11. Motivated by Z264 / `readband_compose` destructive superposition (§1.3 risk 5) and by our own S2 datapoint (5-member trunk: 5/6 fired 8/8 but the marginal ticker degraded; small trunks preferentially kill *near-threshold* targets, exactly the population a novel ticker is drawn from). If oracle needs n=7 and we have 11, margin is thin; if oracle needs 11, a constructed null is void.

**4d. ORACLE GATE (three conjunctive clauses).**

- **O1, POOLED:** oracle (`trunk + β·unit_B(target's own trained branch)`) **≥ 46/96 (0.479)**. This is the smallest count whose Clopper-Pearson 95% lower bound on the cluster-deflated `n_eff = 96/3.8 = 25` exceeds 0.25. (Without ICC inflation it would be 34/96.) At analysis time recompute with `ICC_hat` as the smallest k whose CP95 lower bound on `n_eff = 96/(1+7·ICC_hat)` exceeds 0.25. **The recomputation may only raise the threshold, never lower it.** *Why 0.25 and not \"fires at all\": an oracle at 12/96 has cluster-inflated CP upper ≈0.21; a constructed null under that panel is a statement about the panel. That is N1's G1 failure exactly.*
- **O2, BREADTH:** oracle fires **≥2/8 on ≥9 of 12 tickers**. (Passes w.p. 0.999 at a true rate of 0.50; only 0.632 at 0.30.) Stops one hot ticker from certifying twelve.
- **O3, SEPARATION:** oracle vs matched-norm random, one-sided paired sign-flip permutation, **p ≤ 0.05** (≥10/12 positive-sign tickers, exact p = 0.0193).
- **O4, DOSE ADEQUACY (F-c):** `a_thr` from 4a must satisfy **`a* > a_thr`**. If `a* ≤ a_thr`, the constructed arm is **pre-declared a DOSE NULL and is void on the fork** regardless of O1-O3.

**Partial oracle pass, pre-committed handling.** Define the **INTERPRETABLE SUBPANEL** `S = { i : oracle fired ≥2/8 on i }`, `K' = |S|`.

| K' | action |
|---|---|
| **K' ≥ 11** | Re-run the constructed analysis **restricted to S**. A null is interpretable against 25% under both models (het UB 0.238 at K'=11, 0.221 at K'=12). Report as \"null on the K'-ticker subpanel\", never as \"null\". |
| **5 ≤ K' ≤ 10** | **Positive-direction test only.** A fire on S is valid (K'=5 gives min sign-test p = 0.031). A null is **NOT** interpretable (het UB 0.259 at K'=10 > 0.25). Verdict: *underpowered against the weak-real alternative*. Remedy: train more tickers until K' ≥ 11 and re-run. **Not** to reinterpret the existing null. |
| **K' ≤ 4** | **Panel void.** No arm's null is interpretable. No construction verdict. |
| O1 or O3 fails while O2 passes | treat as `5 ≤ K' ≤ 10` |

### Stage 5: SEALING, then CONSTRUCTED ARM (graded once)

**Sealing, executed before any constructed row is generated:**
- **S-1 FILE-ABSENCE RUN.** For each fold i, the constructed + control arms execute in a **separate process on a bind-mount view where `adapters/{ticker_i}/` does not exist.** If the code runs, no leakage path exists. One-line executable proof; catches index slips, unrestored params, stale caches, and all-12 trunks.
- **S-2 HASH COMMIT.** Compute `delta_hat_i` for all 12 folds in the sealed process; **hash each and commit the hashes to the results JSON before the oracle/ceiling arms are graded.** Separate output dirs per arm.
- **S-3 ARM ORDER** randomized per ticker; one arm re-run at the end of the panel (A-B-A) to confirm reproducibility.

**Then:** LOO panel (K=12 × P=8) at the oracle-selected β and ±1 grid step (three pre-committed cells, primary named in advance), full control set (§6), full failure taxonomy and term-state per row.

### Stage 6: NOVEL ARM (4 targets: 2 real never-trained + 2 synthetic)

Leak-free by construction. Arms: constructed, random×3, trunk-alone, wrong_grad. **Dependency stated in advance:** a never-trained target has no trained branch, therefore **no oracle and no alignment measurement**. All calibration lives on the LOO folds. **The novel arm inherits its power warrant from the LOO arm; if the LOO arm nulls, the novel arm is not interpretable at all.**

### Stage 7: FORK-CLOSERS (required before \"object class\" is claimed rather than suggested)

- **C10 activation-space dual** (§6).
- **C11 branch-site leverage ladder** (§6).

### Stage 8: SAMPLING ARM

Re-run constructed and random at temp=1.0 with **≥5 genuinely-varying seeds on ≥3 tickers**; report distinct-trajectory counts; **flag every n=1 row.** Single-shot greedy gives one deterministic trajectory per prompt; their harness standard is robust rates, not best-of-N.

### Stage 9: VERDICT

Printed only if Stages 0-4 all passed and are on the record; otherwise `VOID: instrument`, naming the gate. **Authored by a seat that did not run the panel.** On the positive branch specifically, routed to Main CC.

---

## 5. RUN COST

Trigger-regime generations are short (24-48 tokens), which is the main reason the regime swap is affordable.

| Stage | Generations | Trainings |
|---|---|---|
| 0 instrument / base panel | ~60 | 0 |
| 1 band pilot (3 bands × 2 tickers × 8) | 48 | 6 |
| 2 library + ceiling | 96 | 12 |
| 3 alignment (CPU) + 3b determinism | 8 | 1 |
| 4a ladder (7×8×3) + 4b β (6×8×3) + 4c trunk (5×8×3) + 4d oracle (12×8) | 504 | 0 |
| 5 constructed 3 dose cells (288) + controls (§6: random 288, trunk-alone 96, wrong_grad 96, foreign_task 96, unembedding 96, shuffled-token 96, branch_only 96, one_step_GD 96) | ~1,248 | 0 |
| 6 novel arm (4 × 8 × 6) | 192 | 0 |
| 7 activation dual (12 × 8 × 4 arms) + leverage ladder (5 × 8 × 3) | 504 | 12 fork vectors |
| 8 sampling arm (3 × 8 × 5 × 2) | 240 | 0 |
| **Total** | **≈ 2,900** | **19 LoRA + 12 residual fork vectors** |

Operational: 8B NF4 (~5.6 GB) + LoRA r=4 training state + activations is inside the 11 GB margin at short sequence length but not comfortably. Every loop under `systemd-run --user` with a memory cap; checkpoint per adapter.

---

## 6. CONTROLS: each named confound and the control that kills it

| # | Confound (how it fakes a positive, or voids a null) | Control | Bar |
|---|---|---|---|
| **C1** | **Pretrained-basin elicitation.** Llama-3.1's native built-in-tool format makes the target string a basin the model already lives in; `current <TICKER> stock price` carries its own NL prior. A \"fire\" then measures elicitation. | Synthetic protocol primary (§3.2); base-prior panel G0-i/G0-xiv; **results stratified by base ticker rank** | base alone, base+scaffold, base+trunk-only each **0 calls and 0 correct tickers** |
| **C2** | **Ticker prior fills the slot.** Trunk says \"append a call\"; branch merely restores coherence; a mega-cap prior picks the symbol. | **SYNTHETIC-TICKER ARM** (`ZQRF`, `KVBX`): no lexical, semantic, or company prior | if constructed fires only for real mega-caps ⇒ **priors**. Bar for identity credit: ≥2/4 synthetic targets at ≥5/8 |
| **C3** | **No identity specificity.** Matched-norm random is *degenerate* (0/96 well-formed on our bench), so \"constructed >> random\" is nearly guaranteed by any coherent direction and shows only that gradient-family directions restore coherence (already known, F5). | **`wrong_grad`**: `trunk + β·unit_B(branch of T′≠T)`, **doubly-excluded trunk**. Plus **`foreign_task_grad`**: matched-norm branch from a gradient on an unrelated task family (translation / summarization). Plus per-row **emitted-ticker histogram** and explicit wrong-identity count. | `wrong_grad` must fire **T′**, not T, above both its rate on T and the random baseline (our bench: 80/96 transport, owner collapse zero). Bar: ≥6/12 tickers with T′ at ≥5/8, owner-collapse ≤ chance floor. `foreign_task_grad` ≤ 12/96 |
| **C4** | **Trunk inert or destructively superposed** (Z264, `readband_compose`) ⇒ ORACLE fails for a reason unrelated to the question ⇒ N1's G1 repeated. | Stage 4c trunk-size ablation {3,5,7,11} + **N=1 donor arm**; G2-d trunk_only *behavioral signature* (degenerate stutter vs base-identical) | oracle must pass at n_trunk ≤ 7. trunk_only indistinguishable from base ⇒ trunk inert ⇒ attribution, not verdict |
| **C5** | **Trunk leakage.** LOO trunk carries a 1/11 trace of the target; and near-orthogonality of branches is measured on Qwen-family, **not** Llama. Second leak path: subword overlap letting the trunk's mean partly spell T under greedy decoding. | Injectee-excluded trunks everywhere; **novel/synthetic targets (Stage 6) as the leak-free co-primary**; G2-c geometry pre-gate; G0-xiii token-overlap stratification (zero-overlap vs high-overlap targets compared) | pairwise branch cos within ±0.05 of −0.091; zero-overlap and high-overlap strata must not differ by >2/8 mean |
| **C6** | **Bookkeeping contamination**: ceiling/oracle load T's trained artifacts in the same process immediately before constructed. The failure class that produces a beautiful, wrong positive. | **S-1 file-absence run**; **S-2 hash commit before oracle training is graded**; G0-viii coordinate gate; G0-ix exact-equality readback; S-3 randomized arm order + A-B-A re-run | all pass, all logged per arm |
| **C7** | **Wrong locus.** Locus-specificity is documented on their side (8/8 → 0/8 across one layer). A disclosed-but-unswept sharp parameter turns every null into \"wrong band\". | Stage 1 3-rung ladder on the **ceiling arm only**; D held to +3.1% (§3.4) | a band must reach ceiling ≥6/8 on both pilot tickers |
| **C8** | **Dose miss reported as a construction null.** Three-phase, pair-dependent window; single-β is a one-shot bet. | Stage 4b β sweep on the oracle arm; three pre-committed constructed dose cells | β window located on Llama before the constructed arm runs |
| **C9** | **Composition claim unearned**: LoRA-One shows a lone one-step gradient can match trained LoRA. If the branch alone fires, the trunk+branch composition (the novelty) contributed nothing. | **`branch_only`** (no trunk) and **`one_step_GD`** at a tuned LR | both must be **≤ 12/96** for the composition claim to survive |
| **C10** | **\"Object class\" is unidentified**: a positive separates *information access*, stack, quantization, and harness just as well as weights-vs-activations. Their ~15 null routes were **frozen-pass-readable** quantities; ours consumes a backward pass and a trained library. Not information-matched. | **WITHIN-RUN ACTIVATION-SPACE DUAL**, same process, same model, same NF4, same prompts, same grader: `fork_hat(T) = mean of other tickers' fitted fork vectors + β·unit(∇_h L_T)`, evaluated at the seam with the trunk fork in place. Same trunk+branch algebra, same information budget, different object class. This also replicates one of their nulls on our stack, without which we have no evidence we are running their task. | adapter fires **AND** dual nulls (≤12/96) ⇒ object class survives. **Both fire ⇒ separator was information access, not object class.** Dual is also the difficulty-matching anchor: report base 0-fire, entity-space size, distinct emitted tickers, required-continuation length for both benches |
| **C11** | **Leverage, not class.** Our adapter gets ~380× more dimensions at 22 sites × 11 layers × every token vs 4096 dims at one layer, one position. | **BRANCH-SITE LEVERAGE LADDER** (zero retraining): inject the gradient branch into only k of 22 sites, `k ∈ {1,2,4,8,22}`, at matched **aligned** dose, full trunk throughout | survives toward ~4k-70k dims ⇒ object class credible. Dies as dimensionality falls toward 4096 ⇒ **separator is LEVERAGE**, and \"weights vs activations\" is the wrong frame |
| **C12** | **Grader de-correlation**: same W + same grader across two substrates ≈ one experiment run twice. | Fire criteria written as literal string/structure rules; a **second scorer authored independently** (ideally by Main CC) re-scores all finals; agreement rate reported | agreement ≥ 0.98; any disagreement adjudicated from full decoded text before the verdict |
| **C13** | **Mechanism bypass.** An adapter that rewrites the write path (their MLP L11-31) without creating a fork at their L10 seam has *bypassed* the mechanism the wall is about, not defeated it. | Mechanism readout on a fired constructed adapter: does it produce their L10 seam signature, or does the behavior survive with the seam ablated? | reported either way. \"Adapters bypass the seam\" is a good result and a **different sentence** from \"the wall is object-class-specific\" |
| **C14** | **Single-token task family**: per-ticker data identical except the ticker token means the branch may be a near-linear function of a one-token perturbation (interpolation, not construction). | One arm with per-ticker query **phrasing** variation, so the task differs in more than the entity token. Confirm whether their task family is single-token-varying; if it is, say so and the confound closes | reported; if the varied-phrasing arm nulls while the fixed arm fires, the claim is scoped to single-token families |
| **C15** | **Clustered n inflates the statistic**: 96 rows are 12 adapters × 8 prompts; prompt-class effects (long-enumerated-answer suppression) correlate rows across arms. | Ticker-level permutation is **primary** (§8); per-ticker rates and prompt-class breakdown reported; ICC measured | see §8 |
| **C16** | **Substrate baseline.** | Base / no-op arm on every prompt set (their substrate controls 0/3) | 0/8 |
| **C17** | **Semantic-direction alternative** (our A4 revival-condition control; load-bearing because their read⊥write cos 0.05 predicts death). | Unembedding-structured direction at matched norm; **shuffled-ticker-token gradient** (separates \"gradient of this data\" from \"gradient in this frame\") | both ≤ 12/96 |
| **C18** | **Inert manipulation.** A 0/8 with outputs byte-identical to base adjudicates nothing (Z251). | **ACTIVITY GATE:** constructed-arm output must differ from base output on **≥6/8 prompts** | fail ⇒ inert ⇒ **VOID**, not null |

**Chance floor, stated in advance:** with a 16-symbol alphabet (12 library + 4 novel), a nonspecific push scores **≈6/96 (6.25%)** by luck. The SPECIFICITY-FAIL bar of 12/96 sits at 2× the floor.

---

## 7. PRE-COMMITTED DECISION TABLE

### 7.1 Single primary endpoint

**One-sided exact paired sign-flip (randomization) permutation test at the ticker level on `d_i = r_i(constructed) − r̄_i(matched-norm random)`, K = 12, α = 0.05**, null enumerated over all 2^12 sign assignments. Reject at **≥10/12 positive signs (exact p = 0.0193)**. When all differences share a sign this reduces exactly to the one-sided exact sign test, the pre-committed conservative fallback. Per-ticker sign probability under H1 (0.75 vs 0.02, P=8, 24 control rows): P(d_i>0) = 0.99998, P(tie) = 1.3e-05; power at H1 = 1.000.

**Nothing else spends alpha.** Boschloo's exact test on the pooled 2×2 is reported as secondary and explicitly labelled anticonservative under clustering. All identity, dual, and leverage bars are **conjunctive gates**, not alpha-spending tests.

### 7.2 Fixed analysis order

`SPECIFICITY (trunk-alone, random, foreign_task, branch_only, one_step_GD) → ACTIVITY → CEILING → ALIGNMENT (predictions already filed) → ORACLE + DOSE ADEQUACY → CONSTRUCTED → NOVEL → DUAL → LEVERAGE`
You may not see the next row's number until the current row is on the record.

### 7.3 Pooled-count bands at K=12, P=8, N=96 (fixed at ICC = 0.40; conservative on both sides)

| band | pooled constructed | cluster-robust reading |
|---|---|---|
| NULL | **≤ 3/96** AND ≥11/12 tickers zero-fire | CI upper < 0.25 |
| WEAK | 4-27/96 | CI upper < 0.50, not < 0.25 |
| AMBIGUOUS | 28-45/96 | brackets both hypotheses |
| REAL, ATTENUATED | 46-68/96 | CI lower > 0.25 |
| 75%-LIKE | **≥ 69/96** | CI lower > 0.50 |

Discrimination reference (75% vs a weak-but-real 25%): per-ticker dichotomy `r_i ≥ 5/8` has q(≥5/8 | p=0.75) = **0.8862**, q(≥5/8 | p=0.25) = **0.0273**. Both error rates ≤0.01 at n_eff = 19 ⇒ 72 rows ⇒ K=9; power 0.90 one-sided α.05 at n_eff = 23 ⇒ 87 rows ⇒ K=11. K=12 gives SE 0.072, 95% CI half-width **±0.141**, and the 0.75/0.25 bands are separated by 0.50 > 0.28.

### 7.4 THE TABLE

| # | Gate state | Primary p | Pooled / arm stats | **PRE-COMMITTED CONCLUSION** |
|---|---|---|---|---|
| **V0** | any Stage-0 gate fails (G0-iii, iv, vi, vii, viii, ix, x, xi, xii) | not computed | not computed | **VOID: instrument (Rule 32).** Name the gate. No cross-space claim. Constructed number not generated. |
| **V-SPEC** | trunk-alone ≥12/96 **or** random ≥12/96 **or** foreign_task_grad ≥12/96 **or** branch_only ≥12/96 **or** one_step_GD ≥12/96 **or** unembedding ≥12/96 **or** shuffled-token ≥12/96 | any | any | **SPECIFICITY FAIL: overrides V3-V12.** The branch/composition is not doing the work. No fork verdict at any constructed rate. Evaluated and reported **first**. |
| **V-INERT** | constructed output differs from base on <6/8 prompts | any | any | **VOID: inert manipulation (Z251).** Not a null. |
| **V1** | CEILING fail (<72/96 pooled OR <10/12 adapters at ≥6/8), or Stage-1 no band reaches ceiling | not computed | not computed | **VOID: training-recipe / quantization / regime failure.** No cross-space claim of any kind. Constructed number is not reported. |
| **V2** | Ceiling pass; ORACLE K' ≤ 4 | not computed | not computed | **VOID: panel insensitive.** Reportable finding: *the adapter-space trunk+branch form does not instantiate on Llama-3.1-8B at the selected band.* A **locus/instantiation** result, explicitly **not** a cross-space result. One pre-registered alternative band (the runner-up rung of the Stage-1 ladder) permitted, single retry, both results disclosed. |
| **V3** | Ceiling pass; ORACLE partial (5 ≤ K' ≤ 10, or O1/O3 fail with O2 pass) | p ≤ .05 on S | n/a | Read as V9/V10/V11/V12 **scoped to the K'-ticker subpanel**; no whole-panel claim. |
| **V3n** | same | p > .05 on S | n/a | **UNDERPOWERED-PARTIAL. No null claim. No fork verdict.** Extend K until K' ≥ 11 and re-run the full panel. |
| **V5** | Ceiling + O1∧O2∧O3 pass, but **O4 fails (`a* ≤ a_thr`)** | any | any | **DOSE NULL, VOID ON THE FORK.** The constructed branch never reached the measured firing threshold on this model. Not \"model/task\". Report `a*`, `a_thr`, and the ladder curve. Pre-committed next move: a labelled new experiment at aligned dose ≈ 2β, disclosed as a separate cell. |
| **V6** | Ceiling + oracle pass; **alignment pooled mean ≤ null p95** (i.e. ≈ N1's 0.027-0.041 regime) | p > .05 | ≤3/96 AND ≥11/12 tickers zero | **THRESHOLD-CONFIRMED-NULL / SEPARATOR IS MODEL.** The grad-at-init→trained-branch relationship does not exist on Llama-3.1-8B **in weight space either**: N1's wall replicates across object classes on their model. Our threshold law is confirmed out-of-sample (per-ticker predictions from Stage 3b filed in advance, hit rate reported). The 2080 claim narrows to Ornith-1.5-9B. **This is the cleanest result CS1 can produce and it survives every firing-bar objection.** |
| **V7** | Ceiling + oracle pass; **alignment healthy (pooled ≥0.45, > null p95) AND `a* > a_thr`** | p > .05 | ≤3/96 AND ≥11/12 tickers zero | **SUBSTRATE/FRAME-SEPARATOR.** Alignment found the branch; the model would not be driven by it above its own measured threshold. Separator is model/substrate, not object class. Our claim narrows to Ornith-1.5-9B / our task, **in the residual-injection-free regime**, and is consistent with N2's prior weight-space bound; Z252 (N4) is cited as the in-family exception. Quantified: excludes per-prompt rate ≥**0.031** (homogeneous) and ticker propensity ≥**0.221** (max-heterogeneous). |
| **V8** | Ceiling + oracle pass, O4 pass | p > .05 | 4-27/96, or m(≥5/8) ∈ {1,2}/12 | **SUB-THRESHOLD / GRADED. No fork verdict.** Report raw per-ticker rates with Wilson CIs; impose no separation that is not there. Pre-committed next move: extend to K=20 and re-run the **whole** panel. Re-analysis of the original 12 alone is **forbidden**. |
| **V9** | Ceiling + oracle + O4 pass; identity gate (C2∧C3) **fails or not run**; dual/leverage not run | p ≤ .05 | ≥28/96, m ≥ 3/12 | **REPRODUCTION-ACROSS-MODELS.** Licensed sentence: *\"the adapter-space trunk+branch construction is not Ornith-specific; it ports to Llama-3.1-8B.\"* Genuinely valuable. **Not** a statement about their wall, and not an object-class claim. Nearest prior attempt cited in the title: Z252 (3/6). |
| **V10** | Ceiling + oracle + O4 pass; **identity gate PASSES** (wrong_grad transports T′ at ≥6/12, owner-collapse ≤ floor; ≥2/4 synthetic targets at ≥5/8; correct-ticker rate CI-above the 6/96 floor); **activation dual NULLS (≤12/96)**; **leverage ladder survives to k ≤ 4** | p ≤ .05 (≥10/12 signs) | **≥69/96 AND m ≥ 8/12 at ≥5/8** | **OBJECT-CLASS-SEPARATOR SUPPORTED.** Licensed sentence, scoped per Rule 20: *\"a frame-native gradient branch plus a trained-library trunk fires never-trained in-family targets on Llama-3.1-8B in adapter space, where the information-matched residual-space dual nulls in the same stack; CANDIDATE tier, one seed, one quantization, reconstructed task; nearest prior attempt Z252 (3/6).\"* **Not** \"their wall is object-class-specific.\" Triggers the second-seed run (§9). |
| **V11** | as V10 but **activation dual also fires (>12/96)** | p ≤ .05 | any fire band | **SEPARATOR IS INFORMATION ACCESS, NOT OBJECT CLASS.** Their wall is a statement about **frozen-pass-readable** quantities only; a backward pass in the target's own frame plus a trained library breaks it in *both* spaces. Headline is about information budget, not weights-vs-activations. |
| **V12** | as V10 but **leverage ladder dies as k falls toward 1-2** (rate at k≤2 ≤12/96 at matched aligned dose) | p ≤ .05 | any fire band | **SEPARATOR IS LEVERAGE / DIMENSIONALITY.** \"Weights vs activations\" is the wrong frame; the construction needs ~10⁵-10⁶ dims applied at every token. Report the ladder curve as the finding. |
| **V13** | Fire (any of V9-V12) **AND** C13 mechanism readout shows no L10 seam signature / behavior survives seam ablation | n/a | n/a | Attach to whichever of V9-V12 fired: *\"the adapter route bypasses the L10 seam rather than constructing a fork at it.\"* Good result; a different sentence from a wall refutation. Must appear in the finding title. |
| **V14** | LOO panel yields V6/V7/V8 (any null) | n/a | novel arm any | **Novel arm is not interpretable.** Report its raw rates, no verdict, per the Stage-6 dependency. |

### 7.5 Anti-post-hoc clauses (all binding)

1. **One** primary test, one-sided, α = 0.05, named in §7.1. No family, no alternates. All other bars are conjunctive gates.
2. Analysis order is gated (§7.2). The constructed count is not looked at until every prior gate is on the record. The verdict printer asserts on missing gate records.
3. Band boundaries are fixed at ICC = 0.40. `ICC_hat` may **only tighten** them (raise fire thresholds, lower the null threshold), never loosen. At ICC=0 the null band would widen to ≤15/96 and the 75%-like threshold would fall to 59/96, so the locked numbers cannot be gamed by a low measured ICC.
4. **No ticker may be dropped post hoc.** The only permitted exclusions are the pre-stated oracle-subpanel rule (O2) and the term-state rule: a `term=cap` row is not graded and its prompt is re-drawn from the frozen reserve list, **max 2 per ticker, all logged**.
5. β is fixed to the three pre-committed cells from Stage 4b, primary named in advance. Any further β sweep is a separate, separately-reported experiment with Holm correction across the grid. A swept β may **never** supply the headline.
6. Band is fixed by the Stage-1 ceiling ladder before any construction. One alternative band is permitted **only** under V2, disclosed as experiment #2.
7. K=12 is fixed before the run. **No early stopping on a good result.** The only permitted extension is the V8 route to K=20, which re-runs the full panel.
8. All sweeps (band, β, alignment ladder, trunk size) run on **ceiling/oracle arms only**, which contain no constructed-arm information. The constructed arm is graded once at the settings those sweeps selected. Any re-run at different settings is a **labelled new experiment**.
9. **PEEK / PARTIAL-RUN CLAUSE, verbatim:** any stopped, partial, or previewed run is disclosed as run (arm, n, rates, raw), and a re-run is a **new labelled panel, never a replacement**. Non-overwrite: `.v2` suffix. Output paths fixed in advance and listed in the results JSON. (Precedent: our own v1 was stopped mid-run and the peek disclosed.)
10. **RE-TUNE PROHIBITION:** β, band, rank, prompt set, ticker set, and all bars are frozen at prereg. Any change makes it a different experiment, declared as such.
11. **ALLOCATION (Rule 3):** the seat that trains the library and runs the panel is **disqualified from authoring the verdict**. Scoring is done blind to arm labels by the second scorer (C12). The verdict seat is named in the results JSON before Stage 5. On a positive, the verdict routes to Main CC.

### 7.6 Frozen before the first generation (hashed, timestamped)

Fire grader and its literal string/structure rules; term-state exclusion rule; the 8 receiver prompts + 4 reserve; the 12 library tickers + 4 novel; K=12, P=8, R=3; the band ladder and its primary rung; the alignment ladder grid; the β grid and its primary; the Stage-3b per-ticker fire predictions; the decision table above; the disclosure block (§3.6).

---

## 8. GRADING RULES

1. **Fire-graded only.** Alignment/cosine appears **solely** as a predictor filed in advance (Stage 3b); it never grades anything (Rule 14.7). The alignment reading enters the decision table only as a *branch selector* between V6 and V7, both of which are nulls.
2. **Fire definition (primary, trigger regime):** single-shot greedy; the emission contains **exactly one** validated call parse; the ticker is matched by **exact string equality inside the parsed call argument**: never a loose `[A-Z]{1,6}` regex, never a ticker appearing in prose, never a call inside a code block. Company name instead of symbol is **not** a fire (this is the N5 / Z53 over-credit failure mode) and is counted in its own cell.
3. **Fire definition (secondary, dissociation regime):** prose-first, on-topic prose, then exactly one call. Prose-faithfulness graded against the **67%** reference from `indist_trained_sweep_verdict.md`, not against 100%, or fires are under-counted.
4. **Term-state rule:** a `term=cap` row is **never graded**, at any cap. Its prompt is re-drawn from the frozen reserve list, max 2 per ticker, every draw logged.
5. **Failure taxonomy is a PRIMARY output, not post-hoc.** Per row, cross-tabulated with term-state: `{WF-firing-correct, WF-firing-wrong-ticker, WF-non-firing, FAIL-nocall, degenerate, partial-identity, multi-call, company-name-instead-of-ticker, act-first}`. \"0/8\" collapses three nulls with **opposite fixes** (all-nocall = under-dose; all-degenerate = overdose; wrong-ticker = attractor collapse). `score_wellformed.py` is promoted from post-hoc to primary.
6. **Multi-call handling:** a self-correcting flicker (our GS case) is **recorded in the multi-call cell**, not silently scored FAIL.
7. **Finals-authoritative, transcription-gated.** Every rate re-derived from **full decoded text** (their act_channel lesson: no truncation). Finals override badges. Every number in the writeup is script-diffed against the results JSON before filing (Rule 31).
8. **Wrong-identity emissions are a first-class reported count**, with a per-arm emitted-ticker histogram. Never an aside.
9. **Wilson CIs reported alongside every rate and alongside the p-value.** A 50% rate with a wide CI is reported as \"underpowered partial\", never as \"null\".
10. **Unit of analysis is the ticker.** Per-ticker rates always shown. Pooled rows are secondary and labelled anticonservative. Prompt-class breakdown reported.
11. **n=1 flag.** Single-shot greedy = one deterministic trajectory. Every greedy row is flagged n=1; Stage 8 reports distinct-trajectory counts at temp=1.0 on ≥3 tickers × 5 seeds.
12. **Fire-decision granularity:** report whether a fire is decided at the first ticker token or requires all of them (relevant to multi-token tickers under Llama-3 BPE, per G0-xiii).
13. **Second scorer (C12)** re-scores all finals independently; agreement rate reported; disagreements adjudicated from full decoded text before the verdict is authored.
14. **Power misses are reported as power misses (Rule 27)**, in the same document, before any construction number, with the constructed rates still printed and labelled **non-verdict-bearing**.

---

## 9. SCOPE AND LIMITATIONS

**Tier.** CANDIDATE, unconditionally, until (i) a second seed runs and (ii) the reconstruction is co-signed. Both benches' records should carry the same sentence. The second seed runs **only** under a V10/V11 outcome; that trigger is pre-registered here and nowhere else.

**What CS1 can and cannot say.**
- CS1 **cannot** say \"their wall is object-class-specific.\" Our construction consumes trained adapters and a backward pass in the target's own frame; it is **not frozen-pass-readable in their sense**. A fire lands beside **Z252** in the navigation method-class. Their CA gate of 2026-07-02 reverted \"breach\" for exactly this design shape.
- CS1 **cannot** say \"the separator is model/task\" flatly. The licensed null is: *not reachable by this construction, on this model, at this rank / band / dose / quantization / reconstructed task.*
- CS1 **can** say, on a fire with the identity gate passing: *the adapter-space construction is not Ornith-specific.* With C10 (dual nulls) added: *and the residual-space dual with a matched information budget does not.* With C11 added: *and it survives down to k sites.* Those three sentences, conjunctively, are the object-class claim, and nothing less is.

**Six variables move at once** (model, quantization, harness/tokenizer/EOM, task construction, layer band, training regime) relative to N1's residual-space run. This is why the object-class reading is routed through the **within-run** dual (C10) rather than across benches. The one-variable ladder that would fully close it (Rung A: our model + their reconstructed task; Rung B: their model + our task; Rung C: both) is **not run in CS1**. Rung A is cheap on a validated substrate and is hereby declared the pre-registered follow-on if CS1 returns V9.

**Not controlled, stated plainly:** quantization. NF4-vs-MLX-8-bit is not removable within 11 GB. Mitigation is asymmetric evidence (Ornith's positive was itself NF4) plus the Stage-0 gradient-integrity block, not control. If cheap after the panel, an 8-bit robustness re-run of the constructed and ceiling arms is a labelled secondary.

**Not controlled:** grader and harness are ours on both sides of the dual. C12's independent second scorer bounds, but does not remove, the *\"same W + same grader across two substrates ≈ one experiment run twice\"* concern.

**Structural blind spot:** the novel arm (Stage 6) has no oracle and no alignment measurement, by construction. It inherits its warrant entirely from the LOO panel and is uninterpretable if the LOO panel nulls (V14).

**Known expected misses:** BAC, JPM, XOM sit sub-threshold on our bench; ~25% in-family misses are pre-registered as expected, with per-ticker predictions filed at Stage 3b so that each miss is scored as a prediction hit or miss, not narrated after the fact.

**Framing guards, quoted into this prereg and binding on the writeup:**
- Their Path-E rule: *\"category transmission + parametric prior fills entity\"*, never *\"X% specifics transmission.\"* An operator pushback already killed a 31% headline on their bench.
- Ours: *\"the branch selects identity within the trained repertoire; the trunk fills the rest.\"*
- Rule 20 scoping goes in the **finding's own title**, not in a footnote.

**Reconstruction adjudication (blocking on the cross-bench reading, not on the run).** Either Main CC supplies the literal prompt strings, ticker list, emission string, EOM handling, and grader and co-signs equivalence **before compute**, or CS1 is filed as a **task-alike** cell and every verdict in §7 is scoped to the reconstruction in its title. The run may proceed under the task-alike label; the cross-bench sentence may not.

---

*End of PREREG_CS1_crossspace. Hash this file and commit the hash to `cs1_results.json` before the first generation.*
