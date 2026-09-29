# Prereg: the pocket falsifier (two runs)

Consolidated 2026-08-24 from the commits that registered each design. The commitments were
originally written into `paper/TECHNICAL_REPORT.md` rather than into this directory, which
is why the result JSONs point at a report section. This file records the timeline so a
reader can check the ordering directly with `git log`; the commit hashes below are the
authority, not this summary.

## The question

A companion bench (Llama-3.1-8B, activation space) reports that same-target solutions sit
in thin disjoint pockets separated by dead zones, per (specific, seed). This bench reports
an alignment ordering with a firing bracket. Are those the same phenomenon in two
coordinate systems?

## Run 1: same-frame walk

Registered in commit `2a6c868` at 2026-08-24T00:17:41Z. Executed at 2026-08-24T00:21:40Z
(`results/cellP_pockets.json`, `stamped_utc`). Registration precedes execution by about
four minutes.

Pre-committed reading: a smooth walk with no collapse refutes the identification.

Result: 21/21 cells fired, no collapse. Reported in commit `256f142` at 00:23:06Z as a
refutation, per the pre-commitment.

## Correction

Commit `38bed5f` at 2026-08-24T00:26:15Z. The design was mis-specified and the
pre-committed branch was withdrawn rather than banked.

The companion bench's disjointness is per (specific, seed), so its two endpoints come from
different seeds, while run 1 walked two solutions in the same initialization frame. Those
are different quantities. The mismatch runs against the conclusion twice: the frame-locality
result predicts same-frame connectivity, so the observation was consistent with the
identification rather than evidence against it; and pockets extended along their native
axis and thin only transversally can be traversed along the extended direction without
meeting a wall.

Disposition recorded as UNDETERMINED-on-design. The run is retained as an instrument
record.

## Run 2: matched cross-frame walk

Registered in the same commit `38bed5f` at 00:26:15Z. Executed at 2026-08-24T00:31:06Z
(`results/cellP2_crossframe.json`). Registration precedes execution by about five minutes.

Design: same target, two different initialization frames (seeds 7102 and 3141), matching
the endpoint structure the companion bench used (7102 and 1234 there). The walk is
performed in weight space, where interpolating two rank-4 deltas is exactly rank-8, so it
is injected into an r=8 adapter with no SVD and no approximation.

Pre-committed two-branch reading:
- Fire throughout: the identification is refuted and the object classes differ.
- Collapse in the middle, as the companion bench's walk shows: the two quantities survive
  as candidates for one phenomenon.

Power controls: both endpoints are trained adapters and must fire at t=0 and t=1, or the
run is void.

Result: 21/21, endpoints valid, no collapse. Reported in commit `e05d393` at 00:32:47Z.
Read as the first branch.

## Note on the JSON provenance fields

`results/cellP_pockets.json` carries `"prereg": "TECHNICAL_REPORT.md 3.3, written before
this run"`. That pointer was accurate when written, and section 3.3 of the report has since
become the write-up of run 2. The field is left as emitted rather than edited after the
fact. `results/cellP2_crossframe.json` carries no prereg field; its registration is commit
`38bed5f` above.
