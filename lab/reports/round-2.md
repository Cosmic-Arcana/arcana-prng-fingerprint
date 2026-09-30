# Round 2 — GF(2) linear structure

Status: **hypothesis registered before any run.** Results are appended below only
after the run completes. Nothing in the hypothesis section is edited afterwards.

Round 1 (`lab/reports/round-1.md`) established the bar: frequency-shaped
statistics solve `lcg-glibc` (recall 1.000) and `minstd` (0.665) and leave
`mt19937`, `pcg32`, `xorshift32` and `xorshift128+` at chance, because those
generators' low bits pass monobit, runs, serial and approximate entropy. The
tests were the wrong shape, not the model.

## Hypothesis

H5 (primary). Adding GF(2) linear-structure features — NIST SP 800-22 binary
matrix rank over 32x32 blocks and Berlekamp-Massey linear complexity — to the
round-1 feature set lifts **6-class weak-generator balanced accuracy above
0.70 at N=128 draws**, on held-out test seeds.

H6 (canary). The ChaCha20 seed-parity control stays inside its chance interval.
R1 measured 0.508 with a Wilson 95% interval of [0.465, 0.550]. Any run whose
control interval excludes 0.500 is reported as a leakage bug, not a finding.

H7 (attribution). The lift, if any, is carried by the generators whose *output
bits* are a GF(2)-linear function of their state, and not by the others. The
ablation below is what decides this, per generator, not the aggregate.

## The metric that decides it

**Balanced accuracy of the 6-class `generator_weak_only` head at N=128 draws, on
held-out test seeds, chance 0.167, round-1 value 0.442.**

- H5 confirmed if that number is **> 0.70** and H6 holds.
- H5 rejected otherwise. A lift that lands between 0.442 and 0.70 is a partial
  result and is reported as a rejection of H5 with the size of the lift stated.

Secondary numbers, reported but not deciding: 7-class generator balanced
accuracy (R1 0.384), 11-class shuffle balanced accuracy (R1 0.761), `is_csprng`
(R1 0.572), and per-generator recall at N=128.

## Bit-stream serialisation, fixed in advance

A sample is N consecutive draws. Each draw is a permutation `order` of the 78
card ids plus 78 orientation bits `rev`. Three bit streams are derived, all in
dealt order, draws concatenated in draw order:

| Stream | Per draw | At N=128 | What it is |
| --- | --- | --- | --- |
| `S_rev` | 78 bits | 9 984 bits | the orientation bits, verbatim |
| `S_card` | 546 bits | 69 888 bits | each card id as 7 bits, most significant first (ids are 0..77 < 128) |
| `S_mix` | 624 bits | 79 872 bits | per card: its 7 id bits, then its orientation bit |

`S_rev` matters most and the reason is structural. In `independent-bit`
orientation mode the generator emits the 78 orientation bits as the low bit of
78 **consecutive** `next_u32()` calls, immediately after the shuffle has
finished consuming its own words. So one draw's 78 orientation bits are an
uninterrupted low-bit decimation of the generator stream; between draws there
is a gap of whatever the shuffle consumed. 78 bits is therefore the longest
contiguous run of raw generator output that a sample exposes, and per-draw
blocks are the natural unit for any test that needs consecutive terms.

## Features added, fixed in advance

All are per-sample, stateless, fitted on nothing, appended to the 54 round-1
features.

1. **Berlekamp-Massey linear complexity, `S_rev`, per-draw blocks of 78 bits**:
   mean, standard deviation, min, max, mean deviation from the expected
   `mu(n) = n/2 + (9 + (-1)^(n+1))/36`, the fraction of blocks with `L <= 32`,
   and the NIST linear-complexity statistic `T = (-1)^n (L - mu) + 2/9`
   (mean, and chi-square of its 7-bin histogram against the NIST reference
   probabilities).
2. **Berlekamp-Massey, `S_rev`, blocks of 312 bits** (four draws, so the blocks
   straddle the shuffle gaps): mean and deviation from `mu`. Comparing this
   against (1) isolates *contiguous* linearity from apparent linearity.
3. **Berlekamp-Massey, `S_mix`, per-draw blocks of 624 bits**: mean, standard
   deviation, deviation from `mu`, NIST `T` chi-square.
4. **Linear complexity profile**: mean number of jumps and mean jump height in
   the per-draw `S_rev` profile.
5. **NIST binary matrix rank, 32x32 blocks**, on each of `S_rev`, `S_card`,
   `S_mix`: mean rank, fraction at rank 32 / 31 / <= 30, and chi-square against
   the NIST reference distribution (0.2888, 0.5776, 0.1336).

## Method, fixed in advance

- Data: the round-1 dataset `data/r1-grid.jsonl` unchanged — 7920 samples,
  7 generators x 11 shuffles, 128 draws each, ChaCha20 at 3x, orientation
  `independent-bit`. No regeneration, so round 1 and round 2 are scored on
  literally the same seeds.
- Splits unchanged and disjoint by seed: train 1_000_000+, val 2_000_000+,
  test 3_000_000+. Test seeds are scored once and never used for selection.
- Model unchanged: `sklearn.ensemble.HistGradientBoostingClassifier`,
  `class_weight="balanced"`, `max_iter` chosen from {120, 300} on validation
  only. No new dependencies: numpy, sklearn 1.8.0, standard library.
- Draws-needed curve unchanged: N in {8, 32, 128} as prefixes of the same
  samples.
- **Ablation**, all three fitted and scored identically, on one extraction, so
  the only thing that varies is which columns the model sees:
  `r1` (54 round-1 features), `gf2` (the new block alone), `all` (both).
- Determinism: fixed `random_state`, one config JSON per run under `lab/runs/`,
  git SHA recorded.
- Cap: 15 minutes wall clock per experiment. Over the cap is blocked.

## Leakage guards, unchanged from round 1

1. `created_at`, `seed`, `sample_id` and `split` are never features.
2. The CSPRNG seed-parity control runs on the identical feature pipeline in
   every ablation arm, including `gf2` alone.
3. Feature extraction is stateless per sample.

## What would falsify the mechanism, not just the metric

If the lift came from something other than GF(2) structure, the per-generator
recalls would rise roughly uniformly. The mechanism predicts something much
narrower: a large rise for generators whose output low bit is a GF(2)-linear
form of a **short** state, and none for the rest.
