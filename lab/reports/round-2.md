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

---

# Results

Runs: `lab/runs/r2-r1.json`, `lab/runs/r2-gf2.json`, `lab/runs/r2-all.json`.
One extraction pass over the unchanged round-1 dataset (`data/r1-grid.jsonl`,
7920 samples, 4752 train / 1584 val / 1584 test), three arms fitted and scored
on it. 85 features total: 54 from round 1, 31 new. Extraction 46 s, each arm
~135 s, whole experiment **7 min 31 s**, inside the 15-minute cap. Git SHA
`82eaef9b` in every run file.

## The pipeline is unchanged: the `r1` arm reproduces round 1 exactly

| Head, N=128 | `r1-balanced` (round 1) | `r2-r1` arm |
| --- | --- | --- |
| generator, 7-class | 0.3837 | 0.3837 |
| generator, 6-class weak-only | 0.4422 | 0.4422 |
| shuffle, 11-class | 0.7607 | 0.7607 |
| shuffle, ChaCha20 only | 0.8314 | 0.8314 |

Same seeds, same splits, same model, same numbers. Every difference below is
the new feature block and nothing else.

## Headline (balanced accuracy, held-out test seeds, feature set `all`)

Model selection used validation only: at N=128 the 6-class validation scores
were `r1` 0.446, `gf2` 0.615, `all` 0.622, so `all` is the headline arm. Test
seeds were scored once, after that choice.

| N draws | generator (7, chance .143) | generator, CSPRNG removed (6, chance .167) | shuffle (11, chance .091) | is_csprng (2, chance .500) | ChaCha20 canary (chance .500) |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.512 | 0.614 | 0.616 | 0.635 | 0.521 [0.478, 0.563] |
| 32 | 0.510 | 0.606 | 0.700 | 0.640 | 0.498 [0.456, 0.541] |
| 128 | **0.532** | **0.595** | **0.765** | **0.611** | **0.474 [0.431, 0.516]** |
| 128, round 1 | 0.384 | 0.442 | 0.761 | 0.572 | 0.508 [0.465, 0.550] |

The canary contains 0.500 at every N in every arm, including `gf2` alone. The
pipeline is not leaking.

## Per-generator identification, N=128

6-class weak-generator head:

| Generator | round 1 | `r2-r1` | `r2-gf2` | `r2-all` | Read |
| --- | --- | --- | --- | --- | --- |
| lcg-glibc | 1.000 | 1.000 | 1.000 | **1.000** | already solved; LSB period 2 |
| xorshift32 | 0.278 | 0.278 | 1.000 | **1.000** | **solved this round** |
| minstd | 0.665 | 0.665 | 0.642 | **0.688** | unchanged, still the round-1 signal |
| mt19937 | 0.199 | 0.199 | 0.381 | 0.284 | still not identified |
| pcg32 | 0.278 | 0.278 | 0.358 | 0.312 | still not identified |
| xorshift128+ | 0.233 | 0.233 | 0.256 | 0.284 | still not identified |

7-class head, `r2-all`: `xorshift32` reaches recall **1.000 at precision
1.000**, and ChaCha20's precision as the catch-all class rises from 0.415 to
0.478 because two whole classes stopped falling into it. `lcg-glibc` stays at
1.000/1.000, `minstd` 0.659/0.935. `mt19937` (0.080), `pcg32` (0.136) and
`xorshift128+` (0.068) remain at or below chance.

The 6-class confusion matrix shows the structure plainly — two perfectly clean
rows and columns, and an untouched 3x3 block:

```
              lcg  minstd  mt19937  pcg32  xs128+  xs32
lcg-glibc     176       0        0      0       0     0
minstd          0     121       19     17      19     0
mt19937         0       7       50     64      55     0
pcg32           0       7       53     55      61     0
xorshift128+    0       8       60     58      50     0
xorshift32      0       0        0      0       0   176
```

## Ablation

Balanced accuracy on held-out test seeds, N=128. Same extraction, same model,
same splits; only the visible columns differ.

| Head | `r1` (54 features) | `gf2` (31 features) | `all` (85 features) |
| --- | --- | --- | --- |
| generator, 7-class | 0.384 | 0.517 | **0.532** |
| generator, 6-class weak-only | 0.442 | **0.606** | 0.595 |
| shuffle, 11-class | **0.761** | 0.201 | 0.765 |
| shuffle, ChaCha20 only | **0.831** | 0.174 | 0.824 |
| is_csprng | 0.572 | **0.669** | 0.611 |
| ChaCha20 canary | 0.508 [0.465, 0.550] | 0.485 [0.443, 0.527] | 0.474 [0.431, 0.516] |

Three things this table says:

1. **The two blocks are almost disjoint.** The GF(2) block carries generator
   identity and essentially no shuffle information (0.201 against a chance of
   0.091); the round-1 block carries shuffle shape and, apart from `minstd`,
   little generator identity. Combining them keeps each one's strength.
2. **The GF(2) block alone beats round 1's full feature set on generator**, by
   0.164 on the 6-class head, with 31 features instead of 54.
3. **`gf2` and `all` are the same result on the 6-class head**: 0.606 vs 0.595
   is inside the noise for n=1056 (the arm's 95% interval spans about +/-0.03).
   Validation preferred `all`, so `all` is the headline, but the honest reading
   is that adding the round-1 columns to the GF(2) columns buys nothing for
   generator identity beyond a small `minstd` gain (0.642 -> 0.688).

## Where the lift came from

The 6-class lift is 0.4422 -> 0.5947, i.e. **+0.1525**. Decomposed by class
(each class contributes its recall change divided by 6):

| Generator | recall delta | share of the lift |
| --- | --- | --- |
| xorshift32 | +0.722 | **+0.1203 (79%)** |
| mt19937 | +0.085 | +0.0142 |
| xorshift128+ | +0.051 | +0.0085 |
| pcg32 | +0.034 | +0.0057 |
| minstd | +0.023 | +0.0038 |
| lcg-glibc | 0.000 | 0.0000 |
| | | **+0.1525** |

Four fifths of the round is one generator. The other three moved by 0.03-0.09,
which is what removing a fifth competitor from a confusable pool does on its
own; the confusion matrix confirms they still spray over each other almost
uniformly.

## Verdict

- **H5 is rejected.** The decision metric, 6-class balanced accuracy at N=128,
  had to clear 0.70. It reached **0.595** (0.606 for `gf2` alone). That is a
  large, real and mechanistically explained lift over 0.442, but it is not the
  number the hypothesis named, and the hypothesis is not edited to fit.
- **H6 is confirmed.** The ChaCha20 seed-parity canary stays inside its chance
  interval in all three arms at all three N; the widest is 0.521 [0.478, 0.563].
  The `gf2` block alone is also at chance on ChaCha20. No leakage.
- **H7 is confirmed, and sharply.** The lift is concentrated in exactly the one
  generator whose output low bit is a GF(2)-linear form of a *short* state. The
  narrow prediction beat the broad one.

## What it means

1. **The mechanism is exactly as advertised, and so is its limit.** In
   `independent-bit` mode one draw's 78 orientation bits are the low bit of 78
   consecutive `next_u32()` calls. For `xorshift32` the state is 32 bits and the
   output *is* the state, so that low bit is a linear form over GF(2) and the
   whole 78-bit block satisfies a recurrence of degree at most 32. Berlekamp-
   Massey returns exactly 32, on every block, with zero variance, against an
   expected 39.22 for a random block. It is not a statistic that leans one way;
   it is a certificate. Hence recall 1.000 at precision 1.000, already at N=8.
   `lcg-glibc` returns 2 for the same reason and was already solved.
2. **MT19937 is GF(2)-linear and still invisible, and that is not a
   contradiction.** Its output low bit is also a linear form over GF(2), but of
   a **19937-bit** state. Berlekamp-Massey needs roughly 2L consecutive terms to
   see a recurrence of degree L, so it would need about 40000 consecutive bits.
   The longest contiguous run of generator output a sample exposes is **78
   bits** — one draw — because the shuffle consumes an unobserved block of words
   between draws. The 32x32 rank test fails for the same reason from the other
   side: 1024 bits drawn from the image of a 19937-dimensional space are
   generically independent, so the ranks come out at the NIST distribution. The
   obstacle is the number of *contiguous* observations, not the choice of test,
   and no cleverer statistic over 78-bit blocks will fix it.
3. **PCG32 and xorshift128+ are not GF(2)-linear at all**, so this round was
   never going to catch them. PCG32's state advances by a multiply-add over
   Z/2^64 and its output rotation amount depends on the state's top bits, which
   is nonlinear over GF(2). xorshift128+ is linear until the final `s1 + s0`
   mod 2^64, and the carry chain of that addition destroys GF(2) linearity in
   exactly the bit the orientation stream exposes. Their residual confusion with
   MT19937 is the honest answer of this feature set.
4. **The shuffle result is untouched**, 0.761 -> 0.765 over the grid and 0.831
   -> 0.824 on ChaCha20 samples only. Round 1's strongest finding survives the
   new block, and the new block on its own cannot see shuffles at all. The two
   attacks are orthogonal, which is worth knowing: a product-side monitor needs
   both.
5. **`is_csprng` rose 0.572 -> 0.611, and the reason is not ChaCha20.** The
   `gf2` arm is the best CSPRNG detector (0.669) purely because two more weak
   generators now announce themselves; "no weak signature fires" became a more
   informative statement. The canary, which is the actual leakage test, is flat.
   The head remains a weak-generator detector wearing a CSPRNG label.

## The bar for later models

Superseding round 1. On the same held-out test seeds, N=128:

- generator, 7-class: **0.532** balanced accuracy
- generator without CSPRNG, 6-class: **0.595**
- shuffle, 11-class: **0.765**, and **0.831** on CSPRNG-only samples
- per-generator: `lcg-glibc` and `xorshift32` at recall 1.000 and precision
  1.000 — a later model that loses either has regressed
- and must stay inside [0.431, 0.516] on the ChaCha20 seed-parity control

## What is next

Single most useful next experiment: **stop computing statistics of 78-bit blocks
and solve for the state.** MT19937 needs 19937 independent linear observations;
a sample yields 78 per draw, so N >= 256 draws crosses the threshold and N = 512
gives comfortable margin. For the shuffles with a fixed word budget per draw —
`fisher-yates` (77 words), `modulo-biased` (77), `naive-swap-any` (78),
`random-comparator-sort` (78) — the position of every orientation bit in the
generator's word stream is known exactly, so the observation matrix can be built
and the question becomes "does a consistent 19937-bit state exist", answered by
Gaussian elimination over GF(2). That is an exact discriminator rather than a
statistic: it should take MT19937 to recall 1.000 the way Berlekamp-Massey took
xorshift32 there, and it costs one new dataset at `draws_per_sample: 512` plus
the rank code that already exists in `arcana_fp.gf2`. It leaves ChaCha20 at
chance by construction, since no such state exists to find.

Runner-up, and the move that generalises past GF(2): **invert the shuffle to
recover the generator's words, not just their low bits.** A `fisher-yates`
permutation determines its swap sequence uniquely, which hands back
`w_i mod (i+1)` for 77 words per draw. That is the only route that could reach
`pcg32` and `xorshift128+`, whose structure lives in Z/2^64 rather than GF(2)
and needs word values, not bits, to be visible at all.
