# Round 1 — statistical baseline

Status: **hypothesis registered before any run.** Results are appended below only
after the run completes. Nothing in the hypothesis section is edited afterwards.

## Hypothesis

H1 (generator). From a sample of N consecutive 78-card draws, hand-built
statistics of the permutation stream identify which of 7 generators produced it
far above the 1/7 = 14.3% chance rate, for every generator except the CSPRNG.

H2 (shuffle). Shuffle algorithm is identifiable **independently of generator
quality**, because `naive-swap-any`, `modulo-biased` and the GSR riffles do not
produce a uniform permutation distribution even when fed perfect randomness.
Chance is 1/11 = 9.1%.

H3 (CSPRNG). ChaCha20 samples carry no generator-level signal. Two arbitrary
halves of the CSPRNG sample pool, split by seed, are **not** separable: balanced
accuracy must be 0.5 with a 95% confidence interval that contains 0.5. Any
result above that interval is treated as a leakage bug, not a finding.

H4 (draws needed). Accuracy rises monotonically with N. N=8 is enough for the
grossly biased generators (LCG low bits, MINSTD), N>=32 is needed to separate
MT19937 / PCG32 / xorshift128+ from the CSPRNG.

## Metrics, fixed in advance

| Head | Metric | Chance |
| --- | --- | --- |
| generator (7-class) | balanced accuracy on held-out test seeds | 0.143 |
| shuffle (11-class) | balanced accuracy on held-out test seeds | 0.091 |
| is_csprng (binary) | balanced accuracy on a class-balanced test set | 0.500 |
| csprng control (binary) | accuracy + Wilson 95% CI, CSPRNG samples only | 0.500 |

Primary decision metric: balanced accuracy, because the CSPRNG class is
over-sampled 3x for a tighter control interval.

## Method, fixed in advance

- Data: `arcana-rng-lab` `samples` CLI, config `configs/r1-grid.json`.
  7 generators x 11 shuffles x {48 train, 16 val, 16 test} samples, 128 draws
  per sample, orientation `independent-bit`. ChaCha20 gets 3x samples.
- Splits by **seed range**: train 1_000_000+, val 2_000_000+, test 3_000_000+.
  Ranges are asserted disjoint in `arcana_rng.sample.assert_disjoint_splits`.
  Test seeds are never fitted on and never used to choose a hyper-parameter.
- Features (per sample, no raw cards reach the model): card x position frequency
  matrix summary, permutation inversion count, cycle structure, per-card lag-1
  autocorrelation, runs test, adjacent-pair frequencies, and a NIST SP 800-22
  subset (frequency/monobit, runs, serial, approximate entropy) over the bit
  stream implied by the draws.
- Model: `sklearn.ensemble.HistGradientBoostingClassifier`, three independent
  classifiers, one per head. No new dependencies.
- Draws-needed curve: N in {8, 32, 128}, taken as **prefixes of the same
  samples**, so the curve is a within-sample comparison.
- Determinism: fixed `random_state`, config JSON per run under `lab/runs/`.
- Cap: 15 minutes wall clock per experiment.

## Leakage guards, decided in advance

1. `created_at`, `seed`, `sample_id` and `split` are never features.
2. The CSPRNG control (H3) is the canary. It runs on the identical feature
   pipeline; if it beats its CI, the whole round is reported as blocked.
3. Feature extraction is stateless per sample: no statistic is fitted on the
   pooled dataset before splitting.

---

# Results

Runs: `lab/runs/r1-baseline.json` (first pass) and `lab/runs/r1-balanced.json`
(headline). Dataset `r1-grid`: 7920 samples, 4752 train / 1584 val / 1584 test,
54 features, 128 draws per sample. Feature extraction 8 s, full run 110 s, both
well inside the 15-minute cap. Git SHA recorded in each run file.

## Design flaw found and fixed mid-round

The first pass (`r1-baseline`) used unweighted boosting. Because ChaCha20 is
deliberately over-sampled 3x, the class prior alone decided every ambiguous
sample and the four strongest weak generators collapsed into the CSPRNG class.
The fix was `class_weight="balanced"`, chosen on the validation split only; test
seeds were scored once, after the choice. Both runs are kept.

## Headline table (balanced accuracy on held-out test seeds)

| N draws | generator (7, chance .143) | generator, CSPRNG removed (6, chance .167) | shuffle (11, chance .091) | is_csprng (2, chance .500) | CSPRNG control (chance .500) |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.374 | 0.427 | 0.612 | 0.540 | 0.525 [0.482, 0.567] |
| 32 | 0.363 | 0.440 | 0.699 | 0.569 | 0.532 [0.490, 0.574] |
| 128 | **0.384** | **0.442** | **0.761** | **0.572** | **0.508 [0.465, 0.550]** |

## Per-generator identification, N=128

| Generator | Recall | Precision | Read |
| --- | --- | --- | --- |
| lcg-glibc | **1.000** | 1.000 | solved; its LSB stream has period 2 |
| minstd | 0.648 | 0.905 | partially solved, high precision |
| chacha20 | 0.799 | 0.415 | **not** identified: it is the default bucket |
| pcg32 | 0.102 | 0.207 | at chance |
| xorshift32 | 0.062 | 0.167 | at chance |
| mt19937 | 0.051 | 0.184 | at chance |
| xorshift128+ | 0.023 | 0.062 | below chance |

ChaCha20's high recall with 0.415 precision is the signature of a catch-all
class, not of positive identification. Four of the seven generators are not
separated at all by this feature set.

## Shuffle identification, N=128

| Shuffle | Recall (all generators) | Recall (ChaCha20 samples only) |
| --- | --- | --- |
| modulo-biased | 1.000 | 1.000 |
| naive-swap-any | 0.993 | 1.000 |
| gsr-riffle-1 | 0.917 | 1.000 |
| gsr-riffle-2..5 | 0.785 - 0.861 | 1.000 |
| gsr-riffle-6 | 0.764 | 0.875 |
| gsr-riffle-7 | 0.535 | 0.604 |
| fisher-yates | 0.479 | 0.396 |
| random-comparator-sort | 0.451 | 0.271 |

Balanced accuracy restricted to ChaCha20 samples is **0.831**, higher than the
0.761 over the whole grid.

## CSPRNG: is it at chance?

The canary is `csprng_control`: ChaCha20 samples only, labelled by an arbitrary
seed parity, trained on train seeds and scored on test seeds. At every N the
Wilson 95% interval contains 0.500 (widest miss: 0.532 [0.490, 0.574] at N=32,
n=528). **No generator-level signal is recoverable from ChaCha20 draws.** The
pipeline is not leaking.

The `is_csprng` head sits at 0.572 balanced accuracy, above chance. That is not
a contradiction and not a bug: it detects LCG and MINSTD, never ChaCha20. Its
CSPRNG-class recall is 0.485 with a 95% interval of [0.442, 0.527] — exactly
chance — while its weak-class recall is 0.659. The head is a weak-generator
detector wearing a CSPRNG label.

## What it means

1. **Hypothesis H1 is rejected.** Frequency-shaped statistics identify only the
   two generators whose *low bits* are structured. glibc's LCG has a period-2
   least significant bit, so the orientation bit stream alone gives it away
   perfectly at N=8. MINSTD leaks through the same channel, more weakly.
   MT19937, PCG32, xorshift32 and xorshift128+ are GF(2)-linear or tempered;
   their low bits pass monobit, runs, serial and approximate entropy, so this
   feature set sees nothing. They are the wrong tests for these generators.
2. **Hypothesis H2 is confirmed, and is the strongest result of the round.**
   Shuffle algorithm is identifiable *from permutation shape alone*, at 0.831
   balanced accuracy even when the generator is a real CSPRNG. `modulo-biased`
   and `naive-swap-any` are perfectly identified. This matters for the product:
   a broken shuffle is detectable no matter how good the RNG behind it is.
3. **Fisher-Yates and sort-with-random-comparator are confusable on purpose.**
   Both produce a uniform permutation distribution, so no statistic of the
   permutation can separate them. Their ~0.4 / ~0.3 recalls are the correct
   answer, not a modelling failure, and they are the ceiling for that pair.
4. **Hypothesis H3 is confirmed.** ChaCha20 is at chance under the control.
5. **Hypothesis H4 is half right.** The draws-needed curve rises for shuffle
   (0.612 -> 0.699 -> 0.761) and is flat for generator (0.374 -> 0.363 ->
   0.384). More draws buy permutation-shape evidence, and buy nothing at all
   for generator identity, because the generator signal that exists (LCG) is
   already saturated at N=8 and the signal that is missing does not appear with
   more samples of the same statistic.

## The bar for later neural models

Any neural model in a later round must beat, on the same held-out test seeds:

- generator, 7-class: **0.384** balanced accuracy (N=128)
- generator without CSPRNG, 6-class: **0.442**
- shuffle, 11-class: **0.761**, and **0.831** on CSPRNG-only samples
- and must stay inside [0.465, 0.550] on the CSPRNG seed-parity control

## What is next

Single most useful next experiment: **add GF(2) linear-structure features** —
the NIST binary matrix rank test (32x32 blocks) and Berlekamp-Massey linear
complexity, computed over the bit stream implied by the draws. Every generator
this round failed on (MT19937, xorshift32, xorshift128+, and PCG32's LCG core)
is linear over GF(2); linear complexity is the standard test that catches them
and costs no new dependency. Hypothesis to register for round 2: linear
complexity alone lifts 6-class weak-generator balanced accuracy from 0.442 to
above 0.70 at N=128, while leaving the ChaCha20 control inside its interval.

Runner-up: raise the orientation ablation (`configs/r1-orientation.json`,
`derived` mode) into a full run, to measure how much a shared-random-value
orientation bug widens the attack surface.
