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
