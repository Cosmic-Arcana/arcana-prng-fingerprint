# Round 6 — draws-needed ceiling for pcg32 and xorshift128+ (detection only)

Status: **hypotheses registered before any run.** Results are appended below only
after the run completes. Nothing in the hypothesis section is edited afterwards.

Boundary: this round measures whether the existing statistical detectors see
`pcg32` or `xorshift128+` at all. It reads no generator words, recovers no state
and predicts no output. Whatever the outcome, no state-reconstruction follow-up
is opened (`lab/CAPABILITIES.md`, `lab/reports/round-5.md`).

## Question

Both generators sat at chance in rounds 1-3 at N = 128 draws. Round 4 showed that
a ceiling set by too few observations can lift sharply with more draws (MT19937,
exactly at 256). Is the `pcg32` / `xorshift128+` ceiling of that kind, or is it
structural? Measure at 1x, 2x and 4x the round-2 draw count.

## Design

- Data: existing `data/r4-state-recovery.jsonl` (512 draws per sample,
  `independent-bit` orientation, four fixed-budget shuffles). No new generation.
  Selected: `pcg32`, `xorshift128+`, `chacha20`. Samples per class (train / val /
  test): 192 / 64 / 64 for each weak generator, 576 / 192 / 192 for `chacha20`.
- Features: the full 85-column set of round 2 (r1 statistics + GF(2) block),
  computed on the first N draws, N in {128, 256, 512}.
- Model: `HistGradientBoostingClassifier`, balanced class weights, `max_iter`
  from {120, 300} chosen on validation only. Test seeds are scored once.
- Heads: `pcg32` vs `chacha20`; `xorshift128+` vs `chacha20` (both binary); plus
  `pcg32` vs `xorshift128+` and the 3-class head, reported but not decided on.
- Metric: balanced accuracy on the test split, with a 95% bootstrap interval over
  test samples (2000 resamples, seed 20261031).
  **Signal** = the interval's lower bound is above 0.500.
  **At chance** = the interval contains 0.500.

## Hypotheses

H14 (primary). At N = 512 neither binary head separates its generator from
ChaCha20: both intervals contain 0.500. Prior, not evidence: expected to hold.

H15 (scale). There is no draws-needed gain: for each binary head, the interval at
N = 512 overlaps the interval at N = 128.

H16 (control). The ChaCha20 seed-parity control stays at chance at every N: its
Wilson 95% interval contains 0.500. A failure invalidates the round, not H14.

H17 (what a miss would mean). If any binary head shows signal at any N, that is a
detection signal for a generator the lab listed as invisible. The follow-up is
to attribute it to a feature family (r1 statistics or GF(2) block) by ablation.
It is a measurement follow-up only.
