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

---

# Results

Run `r6-weak-ceiling` at `61202d5`, 87 s, test seeds scored once.
Record: `lab/runs/r6-weak-ceiling.json`.

Balanced accuracy on held-out test seeds, 95% bootstrap interval in brackets.
Chance is 0.500 for binary heads, 0.333 for the 3-class head.

| Head | N = 128 | N = 256 | N = 512 |
| --- | --- | --- | --- |
| `pcg32` vs `chacha20` (decided) | 0.484 [0.453, 0.518] | 0.526 [0.487, 0.573] | 0.479 [0.453, 0.508] |
| `xorshift128+` vs `chacha20` (decided) | 0.492 [0.456, 0.531] | 0.521 [0.479, 0.562] | 0.518 [0.482, 0.560] |
| `pcg32` vs `xorshift128+` (reference) | 0.453 [0.367, 0.539] | 0.539 [0.445, 0.625] | 0.422 [0.336, 0.508] |
| 3-class (reference) | 0.316 [0.288, 0.344] | 0.325 [0.293, 0.358] | 0.349 [0.312, 0.389] |
| ChaCha20 seed-parity control (Wilson) | 0.516 [0.445, 0.585] | 0.479 [0.410, 0.550] | 0.438 [0.369, 0.508] |

| Hypothesis | Verdict |
| --- | --- |
| H14 at N = 512 neither binary head separates its generator from ChaCha20 | **Confirmed.** Both intervals contain 0.500. |
| H15 no draws-needed gain | **Confirmed.** Each head's N = 512 interval overlaps its N = 128 interval; point estimates wander within +-0.04 with no trend. |
| H16 control stays at chance at every N | **Confirmed.** All three Wilson intervals contain 0.500, so the round is valid. |
| H17 a signal at any N | **Not triggered.** No interval has a lower bound above chance. |

## Reading

- For these detectors the `pcg32` / `xorshift128+` ceiling is **not a draw-count
  limit**. MT19937 flipped from invisible to exact at 256 draws because more draws
  added linear equations the detector could use. The zoo's analysis says these two
  generators expose no such equations in the orientation bit, and quadrupling the
  draws moved nothing, which fits that analysis but does not prove it.
- The 3-class head drifts from 0.316 to 0.349 across N, and its N = 512 interval
  [0.312, 0.389] still contains chance (0.333). Read as noise, not a trend.
- Power: with 64 test samples per weak generator and 192 for ChaCha20, intervals
  are about +-0.03 to +-0.05 wide on the binary heads. A signal larger than about
  0.06 over chance would have shown. A smaller one is not excluded. This is a
  statement about these features and this sample size, not about the generators.
- The control at N = 512 reads 0.438 with an upper bound of 0.508, so it passes
  H16 narrowly. It is below chance, and a leak would show above chance, so it
  does not threaten the round. A rerun with more control samples would settle it.

## Consequence

The existing statistical detectors show no draw-count lift for these two
generators. Further progress needs either more samples per weak class to tighten
the intervals, or features of a different kind. Features of the kind that read
generator words are the state-reconstruction family and stay out of scope
(`lab/CAPABILITIES.md`). Nothing in R6 changes that boundary.
