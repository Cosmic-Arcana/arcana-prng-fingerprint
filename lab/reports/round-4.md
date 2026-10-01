# Round 4 — MT19937 state consistency over GF(2)

Status: **hypothesis registered before any run.** Results are appended below only
after the run completes. Nothing in the hypothesis section is edited afterwards.

Round 2 (`lab/reports/round-2.md`) left `mt19937` at chance and named the reason:
its output low bit is a GF(2)-linear form of a 19937-bit state, and per-draw
statistics never see enough of it. This round stops computing statistics and
asks the algebraic question directly: **is there any MT19937 state that
produces exactly these orientation bits?**

## Why this is decidable

In `independent-bit` mode, draw `d` consumes `w` shuffle words and then 78
orientation words, whose bit 0 is the card's `reversed` flag. Every sample is
generated from a freshly seeded generator. So with a fixed word budget, the
observed bit `(d, k)` is the low bit of tempered word number
`d * (w + 78) + w + k` of the stream, exactly.

MT19937's twist and tempering are both GF(2)-linear. Each observed bit is
therefore a known linear functional of the unknown initial state, which is 624
words but only 19937 effective bits (the low 31 bits of `mt[0]` never reach an
output). `n` observations give a system `A s = b` with `A` known per budget and
`b` read from the sample.

- If `n > rank(A)`, the left null space of `A` is non-empty and every null
  vector `y` is a parity check: `y · b = 0` for **every** MT19937 sample with
  that budget, and holds for a non-MT stream with probability 1/2 per
  independent check.
- `rank(A) <= 19937`, so checks must exist once `78 N > 19937`, i.e. **N >= 256**.

Fixed word budgets reduce to two: `fisher-yates` and `modulo-biased` consume
77 words per draw, `naive-swap-any` and `random-comparator-sort` consume 78.
The GSR riffles are variable and are out of scope (already excluded from the
R4 dataset).

## Hypotheses

H8 (primary). The detector — "flag `mt19937` if the sample satisfies every
parity check under the 77-word or the 78-word budget" — reaches **recall 1.000
on `mt19937` test samples and flags zero non-MT test samples** at N = 260.

H9 (threshold). The number of independent checks is exactly
`max(0, 78 N - 19937)`: zero at N <= 255, so recall is 0.000 there; 31 at
N = 256, so recall is already 1.000 there. The round-2 ceiling is a sharp
transition at 256 draws, not a gradual curve.

H10 (canary). ChaCha20 is never flagged. One flagged ChaCha20 sample at N >= 256
is a bug in the detector, not a finding.

H11 (budget). For flagged samples, the budget that matched equals the true
shuffle's budget. A match under the wrong budget would mean the observation
model is wrong.

## The metric that decides it

**Recall on `mt19937` test samples, and the count of flagged non-MT test
samples, at N = 260.** H8 is confirmed only if both are exactly 1.000 and 0.
Anything else rejects H8.

Secondary: rank and check count per budget and N, recall and false-flag count
per N in {128, 255, 256, 257, 260}, and per-generator flag rates.

## Method, fixed in advance

- Data: `data/r4-state-recovery.jsonl`, generated from the committed
  `arcana-rng-lab/configs/r4-state-recovery.json` unchanged — 5 generators
  (`xorshift32`, `xorshift128+`, `mt19937`, `pcg32`, `chacha20` at 3x) x 4
  fixed-budget shuffles, 512 draws per sample, seeds 1_000_000+ / 2_000_000+ /
  3_000_000+.
- Observations: the first 260 draws of each sample (20 280 bits). That leaves
  at least 343 checks per budget, a false-flag bound of 2^-343 per sample. The
  remaining 252 draws are unused; 512 in the config was a pre-round-4 estimate
  of where the ceiling lies, and this round measures the actual ceiling.
- The observation matrix is built by running MT19937 symbolically over GF(2),
  one 19 968-bit vector per state bit, and verified in tests against the real
  generator on concrete seeds. Null space by Gauss-Jordan elimination of
  `A^T` with 8-column table lookups, in numpy. Columns are processed in stream
  order, so the checks for every prefix N fall out of one elimination.
- Nothing is fitted. There is no parameter to choose, so no split is used for
  selection; the test split is the headline and train/val are reported
  alongside as additional evidence.
- No new dependencies: numpy and the standard library.
- Determinism: no randomness in the detector. One config JSON per run under
  `lab/runs/`, git SHA recorded.
- Cap: 15 minutes wall clock. Over the cap is blocked.

## What would falsify the mechanism, not just the metric

- An `mt19937` sample failing any check means the observation model (positions,
  tempering functional, twist) is wrong, and every other number is void.
- Checks appearing at N < 256 would mean the observations are linearly
  dependent earlier than state size predicts, i.e. MT19937 leaks more per draw
  than one bit per orientation word.
- Any non-MT generator flagged at N >= 256 with hundreds of checks would mean
  that generator is itself consistent with MT19937 at those positions, which is
  not a plausible outcome; it would be treated as a bug.

---

# Results

Run: `lab/runs/r4-mt-consistency.json`. Dataset `data/r4-state-recovery.jsonl`,
2240 samples (1344 train / 448 val / 448 test), first 260 of 512 draws read.
Symbolic model 1.4 s and elimination ~18 s per budget, both budgets in
parallel; whole run **24 s** wall clock, inside the 15-minute cap. Git SHA in
the run file. Nothing fitted, nothing selected.

## Headline (held-out test seeds, N = 260)

| | Result | Target |
| --- | --- | --- |
| `mt19937` recall | **1.000** (64 / 64) | 1.000 |
| non-MT samples flagged | **0** / 384 | 0 |
| ChaCha20 flagged | **0** / 192 | 0 |

**H8 confirmed.** H10 holds: the canary was never flagged at any N.

## The threshold is exactly where the state size puts it (H9)

Checks available per budget, measured against `max(0, 78 N - 19937)`:

| N draws | 128 | 255 | 256 | 257 | 260 |
| --- | --- | --- | --- | --- | --- |
| predicted | 0 | 0 | 31 | 109 | 343 |
| measured, budget 77 | 0 | 0 | 31 | 109 | 343 |
| measured, budget 78 | 0 | 0 | 31 | 109 | 343 |
| `mt19937` recall, test | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| non-MT flagged, test | 0 | 0 | 0 | 0 | 0 |

Rank is 19937 under both budgets — the observations reach every effective state
bit and leak nothing beyond one bit per orientation word. **H9 confirmed**: the
ceiling round 2 hit is a step function at 256 draws. Train and val give the
same numbers (192/192 and 64/64 recall, 0 false flags), as they must for a
detector with no parameters.

## Budget agreement (H11)

All 64 test `mt19937` samples matched under their true shuffle's budget and
under that budget only. **H11 confirmed**; the observation model is right about
where every bit sits in the stream.

## What this changes

- `mt19937` moves from "at chance" to **solved at N >= 256**, deterministically,
  with a false-flag bound of 2^-31 per budget at 256 and 2^-343 at 260.
- The 512-draw config over-provisioned by 2x. 256 is the floor; 260 buys
  hundreds of checks.
- Below 256 draws no algorithm can do this from orientation bits alone: the
  system is underdetermined, so every bit string is consistent with some state.
- The detector needs a fixed word budget. Variable-budget shuffles (GSR
  riffles) and `derived` orientation are outside it.
- `pcg32` and `xorshift128+` remain at chance; they are not GF(2)-linear in the
  exposed bit, which is R5's problem.
