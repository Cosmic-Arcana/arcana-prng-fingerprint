# Track B board — PRNG fingerprinting from tarot draws

Active question: which generator and which shuffle produced a run of draws, and
is the stream distinguishable from a CSPRNG?

Data lives in the sibling checkout `arcana-rng-lab` (`samples` CLI,
`sample-dataset.v1`). `data/` and `artifacts/` are gitignored. No training in CI.

| id | experiment | metric | result | status |
| --- | --- | --- | --- | --- |
| B0 | generator + shuffle library, reference vectors | published vectors reproduce | 6 of 8 VERIFIED, 2 UNVERIFIED | done |
| R1a | `r1-baseline` — unweighted boosting | balanced acc, held-out seeds | prior-dominated, superseded | superseded |
| R1b | `r1-balanced` — headline statistical baseline | balanced acc, held-out seeds | gen 0.384 / shuffle 0.761 | done |
| R1c | CSPRNG seed-parity control | Wilson 95% CI contains 0.500 | 0.508 [0.465, 0.550] | done — at chance |
| R2a | `r2-all` — GF(2) block on top of round 1 | 6-class weak-generator balanced acc | 0.595 (target was > 0.70) | done — H5 rejected, large lift |
| R2b | `r2-gf2` / `r2-r1` — ablation arms | same, one block at a time | gf2 alone 0.606, r1 alone 0.442 | done |
| R2c | CSPRNG seed-parity control, all three arms | Wilson 95% CI contains 0.500 | 0.474 [0.431, 0.516] | done — at chance |
| R4 | `r4-mt-consistency` — MT19937 state consistency over GF(2) | test recall / non-MT flagged | 1.000 / 0 of 384 at N>=256; 0.000 below | done — H8-H11 confirmed, threshold exactly 256 |
| R5 | shuffle inversion to recover generator words | reach pcg32 / xorshift128+ | — | next |
| R3 | `derived` orientation ablation | delta vs `independent-bit` | — | queued |

## Baseline to beat (held-out test seeds, N=128)

Round 2 supersedes round 1 on every generator head. Round 1 numbers in brackets.

| Head | Chance | Baseline |
| --- | --- | --- |
| generator, 7-class | 0.143 | **0.532** (r1: 0.384) |
| generator, CSPRNG removed, 6-class | 0.167 | **0.595** (r1: 0.442) |
| shuffle, 11-class | 0.091 | **0.765** (r1: 0.761) |
| shuffle, ChaCha20 samples only | 0.091 | **0.831** |
| is_csprng, binary | 0.500 | **0.611** (r1: 0.572) |
| CSPRNG control | 0.500 | **must stay in [0.431, 0.516]** |

Solved generators, recall and precision both 1.000 at N=128: `lcg-glibc`,
`xorshift32`. Losing either is a regression. `mt19937` is solved separately by
the R4 algebraic detector at N>=256 with a fixed-budget shuffle (recall 1.000,
0 false flags); below 256 draws it is provably undetectable from orientation
bits. `pcg32` and `xorshift128+` are still at chance and are what R5 is for.

## Standing rules for this track

- Hypothesis and metric go in `lab/reports/round-<n>.md` before the run starts,
  and are never edited afterwards to match the result.
- Seed ranges are disjoint per split: train 1_000_000+, val 2_000_000+,
  test 3_000_000+. Test seeds are scored once, never tuned on.
- Above chance on the CSPRNG control is a bug until proven otherwise.
- 15 minutes wall clock per experiment. Over the cap: mark blocked, move on.

## Blocked

Nothing blocked. `crypto.getRandomValues` stays out of the seeded path by
design, so there is no unseeded-CSPRNG arm.
