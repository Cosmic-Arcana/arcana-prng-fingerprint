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
| R2 | GF(2) linear structure: matrix rank + Berlekamp-Massey | 6-class weak-generator balanced acc | target > 0.70 at N=128 | next |
| R3 | `derived` orientation ablation | delta vs `independent-bit` | — | queued |

## Baseline to beat (held-out test seeds, N=128)

| Head | Chance | Baseline |
| --- | --- | --- |
| generator, 7-class | 0.143 | **0.384** |
| generator, CSPRNG removed, 6-class | 0.167 | **0.442** |
| shuffle, 11-class | 0.091 | **0.761** |
| shuffle, ChaCha20 samples only | 0.091 | **0.831** |
| is_csprng, binary | 0.500 | **0.572** |
| CSPRNG control | 0.500 | **must stay in [0.465, 0.550]** |

## Standing rules for this track

- Hypothesis and metric go in `lab/reports/round-<n>.md` before the run starts.
- Seed ranges are disjoint per split: train 1_000_000+, val 2_000_000+,
  test 3_000_000+. Test seeds are scored once, never tuned on.
- Above chance on the CSPRNG control is a bug until proven otherwise.
- 15 minutes wall clock per experiment. Over the cap: mark blocked, move on.

## Blocked

Nothing blocked. `crypto.getRandomValues` stays out of the seeded path by
design, so there is no unseeded-CSPRNG arm.
