# Zoo — runs

Generator and shuffle variants are catalogued in the sibling checkout,
`arcana-rng-lab/lab/ZOO.md`. This file tracks experiment runs.

| Run id | Hypothesis | Outcome |
| --- | --- | --- |
| `r1-baseline` | Statistics of the permutation stream identify all 7 generators far above chance | Rejected, and the run was prior-dominated: ChaCha20 is over-sampled 3x and unweighted boosting dumped every ambiguous sample into it. Superseded by `r1-balanced`. |
| `r1-balanced` | Same, with `class_weight="balanced"` chosen on validation only | Generator 0.384 balanced accuracy (chance 0.143) but carried almost entirely by lcg-glibc (1.000) and minstd (0.648). MT19937, PCG32, xorshift32, xorshift128+ are at chance. |
| `r1-balanced` / shuffle head | Shuffle algorithm is identifiable independently of generator quality | Confirmed, strongest result of the round: 0.761 over the grid and 0.831 on ChaCha20 samples only. `modulo-biased` and `naive-swap-any` at 1.000. |
| `r1-balanced` / csprng control | ChaCha20 samples carry no generator-level signal | Confirmed at every N. Widest interval 0.532 [0.490, 0.574], all contain 0.500. No leakage. |
| `r1-balanced` / draws-needed | Accuracy rises with N for both heads | Half right. Shuffle rises 0.612 -> 0.699 -> 0.761. Generator is flat at ~0.37-0.38. |
| `r2-r1` | Round-1 columns only, re-run on the round-2 code | Reproduces `r1-balanced` to four decimals on every head. Kept as the ablation floor and as proof that the pipeline did not move between rounds. |
| `r2-gf2` | GF(2) columns only: matrix rank + Berlekamp-Massey, 31 features | Generator 0.606 6-class, beating round 1's whole 54-feature set by 0.164. Blind to shuffle (0.201 against chance 0.091), which is the expected negative control. |
| `r2-all` | Both blocks, 85 features. Headline: GF(2) structure lifts the 6-class head above 0.70 | **Rejected** at 0.595, but a +0.153 lift over round 1 and the largest move of the project so far. Validation chose this arm over `gf2` alone; on test the two are inside each other's noise. |
| `r2-all` / attribution | The lift comes from GF(2)-linear generators specifically | Confirmed, sharply. `xorshift32` 0.278 -> 1.000 is 79% of the lift. `mt19937`, `pcg32`, `xorshift128+` moved 0.03-0.09, which is what losing a competitor does on its own. |
| `r2-*` / csprng control | ChaCha20 still carries no generator-level signal under the new features | Confirmed in all three arms at all three N. Widest 0.521 [0.478, 0.563]. |

## Known ceilings

`fisher-yates` and `random-comparator-sort` both produce uniform permutations,
so no statistic of the permutation can separate them. Their mutual confusion is
the correct answer and should not be "fixed" by a later model.

MT19937's output low bit is a GF(2)-linear form of a **19937-bit** state, and a
sample exposes at most **78 contiguous** generator bits (one draw; the shuffle
consumes an unobserved block between draws). Berlekamp-Massey needs ~2L
consecutive terms for degree L, so no statistic over per-draw blocks can reach
it. This is a ceiling of the 128-draw dataset, not of the method: it lifts by
collecting more observations (R4), not by choosing a better test.

`pcg32` and `xorshift128+` are not GF(2)-linear in the exposed bit at all —
PCG32's output rotation is state-dependent, and xorshift128+'s final addition
mod 2^64 carries. GF(2) tools are the wrong algebra for them; they need word
values, which means R5.
