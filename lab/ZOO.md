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

## Known ceilings

`fisher-yates` and `random-comparator-sort` both produce uniform permutations,
so no statistic of the permutation can separate them. Their mutual confusion is
the correct answer and should not be "fixed" by a later model.
