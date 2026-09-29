# Fingerprint lab

Purpose: classify PRNG/shuffle fingerprints. Delete `py/` or `ts/` after ENTROPY picks a stack.

Do not train in CI. Do not commit `data/` or `artifacts/`.
Generate JSONL with `arcana-rng-lab` CLI, then point experiments at those files.
Fixed seeds. Disjoint seed ranges per split. Record git SHA in `lab/runs/<id>.json`.
