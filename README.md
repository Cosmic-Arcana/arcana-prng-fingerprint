# arcana-prng-fingerprint

Mission: from `draw-dataset.v1` JSONL, infer **which generator and shuffle** produced a draw, and whether `is_csprng` is true — especially on production-export rows where generator/shuffle labels are null.

This repo does not emit draws. Datasets come from **`arcana-rng-lab` CLI** (sibling checkout), not a git dependency: one install path, no nested package version hell. Example:

```bash
cd ../arcana-rng-lab/py && pip install -e .
python -m arcana_rng.cli generate --config ../configs/example.json --out ../../arcana-prng-fingerprint/data/lab.jsonl
```

`data/` is gitignored. Training stack is ENTROPY’s choice (see `~/code/arcana-ml/MACHINE.md`).
