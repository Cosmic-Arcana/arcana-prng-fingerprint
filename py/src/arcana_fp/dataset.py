"""Streaming loader: JSONL samples in, feature matrices out.

Feature extraction runs in a worker pool over chunks of raw lines, so each line
is parsed exactly once and features for every N in the draws-needed curve come
out of that single parse.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arcana_fp.features import extract

LABEL_FIELDS = ("generator", "shuffle", "orientation", "is_csprng")


@dataclass(frozen=True)
class FeatureSet:
    n_draws: int
    x: np.ndarray
    generator: np.ndarray
    shuffle: np.ndarray
    is_csprng: np.ndarray
    split: np.ndarray
    seed: np.ndarray

    def mask(self, split: str) -> np.ndarray:
        return self.split == split


def _extract_chunk(args: tuple[list[str], tuple[int, ...]]) -> tuple[list[list[float]], list[dict]]:
    lines, n_list = args
    rows: list[list[float]] = []
    meta: list[dict] = []
    for line in lines:
        sample = json.loads(line)
        available = sample["draws_per_sample"]
        row: list[float] = []
        for n in n_list:
            if n > available:
                raise ValueError(f"sample has {available} draws, need {n}")
            row.append(extract(sample, n))
        rows.append(row)
        meta.append(
            {
                "generator": sample["labels"]["generator"],
                "shuffle": sample["labels"]["shuffle"],
                "is_csprng": bool(sample["labels"]["is_csprng"]),
                "split": sample["split"],
                "seed": int(sample["seed"]),
            }
        )
    return rows, meta


def _chunks(path: Path, size: int):
    buffer: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if len(buffer) >= size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def load(
    path: Path, n_list: tuple[int, ...], workers: int = 8, chunk_size: int = 64
) -> dict[int, FeatureSet]:
    xs: dict[int, list] = {n: [] for n in n_list}
    meta: list[dict] = []
    payloads = ((chunk, n_list) for chunk in _chunks(path, chunk_size))
    if workers <= 1:
        results = (_extract_chunk(payload) for payload in payloads)
        for rows, chunk_meta in results:
            _absorb(rows, chunk_meta, n_list, xs, meta)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rows, chunk_meta in pool.map(_extract_chunk, payloads, chunksize=1):
                _absorb(rows, chunk_meta, n_list, xs, meta)

    generator = np.array([m["generator"] for m in meta])
    shuffle = np.array([m["shuffle"] for m in meta])
    is_csprng = np.array([m["is_csprng"] for m in meta])
    split = np.array([m["split"] for m in meta])
    seed = np.array([m["seed"] for m in meta], dtype=np.int64)
    return {
        n: FeatureSet(
            n_draws=n,
            x=np.asarray(xs[n], dtype=np.float64),
            generator=generator,
            shuffle=shuffle,
            is_csprng=is_csprng,
            split=split,
            seed=seed,
        )
        for n in n_list
    }


def _absorb(rows, chunk_meta, n_list, xs, meta) -> None:
    for row in rows:
        for i, n in enumerate(n_list):
            xs[n].append(row[i])
    meta.extend(chunk_meta)
