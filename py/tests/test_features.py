import numpy as np

from arcana_fp.features import N_FEATURES, extract, feature_names


def _sample(seed: int, n_draws: int = 16, identity: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        order = np.arange(78) if identity else rng.permutation(78)
        rev = "".join(str(int(b)) for b in rng.integers(0, 2, size=78))
        draws.append({"order": order.tolist(), "rev": rev})
    return {
        "schema": "sample-dataset.v1",
        "sample_id": f"synthetic|{seed}",
        "split": "train",
        "seed": seed,
        "deck_size": 78,
        "draws_per_sample": n_draws,
        "draws": draws,
        "labels": {
            "generator": "synthetic",
            "shuffle": "synthetic",
            "orientation": "independent-bit",
            "is_csprng": False,
        },
        "source": "lab",
        "created_at": "2026-09-29T00:00:00Z",
    }


def test_feature_vector_matches_the_declared_names() -> None:
    vector = extract(_sample(1), 16)
    assert vector.shape == (N_FEATURES,)
    assert len(feature_names()) == N_FEATURES
    assert len(set(feature_names())) == N_FEATURES


def test_extraction_is_deterministic_and_finite() -> None:
    sample = _sample(2)
    a = extract(sample, 16)
    b = extract(sample, 16)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_prefix_of_a_sample_is_a_prefix_of_the_stream() -> None:
    sample = _sample(3, n_draws=32)
    short = _sample(3, n_draws=32)
    short["draws"] = short["draws"][:8]
    assert np.allclose(extract(sample, 8), extract(short, 8))


def test_identity_permutations_are_separated_from_random_ones() -> None:
    names = feature_names()
    idx = names.index("inversions_mean")
    fixed = names.index("fixed_points_mean")
    identity = extract(_sample(4, identity=True), 16)
    shuffled = extract(_sample(4), 16)
    assert identity[idx] < shuffled[idx]
    assert identity[fixed] > shuffled[fixed]


def test_labels_never_enter_the_feature_vector() -> None:
    sample = _sample(5)
    baseline = extract(sample, 16)
    for field, value in (("generator", "lcg-glibc"), ("shuffle", "modulo-biased")):
        sample["labels"][field] = value
    sample["seed"] = 999_999
    sample["split"] = "test"
    sample["sample_id"] = "leaky"
    sample["created_at"] = "2030-01-01T00:00:00Z"
    assert np.array_equal(baseline, extract(sample, 16))
