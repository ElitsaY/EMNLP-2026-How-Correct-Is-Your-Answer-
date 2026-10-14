import json

import numpy as np
import pandas as pd
import pytest

from cap_eval.evaluation import (
    ROW_ID,
    PairCounts,
    bootstrap,
    check_same_input,
    class_pairs,
    compare_groups,
    evaluate,
    load_correctness_data,
    pairwise_ranking,
    percentile_ci,
    write_manifest,
)

SEVERITY = {"exact": 2, "equivalent": 2, "partial": 1, "invalid": 0}
HARD_PAIRS = [("equivalent", "partial")]


# --- statistics against brute force -------------------------------------------

def brute_force_groups(better, worse):
    wins = sum(b > w for b in better for w in worse)
    ties = sum(b == w for b in better for w in worse)
    return PairCounts(wins, ties, len(better) * len(worse) - wins - ties)


def brute_force_ranking(scores, severity):
    total = PairCounts(0, 0, 0)
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if severity[i] != severity[j]:
                hi, lo = (i, j) if severity[i] > severity[j] else (j, i)
                total = total + brute_force_groups([scores[hi]], [scores[lo]])
    return total


@pytest.mark.parametrize("seed", range(5))
def test_compare_groups_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    better, worse = rng.normal(size=40).round(1), rng.normal(size=30).round(1)  # rounding -> ties
    assert compare_groups(better, worse) == brute_force_groups(better, worse)


@pytest.mark.parametrize("seed", range(5))
def test_pairwise_ranking_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    severity = rng.integers(0, 5, size=120)
    scores = (severity + rng.normal(scale=2, size=120)).round(0)
    assert pairwise_ranking(scores, severity) == brute_force_ranking(scores, severity)


def test_pair_counts_rates():
    counts = PairCounts(wins=6, ties=2, losses=2)
    assert (counts.pairs, counts.accuracy, counts.auc, counts.violation_rate) == pytest.approx(
        (10, 0.6, 0.7, 0.2)
    )
    assert np.isnan(PairCounts(0, 0, 0).accuracy)
    assert compare_groups(np.array([]), np.array([1.0])).pairs == 0


def test_bootstrap_is_reproducible():
    values = np.arange(50, dtype=float)
    first = bootstrap(lambda idx: {"mean": values[idx].mean()}, n=50, n_boot=20, seed=7)
    second = bootstrap(lambda idx: {"mean": values[idx].mean()}, n=50, n_boot=20, seed=7)
    np.testing.assert_array_equal(first["mean"], second["mean"])


def test_percentile_ci_ignores_nan():
    low, high = percentile_ci(np.array([np.nan, *range(101)], dtype=float), level=0.9)
    assert (low, high) == pytest.approx((5.0, 95.0))


# --- data loading ---------------------------------------------------------------

def _frame(labels, answers=None):
    n = len(labels)
    return pd.DataFrame({
        "question": [f"q{i}" for i in range(n)],
        "gold_answer": ["gold"] * n,
        "answer": answers or ["ans"] * n,
        "premise": ["p"] * n,
        "hypothesis": ["h"] * n,
        "label_semantic": labels,
    })


def test_load_drops_incomplete_rows_and_keeps_row_ids(tmp_path):
    path = tmp_path / "data.csv"
    _frame([" Exact", "partial", "invalid"], answers=["a", None, "c"]).to_csv(path, index=False)
    df = load_correctness_data(path, severity=SEVERITY)
    assert df[ROW_ID].tolist() == [0, 2]
    assert df["label"].tolist() == ["exact", "invalid"]
    assert df["severity"].tolist() == [2, 0]


def test_load_rejects_unknown_labels(tmp_path):
    path = tmp_path / "data.csv"
    _frame(["exact", "mystery"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="mystery"):
        load_correctness_data(path, severity=SEVERITY)


# --- benchmark tables -------------------------------------------------------------

def test_class_pairs_skip_equal_severity():
    pairs = class_pairs(SEVERITY, SEVERITY)
    assert ("exact", "equivalent") not in pairs
    assert ("exact", "partial") in pairs
    assert all(SEVERITY[a] > SEVERITY[b] for a, b in pairs)


def _scored_frame():
    rng = np.random.default_rng(0)
    labels = rng.choice(list(SEVERITY), size=200)
    df = pd.DataFrame({"label": labels, "severity": [SEVERITY[l] for l in labels]})
    df["perfect"] = df["severity"] + rng.uniform(0, 0.5, size=200)
    df["noisy"] = rng.uniform(size=200)
    return df


def test_evaluate_is_deterministic_and_ranks_a_perfect_metric_first():
    df = _scored_frame()
    kwargs = dict(metrics=["perfect", "noisy"], reference_metric="perfect", severity=SEVERITY,
                  hard_pairs=HARD_PAIRS, n_bootstrap=50, seed=1)
    first, second = evaluate(df, **kwargs), evaluate(df, **kwargs)
    for name, table in first.tables.items():
        pd.testing.assert_frame_equal(table, second.tables[name])

    corr = first.tables["correlations"].set_index("metric")
    assert corr.loc["perfect", "pairwise_accuracy"] == 1.0
    assert corr.loc["perfect", "pairwise_accuracy_ci_low"] == 1.0
    assert first.tables["monotonicity"].set_index("metric").loc["perfect", "mean_violations"] == 0
    assert first.tables["hard_pairs"]["better"].eq("equivalent").all()
    assert (first.tables["paired_differences"]["mean_difference"] > 0).all()


def test_evaluate_rejects_inconsistent_hard_pairs():
    with pytest.raises(ValueError, match="contradicts"):
        evaluate(_scored_frame(), ["noisy"], severity=SEVERITY,
                 hard_pairs=[("partial", "equivalent")], n_bootstrap=1)


# --- run records --------------------------------------------------------------------

def test_manifest_detects_changed_input(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    manifest = tmp_path / "out" / "run.manifest.json"
    write_manifest(manifest, settings={"x": 1}, inputs={"data": data})

    recorded = json.loads(manifest.read_text())
    assert recorded["settings"] == {"x": 1}
    assert recorded["constants"]["SEED"] == 42
    check_same_input(manifest, "data", data)

    data.write_text("a\n2\n")
    with pytest.raises(ValueError, match="different"):
        check_same_input(manifest, "data", data)


def test_floating_point_noise_counts_as_a_tie():
    df = pd.DataFrame({
        "label": ["exact", "partial"],
        "severity": [SEVERITY["exact"], SEVERITY["partial"]],
        "noisy_tie": [0.49999999999999994, 0.5],
    })
    result = evaluate(df, ["noisy_tie"], severity=SEVERITY, hard_pairs=[], n_bootstrap=1)
    row = result.tables["correlations"].iloc[0]
    assert (row.pair_wins, row.pair_ties, row.pair_losses) == (0, 1, 0)
