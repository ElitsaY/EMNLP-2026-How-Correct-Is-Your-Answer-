"""Benchmark metric scores against ordinal human correctness labels.

Sections: data loading, agreement statistics, the benchmark tables, and run
records (manifests) that make every output traceable to its inputs.

Conventions for the pairwise statistics:

* A pair of examples is *comparable* when their severities differ.
* A *win* means the more-correct example has the strictly higher score; score
  ties are counted separately and never as wins.
* ``accuracy = wins / pairs``, ``auc = (wins + 0.5 * ties) / pairs`` and
  ``violation_rate = losses / pairs``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm

from . import constants as C

logger = logging.getLogger(__name__)

ROW_ID = "row_id"
LABEL = "label"
SEVERITY = "severity"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Rows missing any of these are dropped so every metric sees the same examples.
REQUIRED_TEXT_COLUMNS = ("question", "gold_answer", "answer")
STATEMENT_COLUMNS = ("premise", "hypothesis")


def load_correctness_data(
    path: str | Path,
    label_column: str = C.LABEL_COLUMN,
    severity: Mapping[str, int] = C.SEVERITY,
    require_statements: bool = True,
) -> pd.DataFrame:
    """Load CAP-Correctness with a stable ``row_id`` and ordinal ``severity``.

    ``row_id`` is the 0-based row position in the original CSV (kept if the file
    already has one), so score files can always be joined back to the data.
    """
    df = pd.read_csv(path)
    required = [*REQUIRED_TEXT_COLUMNS, label_column]
    if require_statements:
        required += STATEMENT_COLUMNS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {missing_cols}")

    if ROW_ID not in df.columns:
        df.insert(0, ROW_ID, range(len(df)))
    if not df[ROW_ID].is_unique:
        raise ValueError(f"{path}: {ROW_ID} values must be unique")

    text = df[required].astype(str).apply(lambda col: col.str.strip())
    incomplete = df[required].isna().any(axis=1) | (text == "").any(axis=1)
    if incomplete.any():
        logger.warning(
            "Dropping %d row(s) with missing %s (row_id: %s)",
            int(incomplete.sum()), required, df.loc[incomplete, ROW_ID].tolist(),
        )
        df = df.loc[~incomplete].copy()

    df[LABEL] = df[label_column].astype(str).str.strip().str.lower()
    unknown = sorted(set(df[LABEL]) - set(severity))
    if unknown:
        raise ValueError(f"Labels without a severity: {unknown}")
    df[SEVERITY] = df[LABEL].map(severity).astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCounts:
    wins: int
    ties: int
    losses: int

    @property
    def pairs(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def accuracy(self) -> float:
        return self.wins / self.pairs if self.pairs else np.nan

    @property
    def auc(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.pairs if self.pairs else np.nan

    @property
    def violation_rate(self) -> float:
        return self.losses / self.pairs if self.pairs else np.nan

    def __add__(self, other: PairCounts) -> PairCounts:
        return PairCounts(
            self.wins + other.wins, self.ties + other.ties, self.losses + other.losses
        )


def compare_groups(better: np.ndarray, worse: np.ndarray) -> PairCounts:
    """Count all (b, w) pairs with b > w, b == w and b < w in O(n log n)."""
    if len(better) == 0 or len(worse) == 0:
        return PairCounts(0, 0, 0)
    worse_sorted = np.sort(worse)
    below = np.searchsorted(worse_sorted, better, side="left")
    below_or_equal = np.searchsorted(worse_sorted, better, side="right")
    wins = int(below.sum())
    ties = int((below_or_equal - below).sum())
    return PairCounts(wins, ties, len(better) * len(worse) - wins - ties)


def pairwise_ranking(scores: np.ndarray, severity: np.ndarray) -> PairCounts:
    """Compare every pair of examples with different severity."""
    total = PairCounts(0, 0, 0)
    lower: list[np.ndarray] = []
    for level in np.unique(severity):  # ascending
        current = scores[severity == level]
        if lower:
            total = total + compare_groups(current, np.concatenate(lower))
        lower.append(current)
    return total


def correlations(scores: np.ndarray, severity: np.ndarray) -> dict[str, float]:
    """Spearman rho and Kendall tau-b between scores and severity."""
    if len(scores) < 3 or np.ptp(scores) == 0 or np.ptp(severity) == 0:
        return {"spearman": np.nan, "spearman_p": np.nan, "kendall": np.nan, "kendall_p": np.nan}
    sp = spearmanr(scores, severity)
    kt = kendalltau(scores, severity)
    return {
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def bootstrap(
    stat_fn: Callable[[np.ndarray], dict[str, float]],
    n: int,
    n_boot: int = C.N_BOOTSTRAP,
    seed: int = C.SEED,
    progress: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Nonparametric bootstrap over row indices.

    ``stat_fn`` receives an index array of length ``n`` and returns named
    statistics. The same ``seed`` and ``n`` always draw the same resamples, so
    bootstraps of different metrics are paired.
    """
    rng = np.random.default_rng(seed)
    iterator = range(n_boot) if progress is None else progress(range(n_boot))
    replicates: dict[str, list[float]] = {}
    for _ in iterator:
        idx = rng.integers(0, n, size=n)
        for key, value in stat_fn(idx).items():
            replicates.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=float) for key, values in replicates.items()}


def percentile_ci(replicates: np.ndarray, level: float = C.CI_LEVEL) -> tuple[float, float]:
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return np.nan, np.nan
    alpha = (1 - level) / 2 * 100
    low, high = np.percentile(finite, [alpha, 100 - alpha])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Benchmark tables
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    tables: dict[str, pd.DataFrame]
    replicates: dict[str, dict[str, np.ndarray]]


def class_pairs(severity: Mapping[str, int], present: Iterable[str]) -> list[tuple[str, str]]:
    """All (better, worse) label pairs with different severity, most correct first."""
    present = set(present)
    ordered = sorted((l for l in severity if l in present), key=lambda l: -severity[l])
    return [(a, b) for a, b in combinations(ordered, 2) if severity[a] > severity[b]]


def evaluate(
    df: pd.DataFrame,
    metrics: Sequence[str],
    reference_metric: str | None = None,
    severity: Mapping[str, int] = C.SEVERITY,
    hard_pairs: Iterable[tuple[str, str]] = C.HARD_PAIRS,
    n_bootstrap: int = C.N_BOOTSTRAP,
    ci_level: float = C.CI_LEVEL,
    seed: int = C.SEED,
) -> EvaluationResult:
    """Benchmark each metric column of ``df`` against its ``label``/``severity``.

    Scores are rounded to ``constants.SCORE_DECIMALS`` so floating-point noise
    cannot turn ties into wins or losses.

    All metrics share the same rows and the same bootstrap resamples, so
    ``paired_differences`` compares ``reference_metric`` to every other metric.
    """
    score_arrays = {}
    for metric in metrics:
        values = df[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {metric!r} has missing or non-finite scores")
        score_arrays[metric] = np.round(values, C.SCORE_DECIMALS)
    hard_pairs = {tuple(p) for p in hard_pairs}
    for better, worse in hard_pairs:
        if severity[better] <= severity[worse]:
            raise ValueError(f"Hard pair {better!r} > {worse!r} contradicts the severity map")

    labels = df[LABEL].to_numpy()
    label_names = [l for l in severity if l in set(labels)]
    label_codes = pd.Categorical(labels, categories=label_names).codes
    sev = df[SEVERITY].to_numpy()
    pairs = class_pairs(severity, label_names)

    def stats_for(scores: np.ndarray, idx: np.ndarray) -> dict[str, float]:
        s, v, codes = scores[idx], sev[idx], label_codes[idx]
        out = {k: x for k, x in correlations(s, v).items() if not k.endswith("_p")}
        out["pairwise_accuracy"] = pairwise_ranking(s, v).accuracy
        by_label = [s[codes == i] for i in range(len(label_names))]
        for name, values in zip(label_names, by_label):
            out[f"mean:{name}"] = values.mean() if values.size else np.nan
        for better, worse in pairs:
            counts = compare_groups(
                by_label[label_names.index(better)], by_label[label_names.index(worse)]
            )
            out[f"acc:{better}>{worse}"] = counts.accuracy
            out[f"auc:{better}>{worse}"] = counts.auc
        return out

    replicates = {}
    rows: dict[str, list[dict]] = {"correlations": [], "descriptives": [], "class_pairs": []}
    for metric in metrics:
        scores = score_arrays[metric]
        boot = bootstrap(
            lambda idx: stats_for(scores, idx),
            n=len(scores),
            n_boot=n_bootstrap,
            seed=seed,
            progress=lambda it, m=metric: tqdm(it, desc=f"Bootstrap {m}", leave=False),
        )
        replicates[metric] = boot

        def ci(key: str) -> dict[str, float]:
            low, high = percentile_ci(boot[key], ci_level)
            return {f"{key}_ci_low": low, f"{key}_ci_high": high}

        corr = correlations(scores, sev)
        ranking = pairwise_ranking(scores, sev)
        rows["correlations"].append({
            "metric": metric,
            "n": len(scores),
            "spearman": corr["spearman"], **ci("spearman"), "spearman_p": corr["spearman_p"],
            "kendall": corr["kendall"], **ci("kendall"), "kendall_p": corr["kendall_p"],
            "pairwise_accuracy": ranking.accuracy, **ci("pairwise_accuracy"),
            "comparable_pairs": ranking.pairs,
            "pair_wins": ranking.wins,
            "pair_ties": ranking.ties,
            "pair_losses": ranking.losses,
        })

        for name in label_names:
            values = scores[labels == name]
            low, high = percentile_ci(boot[f"mean:{name}"], ci_level)
            rows["descriptives"].append({
                "metric": metric,
                "label": name,
                "severity": severity[name],
                "n": values.size,
                "mean": values.mean(),
                "mean_ci_low": low,
                "mean_ci_high": high,
                "std": values.std(ddof=1) if values.size > 1 else np.nan,
                "min": values.min(),
                "p05": np.percentile(values, 5),
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "max": values.max(),
            })

        for better, worse in pairs:
            b, w = scores[labels == better], scores[labels == worse]
            counts = compare_groups(b, w)
            key = f"{better}>{worse}"
            acc_low, acc_high = percentile_ci(boot[f"acc:{key}"], ci_level)
            auc_low, auc_high = percentile_ci(boot[f"auc:{key}"], ci_level)
            rows["class_pairs"].append({
                "metric": metric,
                "better": better,
                "worse": worse,
                "hard_pair": (better, worse) in hard_pairs,
                "n_better": b.size,
                "n_worse": w.size,
                "pairs": counts.pairs,
                "wins": counts.wins,
                "ties": counts.ties,
                "losses": counts.losses,
                "accuracy": counts.accuracy,
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "auc": counts.auc,
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "violation_rate": counts.violation_rate,
                "mean_better": b.mean(),
                "mean_worse": w.mean(),
                "mean_violation": bool(w.mean() > b.mean()),
            })

    class_pair_df = pd.DataFrame(rows["class_pairs"])
    tables = {
        "correlations": pd.DataFrame(rows["correlations"]),
        "descriptives": pd.DataFrame(rows["descriptives"]),
        "class_pairs": class_pair_df,
        "hard_pairs": class_pair_df[class_pair_df["hard_pair"]].reset_index(drop=True),
        "monotonicity": _monotonicity_summary(class_pair_df),
        "paired_differences": _paired_differences(replicates, metrics, reference_metric, ci_level),
    }
    return EvaluationResult(tables=tables, replicates=replicates)


def _monotonicity_summary(class_pair_df: pd.DataFrame) -> pd.DataFrame:
    """Share of class pairs whose mean scores are in the wrong order."""
    grouped = class_pair_df.groupby("metric", sort=False)
    sums = grouped[["losses", "pairs"]].sum()
    return pd.DataFrame({
        "class_pairs": grouped.size(),
        "mean_violations": grouped["mean_violation"].sum(),
        "mean_violation_rate": grouped["mean_violation"].mean(),
        "row_level_violation_rate": sums["losses"] / sums["pairs"],
    }).reset_index()


def _paired_differences(
    replicates: dict[str, dict[str, np.ndarray]],
    metrics: Sequence[str],
    reference: str | None,
    ci_level: float,
) -> pd.DataFrame:
    """Reference metric minus each other metric, from the paired bootstrap."""
    rows = []
    if reference is None:
        return pd.DataFrame(rows)
    if reference not in metrics:
        raise ValueError(f"reference_metric {reference!r} is not among {list(metrics)}")
    for metric in metrics:
        if metric == reference:
            continue
        for stat in ("spearman", "kendall", "pairwise_accuracy"):
            diff = replicates[reference][stat] - replicates[metric][stat]
            diff = diff[np.isfinite(diff)]
            low, high = percentile_ci(diff, ci_level)
            rows.append({
                "reference": reference,
                "metric": metric,
                "statistic": stat,
                "mean_difference": diff.mean(),
                "ci_low": low,
                "ci_high": high,
                # One-sided bootstrap p-value for "reference is not better".
                "p_reference_not_better": float((diff <= 0).mean()),
            })
    return pd.DataFrame(rows)


def write_tables(result: EvaluationResult, output_dir: str | Path, ci_level: float = C.CI_LEVEL) -> Path:
    """Write every table as CSV plus a ``summary.md``; returns the summary path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    summary = output_dir / "summary.md"
    summary.write_text(_summary_markdown(result.tables, ci_level), encoding="utf-8")
    logger.info("Wrote evaluation tables to %s", output_dir)
    return summary


def _summary_markdown(tables: dict[str, pd.DataFrame], ci_level: float) -> str:
    def with_ci(row: pd.Series, key: str) -> str:
        return f"{row[key]:.4f} [{row[key + '_ci_low']:.4f}, {row[key + '_ci_high']:.4f}]"

    corr = tables["correlations"].sort_values("spearman", ascending=False)
    lines = [
        "# Evaluation summary",
        "",
        f"Point estimates with {ci_level:.0%} bootstrap confidence intervals.",
        "",
        "## Agreement with human severity",
        "",
        _md_table(
            ["Metric", "Spearman", "Kendall tau-b", "Pairwise accuracy"],
            [[r.metric, with_ci(r, "spearman"), with_ci(r, "kendall"), with_ci(r, "pairwise_accuracy")]
             for _, r in corr.iterrows()],
        ),
        "",
        "## Hard pairs (accuracy: better class scored strictly higher)",
        "",
        _md_table(
            ["Metric", "Pair", "Accuracy", "AUC"],
            [[r.metric, f"{r.better} > {r.worse}", with_ci(r, "accuracy"), with_ci(r, "auc")]
             for _, r in tables["hard_pairs"].iterrows()],
        ),
        "",
        "## Monotonicity",
        "",
        _md_table(
            ["Metric", "Class pairs with mean in wrong order", "Row-level violation rate"],
            [[r.metric, f"{r.mean_violations}/{r.class_pairs}", f"{r.row_level_violation_rate:.4f}"]
             for _, r in tables["monotonicity"].iterrows()],
        ),
    ]
    diffs = tables["paired_differences"]
    if not diffs.empty:
        lines += [
            "",
            f"## Paired differences ({diffs['reference'].iloc[0]} minus metric)",
            "",
            _md_table(
                ["Metric", "Statistic", "Difference", "p (reference not better)"],
                [[r.metric, r.statistic, f"{r.mean_difference:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]",
                  f"{r.p_reference_not_better:.4f}"] for _, r in diffs.iterrows()],
            ),
        ]
    return "\n".join(lines) + "\n"


def _md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

TRACKED_PACKAGES = (
    "cap-eval", "numpy", "pandas", "scipy", "torch", "transformers", "datasets",
    "accelerate", "sentencepiece", "scikit-learn", "sacrebleu", "rouge-score",
    "nltk", "bert-score", "unbabel-comet",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    *,
    settings: Mapping[str, Any],
    inputs: Mapping[str, str | Path],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Record everything needed to reproduce an output, next to the output.

    Stores the command line, ``settings``, all values in ``constants``, SHA-256
    of each input file, git commit, package versions and hardware.
    """
    manifest = {
        "argv": sys.argv,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "hardware": _hardware(),
        "inputs": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in inputs.items()},
        "settings": dict(settings),
        "constants": {k: v for k, v in vars(C).items() if k.isupper()},
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Wrote run manifest to %s", path)


def check_same_input(manifest_path: str | Path, input_name: str, data_path: str | Path) -> None:
    """Fail if an output was computed from a different version of ``data_path``."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inputs"][input_name]["sha256"] != sha256_file(data_path):
        raise ValueError(f"{manifest_path.parent} was computed from a different {data_path}")


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": False}
    info: dict[str, Any] = {"cuda": torch.cuda.is_available()}
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info
