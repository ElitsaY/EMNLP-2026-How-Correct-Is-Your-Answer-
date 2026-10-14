"""Score CAP-Correctness with CAP and benchmark it against the baseline metrics.

    python scripts/evaluate_dataset.py                    # CAP + all baselines
    python scripts/evaluate_dataset.py --baselines        # CAP only
    python scripts/evaluate_dataset.py --statement-model outputs/statement_model/final

Baseline scores come from scripts/score_baselines.py. With --statement-model,
premise/hypothesis statements are regenerated before scoring CAP; otherwise the
released statements in the data file are used.

Outputs (under --output-dir):
    scores/cap.csv            CAP score and NLI probabilities per row
    statements.csv            regenerated statements (only with --statement-model)
    evaluation/<split>/       benchmark tables and summary.md (default split: test)
Each output folder gets a *.manifest.json describing how it was produced.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cap_eval import constants as C
from cap_eval.evaluation import (
    ROW_ID,
    check_same_input,
    evaluate,
    load_correctness_data,
    sha256_file,
    write_manifest,
    write_tables,
)

logger = logging.getLogger("evaluate_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/CAP-Correctness.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split", choices=["test", "validation", "all"], default="test",
                        help="rows to evaluate on; the paper reports the test split (default)")
    parser.add_argument("--splits-file", type=Path, default=Path("data/CAP-Correctness-splits.csv"),
                        help="row_id,split assignment of the data file")
    parser.add_argument("--baselines", nargs="*", choices=C.BASELINE_METRICS,
                        default=list(C.BASELINE_METRICS),
                        help="baseline metrics to compare against (default: all)")
    parser.add_argument("--statement-model", type=Path,
                        help="regenerate premise/hypothesis with this statement model")
    parser.add_argument("--reuse-cap-scores", action="store_true",
                        help="reuse scores/cap.csv if it was computed with the same data and settings")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--batch-size", type=int, default=C.NLI_BATCH_SIZE)
    parser.add_argument("--neutral-weight", type=float, default=C.NEUTRAL_WEIGHT)
    parser.add_argument("--forward-weight", type=float, default=C.FORWARD_WEIGHT)
    parser.add_argument("--no-fp16", action="store_true", help="run the NLI model in fp32 on CUDA")
    parser.add_argument("--n-bootstrap", type=int, default=C.N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=C.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    scores_dir = args.output_dir / "scores"

    df = load_correctness_data(args.data, require_statements=args.statement_model is None)
    cap = score_cap(df, args, scores_dir)
    df = df.merge(cap[[ROW_ID, "cap"]], on=ROW_ID, validate="one_to_one")

    for metric in args.baselines:
        path = scores_dir / f"{metric}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No {metric} scores: run `python scripts/score_baselines.py --metrics {metric}` "
                "or leave it out with --baselines"
            )
        check_same_input(scores_dir / f"{metric}.manifest.json", "data", args.data)
        scores = pd.read_csv(path, usecols=[ROW_ID, metric])
        df = df.merge(scores, on=ROW_ID, how="left", validate="one_to_one")
        if df[metric].isna().any():
            raise ValueError(f"{metric} scores are missing for some rows; re-run score_baselines.py")

    inputs = {"data": args.data}
    if args.split != "all":
        splits = pd.read_csv(args.splits_file, usecols=[ROW_ID, "split"])
        df = df.merge(splits, on=ROW_ID, how="left", validate="one_to_one")
        if df["split"].isna().any():
            raise ValueError(f"{args.splits_file} has no split for some rows of {args.data}")
        df = df[df["split"] == args.split].reset_index(drop=True)
        inputs["splits"] = args.splits_file
        logger.info("Evaluating on the %s split (%d rows)", args.split, len(df))

    metrics = ["cap", *args.baselines]
    result = evaluate(
        df, metrics, reference_metric="cap", n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    eval_dir = args.output_dir / "evaluation" / args.split
    summary = write_tables(result, eval_dir)
    write_manifest(
        eval_dir / "evaluate.manifest.json",
        settings=settings,
        inputs=inputs | {f"scores_{m}": scores_dir / f"{m}.csv" for m in metrics},
        extra={"rows": len(df), "metrics": metrics, "split": args.split},
    )
    print(summary.read_text(encoding="utf-8"))


def score_cap(df: pd.DataFrame, args: argparse.Namespace, scores_dir: Path) -> pd.DataFrame:
    """Compute CAP for every row (or reuse matching cached scores)."""
    cap_settings = {
        "nli_model": C.NLI_MODEL,
        "nli_revision": C.NLI_REVISION,
        "max_length": C.NLI_MAX_LENGTH,
        "fp16": not args.no_fp16,
        "neutral_weight": args.neutral_weight,
        "forward_weight": args.forward_weight,
        "statement_model": str(args.statement_model) if args.statement_model else None,
    }
    out_path = scores_dir / "cap.csv"
    manifest_path = scores_dir / "cap.manifest.json"
    if args.reuse_cap_scores and _can_reuse(manifest_path, args.data, cap_settings):
        logger.info("Reusing %s", out_path)
        return pd.read_csv(out_path)

    from cap_eval.scorer import CAPScorer, set_seed

    set_seed(args.seed)
    scores_dir.mkdir(parents=True, exist_ok=True)
    scorer = CAPScorer(
        device=args.device,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
        neutral_weight=args.neutral_weight,
        forward_weight=args.forward_weight,
        statement_model=args.statement_model,
    )
    if args.statement_model:
        scores = scorer.score_answers(df["question"], df["gold_answer"], df["answer"])
        statements = df.drop(columns=["label", "severity"]).assign(
            premise=scores["premise"].to_numpy(), hypothesis=scores["hypothesis"].to_numpy()
        )
        statements.to_csv(args.output_dir / "statements.csv", index=False)
    else:
        scores = scorer.score_detailed(df["premise"], df["hypothesis"])
    scores.insert(0, ROW_ID, df[ROW_ID].to_numpy())
    scores.to_csv(out_path, index=False)
    write_manifest(
        manifest_path,
        settings=cap_settings,
        inputs={"data": args.data},
        extra={"rows": len(scores), "device": scorer.device},
    )
    return scores


def _can_reuse(manifest_path: Path, data: Path, settings: dict) -> bool:
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["settings"] == settings and manifest["inputs"]["data"]["sha256"] == sha256_file(data)


if __name__ == "__main__":
    main()
