"""Score CAP-Correctness with the reference-based baselines.

    python scripts/score_baselines.py                      # all baselines
    python scripts/score_baselines.py --metrics bleu meteor

Each metric compares the candidate answer to the gold answer (COMET also gets
the question as source) and is written to <output-dir>/scores/<metric>.csv with
a matching <metric>.manifest.json.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cap_eval import constants as C
from cap_eval.evaluation import ROW_ID, load_correctness_data, write_manifest

logger = logging.getLogger("score_baselines")

NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/CAP-Correctness.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metrics", nargs="+", choices=C.BASELINE_METRICS,
                        default=list(C.BASELINE_METRICS))
    parser.add_argument("--device", help="device for BERTScore (default: best available)")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=64)
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=32)
    parser.add_argument("--comet-gpus", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    df = load_correctness_data(args.data, require_statements=False)
    refs = df["gold_answer"].astype(str).tolist()
    hyps = df["answer"].astype(str).tolist()
    scores_dir = args.output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        logger.info("Scoring %s on %d rows", metric, len(df))
        if metric == "bleu":
            values = score_bleu(refs, hyps)
        elif metric == "rouge_l":
            values = score_rouge_l(refs, hyps)
        elif metric == "meteor":
            values = score_meteor(refs, hyps)
        elif metric == "bertscore":
            values = score_bertscore(refs, hyps, args)
        else:
            values = score_comet(df["question"].astype(str).tolist(), refs, hyps, args)

        pd.DataFrame({ROW_ID: df[ROW_ID], metric: values}).to_csv(
            scores_dir / f"{metric}.csv", index=False
        )
        write_manifest(
            scores_dir / f"{metric}.manifest.json",
            settings=settings,
            inputs={"data": args.data},
            extra={"metric": metric, "rows": len(df)},
        )


def score_bleu(refs: list[str], hyps: list[str]) -> np.ndarray:
    """Smoothed sentence BLEU (NLTK, method1) on lowercased word tokens."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ensure_nltk_resources()
    smoother = SmoothingFunction().method1

    def bleu(ref: str, hyp: str) -> float:
        hyp_tokens = tokenize(hyp)
        if not hyp_tokens:
            return 0.0
        return sentence_bleu([tokenize(ref)], hyp_tokens, smoothing_function=smoother)

    return pairwise(refs, hyps, bleu)


def score_rouge_l(refs: list[str], hyps: list[str]) -> np.ndarray:
    """ROUGE-L F-measure with Porter stemming."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return pairwise(refs, hyps, lambda ref, hyp: scorer.score(ref, hyp)["rougeL"].fmeasure)


def score_meteor(refs: list[str], hyps: list[str]) -> np.ndarray:
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk_resources()

    def meteor(ref: str, hyp: str) -> float:
        ref_tokens, hyp_tokens = tokenize(ref), tokenize(hyp)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return meteor_score([ref_tokens], hyp_tokens)

    return pairwise(refs, hyps, meteor)


def score_bertscore(refs: list[str], hyps: list[str], args: argparse.Namespace) -> np.ndarray:
    """BERTScore F1 (no baseline rescaling)."""
    from bert_score import score

    from cap_eval.scorer import resolve_device

    _, _, f1 = score(
        hyps,
        refs,
        model_type=args.bertscore_model,
        lang="en",
        batch_size=args.bertscore_batch_size,
        device=resolve_device(args.device),
        verbose=False,
    )
    return f1.numpy()


def score_comet(
    questions: list[str], refs: list[str], hyps: list[str], args: argparse.Namespace
) -> np.ndarray:
    """COMET with the question as source, candidate as MT output and gold answer as reference."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(args.comet_model))
    samples = [{"src": q, "mt": h, "ref": r} for q, h, r in zip(questions, hyps, refs)]
    output = model.predict(samples, batch_size=args.comet_batch_size, gpus=args.comet_gpus)
    return np.asarray(output.scores)


def pairwise(refs: list[str], hyps: list[str], fn: Callable[[str, str], float]) -> np.ndarray:
    return np.array([fn(r, h) for r, h in zip(refs, hyps)])


def tokenize(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    return word_tokenize(text.lower())


def ensure_nltk_resources() -> None:
    import nltk

    for name, resource in NLTK_RESOURCES.items():
        if _nltk_has(nltk, resource):
            continue
        logger.info("Downloading NLTK resource %s", name)
        if not nltk.download(name, quiet=True):
            raise RuntimeError(
                f"Could not download NLTK resource {name!r}. If this is an SSL certificate "
                "error on macOS, run 'Install Certificates.command' from your Python folder, "
                f"or install it manually: python -m nltk.downloader {name}"
            )


def _nltk_has(nltk, resource: str) -> bool:
    for candidate in (resource, f"{resource}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            pass
    return False


if __name__ == "__main__":
    main()
