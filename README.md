# EMNLP 2026 — How Correct Is Your Answer?

Data and code for **CAP**, an NLI-based metric for judging QA answer correctness, benchmarked against standard NLG metrics (BLEU, ROUGE-L, METEOR, BERTScore, COMET) on questions from ARC-Easy, OpenBookQA, and MMLU.

## How CAP works

A fine-tuned mT5 model turns *(question, gold answer)* into a declarative **premise** and *(question, candidate answer)* into a **hypothesis**. An NLI cross-encoder then scores both directions:

```
dir(a → b) = P(entailment | a, b) + 0.3 · P(neutral | a, b)
CAP        = 0.85 · dir(premise → hypothesis) + 0.15 · dir(hypothesis → premise)
```

## Using the metric

```bash
pip install -e .
```

```python
from cap_eval import CAPScorer

scorer = CAPScorer()  # cross-encoder/nli-deberta-v3-large, pinned revision
scorer.score(
    ["The powerhouse of the cell is the mitochondria."],   # premises (gold)
    ["The mitochondria is the powerhouse of the cell."],   # hypotheses (candidate)
)

# Raw (question, gold answer, answer) triples need a trained statement model:
scorer = CAPScorer(statement_model="outputs/statement_model/final")
scorer.score_answers(["What is the powerhouse of the cell?"], ["mitochondria"], ["the mitochondrion"])
```

`score_detailed()` also returns both directional scores and all NLI probabilities.

## Repository structure

```
.
├── src/cap_eval/
│   ├── constants.py            # every default: models + pinned revisions, CAP weights, severity map, seed
│   ├── scorer.py               # CAPScorer and StatementGenerator
│   └── evaluation.py           # data loading, agreement statistics, benchmark tables, run manifests
├── scripts/
│   ├── evaluate_dataset.py     # score CAP-Correctness with CAP and benchmark it against the baselines
│   ├── score_baselines.py      # BLEU, ROUGE-L, METEOR, BERTScore, COMET
│   ├── run_comet_eval.py       # standalone COMET run with per-label stats
│   └── train_statement_model.py  # fine-tune the question + answer → statement model
├── tests/
│   ├── test_scorer.py
│   └── test_evaluation.py
├── data/
│   ├── CAP-Correctness.csv          # 8,827 labeled QA rows with premise/hypothesis statements
│   ├── CAP-Correctness-splits.csv   # row_id → validation (1,000) / test (7,827), as in the paper
│   └── CAP-Statements.csv           # 11,000 question/answer/statement training examples
├── pyproject.toml
└── requirements.txt
```

## Reproducing the results

```bash
pip install -r requirements.txt

python scripts/score_baselines.py      # -> outputs/scores/<metric>.csv
python scripts/evaluate_dataset.py     # -> outputs/scores/cap.csv, outputs/evaluation/
```

Results are reported on the test split by default, as in the paper (`--split validation` or `--split all` for the others). This uses the released premise/hypothesis statements. To retrain the statement model and regenerate the statements first:

```bash
python scripts/train_statement_model.py                                          # -> outputs/statement_model/final
python scripts/evaluate_dataset.py --statement-model outputs/statement_model/final
```

To run COMET on its own and print its score stats per label:

```bash
python scripts/run_comet_eval.py      # -> outputs/scores/comet_eval.csv
```

Run each script with `--help` for its options, e.g. `--baselines` (compare against a subset, or none), `--device`, `--n-bootstrap`, `--reuse-cap-scores`.

### Outputs in `outputs/evaluation/`

| File | Contents |
|---|---|
| `summary.md` | Headline tables |
| `correlations.csv` | Spearman ρ, Kendall τ-b and pairwise ranking accuracy per metric, with bootstrap CIs |
| `descriptives.csv` | Per-label score distribution per metric |
| `class_pairs.csv` | For every (better, worse) label pair: accuracy, AUC, violation rate, class means |
| `hard_pairs.csv` | The subset of `class_pairs.csv` in `constants.HARD_PAIRS` |
| `monotonicity.csv` | How many class pairs have their mean scores in the wrong order |
| `paired_differences.csv` | Paired-bootstrap difference between CAP and each baseline |

**Definitions.** Severity follows `constants.SEVERITY`. A pair of examples is comparable when their severities differ. *Pairwise accuracy* is the share of comparable pairs where the more-correct answer scores strictly higher (score ties count as not correct). *AUC* counts ties as half. Confidence intervals are percentile bootstrap intervals over rows (10,000 resamples). Every metric uses the same resamples, which is what makes the paired differences valid.

## Data

`CAP-Correctness.csv` has one row per (question, gold answer, candidate answer) triple. Each row has a human-assigned correctness label (`exact`, `equivalent`, `alternative_correct`, `overinclusive_valid`, `partial`, `overinclusive_invalid`, `invalid`, `contradictory`) and the generated `premise`/`hypothesis` statements. The questions come from ARC-Easy, OpenBookQA and MMLU. One row has no candidate answer and is excluded from evaluation.

`CAP-Statements.csv` has 11k examples used to train the question + answer → declarative statement model.

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for licensing and attribution of the underlying QA benchmarks.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The pairwise statistics are checked against brute-force implementations.

## License

Code is released under the [MIT License](LICENSE). Data licensing (including third-party attribution for ARC-Easy, OpenBookQA, and MMLU) is described in [`DATA_LICENSES.md`](DATA_LICENSES.md).
