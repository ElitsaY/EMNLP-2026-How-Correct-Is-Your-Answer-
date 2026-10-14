"""Fine-tune the question + answer -> declarative statement model (mT5).

    python scripts/train_statement_model.py
    python scripts/evaluate_dataset.py --statement-model outputs/statement_model/final

Outputs (under --output-dir):
    final/                  best model + tokenizer, loadable by CAPScorer(statement_model=...)
    splits.csv              train/validation/test assignment of every source row
    test_metrics.json       BLEU / ROUGE-L on the held-out test split
    train.manifest.json     settings, input hash, package versions, git commit
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cap_eval import constants as C
from cap_eval.evaluation import write_manifest
from cap_eval.scorer import format_inputs, normalize_text, set_seed

logger = logging.getLogger("train_statement_model")

# Hyperparameters used in the paper.
TRAINING = {
    "num_train_epochs": 15,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "logging_steps": 5,
    "save_total_limit": 2,
}
EARLY_STOPPING_PATIENCE = 8
HOLDOUT_FRACTION = 0.2  # split evenly into validation and test
STRATIFY_COLUMN = "language"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/CAP-Statements.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/statement_model"))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--epochs", type=int, default=TRAINING["num_train_epochs"])
    parser.add_argument("--full-determinism", action="store_true",
                        help="deterministic CUDA kernels (slower)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed, deterministic=args.full_determinism)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_statement_data(args.data)
    splits = split_data(df, args.seed)
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})
    pd.concat(
        [pd.DataFrame({"source_row": frame.index, "split": name}) for name, frame in splits.items()]
    ).sort_values("source_row").to_csv(args.output_dir / "splits.csv", index=False)

    test_metrics = train(splits, args)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    logger.info("Test metrics: %s", test_metrics)

    write_manifest(
        args.output_dir / "train.manifest.json",
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"training": TRAINING | {"num_train_epochs": args.epochs},
           "early_stopping_patience": EARLY_STOPPING_PATIENCE,
           "holdout_fraction": HOLDOUT_FRACTION,
           "stratify_column": STRATIFY_COLUMN},
        inputs={"data": args.data},
        extra={"test_metrics": test_metrics},
    )


def load_statement_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    columns = ["question", "answer", "statement"]
    df = df.dropna(subset=columns).copy()
    for col in columns:
        df[col] = normalize_text(df[col]).to_numpy()
    df = df[(df[columns] != "").all(axis=1)].reset_index(drop=True)
    df[STRATIFY_COLUMN] = df[STRATIFY_COLUMN].astype(str).str.strip()
    df["input_text"] = format_inputs(df["question"], df["answer"])
    df["target_text"] = df["statement"]
    return df


def split_data(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, holdout = train_test_split(
        df, test_size=HOLDOUT_FRACTION, random_state=seed, stratify=df[STRATIFY_COLUMN]
    )
    val_df, test_df = train_test_split(
        holdout, test_size=0.5, random_state=seed, stratify=holdout[STRATIFY_COLUMN]
    )
    return {"train": train_df, "validation": val_df, "test": test_df}


def train(splits: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, float]:
    import sacrebleu
    import torch
    from datasets import Dataset, DatasetDict
    from rouge_score import rouge_scorer
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        C.STATEMENT_BASE_MODEL, revision=C.STATEMENT_BASE_REVISION
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.STATEMENT_BASE_MODEL, revision=C.STATEMENT_BASE_REVISION
    )
    max_length = C.STATEMENT_TRAIN_MAX_LENGTH

    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], max_length=max_length, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=max_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    columns = ["input_text", "target_text"]
    tokenized = DatasetDict(
        {name: Dataset.from_pandas(frame[columns].reset_index(drop=True))
         for name, frame in splits.items()}
    ).map(preprocess, batched=True, remove_columns=columns)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [s.strip() for s in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [s.strip() for s in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        rouge_l = np.mean(
            [rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(decoded_labels, decoded_preds)]
        )
        return {"bleu": round(bleu, 2), "rougeL": round(rouge_l * 100, 2)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        **(TRAINING | {"num_train_epochs": args.epochs}),
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=C.STATEMENT_NUM_BEAMS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return trainer.evaluate(tokenized["test"], metric_key_prefix="test")


if __name__ == "__main__":
    main()
