"""Score CAP-Correctness with COMET and report COMET stats per label.

    python scripts/run_comet_eval.py
    python scripts/run_comet_eval.py --input my_data.csv --output my_scores.csv

COMET gets the question as source, the candidate answer as MT output and the
gold answer as reference. The input CSV needs question, gold_answer, answer and
label_semantic columns. The output CSV is the input plus a comet_score column.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from comet import download_model, load_from_checkpoint

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", type=Path, default=Path("data/CAP-Correctness.csv"))
parser.add_argument("--output", type=Path, default=Path("outputs/scores/comet_eval.csv"))
parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--gpus", type=int, default=0)
args = parser.parse_args()

print("Loading dataset...")
df = pd.read_csv(args.input)
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# One row in CAP-Correctness has no candidate answer.
df = df.dropna(subset=["question", "gold_answer", "answer"]).reset_index(drop=True)
print(f"Rows with question, gold answer and answer: {len(df)}")

print(f"\nDownloading/loading COMET model ({args.model})...")
model_path = download_model(args.model)
model = load_from_checkpoint(model_path)

data = [
    {
        "src": str(row["question"]),
        "mt": str(row["answer"]),
        "ref": str(row["gold_answer"]),
    }
    for _, row in df.iterrows()
]

print(f"\nRunning COMET on {len(data)} rows...")
accelerator = "gpu" if args.gpus > 0 else "cpu"
output = model.predict(data, batch_size=args.batch_size, gpus=args.gpus, num_workers=0, accelerator=accelerator)

scores = output.scores
df["comet_score"] = scores

args.output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output, index=False)
print(f"\nSaved output CSV to: {args.output}")

print("\n=== Report: COMET stats by label_semantic ===\n")
report = df.groupby("label_semantic")["comet_score"].agg(
    avg="mean",
    min="min",
    max="max",
    std="std",
    count="count",
).round(4)
print(report.to_string())
