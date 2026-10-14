"""The CAP metric and the statement generator it relies on.

    >>> scorer = CAPScorer()
    >>> scorer.score(["The capital of France is Paris."], ["Paris is the capital of France."])

To score raw (question, gold answer, answer) triples, pass a trained statement
model: ``CAPScorer(statement_model="outputs/statement_model/final")``.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from . import constants as C

logger = logging.getLogger(__name__)

_LABEL_PATTERNS = {"entailment": "entail", "neutral": "neutral", "contradiction": "contrad"}


def directional_score(probs: np.ndarray, neutral_weight: float = C.NEUTRAL_WEIGHT) -> np.ndarray:
    """``probs`` has columns ordered as ``constants.NLI_CLASSES``."""
    return probs[:, 0] + neutral_weight * probs[:, 1]


def cap_score(
    forward_probs: np.ndarray,
    reverse_probs: np.ndarray,
    neutral_weight: float = C.NEUTRAL_WEIGHT,
    forward_weight: float = C.FORWARD_WEIGHT,
) -> np.ndarray:
    forward = directional_score(forward_probs, neutral_weight)
    reverse = directional_score(reverse_probs, neutral_weight)
    return forward_weight * forward + (1 - forward_weight) * reverse


class CAPScorer:
    """Scores candidate statements (hypotheses) against gold statements (premises)."""

    def __init__(
        self,
        model: str = C.NLI_MODEL,
        revision: str | None = C.NLI_REVISION,
        device: str | None = None,
        batch_size: int = C.NLI_BATCH_SIZE,
        max_length: int = C.NLI_MAX_LENGTH,
        fp16: bool = True,
        neutral_weight: float = C.NEUTRAL_WEIGHT,
        forward_weight: float = C.FORWARD_WEIGHT,
        statement_model: str | Path | None = None,
    ):
        for name, value in (("neutral_weight", neutral_weight), ("forward_weight", forward_weight)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.neutral_weight = neutral_weight
        self.forward_weight = forward_weight

        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.class_ids = resolve_class_ids(self.model.config.id2label)
        logger.info("Loaded %s (%s) on %s", model, dtype, self.device)

        self.statements = (
            StatementGenerator(statement_model, device=self.device) if statement_model else None
        )

    def score(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """CAP score for each (premise, hypothesis) pair."""
        return self.score_detailed(premises, hypotheses)["cap"].to_numpy()

    def score_detailed(self, premises: Sequence[str], hypotheses: Sequence[str]) -> pd.DataFrame:
        """CAP plus both directional scores and all NLI probabilities."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        premises, hypotheses = list(map(str, premises)), list(map(str, hypotheses))
        forward = self.nli_probabilities(premises, hypotheses, desc="NLI premise->hypothesis")
        reverse = self.nli_probabilities(hypotheses, premises, desc="NLI hypothesis->premise")

        out = pd.DataFrame(
            {"cap": cap_score(forward, reverse, self.neutral_weight, self.forward_weight)}
        )
        for direction, probs in (("forward", forward), ("reverse", reverse)):
            out[f"cap_{direction}"] = directional_score(probs, self.neutral_weight)
            for j, name in enumerate(C.NLI_CLASSES):
                out[f"p_{name}_{direction}"] = probs[:, j]
            out[f"nli_label_{direction}"] = np.asarray(C.NLI_CLASSES)[probs.argmax(axis=1)]
        return out

    def score_answers(
        self, questions: Sequence[str], gold_answers: Sequence[str], answers: Sequence[str]
    ) -> pd.DataFrame:
        """Generate statements for raw QA triples, then score them."""
        if self.statements is None:
            raise ValueError("score_answers needs CAPScorer(statement_model=...)")
        premises = self.statements.generate(questions, gold_answers, desc="Premises")
        hypotheses = self.statements.generate(questions, answers, desc="Hypotheses")
        out = self.score_detailed(premises, hypotheses)
        out.insert(0, "premise", premises)
        out.insert(1, "hypothesis", hypotheses)
        return out

    def nli_probabilities(
        self, premises: list[str], hypotheses: list[str], desc: str = "NLI"
    ) -> np.ndarray:
        """Probabilities with columns ordered as ``constants.NLI_CLASSES``."""
        out = []
        for start in tqdm(range(0, len(premises), self.batch_size), desc=desc):
            enc = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, self.class_ids].cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(C.NLI_CLASSES)))


class StatementGenerator:
    """Turns (question, answer) into a declarative statement with a fine-tuned mT5."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        batch_size: int = C.STATEMENT_BATCH_SIZE,
        max_input_length: int = C.STATEMENT_MAX_INPUT_LENGTH,
        max_new_tokens: int = C.STATEMENT_MAX_NEW_TOKENS,
        num_beams: int = C.STATEMENT_NUM_BEAMS,
        template: str = C.STATEMENT_INPUT_TEMPLATE,
    ):
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device).eval()

    def generate(
        self, questions: Sequence[str], answers: Sequence[str], desc: str = "Statements"
    ) -> list[str]:
        inputs = format_inputs(questions, answers, self.template)
        outputs: list[str] = []
        for start in tqdm(range(0, len(inputs), self.batch_size), desc=desc):
            enc = self.tokenizer(
                inputs[start : start + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    early_stopping=True,
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def normalize_text(series: pd.Series | Sequence[str]) -> pd.Series:
    """Collapse whitespace; used identically for training and generation inputs."""
    return pd.Series(series, dtype=str).str.replace(r"\s+", " ", regex=True).str.strip()


def format_inputs(
    questions: Sequence[str], answers: Sequence[str], template: str = C.STATEMENT_INPUT_TEMPLATE
) -> list[str]:
    return [
        template.format(question=q, answer=a)
        for q, a in zip(normalize_text(questions), normalize_text(answers))
    ]


def resolve_class_ids(id2label: dict[int, str]) -> list[int]:
    """Model output indices for ``constants.NLI_CLASSES``."""
    ids = []
    for name in C.NLI_CLASSES:
        matches = [i for i, label in id2label.items() if _LABEL_PATTERNS[name] in label.lower()]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify the {name!r} class in {id2label}")
        ids.append(int(matches[0]))
    return ids


def resolve_device(preference: str | None = None) -> str:
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = C.SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Required by cuBLAS for deterministic matmuls.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
