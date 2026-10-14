"""Default settings for CAP and its evaluation.

These are the values used in the paper. Scripts expose the ones worth changing
as command-line flags, and every run records the values it used in its manifest.
"""

SEED = 42

# --- CAP: bidirectional NLI -------------------------------------------------
# dir(a -> b) = P(entailment | a, b) + NEUTRAL_WEIGHT * P(neutral | a, b)
# CAP         = FORWARD_WEIGHT * dir(premise -> hypothesis)
#             + (1 - FORWARD_WEIGHT) * dir(hypothesis -> premise)
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_REVISION = "bab4bc7178836f731dcfd18c06ca9def0a137712"  # pinned Hugging Face commit
NLI_MAX_LENGTH = 512
NLI_BATCH_SIZE = 16
NLI_CLASSES = ("entailment", "neutral", "contradiction")
NEUTRAL_WEIGHT = 0.3
FORWARD_WEIGHT = 0.85

# --- Statement model: question + answer -> declarative statement -------------
STATEMENT_BASE_MODEL = "google/mt5-small"
STATEMENT_BASE_REVISION = "73fb5dbe4756edadc8fbe8c769b0a109493acf7a"  # pinned Hugging Face commit
STATEMENT_INPUT_TEMPLATE = "question: {question} answer: {answer}"
STATEMENT_TRAIN_MAX_LENGTH = 512  # input and target length during training
# The released CAP-Correctness statements were generated with 128-token inputs.
STATEMENT_MAX_INPUT_LENGTH = 128
STATEMENT_MAX_NEW_TOKENS = 512
STATEMENT_NUM_BEAMS = 4
STATEMENT_BATCH_SIZE = 16

# --- Human labels -------------------------------------------------------------
LABEL_COLUMN = "label_semantic"
# Ordinal severity, higher = more correct. Rank statistics only depend on the
# ordering and ties, not on the values themselves.
SEVERITY = {
    "exact": 6,
    "equivalent": 6,
    "alternative_correct": 6,
    "overinclusive_valid": 5,
    "partial": 4,
    "overinclusive_invalid": 2,
    "invalid": 1,
    "contradictory": 0,
}
# (better, worse) label pairs that are hard to separate; reported separately.
HARD_PAIRS = (
    ("equivalent", "partial"),
    ("overinclusive_valid", "partial"),
    ("overinclusive_valid", "overinclusive_invalid"),
    ("alternative_correct", "invalid"),
)

# --- Evaluation -----------------------------------------------------------------
N_BOOTSTRAP = 10_000
# Scores are rounded to this many decimals before comparison, so values that differ
# only by floating-point noise (e.g. 0.5 vs 0.49999999999999994) count as ties.
SCORE_DECIMALS = 10
CI_LEVEL = 0.95
BASELINE_METRICS = ("bleu", "rouge_l", "meteor", "bertscore", "comet")
