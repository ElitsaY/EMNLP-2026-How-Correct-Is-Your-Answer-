"""CAP: an NLI-based metric for judging QA answer correctness.

    from cap_eval import CAPScorer
    scores = CAPScorer().score(premises, hypotheses)
"""

from .evaluation import evaluate, load_correctness_data, write_tables

__version__ = "0.1.0"
__all__ = ["CAPScorer", "StatementGenerator", "evaluate", "load_correctness_data", "write_tables"]


def __getattr__(name):
    # Imported lazily so that evaluation-only use does not load torch/transformers.
    if name in {"CAPScorer", "StatementGenerator"}:
        from . import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
