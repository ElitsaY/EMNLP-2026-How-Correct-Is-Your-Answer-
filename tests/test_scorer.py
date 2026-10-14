import numpy as np
import pandas as pd
import pytest

from cap_eval import constants as C
from cap_eval.scorer import cap_score, directional_score, format_inputs, resolve_class_ids


def test_directional_score():
    probs = np.array([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(directional_score(probs, 0.3), [0.76, 0.3])


def test_cap_score_weights_directions():
    entail = np.array([[1.0, 0.0, 0.0]])
    contradict = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(cap_score(entail, contradict, 0.3, 0.85), [0.85])
    np.testing.assert_allclose(cap_score(contradict, entail, 0.3, 0.85), [0.15])


def test_cap_score_defaults_match_constants():
    rng = np.random.default_rng(0)
    fwd, rev = rng.dirichlet(np.ones(3), size=(2, 10))
    np.testing.assert_allclose(
        cap_score(fwd, rev), cap_score(fwd, rev, C.NEUTRAL_WEIGHT, C.FORWARD_WEIGHT)
    )


def test_cap_score_is_bounded():
    rng = np.random.default_rng(0)
    fwd, rev = rng.dirichlet(np.ones(3), size=(2, 1000))
    scores = cap_score(fwd, rev)
    assert scores.min() >= 0 and scores.max() <= 1


def test_resolve_class_ids():
    assert resolve_class_ids({0: "contradiction", 1: "entailment", 2: "neutral"}) == [1, 2, 0]
    with pytest.raises(ValueError):
        resolve_class_ids({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_format_inputs_normalizes_whitespace():
    out = format_inputs(pd.Series(["What  is\nit?"], index=[7]), [" x "])
    assert out == ["question: What is it? answer: x"]
