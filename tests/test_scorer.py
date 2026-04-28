
"""Unit tests for faithfulness scoring behavior."""

import numpy as np

from hallucination_lens.scorer import FaithfulnessResult, HallucinationScorer, SentenceScore


class FakeEmbeddingModel:
    """Tiny deterministic embedding backend for fast tests."""

    def encode(self, texts, normalize_embeddings=True):
        """Return deterministic 3D vectors keyed by topic words."""

        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            if "paris" in lower or "france" in lower:
                vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "berlin" in lower or "germany" in lower:
                vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif "ocean" in lower:
                vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append(vector)

        return np.vstack(vectors)


def test_faithful_response_scores_high():
    """Response aligned with context should score above threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France.",
    )

    assert result.verdict == "faithful"
    assert result.score >= 0.9
    assert len(result.sentence_scores) == 1


def test_hallucinated_response_scores_low():
    """Response unrelated to context should score below threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Berlin is in Germany.",
    )

    assert result.verdict == "hallucinated"
    assert result.score < 0.6


def test_empty_context_returns_zero_score():
    """Empty context should always produce a hallucinated verdict."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="",
        response="Paris is in France.",
    )

    assert result.score == 0.0
    assert result.verdict == "hallucinated"
    assert result.sentence_scores[0].max_similarity == 0.0


def test_threshold_changes_verdict():
    """Higher thresholds should tighten the faithful decision boundary."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.95)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France. The ocean is blue.",
    )

    assert result.score < 0.95
    assert result.verdict == "hallucinated"


def test_batch_scoring_preserves_input_order_and_returns_results():
    """Batch scoring should return one result per pair in original order."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    results = scorer.batch_faithfulness_scores(
        pairs=[
            ("Paris is the capital of France.", "Paris is in France."),
            ("Paris is the capital of France.", "Berlin is in Germany."),
        ]
    )

    assert len(results) == 2
    assert results[0].verdict == "faithful"
    assert results[1].verdict == "hallucinated"


def test_threshold_override_applies_without_mutating_default_threshold():
    """Per-call threshold override should not mutate scorer default threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France. The ocean is blue.",
        threshold=0.95,
    )

    assert result.threshold == 0.95
    assert scorer.threshold == 0.6


def test_empty_response_returns_zero_and_no_sentence_evidence():
    """Empty response should return hallucinated verdict with no sentence scores."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="   ",
    )

    assert result.score == 0.0
    assert result.verdict == "hallucinated"
    assert result.sentence_scores == []


def test_constructor_rejects_threshold_outside_closed_interval():
    """Constructor should reject thresholds outside the [0, 1] interval."""

    for invalid in (-0.01, 1.01):
        try:
            HallucinationScorer(model=FakeEmbeddingModel(), threshold=invalid)
        except ValueError as exc:
            assert "threshold" in str(exc)
        else:  # pragma: no cover - safety branch
            raise AssertionError("Expected ValueError for invalid threshold")


def test_per_call_threshold_override_rejects_invalid_values():
    """Per-call threshold should validate bounds without mutating scorer state."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)

    for invalid in (-0.5, 1.2):
        try:
            scorer.faithfulness_score(
                context="Paris is the capital of France.",
                response="Paris is in France.",
                threshold=invalid,
            )
        except ValueError as exc:
            assert "threshold" in str(exc)
        else:  # pragma: no cover - safety branch
            raise AssertionError("Expected ValueError for invalid threshold override")


def test_faithfulness_result_to_dict_preserves_sentence_fields():
    """Serialization helper should expose score, verdict, threshold, and evidence list."""

    result = FaithfulnessResult(
        score=0.73,
        verdict="faithful",
        threshold=0.6,
        sentence_scores=[SentenceScore(sentence="Paris is in France.", max_similarity=0.73)],
    )

    payload = result.to_dict()
    assert payload["score"] == 0.73
    assert payload["verdict"] == "faithful"
    assert payload["threshold"] == 0.6
    assert payload["sentence_scores"][0]["sentence"] == "Paris is in France."
    assert payload["sentence_scores"][0]["max_similarity"] == 0.73
