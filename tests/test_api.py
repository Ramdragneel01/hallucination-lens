
"""Integration tests for hallucination-lens FastAPI service wrapper."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from hallucination_lens import api
from hallucination_lens.scorer import FaithfulnessResult, SentenceScore


class FakeScorer:
    """Deterministic scorer stub for API tests."""

    model_name = "fake-embedding"

    def faithfulness_score(self, context: str, response: str, threshold: float | None = None) -> FaithfulnessResult:
        """Return predictable single-score payload for endpoint contract testing."""

        active_threshold = 0.6 if threshold is None else threshold
        return FaithfulnessResult(
            score=0.82,
            verdict="faithful" if 0.82 >= active_threshold else "hallucinated",
            threshold=active_threshold,
            sentence_scores=[SentenceScore(sentence=response, max_similarity=0.82)],
        )

    def batch_faithfulness_scores(
        self,
        pairs: list[tuple[str, str]],
        threshold: float | None = None,
    ) -> list[FaithfulnessResult]:
        """Return deterministic batch result list aligned to input order."""

        active_threshold = 0.6 if threshold is None else threshold
        return [
            FaithfulnessResult(
                score=0.82,
                verdict="faithful" if 0.82 >= active_threshold else "hallucinated",
                threshold=active_threshold,
                sentence_scores=[SentenceScore(sentence=response, max_similarity=0.82)],
            )
            for _, response in pairs
        ]


def _client_with_fake_scorer(monkeypatch):
    """Create API test client with fake scorer to avoid model downloads."""

    monkeypatch.setattr(api, "get_scorer", lambda: FakeScorer())
    return TestClient(api.app)


def _override_settings(monkeypatch, **changes):
    """Apply temporary runtime settings overrides for API security tests."""

    monkeypatch.setattr(api, "settings", replace(api.settings, **changes))


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch):
    """Reset limiter and auth controls between tests for deterministic behavior."""

    api.limiter.clear()
    _override_settings(monkeypatch, api_key="", rate_limit_per_minute=120)

    yield

    api.limiter.clear()


def test_health_endpoint_exposes_limits_and_version(monkeypatch):
    """Health endpoint should return runtime metadata for observability."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["max_batch_items"] > 0


def test_score_endpoint_returns_contract(monkeypatch):
    """Single score endpoint should return verdict and sentence-level evidence."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "faithful"
    assert payload["model_name"] == "fake-embedding"
    assert len(payload["sentence_scores"]) == 1


def test_batch_endpoint_returns_aggregate(monkeypatch):
    """Batch endpoint should return aggregate fields and per-item results."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/batch",
        json={
            "items": [
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
            ],
            "threshold": 0.6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_count"] == 2
    assert len(payload["results"]) == 2
    assert payload["verdict_counts"]["faithful"] == 2


def test_batch_threshold_outside_governance_is_rejected(monkeypatch):
    """Threshold values outside governance range should return 422 error payload."""

    client = _client_with_fake_scorer(monkeypatch)
    response = client.post(
        "/batch",
        json={
            "items": [
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                }
            ],
            "threshold": 0.95,
        },
    )

    assert response.status_code == 422
    assert "governance limits" in response.json()["detail"]


def test_score_requires_api_key_when_configured(monkeypatch):
    """Score endpoint should require API key when runtime key is configured."""

    client = _client_with_fake_scorer(monkeypatch)
    _override_settings(monkeypatch, api_key="secret-key")

    unauthorized = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/score",
        headers={"X-API-Key": "secret-key"},
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert authorized.status_code == 200


def test_health_is_public_when_api_key_enabled(monkeypatch):
    """Health endpoint should remain public for probes."""

    client = _client_with_fake_scorer(monkeypatch)
    _override_settings(monkeypatch, api_key="secret-key")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_rate_limit_returns_429(monkeypatch):
    """Score endpoint should return 429 once request quota is exceeded."""

    client = _client_with_fake_scorer(monkeypatch)
    _override_settings(monkeypatch, rate_limit_per_minute=1)

    first = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/score",
        json={
            "context": "Paris is the capital of France.",
            "response": "Paris is in France.",
            "threshold": 0.6,
        },
    )
    assert second.status_code == 429
