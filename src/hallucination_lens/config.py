
"""Runtime settings for hallucination-lens API service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

from .validators import validate_threshold


@dataclass(frozen=True)
class Settings:
    """Represents runtime configuration for API, validation, and governance limits."""

    app_name: str
    app_version: str
    model_name: str
    default_threshold: float
    min_threshold: float
    max_threshold: float
    max_batch_items: int
    max_context_chars: int
    max_response_chars: int
    rate_limit_per_minute: int
    cors_origins: list[str]
    allowed_hosts: list[str]
    api_key: str
    metrics_api_key: str
    max_request_bytes: int
    secure_headers_enabled: bool
    hsts_max_age_seconds: int
    preload_model_on_startup: bool
    host: str
    port: int
    web_concurrency: int


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse environment booleans from common true/false string forms."""

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str) -> list[str]:
    """Split comma-separated environment values into normalized string list."""

    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_settings(settings: Settings) -> Settings:
    """Validate settings integrity so invalid deployments fail fast at startup."""

    validate_threshold(settings.default_threshold)
    validate_threshold(settings.min_threshold)
    validate_threshold(settings.max_threshold)

    if settings.min_threshold > settings.max_threshold:
        raise ValueError("MIN_THRESHOLD must be less than or equal to MAX_THRESHOLD")

    if not (settings.min_threshold <= settings.default_threshold <= settings.max_threshold):
        raise ValueError("FAITHFULNESS_THRESHOLD must be within MIN_THRESHOLD and MAX_THRESHOLD")

    if settings.max_batch_items <= 0:
        raise ValueError("MAX_BATCH_ITEMS must be greater than 0")

    if settings.max_context_chars <= 0 or settings.max_response_chars <= 0:
        raise ValueError("MAX_CONTEXT_CHARS and MAX_RESPONSE_CHARS must be greater than 0")

    if settings.rate_limit_per_minute <= 0:
        raise ValueError("RATE_LIMIT_PER_MINUTE must be greater than 0")

    if settings.max_request_bytes < 1024:
        raise ValueError("MAX_REQUEST_BYTES must be at least 1024")

    if not (1 <= settings.port <= 65535):
        raise ValueError("PORT must be between 1 and 65535")

    if settings.web_concurrency <= 0:
        raise ValueError("WEB_CONCURRENCY must be greater than 0")

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables using safe production defaults."""

    origin_value = os.getenv("CORS_ORIGINS", "http://127.0.0.1:4176,http://localhost:4176")
    allowed_hosts_value = os.getenv("ALLOWED_HOSTS", "*")

    settings = Settings(
        app_name=os.getenv("APP_NAME", "hallucination-lens"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        default_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.6")),
        min_threshold=float(os.getenv("MIN_THRESHOLD", "0.3")),
        max_threshold=float(os.getenv("MAX_THRESHOLD", "0.9")),
        max_batch_items=int(os.getenv("MAX_BATCH_ITEMS", "50")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "50000")),
        max_response_chars=int(os.getenv("MAX_RESPONSE_CHARS", "50000")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")),
        cors_origins=_split_csv(origin_value),
        allowed_hosts=_split_csv(allowed_hosts_value) or ["*"],
        api_key=os.getenv("HALLUCINATION_API_KEY", "").strip(),
        metrics_api_key=os.getenv("METRICS_API_KEY", "").strip(),
        max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", "1048576")),
        secure_headers_enabled=_as_bool(os.getenv("SECURE_HEADERS_ENABLED"), default=True),
        hsts_max_age_seconds=int(os.getenv("HSTS_MAX_AGE_SECONDS", "31536000")),
        preload_model_on_startup=_as_bool(os.getenv("PRELOAD_MODEL_ON_STARTUP"), default=False),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8003")),
        web_concurrency=int(os.getenv("WEB_CONCURRENCY", "1")),
    )
    return _validate_settings(settings)
