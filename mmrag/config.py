"""
mmrag/config.py
=================

Runtime configuration loaded from environment variables (see
.env.example). Plain frozen dataclass — deliberately NOT pydantic-settings —
so the core stays standard-library-only and config logic is unit-testable.

DEMO_MODE swaps every external provider for a deterministic fake (see
mmrag/providers/fakes.py) so the full system, including the live dashboard,
runs with no API keys. Costs shown in demo mode are SYNTHETIC.
"""

import os
from dataclasses import dataclass


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _env_bool(env: dict, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(env: dict, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(env: dict, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_list(env: dict, name: str, default: tuple) -> tuple:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """
    All tunables for the engine.

    Attributes:
        demo_mode: Use deterministic fake providers (no API keys needed).
        openrouter_api_key / openrouter_base_url: OpenAI-compatible endpoint
            for chat, vision, and embeddings (OpenRouter by default).
        openai_api_key / openai_base_url: Endpoint for speech-to-text
            (Whisper). Kept separate because not every router proxies the
            audio transcription API.
        chat_models: Priority-ordered models for extraction (first = primary,
            rest = fallbacks). Verify slugs at your provider before use.
        vision_models: Priority-ordered vision-capable models.
        embedding_model: ONE embedding model, no fallback — switching
            embedding models mid-stream breaks vector-space consistency
            (the Week 2 invariant).
        stt_model: Speech-to-text model id.
        max_image_bytes / max_image_side: Upload cap and downscale target
            (resolution is a direct cost/latency lever — Week 9 section 3.3).
        max_audio_bytes: Upload cap for audio.
        max_ticket_cost_usd: Hard per-ticket spend cap (Week 8 section 18).
        review_confidence_threshold: Extractions below this confidence are
            forced to human review, whatever the model claims about itself.
        top_k: Chunks retrieved per ticket.
        model_timeout_seconds / model_retries / retry_backoff_seconds:
            Resilience tuning for every provider call.
        kb_dir / pricing_path: Knowledge base directory and price table.
        ledger_max_tickets: Retention bound for the in-memory cost ledger.
        order_id_pattern: Regex a valid order id must match.
    """

    demo_mode: bool = True
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_models: tuple = ("anthropic/claude-sonnet-4.6", "openai/gpt-4o-mini")
    vision_models: tuple = ("anthropic/claude-sonnet-4.6", "openai/gpt-4o-mini")
    embedding_model: str = "openai/text-embedding-3-small"
    stt_model: str = "whisper-1"
    max_image_bytes: int = 5_000_000
    max_image_side: int = 1024
    max_audio_bytes: int = 10_000_000
    max_ticket_cost_usd: float = 0.25
    review_confidence_threshold: float = 0.6
    top_k: int = 4
    model_timeout_seconds: float = 30.0
    model_retries: int = 1
    retry_backoff_seconds: float = 0.5
    kb_dir: str = "data/knowledge_base"
    pricing_path: str = "config/pricing.example.json"
    ledger_max_tickets: int = 5000
    order_id_pattern: str = r"^ORD-\d{3,}$"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        """
        Build Settings from environment variables.

        Args:
            env: Mapping to read from (defaults to os.environ) — injectable
                so tests never depend on the real process environment.

        Raises:
            ConfigError: If a value is malformed, or if demo_mode is off and
                the required API keys are missing.
        """
        env = dict(os.environ) if env is None else env
        defaults = cls()
        settings = cls(
            demo_mode=_env_bool(env, "DEMO_MODE", defaults.demo_mode),
            openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
            openrouter_base_url=env.get("OPENROUTER_BASE_URL", defaults.openrouter_base_url),
            openai_api_key=env.get("OPENAI_API_KEY", ""),
            openai_base_url=env.get("OPENAI_BASE_URL", defaults.openai_base_url),
            chat_models=_env_list(env, "CHAT_MODELS", defaults.chat_models),
            vision_models=_env_list(env, "VISION_MODELS", defaults.vision_models),
            embedding_model=env.get("EMBEDDING_MODEL", defaults.embedding_model),
            stt_model=env.get("STT_MODEL", defaults.stt_model),
            max_image_bytes=_env_int(env, "MAX_IMAGE_BYTES", defaults.max_image_bytes),
            max_image_side=_env_int(env, "MAX_IMAGE_SIDE", defaults.max_image_side),
            max_audio_bytes=_env_int(env, "MAX_AUDIO_BYTES", defaults.max_audio_bytes),
            max_ticket_cost_usd=_env_float(env, "MAX_TICKET_COST_USD", defaults.max_ticket_cost_usd),
            review_confidence_threshold=_env_float(
                env, "REVIEW_CONFIDENCE_THRESHOLD", defaults.review_confidence_threshold
            ),
            top_k=_env_int(env, "TOP_K", defaults.top_k),
            model_timeout_seconds=_env_float(env, "MODEL_TIMEOUT_SECONDS", defaults.model_timeout_seconds),
            model_retries=_env_int(env, "MODEL_RETRIES", defaults.model_retries),
            retry_backoff_seconds=_env_float(env, "RETRY_BACKOFF_SECONDS", defaults.retry_backoff_seconds),
            kb_dir=env.get("KB_DIR", defaults.kb_dir),
            pricing_path=env.get("PRICING_PATH", defaults.pricing_path),
            ledger_max_tickets=_env_int(env, "LEDGER_MAX_TICKETS", defaults.ledger_max_tickets),
            order_id_pattern=env.get("ORDER_ID_PATTERN", defaults.order_id_pattern),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail fast on missing keys or nonsensical limits (see Week 1's config rationale)."""
        if not self.demo_mode:
            if not self.openrouter_api_key:
                raise ConfigError("OPENROUTER_API_KEY is required when DEMO_MODE is off.")
            if not self.openai_api_key:
                raise ConfigError(
                    "OPENAI_API_KEY is required when DEMO_MODE is off (used for speech-to-text)."
                )
        if self.max_ticket_cost_usd <= 0:
            raise ConfigError("MAX_TICKET_COST_USD must be positive.")
        if not 0.0 <= self.review_confidence_threshold <= 1.0:
            raise ConfigError("REVIEW_CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if self.top_k < 1:
            raise ConfigError("TOP_K must be at least 1.")
        if not self.chat_models or not self.vision_models:
            raise ConfigError("CHAT_MODELS and VISION_MODELS must each list at least one model.")
