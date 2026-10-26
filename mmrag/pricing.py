"""
mmrag/pricing.py
===================

Turns metered Usage into dollars — the basis of cost-per-ticket tracking.

IMPORTANT: this module ships NO real vendor prices. config/pricing.example.json
contains obviously fake, illustrative numbers. Put your providers' CURRENT
prices in your own pricing file (PRICING_PATH) before trusting any dashboard
figure. A model with no entry is priced with the table's `default` entry and
FLAGGED as estimated — never silently priced at $0 (see Week 1's pricing rule).
"""

import json
from dataclasses import dataclass

from mmrag.schemas import Usage


class PricingError(Exception):
    """Raised when a price cannot be determined or the price file is invalid."""


@dataclass(frozen=True)
class ModelPrice:
    """
    Per-model price card (USD).

    Attributes:
        input_per_million: Price per 1M input tokens (image tokens are part of input).
        output_per_million: Price per 1M output tokens.
        per_audio_minute: Price per minute of audio (speech-to-text).
    """

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    per_audio_minute: float = 0.0


def _parse_price(name: str, raw: dict) -> ModelPrice:
    if not isinstance(raw, dict):
        raise PricingError(f"Price entry {name!r} must be an object.")
    values = {}
    for key in ("input_per_million", "output_per_million", "per_audio_minute"):
        value = raw.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise PricingError(f"Price entry {name!r} field {key!r} must be a non-negative number.")
        values[key] = float(value)
    return ModelPrice(**values)


class PricingTable:
    """Looks up prices by model id and converts Usage to USD."""

    def __init__(self, prices: dict[str, ModelPrice], default: ModelPrice | None = None) -> None:
        """
        Args:
            prices: model id -> ModelPrice.
            default: Fallback price for unknown models (flagged as estimated).
                If None, an unknown model raises PricingError.
        """
        self._prices = dict(prices)
        self._default = default

    @classmethod
    def from_file(cls, path: str) -> "PricingTable":
        """
        Load a pricing JSON file:
        {"models": {"<id>": {"input_per_million": .., "output_per_million": .., "per_audio_minute": ..}},
         "default": {...}}
        """
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PricingError(f"Could not read pricing file {path!r}: {exc}") from exc

        models = raw.get("models", {})
        if not isinstance(models, dict):
            raise PricingError("'models' must be an object.")
        prices = {name: _parse_price(name, entry) for name, entry in models.items()}
        default = _parse_price("default", raw["default"]) if "default" in raw else None
        return cls(prices, default)

    def cost_for(self, model: str, usage: Usage) -> tuple[float, bool]:
        """
        Price one provider call.

        Args:
            model: The model id that ACTUALLY served the call (after any fallback).
            usage: Metered usage reported by the provider.

        Returns:
            (cost_usd, estimated) — estimated is True when the default
            price was used because the model has no entry.

        Raises:
            PricingError: Unknown model and no default configured.
        """
        price = self._prices.get(model)
        estimated = False
        if price is None:
            if self._default is None:
                raise PricingError(f"No price configured for model {model!r} and no default.")
            price, estimated = self._default, True

        cost = (
            usage.input_tokens / 1_000_000 * price.input_per_million
            + usage.output_tokens / 1_000_000 * price.output_per_million
            + usage.audio_seconds / 60.0 * price.per_audio_minute
        )
        return round(cost, 8), estimated

    def estimate_chat_cost(self, model: str, est_input_tokens: int, est_output_tokens: int) -> float:
        """
        PRE-call cost estimate used by the budget guard (checked BEFORE
        spending, per Week 8 section 18).
        """
        cost, _ = self.cost_for(model, Usage(input_tokens=est_input_tokens, output_tokens=est_output_tokens))
        return cost
