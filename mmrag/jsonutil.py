"""
mmrag/jsonutil.py
====================

Tolerant JSON-object extraction from model output. Models wrap JSON in
markdown fences or add a sentence before it; rather than depend on a
provider-specific "JSON mode" (not portable across routed models), we
parse defensively and let the validation + repair loop handle the rest.
"""

import json


def extract_json_object(text: str) -> dict:
    """
    Extract the first JSON object from model output.

    Args:
        text: Raw model output, possibly fenced or surrounded by prose.

    Returns:
        The parsed dict.

    Raises:
        ValueError: If no JSON object can be found/parsed, or the value found
            is not an object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json) and a trailing fence.
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]

    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            start = stripped.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = stripped.find("{", start + 1)

    raise ValueError("No JSON object found in model output.")
