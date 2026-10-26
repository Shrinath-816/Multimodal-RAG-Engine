"""
mmrag/security.py
====================

Compact versions of the Week 8 defenses, applied to MULTIMODAL inputs:

- Injection scanning + delimiting: text transcribed from a voice call, text
  read out of an image (a "typographic attack" carrier), and ticket text are
  ALL untrusted data — same Week 8 section 2-3 defenses, new carriers.
- PII redaction at the boundary: structured PII is stripped from every
  channel's text before it reaches retrieval, the extraction model, the
  structured record, or any log.

Defense in depth, not a guarantee: pattern matching catches known phrasings
only. It is paired with least-privilege design (this engine's model has NO
tools and cannot take side-effecting actions) and deterministic validation.
"""

import re

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above) instructions",
    r"you are now in (debug|developer|admin) mode",
    r"system\s*override",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as if you (have no|had no) (restrictions|rules|guidelines)",
    r"new instructions?:",
    r"refund every order",
]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Order matters: SSN before the broader card/phone patterns.
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


def scan_for_injection(text: str) -> list[str]:
    """Return the injection patterns that matched `text` (empty list = none found)."""
    return [p.pattern for p in _COMPILED_INJECTION if p.search(text)]


def delimit_untrusted(label: str, content: str) -> str:
    """
    Wrap untrusted content in labeled tags so the model gets a structural
    signal that it is DATA, not instructions.

    Also neutralizes any closing-tag text inside the content, so an attacker
    cannot "break out" of the delimiter by embedding `</untrusted_...>`.
    """
    safe = content.replace("</untrusted_", "<\\/untrusted_")
    return f"<untrusted_{label}>\n{safe}\n</untrusted_{label}>"


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """
    Redact structured PII.

    Returns:
        (redacted_text, counts) — counts per PII type, never the values.
        Unstructured PII (a bare name in prose) is NOT detected here; see
        week-9-notes.md section 3.4 / Week 8 section 5 for NER-based upgrades.
    """
    counts: dict[str, int] = {}
    redacted = text
    for pii_type, pattern in _PII_PATTERNS.items():
        found = len(pattern.findall(redacted))
        if found:
            counts[pii_type] = found
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted, counts
