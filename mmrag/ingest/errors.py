"""
mmrag/ingest/errors.py
=========================

Input-validation errors for the media channels. These represent CLIENT
problems (bad upload), not provider outages — the pipeline maps them to a
"rejected" status (HTTP 422) rather than a failure, and they never consume paid resources.
"""


class MediaError(Exception):
    """Base class for invalid media input."""


class UnsupportedMediaError(MediaError):
    """The payload is not a supported/readable image or audio file."""


class MediaTooLargeError(MediaError):
    """The payload exceeds the configured size (or pixel-count) limit."""
