"""Replace detected PHI spans with Safe Harbor tags.

Dates keep their year (Safe Harbor permits year); everything else is
replaced with a bracketed tag such as [NAME], [SSN], [MRN].
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .detectors import Span, detect


def replacement(span: Span) -> str:
    if span.label == "DATE":
        year = re.search(r"\d{4}", span.text)
        return "[DATE]" + (f" {year.group(0)}" if year else "")
    return f"[{span.label}]"


def redact(text: str, spans: List[Span] | None = None) -> str:
    """Return *text* with every PHI span replaced by its tag."""
    spans = detect(text) if spans is None else spans
    out = text
    for s in sorted(spans, key=lambda sp: sp.start, reverse=True):
        out = out[:s.start] + replacement(s) + out[s.end:]
    return out


def redact_with_spans(text: str) -> Tuple[str, List[Span]]:
    spans = detect(text)
    return redact(text, spans), spans
