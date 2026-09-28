"""Rule-based PHI detectors aligned to the HIPAA Safe Harbor method.

Covers the Safe Harbor identifiers that occur in free-text clinical notes:
names, geographic subdivisions smaller than state, all date elements
except year, phone/fax numbers, email addresses, SSNs, medical record /
health-plan numbers, account numbers, and ages over 89.

Detection is purely regex + small gazetteers over synthetic demo data,
so it runs on CPU with no model downloads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

PHI_LABELS = ["NAME", "DATE", "PHONE", "EMAIL", "SSN", "MRN", "ACCOUNT", "ADDRESS", "AGE"]


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str = ""

    def key(self):
        return (self.start, self.end, self.label)


Detector = Callable[[str], List[Span]]


def _regex_spans(pattern: str, text: str, label: str, group: int | str = 0,
                 flags: int = 0) -> List[Span]:
    out: List[Span] = []
    for m in re.finditer(pattern, text, flags):
        s, e = m.span(group)
        out.append(Span(s, e, label, m.group(group)))
    return out


# ---------------------------------------------------------------- patterns

SSN_PATTERNS = [r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"]

PHONE_PATTERNS = [
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
]

EMAIL_PATTERNS = [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"]

MRN_PATTERNS = [
    r"\b(?:MRN|Medical Record Number)\s*[:#]?\s*[A-Z]{0,3}-?\d{5,10}\b",
]

ACCOUNT_PATTERNS = [
    r"\b(?:Account|Acct)(?:\s+No\.?|\s+Number|\s*#)?\s*[:#]?\s*\d{6,12}\b",
]

# Safe Harbor: only ages > 89 are identifiers. "90-year-old", "Age: 92", "91 y/o".
AGE_PATTERNS = [
    r"\b[Aa]ge[d]?\s*[:\-]?\s*(?:9\d|[1-9]\d{2})\b",
    r"\b(?:9\d|[1-9]\d{2})\s*(?:-?years?-?old|y/?o\.?)\b",
]

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
          r"Nov(?:ember)?|Dec(?:ember)?)")
DATE_PATTERNS = [
    rf"\b{_MONTH}\.?\s+\d{{1,2}},?\s+\d{{4}}\b",   # March 5, 1932
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",                 # 03/15/1957
    r"\b\d{4}-\d{2}-\d{2}\b",                     # 1980-11-02
]

_NAME_WORD = r"[A-Z][a-z]+"
STREET_PATTERNS = [
    rf"\b\d{{1,5}}\s+{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}\s+"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|"
    r"Court|Ct|Circle|Cir|Parkway|Pkwy)\b",
]
ZIP_PATTERNS = [r"\b\d{5}(?:-\d{4})?\b"]

# Fake locations appearing in the synthetic notes (demo gazetteer).
SYNTHETIC_CITY_STATE = ["Springfield, IL", "Riverside, CA", "Madison, WI"]

# Fake person names appearing in the synthetic notes (demo gazetteer).
SYNTHETIC_NAME_GAZETTEER = ["Robert Johnson", "Maria Garcia", "David Kim"]


def detect_ssn(text: str) -> List[Span]:
    return [s for p in SSN_PATTERNS for s in _regex_spans(p, text, "SSN")]


def detect_phone(text: str) -> List[Span]:
    return [s for p in PHONE_PATTERNS for s in _regex_spans(p, text, "PHONE")]


def detect_email(text: str) -> List[Span]:
    return [s for p in EMAIL_PATTERNS for s in _regex_spans(p, text, "EMAIL")]


def detect_mrn(text: str) -> List[Span]:
    return [s for p in MRN_PATTERNS for s in _regex_spans(p, text, "MRN")]


def detect_account(text: str) -> List[Span]:
    return [s for p in ACCOUNT_PATTERNS for s in _regex_spans(p, text, "ACCOUNT")]


def detect_age(text: str) -> List[Span]:
    return [s for p in AGE_PATTERNS for s in _regex_spans(p, text, "AGE")]


def detect_date(text: str) -> List[Span]:
    return [s for p in DATE_PATTERNS for s in _regex_spans(p, text, "DATE")]


def detect_street(text: str) -> List[Span]:
    return [s for p in STREET_PATTERNS for s in _regex_spans(p, text, "ADDRESS")]


def detect_zip(text: str) -> List[Span]:
    return [s for p in ZIP_PATTERNS for s in _regex_spans(p, text, "ADDRESS")]


def detect_city_state(text: str) -> List[Span]:
    out: List[Span] = []
    for city in SYNTHETIC_CITY_STATE:
        out += _regex_spans(rf"\b{re.escape(city)}\b", text, "ADDRESS")
    return out


def detect_names(text: str) -> List[Span]:
    spans: List[Span] = []
    # Titled names: "Dr. Emily Chen", "Mr. James Carter"
    spans += _regex_spans(
        rf"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,2}}\b",
        text, "NAME")
    # "Patient: Robert Johnson"
    spans += _regex_spans(
        rf"\b[Pp]atient\s*:\s*(?P<n>{_NAME_WORD}\s+{_NAME_WORD})\b",
        text, "NAME", group="n")
    # "Electronically signed by Lisa Wang, MD"
    spans += _regex_spans(
        rf"\b[Ss]igned by\s*:?\s*(?P<n>{_NAME_WORD}\s+{_NAME_WORD})"
        r"(?:,\s*(?:MD|DO|NP|PA))?\b",
        text, "NAME", group="n")
    for full in SYNTHETIC_NAME_GAZETTEER:
        spans += _regex_spans(rf"\b{re.escape(full)}\b", text, "NAME")
    return spans


# (priority, detector) — higher priority wins when spans overlap.
DETECTORS: List[tuple[int, Detector]] = [
    (100, detect_ssn),
    (95, detect_email),
    (90, detect_phone),
    (85, detect_mrn),
    (80, detect_account),
    (75, detect_age),
    (70, detect_date),
    (60, detect_names),
    (50, detect_street),
    (45, detect_city_state),
    (40, detect_zip),
]


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def detect(text: str) -> List[Span]:
    """Run all detectors and resolve overlapping spans by priority."""
    candidates: List[tuple[int, Span]] = []
    for priority, fn in DETECTORS:
        for span in fn(text):
            candidates.append((priority, span))
    # Highest priority first; ties broken by earliest start, then longest span.
    candidates.sort(key=lambda c: (-c[0], c[1].start, -(c[1].end - c[1].start)))
    accepted: List[Span] = []
    for _, span in candidates:
        if not any(_overlaps(span, a) for a in accepted):
            accepted.append(span)
    accepted.sort(key=lambda s: s.start)
    return accepted
