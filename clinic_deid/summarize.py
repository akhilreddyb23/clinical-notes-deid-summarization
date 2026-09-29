"""Extractive clinical summarizer (CPU-only, no models).

Scores sentences with clinical cue words and section-header boosts, then
renders a short structured summary: chief complaint, key findings,
assessment/plan. Operates on de-identified text.
"""
from __future__ import annotations

import re
from typing import Dict, List

SECTION_HEADERS = [
    "Chief Complaint", "History of Present Illness", "HPI",
    "Past Medical History", "PMH", "Review of Systems",
    "Vitals", "Physical Exam", "Exam", "Labs", "Laboratory",
    "Imaging", "Assessment", "Plan", "Medications", "Allergies",
]

COMPLAINT_CUES = {"complain", "presents", "c/o", "chief complaint",
                  "admitted for", "here for", "reports", "seen for"}

FINDING_KEYWORDS = {
    "bp", "blood pressure", "hr", "heart rate", "temp", "o2", "bmi",
    "weight", "exam", "labs", "lab", "wbc", "hemoglobin", "a1c",
    "glucose", "cholesterol", "ldl", "x-ray", "ct", "mri", "ecg",
    "ekg", "diagnosed", "history of", "revealed", "showed",
    "elevated", "normal", "ulcer", "pulse", "rhythm", "murmur",
}

PLAN_KEYWORDS = {
    "plan", "prescrib", "start", "continue", "discontinue", "increase",
    "follow up", "follow-up", "return", "refer", "schedule", "advised",
    "counsel", "order", "recheck", "diary",
}

_HEADER_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:",
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _contains_kw(low: str, kw: str) -> bool:
    # Word-boundary match so "start" doesn't fire on "started".
    return re.search(r"\b" + re.escape(kw) + r"\b", low) is not None


def score_sentence(sent: str, idx: int) -> float:
    low = sent.lower()
    score = 1.0
    kw_hits = sum(1 for k in FINDING_KEYWORDS | PLAN_KEYWORDS
                  if _contains_kw(low, k))
    score += 2.0 * min(kw_hits, 4)
    if _HEADER_RE.match(sent):
        score += 6.0
    if idx == 0:
        score += 2.0
    elif idx < 3:
        score += 1.0
    words = len(sent.split())
    if words > 45:
        score -= 4.0
    if words < 4:
        score -= 2.0
    return score


def _is_complaint(sent: str) -> bool:
    low = sent.lower()
    return any(_contains_kw(low, c) for c in COMPLAINT_CUES)


def summarize(redacted_text: str, max_findings: int = 4,
              max_plan: int = 3) -> Dict[str, object]:
    """Return a structured summary dict for de-identified note text."""
    sents = split_sentences(redacted_text)
    if not sents:
        return {"chief_complaint": "", "key_findings": [],
                "plan": [], "summary_text": ""}

    scored = sorted(
        ((score_sentence(s, i), i, s) for i, s in enumerate(sents)),
        key=lambda t: (-t[0], t[1]),
    )

    # Chief complaint: earliest sentence carrying a presenting-problem cue.
    chief = next((s for s in sents if _is_complaint(s)), sents[0])

    plan_sents: List[str] = []
    for _, _, s in scored:
        low = s.lower()
        if s != chief and any(_contains_kw(low, k) for k in PLAN_KEYWORDS):
            plan_sents.append(s)
        if len(plan_sents) >= max_plan:
            break

    findings: List[str] = []
    for _, _, s in scored:
        if s != chief and s not in plan_sents:
            findings.append(s)
        if len(findings) >= max_findings:
            break

    lines = [f"CHIEF COMPLAINT: {chief}", "", "KEY FINDINGS:"]
    lines += [f"- {f}" for f in findings] or ["- (none extracted)"]
    lines += ["", "ASSESSMENT & PLAN:"]
    lines += [f"- {p}" for p in plan_sents] or ["- (none extracted)"]
    summary_text = "\n".join(lines)

    return {
        "chief_complaint": chief,
        "key_findings": findings,
        "plan": plan_sents,
        "summary_text": summary_text,
    }
