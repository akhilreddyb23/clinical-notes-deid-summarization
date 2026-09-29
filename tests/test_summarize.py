"""Summarizer tests."""
from pathlib import Path

from clinic_deid import load_notes, redact, summarize, split_sentences

REPO = Path(__file__).resolve().parent.parent


def note01_redacted():
    notes = load_notes(REPO / "data" / "notes.json")
    return redact(next(n for n in notes if n.id == "note_01").text)


def test_summary_has_all_sections():
    s = summarize(note01_redacted())
    assert s["chief_complaint"]
    assert len(s["key_findings"]) > 0
    assert len(s["plan"]) > 0
    assert "CHIEF COMPLAINT" in s["summary_text"]
    assert "KEY FINDINGS" in s["summary_text"]
    assert "ASSESSMENT & PLAN" in s["summary_text"]


def test_chief_complaint_captures_presenting_problem():
    s = summarize(note01_redacted())
    assert "chest pain" in s["chief_complaint"].lower()


def test_summary_captures_key_meds():
    s = summarize(note01_redacted())
    assert "aspirin" in s["summary_text"].lower()
    assert "atorvastatin" in s["summary_text"].lower()


def test_summary_contains_no_raw_phi():
    s = summarize(note01_redacted())
    assert "Robert Johnson" not in s["summary_text"]
    assert "123-45-6789" not in s["summary_text"]


def test_empty_input_does_not_crash():
    s = summarize("")
    assert s["summary_text"] == ""
    assert s["key_findings"] == []


def test_sentence_split_basic():
    sents = split_sentences("First sentence. Second sentence! Third?")
    assert len(sents) == 3


def test_findings_capped():
    s = summarize(note01_redacted(), max_findings=2, max_plan=1)
    assert len(s["key_findings"]) <= 2
    assert len(s["plan"]) <= 1
