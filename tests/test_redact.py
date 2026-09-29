"""Redaction behavior tests."""
from clinic_deid import detect, redact, redact_with_spans


def test_tags_replace_phi():
    out = redact("Patient: Robert Johnson, SSN: 123-45-6789")
    assert "[NAME]" in out
    assert "[SSN]" in out
    assert "Robert Johnson" not in out
    assert "123-45-6789" not in out


def test_date_keeps_year():
    out = redact("DOB 03/15/1957")
    assert out == "DOB [DATE] 1957"


def test_date_month_name_keeps_year():
    out = redact("DOB: March 5, 1932")
    assert "[DATE] 1932" in out
    assert "March" not in out


def test_bare_year_survives():
    out = redact("diagnosed with diabetes in 2019")
    assert "2019" in out
    assert "[DATE]" not in out


def test_age_88_survives():
    out = redact("His father, age 88, is healthy")
    assert "age 88" in out
    assert "[AGE]" not in out


def test_age_94_redacted():
    out = redact("a 94-year-old female")
    assert "[AGE]" in out
    assert "94-year-old" not in out


def test_redact_with_spans_returns_both():
    text = "Call (555) 234-5678"
    redacted, spans = redact_with_spans(text)
    assert redacted == "Call [PHONE]"
    assert len(spans) == 1 and spans[0].label == "PHONE"


def test_no_phi_text_unchanged():
    text = "Vitals: BP 142/88, HR 78."
    assert redact(text) == text
    assert detect(text) == []


def test_full_address_redacted():
    out = redact("lives at 1234 Maple Street, Springfield, IL 62701")
    assert "1234" not in out and "Maple" not in out
    assert "Springfield" not in out and "62701" not in out
    assert out.count("[ADDRESS]") == 3
