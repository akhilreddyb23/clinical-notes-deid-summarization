"""Detector unit tests — one per PHI category plus edge cases."""
import pytest

from clinic_deid import detect


def labels(text):
    return [(s.label, s.text) for s in detect(text)]


def test_ssn():
    assert labels("SSN: 123-45-6789")[0] == ("SSN", "123-45-6789")


def test_phone_parens():
    assert ("PHONE", "(555) 234-5678") in labels("Call (555) 234-5678 today")


def test_phone_dashes():
    assert ("PHONE", "555-876-1234") in labels("phone: 555-876-1234")


def test_email():
    found = labels("email at robert.johnson@example.com.")
    assert ("EMAIL", "robert.johnson@example.com") in found


def test_mrn_numeric():
    assert ("MRN", "MRN: 482913") in labels("MRN: 482913")


def test_mrn_alphanumeric():
    assert ("MRN", "MRN: DK-883412") in labels("MRN: DK-883412")


def test_account_number():
    assert ("ACCOUNT", "Account No: 88421033") in labels("Account No: 88421033")


def test_age_over_89_hyphenated():
    assert ("AGE", "94-year-old") in labels("a 94-year-old female")


def test_age_ninety_colon():
    assert ("AGE", "Age: 90") in labels("Age: 90")


def test_age_yo_abbrev():
    assert ("AGE", "91 y/o") in labels("91 y/o male")


def test_age_88_not_phi():
    assert labels("His father, age 88, is healthy") == []


def test_age_67_not_phi():
    assert labels("a 67-year-old male") == []


def test_date_slash():
    assert ("DATE", "03/15/1957") in labels("DOB 03/15/1957")


def test_date_month_name():
    assert ("DATE", "March 5, 1932") in labels("DOB: March 5, 1932")


def test_date_iso():
    assert ("DATE", "1980-11-02") in labels("seen on 1980-11-02")


def test_bare_year_not_phi():
    assert labels("diagnosed with diabetes in 2019") == []


def test_name_with_title():
    assert ("NAME", "Dr. Emily Chen") in labels("Dr. Emily Chen evaluated")


def test_name_patient_prefix():
    assert ("NAME", "Robert Johnson") in labels("Patient: Robert Johnson is a male")


def test_name_signed_by():
    found = labels("Electronically signed by Lisa Wang, MD")
    assert ("NAME", "Lisa Wang") in found


def test_street_address():
    assert ("ADDRESS", "1234 Maple Street") in labels("lives at 1234 Maple Street,")


def test_zip():
    assert ("ADDRESS", "62701") in labels("Springfield, IL 62701")


def test_city_state():
    assert ("ADDRESS", "Springfield, IL") in labels("Springfield, IL 62701")


def test_fax_treated_as_phone():
    assert ("PHONE", "555-111-2222") in labels("Fax: 555-111-2222")


def test_no_overlap_double_count():
    # "MRN: 482913" must be one MRN span, not MRN + ZIP fragments.
    spans = detect("MRN: 482913")
    assert len(spans) == 1
    assert spans[0].label == "MRN"


def test_spans_sorted_by_start():
    spans = detect("SSN: 123-45-6789 and phone (555) 234-5678")
    assert [s.start for s in spans] == sorted(s.start for s in spans)


def test_vitals_not_phi():
    assert labels("Vitals: BP 142/88, HR 78, Temp 98.6F") == []


@pytest.mark.parametrize("text", ["", "No phi here, just a normal sentence."])
def test_no_false_positives_on_plain_text(text):
    assert detect(text) == []
