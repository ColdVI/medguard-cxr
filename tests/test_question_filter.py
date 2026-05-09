"""Question-filter branch tests."""

from medguard.safety.question_filter import classify_question, extract_supported_finding


def test_classify_supported_finding_question() -> None:
    assert classify_question("Is there evidence of Pneumothorax?") == "supported_finding_query"
    assert extract_supported_finding("Is there evidence of Pleural Thickening?") == (
        "Pleural_Thickening"
    )


def test_classify_diagnosis_request_question() -> None:
    assert classify_question("What treatment should I prescribe?") == "diagnosis_request"


def test_classify_unsupported_concept_question() -> None:
    assert classify_question("Does this show kidney stones?") == "unsupported_concept"


def test_classify_unparseable_question_defaults_to_abstain() -> None:
    assert classify_question("Please describe the scan.") == "unparseable"


def test_classify_is_case_insensitive() -> None:
    assert classify_question("is there evidence of pneumothorax?") == "supported_finding_query"
