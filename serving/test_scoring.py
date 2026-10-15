"""Check complete-answer scoring for serving/score.py.

These cases keep truncation and malformed numeric text out of correct counts.
Legitimate grouping commas and terminal punctuation remain supported.
"""
import pytest

from score import extract_final_answer, score_response


def test_only_explicit_terminal_answer_is_scored():
    assert extract_final_answer("Work: 3 + 4 = 7\nThe final answer is 7") == "7"
    assert extract_final_answer("The final answer is 7\nMore work follows") is None
    assert extract_final_answer("The answer is 7") is None


def test_truncation_cannot_score_intermediate_or_terminal_number():
    assert extract_final_answer("Work 7", finish_reason="length") is None
    assert not score_response("The final answer is 7", "7", finish_reason="length").correct


def test_number_normalization_is_narrow_and_deterministic():
    assert score_response("The final answer is $1,200.00", "1200").correct
    assert not score_response("The final answer is 12", "21").correct


@pytest.mark.parametrize("answer", ["1,2", "1,,2", "1,23,4", "12,"])
def test_malformed_numeric_grouping_cannot_score_correct(answer):
    result = score_response(f"The final answer is {answer}", answer.replace(",", ""))
    assert result.answer is None
    assert not result.correct
