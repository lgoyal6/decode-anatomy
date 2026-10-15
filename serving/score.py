"""Strict answer scoring helpers for serving comparisons.

Used by serving/test_scoring.py and serving comparison analysis.
Incomplete output and malformed number grouping cannot count as correct answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL = re.compile(r"(?:^|\n)\s*the final answer is\s*\$?\s*(-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?)\s*[.!]?\s*$", re.I)


def normalize_number(value: str) -> str:
    """Normalize grouping commas and insignificant decimal zeros."""
    number = value.replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return "0" if number in {"-0", "-0.0"} else number


def extract_final_answer(text: str, *, finish_reason: str | None = None,
                         max_tokens: int | None = None) -> str | None:
    """Return a closing final answer only when the output was not truncated."""
    if finish_reason in {"length", "max_tokens", "abort", "cancelled", "error"}:
        return None
    if max_tokens is not None and max_tokens <= 0:
        return None
    match = _FINAL.search(text)
    return normalize_number(match.group(1)) if match else None


@dataclass(frozen=True)
class Score:
    answer: str | None
    correct: bool
    truncated: bool


def score_response(text: str, gold: str, *, finish_reason: str | None = None,
                   max_tokens: int | None = None) -> Score:
    answer = extract_final_answer(text, finish_reason=finish_reason,
                                  max_tokens=max_tokens)
    truncated = finish_reason in {"length", "max_tokens", "abort", "cancelled", "error"}
    return Score(answer, answer is not None and answer == normalize_number(gold), truncated)
