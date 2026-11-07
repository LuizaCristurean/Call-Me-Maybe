"""Unit tests for the token rules (no language model needed)."""

from __future__ import annotations

from src.constrained_decoder import (
    StringTokens,
    allowed_for_choices,
    allowed_for_number,
    can_extend_number,
    is_complete_number,
    scan_string_text,
)
from src.pipeline import clean_string


def test_choices_follow_prefixes() -> None:
    vocab = ["fn", "_add", "_a", "x", '"', ""]
    choices = ['fn_add"', 'fn_a"']
    assert allowed_for_choices(vocab, "", choices) == {0}
    assert allowed_for_choices(vocab, "fn", choices) == {1, 2}
    assert allowed_for_choices(vocab, "fn_add", choices) == {4}


def test_number_prefix_and_complete() -> None:
    assert can_extend_number("-3.")
    assert not can_extend_number(".5")
    assert is_complete_number(" -3.5")
    assert not is_complete_number("-3.")
    assert not is_complete_number("")


def test_integer_has_no_dot() -> None:
    assert is_complete_number("12", integer=True)
    assert not can_extend_number("1.", integer=True)


def test_number_tokens_and_end() -> None:
    vocab = ["-", "3", ".", "5", ",", "}", "a", ""]
    end = {4, 5}
    assert allowed_for_number(vocab, "", end) == {0, 1, 3}
    assert allowed_for_number(vocab, "-3", end) == {1, 2, 3, 4, 5}
    assert allowed_for_number(vocab, "-3.", end) == {1, 3}


def test_scan_string_text() -> None:
    assert scan_string_text("hello", False) == (5, False, False)
    assert scan_string_text('",', False) == (0, True, False)
    assert scan_string_text("a\\", False) == (2, False, True)
    assert scan_string_text("\\d", False) is None
    assert scan_string_text("a\nb", False) is None


def test_string_token_groups() -> None:
    vocab = ["hello", " world", '"', "a\nb", "\\", "n", '",', "", "\ufffd"]
    tokens = StringTokens.from_vocab(vocab)
    assert tokens.closers == {2: "", 6: ""}
    assert tokens.dangling == {4}
    assert tokens.open_ids == {0, 1, 2, 4, 5, 6}
    assert tokens.after_escape == {2, 4, 5, 6}


def test_clean_string() -> None:
    assert clean_string(" *", "replace with asterisks") == "*"
    assert clean_string(" world", "append ' world'") == " world"
    assert clean_string(" ", "use a space") == " "
    assert clean_string("hi", "say hi") == "hi"
