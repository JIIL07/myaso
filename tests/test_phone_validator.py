"""Tests for phone toolkit contract."""
from __future__ import annotations

import pytest

from src.toolkit import (
    has_client_phone,
    normalize_and_validate_phone,
    normalize_phone,
    validate_phone,
)


# ---------------------------------------------------------------------------
# normalize_phone
# ---------------------------------------------------------------------------
class TestNormalizePhone:
    def test_empty(self) -> None:
        assert normalize_phone("") == ""

    def test_starts_with_8(self) -> None:
        assert normalize_phone("89123456789") == "+79123456789"

    def test_starts_with_7_no_plus(self) -> None:
        assert normalize_phone("79123456789") == "+79123456789"

    def test_already_plus(self) -> None:
        assert normalize_phone("+79123456789") == "+79123456789"

    def test_ten_digit_starts_with_9(self) -> None:
        assert normalize_phone("9123456789") == "+79123456789"

    def test_strips_formatting(self) -> None:
        assert normalize_phone("+7 (912) 345-67-89") == "+79123456789"


# ---------------------------------------------------------------------------
# validate_phone
# ---------------------------------------------------------------------------
class TestValidatePhone:
    def test_valid_russian(self) -> None:
        assert validate_phone("+79123456789") is True

    def test_valid_from_8(self) -> None:
        assert validate_phone("89123456789") is True

    def test_empty(self) -> None:
        assert validate_phone("") is False

    def test_too_short(self) -> None:
        assert validate_phone("+123") is False

    def test_too_long(self) -> None:
        assert validate_phone("+1" + "0" * 16) is False


# ---------------------------------------------------------------------------
# normalize_and_validate_phone
# ---------------------------------------------------------------------------
class TestNormalizeAndValidatePhone:
    def test_valid(self) -> None:
        normalized = normalize_and_validate_phone("+79123456789")
        assert normalized == "+79123456789"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_and_validate_phone("123")


# ---------------------------------------------------------------------------
# has_client_phone
# ---------------------------------------------------------------------------
class TestHasClientPhone:
    def test_valid(self) -> None:
        assert has_client_phone("+79123456789") is True

    def test_none(self) -> None:
        assert has_client_phone(None) is False

    def test_empty(self) -> None:
        assert has_client_phone("") is False

    def test_whitespace_only(self) -> None:
        assert has_client_phone("   ") is False
