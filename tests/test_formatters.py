"""Tests for src.utils.formatters.formatters."""
from __future__ import annotations

from typing import Any

import pytest

from src.utils.formatters.formatters import (
    filter_products_by_photo,
    has_photo,
    normalize_field_value_sync,
    remove_markdown_symbols,
)


# ---------------------------------------------------------------------------
# remove_markdown_symbols
# ---------------------------------------------------------------------------
class TestRemoveMarkdownSymbols:
    def test_bold(self) -> None:
        assert remove_markdown_symbols("**bold**") == "bold"

    def test_italic_star(self) -> None:
        assert remove_markdown_symbols("*italic*") == "italic"

    def test_italic_underscore(self) -> None:
        assert remove_markdown_symbols("_italic_") == "italic"

    def test_code(self) -> None:
        assert remove_markdown_symbols("`code`") == "code"

    def test_heading(self) -> None:
        assert remove_markdown_symbols("## Heading") == "Heading"

    def test_link(self) -> None:
        assert remove_markdown_symbols("[text](http://example.com)") == "text"

    def test_plain_text_unchanged(self) -> None:
        assert remove_markdown_symbols("hello world") == "hello world"


# ---------------------------------------------------------------------------
# normalize_field_value_sync
# ---------------------------------------------------------------------------
class TestNormalizeFieldValueSync:
    def test_none_returns_default(self) -> None:
        result = normalize_field_value_sync(None)
        assert result == "по запросу"

    def test_empty_string_returns_default(self) -> None:
        assert normalize_field_value_sync("") == "по запросу"

    def test_null_string_returns_default(self) -> None:
        assert normalize_field_value_sync("null") == "по запросу"

    def test_none_string_returns_default(self) -> None:
        assert normalize_field_value_sync("none") == "по запросу"

    def test_zero_int_returns_default(self) -> None:
        assert normalize_field_value_sync(0) == "по запросу"

    def test_valid_text(self) -> None:
        assert normalize_field_value_sync("Москва") == "Москва"

    def test_numeric_field_type_valid(self) -> None:
        assert normalize_field_value_sync("100", "numeric") == "100"

    def test_numeric_field_type_zero_string(self) -> None:
        assert normalize_field_value_sync("0", "numeric") == "по запросу"

    def test_numeric_field_type_float(self) -> None:
        assert normalize_field_value_sync(99.5, "numeric") == "99.5"

    def test_numeric_field_type_float_integer(self) -> None:
        # 100.0 should be shown as "100"
        assert normalize_field_value_sync(100.0, "numeric") == "100"


# ---------------------------------------------------------------------------
# has_photo / filter_products_by_photo
# ---------------------------------------------------------------------------
class TestHasPhoto:
    def test_with_url(self) -> None:
        assert has_photo({"photo": "http://example.com/img.png"}) is True

    def test_empty_string(self) -> None:
        assert has_photo({"photo": ""}) is False

    def test_none(self) -> None:
        assert has_photo({"photo": None}) is False

    def test_whitespace_only(self) -> None:
        assert has_photo({"photo": "   "}) is False

    def test_missing_key(self) -> None:
        assert has_photo({}) is False


class TestFilterProductsByPhoto:
    def test_filters_correctly(self) -> None:
        products: list[dict[str, Any]] = [
            {"id": 1, "photo": "http://img.png"},
            {"id": 2, "photo": ""},
            {"id": 3, "photo": "http://img2.png"},
            {"id": 4, "photo": None},
        ]
        result = filter_products_by_photo(products)
        assert len(result) == 2
        assert [p["id"] for p in result] == [1, 3]

    def test_empty_list(self) -> None:
        assert filter_products_by_photo([]) == []

    def test_no_photos(self) -> None:
        products: list[dict[str, Any]] = [{"id": 1, "photo": None}]
        assert filter_products_by_photo(products) == []
