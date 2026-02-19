"""Tests for src.utils.prices.price_calculator."""
from __future__ import annotations

import pytest

from src.utils.prices.price_calculator import (
    calculate_final_price,
    get_delivery_markup,
    get_markup_from_system_vars,
    parse_markup_value,
)


# ---------------------------------------------------------------------------
# parse_markup_value
# ---------------------------------------------------------------------------
class TestParseMarkupValue:
    def test_percentage_value(self) -> None:
        pct, absolute = parse_markup_value("20%")
        assert pct == 20.0
        assert absolute is None

    def test_absolute_value(self) -> None:
        pct, absolute = parse_markup_value("50")
        assert pct is None
        assert absolute == 50.0

    def test_percentage_with_spaces(self) -> None:
        pct, _ = parse_markup_value("  15 % ")
        assert pct == 15.0

    def test_none_input(self) -> None:
        assert parse_markup_value(None) == (None, None)

    def test_empty_string(self) -> None:
        assert parse_markup_value("") == (None, None)

    def test_no_digits(self) -> None:
        assert parse_markup_value("abc") == (None, None)

    def test_float_percentage(self) -> None:
        pct, _ = parse_markup_value("12.5%")
        assert pct == 12.5


# ---------------------------------------------------------------------------
# get_markup_from_system_vars
# ---------------------------------------------------------------------------
class TestGetMarkupFromSystemVars:
    def test_low_price_markup(self, system_vars_with_markup: dict[str, str]) -> None:
        pct, absolute = get_markup_from_system_vars(50.0, system_vars_with_markup)
        assert pct == 20.0
        assert absolute is None

    def test_high_price_markup(self, system_vars_with_markup: dict[str, str]) -> None:
        pct, absolute = get_markup_from_system_vars(200.0, system_vars_with_markup)
        assert pct == 15.0
        assert absolute is None

    def test_boundary_at_100(self, system_vars_with_markup: dict[str, str]) -> None:
        # price == 100 should hit ">100" branch
        pct, _ = get_markup_from_system_vars(100.0, system_vars_with_markup)
        assert pct == 15.0

    def test_missing_markup_returns_none(self, system_vars_empty: dict[str, str]) -> None:
        assert get_markup_from_system_vars(100.0, system_vars_empty) == (None, None)


# ---------------------------------------------------------------------------
# get_delivery_markup
# ---------------------------------------------------------------------------
class TestGetDeliveryMarkup:
    def test_delivery_found(self, system_vars_with_markup: dict[str, str]) -> None:
        pct, _ = get_delivery_markup(system_vars_with_markup)
        assert pct == 5.0

    def test_delivery_not_found(self, system_vars_empty: dict[str, str]) -> None:
        assert get_delivery_markup(system_vars_empty) == (None, None)


# ---------------------------------------------------------------------------
# calculate_final_price
# ---------------------------------------------------------------------------
class TestCalculateFinalPrice:
    def test_none_price(self, system_vars_with_markup: dict[str, str]) -> None:
        assert calculate_final_price(None, system_vars_with_markup) == "Цена по запросу"

    def test_zero_price(self, system_vars_with_markup: dict[str, str]) -> None:
        assert calculate_final_price(0, system_vars_with_markup) == "Цена по запросу"

    def test_empty_string_price(self, system_vars_with_markup: dict[str, str]) -> None:
        assert calculate_final_price("", system_vars_with_markup) == "Цена по запросу"

    def test_not_specified_string(self, system_vars_with_markup: dict[str, str]) -> None:
        assert calculate_final_price("Не указано", system_vars_with_markup) == "Цена по запросу"

    def test_non_numeric_string(self, system_vars_with_markup: dict[str, str]) -> None:
        assert calculate_final_price("abc", system_vars_with_markup) == "Цена по запросу"

    def test_kit_supplier_no_markup(self, system_vars_with_markup: dict[str, str]) -> None:
        result = calculate_final_price(200.0, system_vars_with_markup, supplier_name="ООО КИТ")
        assert result == "200.00"

    def test_kit_supplier_case_insensitive(self, system_vars_with_markup: dict[str, str]) -> None:
        result = calculate_final_price(100.0, system_vars_with_markup, supplier_name="ООО кит")
        assert result == "100.00"

    def test_percentage_markup_applied(self) -> None:
        """15% markup on 200 = 230, then 5% delivery = 241.50."""
        vars_ = {
            "Наценка на кг/руб (>100 руб)": "15%",
            "Наценка доставки": "5%",
        }
        result = calculate_final_price(200.0, vars_)
        assert result == "241.50"

    def test_absolute_markup_applied(self, system_vars_absolute_markup: dict[str, str]) -> None:
        """Price 50 + absolute markup 30 = 80."""
        result = calculate_final_price(50.0, system_vars_absolute_markup)
        assert result == "80.00"

    def test_string_price_parsed(self, system_vars_with_markup: dict[str, str]) -> None:
        result = calculate_final_price("200", system_vars_with_markup, supplier_name="ООО КИТ")
        assert result == "200.00"

    def test_no_markup_returns_base_price(self, system_vars_empty: dict[str, str]) -> None:
        """When no markup configured, the base price should still be returned."""
        result = calculate_final_price(100.0, system_vars_empty)
        assert result == "100.00"
