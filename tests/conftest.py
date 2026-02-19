"""Shared test fixtures for the myaso project."""
from __future__ import annotations

import pytest


@pytest.fixture()
def system_vars_with_markup() -> dict[str, str]:
    """System variables containing standard markup rules."""
    return {
        "Наценка на кг/руб (<100 руб)": "20%",
        "Наценка на кг/руб (>100 руб)": "15%",
        "Наценка доставки": "5%",
    }


@pytest.fixture()
def system_vars_absolute_markup() -> dict[str, str]:
    """System variables with absolute (fixed) markup instead of percentage."""
    return {
        "Наценка на кг/руб (<100 руб)": "30",
        "Наценка на кг/руб (>100 руб)": "50",
    }


@pytest.fixture()
def system_vars_empty() -> dict[str, str]:
    """Empty system variables (no markup configured)."""
    return {}
