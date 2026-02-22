"""Tests for SQL safety toolkit contract."""
from __future__ import annotations

import pytest

from src.toolkit import validate_sql_conditions, validate_sql_safety


# ---------------------------------------------------------------------------
# validate_sql_safety
# ---------------------------------------------------------------------------
class TestValidateSqlSafety:
    def test_safe_select(self) -> None:
        assert validate_sql_safety("SELECT * FROM products") is True

    def test_safe_where_conditions(self) -> None:
        assert validate_sql_safety("price > 100 AND supplier_name = 'КИТ'") is True

    def test_drop_blocked(self) -> None:
        assert validate_sql_safety("DROP TABLE products") is False

    def test_delete_blocked(self) -> None:
        assert validate_sql_safety("DELETE FROM products WHERE id = 1") is False

    def test_insert_blocked(self) -> None:
        assert validate_sql_safety("INSERT INTO products VALUES (1)") is False

    def test_update_blocked(self) -> None:
        assert validate_sql_safety("UPDATE products SET price = 0") is False

    def test_truncate_blocked(self) -> None:
        assert validate_sql_safety("TRUNCATE TABLE products") is False

    def test_alter_blocked(self) -> None:
        assert validate_sql_safety("ALTER TABLE products ADD COLUMN x int") is False

    def test_execute_blocked(self) -> None:
        assert validate_sql_safety("EXECUTE my_proc") is False

    def test_empty_string(self) -> None:
        assert validate_sql_safety("") is False

    def test_case_insensitive(self) -> None:
        assert validate_sql_safety("drop TABLE products") is False

    def test_keyword_in_value_is_safe(self) -> None:
        # SQL keywords inside string literals are safe
        assert validate_sql_safety("supplier_name = 'DELETE'") is True

    def test_keyword_as_substring_is_safe(self) -> None:
        # "updated_at" contains "update" but as a substring, not a word boundary
        assert validate_sql_safety("updated_at > '2024-01-01'") is True


# ---------------------------------------------------------------------------
# validate_sql_conditions
# ---------------------------------------------------------------------------
class TestValidateSqlConditions:
    @pytest.mark.asyncio
    async def test_valid_conditions(self) -> None:
        # Should not raise
        await validate_sql_conditions("price > 100")

    @pytest.mark.asyncio
    async def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="пустыми"):
            await validate_sql_conditions("")

    @pytest.mark.asyncio
    async def test_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="пустыми"):
            await validate_sql_conditions("   ")

    @pytest.mark.asyncio
    async def test_dangerous_keyword_raises(self) -> None:
        with pytest.raises(ValueError, match="опасная SQL команда"):
            await validate_sql_conditions("DROP TABLE products")
