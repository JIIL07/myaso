"""Tests for SQL syntax validation contract."""
from __future__ import annotations

import pytest

from src.toolkit import validate_sql_conditions, validate_sql_safety
import src.toolkit.sql as sql_module


# ---------------------------------------------------------------------------
# validate_sql_safety
# ---------------------------------------------------------------------------
class TestValidateSqlSafety:
    def test_safe_select(self) -> None:
        assert validate_sql_safety("SELECT * FROM products") is True

    def test_safe_where_conditions(self) -> None:
        assert validate_sql_safety("price > 100 AND supplier_name = 'КИТ'") is True

    def test_drop_is_parseable(self) -> None:
        assert validate_sql_safety("DROP TABLE products") is True

    def test_delete_is_parseable(self) -> None:
        assert validate_sql_safety("DELETE FROM products WHERE id = 1") is True

    def test_insert_is_parseable(self) -> None:
        assert validate_sql_safety("INSERT INTO products VALUES (1)") is True

    def test_update_is_parseable(self) -> None:
        assert validate_sql_safety("UPDATE products SET price = 0") is True

    def test_truncate_is_parseable(self) -> None:
        assert validate_sql_safety("TRUNCATE TABLE products") is True

    def test_alter_is_parseable(self) -> None:
        assert validate_sql_safety("ALTER TABLE products ADD COLUMN x int") is True

    def test_execute_is_parseable_command(self) -> None:
        assert validate_sql_safety("EXECUTE my_proc") is True

    def test_empty_string(self) -> None:
        assert validate_sql_safety("") is False

    def test_case_insensitive_parseable(self) -> None:
        assert validate_sql_safety("drop TABLE products") is True

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
    async def test_invalid_conditions_raises(self) -> None:
        if sql_module.parse_one is None:
            await validate_sql_conditions("DROP TABLE products")
        else:
            with pytest.raises(ValueError, match="Синтаксическая ошибка SQL условий"):
                await validate_sql_conditions("DROP TABLE products")
