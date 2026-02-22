"""Architecture guardrails for toolkit-first validation/formatting."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _python_files() -> list[Path]:
    return [path for path in SRC.rglob("*.py") if path.is_file()]


def test_no_legacy_validator_formatter_imports() -> None:
    forbidden = (
        "src.utils.validators",
        "src.utils.formatters.formatters",
    )

    offenders: list[str] = []
    for path in _python_files():
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, f"Found legacy imports: {offenders}"


def test_execute_sql_has_no_schema_rewrite_rules() -> None:
    execute_sql_path = SRC / "tools" / "execute_sql.py"
    content = execute_sql_path.read_text(encoding="utf-8")

    # Keep SQL execution explicit: no hidden auto-prefixing like myaso.<table>.
    assert "myaso.%s" not in content
    assert "replacement_from" not in content
    assert "replacement_join" not in content
