"""Tests for prompt loading strict mode and cache key behavior."""
from __future__ import annotations

import pytest

from src.services.ai import prompt as prompt_module


class _FakeLangfuseService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, dict | None]] = []

    def is_enabled(self) -> bool:
        return True

    def get_prompt_text(self, *, name: str, variables: dict | None, label: str | None, version: int | None, fallback: str | None) -> str:
        self.calls.append((name, label, variables))
        return f"{name}:{label}:{variables}"


@pytest.mark.asyncio
async def test_prompt_cache_key_depends_on_label_and_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLangfuseService()
    prompt_module._PROMPT_CACHE.clear()
    monkeypatch.setattr(prompt_module, "_get_langfuse_service", lambda: fake)

    first = await prompt_module.get_prompt(
        prompt_name="system",
        langfuse_label="production",
        variables={"a": 1},
    )
    second = await prompt_module.get_prompt(
        prompt_name="system",
        langfuse_label="production",
        variables={"a": 2},
    )
    third = await prompt_module.get_prompt(
        prompt_name="system",
        langfuse_label="staging",
        variables={"a": 1},
    )

    assert first != second
    assert first != third
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_required_prompt_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MissingService:
        def is_enabled(self) -> bool:
            return True

        def get_prompt_text(self, **kwargs):  # type: ignore[no-untyped-def]
            return None

    prompt_module._PROMPT_CACHE.clear()
    monkeypatch.setattr(prompt_module, "_get_langfuse_service", lambda: _MissingService())

    with pytest.raises(ValueError, match="Required prompt"):
        await prompt_module.get_prompt(
            prompt_name="missing",
            default_prompt=None,
            required=True,
        )


@pytest.mark.asyncio
async def test_required_false_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MissingService:
        def is_enabled(self) -> bool:
            return True

        def get_prompt_text(self, **kwargs):  # type: ignore[no-untyped-def]
            return None

    prompt_module._PROMPT_CACHE.clear()
    monkeypatch.setattr(prompt_module, "_get_langfuse_service", lambda: _MissingService())

    value = await prompt_module.get_prompt(
        prompt_name="missing",
        default_prompt=None,
        required=False,
    )
    assert value is None
