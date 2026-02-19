from typing import Any, TypeVar

T = TypeVar("T", bound="AsyncMixin")


class AsyncMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __initobj(self: T) -> T:
        assert not self.async_initialized
        self.async_initialized = True
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self) -> Any:
        return self.__initobj().__await__()
