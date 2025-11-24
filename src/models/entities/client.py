"""Pydantic модель для клиента."""

from typing import Optional

from pydantic import BaseModel, Field


class Client(BaseModel):
    """Модель клиента из таблицы clients."""

    phone: str = Field(..., description="Номер телефона клиента")
    name: Optional[str] = Field(None, description="Имя клиента")
    city: Optional[str] = Field(None, description="Город клиента")
    business_area: Optional[str] = Field(None, description="Бизнес-область клиента")
    org_name: Optional[str] = Field(None, description="Название организации")
    is_it_friend: Optional[bool] = Field(None, description="Является ли клиент другом компании")
    mode: Optional[str] = Field(None, description="Режим работы с клиентом")
    UTC: Optional[int] = Field(None, description="Часовой пояс клиента (UTC offset)")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

