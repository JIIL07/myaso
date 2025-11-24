"""Pydantic модель для заказа."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Order(BaseModel):
    """Модель заказа из таблицы orders."""

    title: Optional[str] = Field(None, description="Название товара в заказе")
    created_at: Optional[datetime] = Field(None, description="Дата и время создания заказа")
    weight_kg: Optional[float] = Field(None, description="Вес заказа в килограммах")
    price_out: Optional[float] = Field(None, description="Итоговая цена заказа")
    destination: Optional[str] = Field(None, description="Пункт назначения заказа")
    client_phone: str = Field(..., description="Номер телефона клиента")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

