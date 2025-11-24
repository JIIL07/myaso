"""Pydantic модель для истории цен."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PriceHistory(BaseModel):
    """Модель истории цен из таблицы myaso.price_history."""

    product: Optional[str] = Field(None, description="Название товара")
    date: Optional[datetime] = Field(None, description="Дата изменения цены")
    price: Optional[float] = Field(None, description="Цена товара")
    suplier_name: Optional[str] = Field(None, description="Название поставщика (опечатка в БД сохранена)")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

