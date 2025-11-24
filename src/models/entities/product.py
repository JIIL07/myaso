"""Pydantic модель для товара."""

from typing import Optional

from pydantic import BaseModel, Field


class Product(BaseModel):
    """Модель товара из таблицы myaso.products."""

    id: int = Field(..., description="Уникальный идентификатор товара")
    title: str = Field(..., description="Название товара")
    supplier_name: Optional[str] = Field(None, description="Название поставщика")
    from_region: Optional[str] = Field(None, description="Регион происхождения")
    photo: Optional[str] = Field(None, description="URL фотографии товара")
    order_price_kg: Optional[float] = Field(None, description="Цена за килограмм для заказа")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

