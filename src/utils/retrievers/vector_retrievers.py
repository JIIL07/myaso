from __future__ import annotations

from typing import Any, List, Sequence, Dict
import os
import re
import logging
import asyncio

import asyncpg
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config.settings import settings
from src.database import get_pool
from src.config.constants import (
    DEFAULT_VECTOR_SEARCH_K,
    EMBEDDING_DELAY_SECONDS,
    EMBEDDING_BATCH_SIZE,
)
from openai import OpenAI

logger = logging.getLogger(__name__)


class SupabaseVectorRetriever(BaseRetriever):
    """Ретривер для семантического поиска по товарам (pgvector).

    Наследуется от BaseRetriever для лучшей интеграции с LangChain экосистемой.
    """

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        db_dsn: str | None = None,
        k: int = DEFAULT_VECTOR_SEARCH_K,
    ) -> None:
        """Инициализация ретривера.

        Args:
            embedding_model: модель эмбеддингов
            db_dsn: DSN Postgres
            k: Количество документов для возврата (по умолчанию 10)
        """
        super().__init__()
        self._embedder = OpenAI(
            api_key=settings.alibaba.alibaba_key,
            base_url=settings.alibaba.base_alibaba_url,
        )
        self._embedding_model = (
            embedding_model
            or settings.alibaba.embedding_model_id
            or "text-embedding-v4"
        )
        self._k = k

        self._db_dsn = db_dsn or os.getenv("POSTGRES_DSN")

    async def _embed(self, text: str) -> List[float]:
        """Создаёт эмбеддинг текста используя Alibaba DashScope API.

        Отправляет текст в модель embeddings и возвращает векторное представление
        в виде списка чисел с плавающей точкой.

        Args:
            text: Текст для создания embedding

        Returns:
            Список float чисел, представляющий векторное представление текста.
            Размерность вектора зависит от модели (для text-embedding-v4 это обычно 1536).

        Raises:
            Exception: Если произошла ошибка при обращении к API embeddings
        """
        completion = self._embedder.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        data = completion.model_dump()
        return data["data"][0]["embedding"]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """Асинхронная версия get_relevant_documents (требуется BaseRetriever).

        Args:
            query: Текстовый запрос для поиска
            run_manager: Менеджер выполнения (опционально)

        Returns:
            Список Document объектов с найденными товарами
        """
        return await self._get_relevant_documents(query, k=self._k)

    async def get_relevant_documents(
        self, query: str, k: int | None = None
    ) -> List[Document]:
        """Возвращает top-k документов по близости (LangChain Document).

        Args:
            query: Текстовый запрос для поиска
            k: Количество документов для возврата (если None, используется значение из __init__)

        Returns:
            Список Document объектов с найденными товарами
        """
        if k is None:
            k = self._k
        return await self._get_relevant_documents(query, k=k)

    async def _get_relevant_documents(self, query: str, k: int) -> List[Document]:
        """Внутренняя реализация получения документов."""
        vector = await self._embed(query)

        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                vector_str = "[" + ",".join(map(str, vector)) + "]"

                rows: Sequence[asyncpg.Record] = await conn.fetch(
                    """
                    SELECT
                      id,
                      title,
                      supplier_name,
                      from_region,
                      photo,
                      pricelist_date,
                      package_weight,
                      order_price_kg,
                      min_order_weight_kg,
                      discount,
                      ready_made,
                      package_type,
                      cooled_or_frozen,
                      product_in_package,
                      embedding <-> ($1::vector) AS distance
                    FROM myaso.products
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> ($1::vector)
                    LIMIT $2
                    """,
                    vector_str,
                    k,
                )
        except Exception as e:
            error_type = type(e).__name__
            error_str = str(e)

            logger.error(
                f"Database connection error: {error_type}: {error_str}", exc_info=True
            )

            raise RuntimeError("Ошибка подключения к базе данных") from e

        documents: List[Document] = []
        for i, row in enumerate(rows):
            row_dict: dict[str, Any] = dict(row)
            content_parts = [
                f"Title: {row_dict.get('title', '')}",
                f"Supplier: {row_dict.get('supplier_name', '')}",
                f"Region: {row_dict.get('from_region', '')}",
                f"Price/kg: {row_dict.get('order_price_kg', '')}",
                f"Min order (kg): {row_dict.get('min_order_weight_kg', '')}",
                f"Cooled/Frozen: {row_dict.get('cooled_or_frozen', '')}",
                f"Ready-made: {row_dict.get('ready_made', '')}",
            ]
            page_content = "; ".join([p for p in content_parts if p])

            metadata = {**row_dict}
            metadata.pop("embedding", None)

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _build_product_text(self, product: Dict[str, Any]) -> str:
        """Создает текстовое описание товара для embedding.

        Args:
            product: Словарь с данными товара из БД

        Returns:
            Текст для создания embedding
        """
        parts = []

        if product.get("title"):
            parts.append(product["title"])

        if product.get("supplier_name"):
            parts.append(f"поставщик: {product['supplier_name']}")

        if product.get("from_region"):
            parts.append(f"регион: {product['from_region']}")

        if product.get("cooled_or_frozen"):
            parts.append(product["cooled_or_frozen"])

        if product.get("ready_made"):
            parts.append("полуфабрикат")

        if product.get("package_type"):
            parts.append(f"упаковка: {product['package_type']}")

        return " ".join(parts)

    async def _embed_products(
        self, delay: float = EMBEDDING_DELAY_SECONDS
    ) -> Dict[str, int]:
        """Создает embedding для всех товаров без embedding в базе данных.

        Args:
            delay: Задержка между запросами к API embedding в секундах (по умолчанию 0.1)

        Returns:
            Словарь с результатами: {"processed": int, "errors": int, "total": int}
        """
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                products = await conn.fetch(
                    """
                    SELECT
                        id,
                        title,
                        supplier_name,
                        from_region,
                        cooled_or_frozen,
                        ready_made,
                        package_type
                    FROM myaso.products
                    WHERE embedding IS NULL
                    ORDER BY id
                    """
                )

                total = len(products)
                if total == 0:
                    logger.info("Все товары уже имеют embedding")
                    return {"processed": 0, "errors": 0, "total": 0}

                logger.info(f"Найдено {total} товаров без embedding. Начинаем обработку...")

                processed = 0
                errors = 0

                for product in products:
                    try:
                        product_dict = dict(product)
                        product_text = self._build_product_text(product_dict)

                        if not product_text.strip():
                            logger.warning(
                                f"Пропущен товар ID={product_dict['id']}: нет текста для embedding"
                            )
                            continue

                        embedding = await self._embed(product_text)

                        vector_str = "[" + ",".join(map(str, embedding)) + "]"

                        await conn.execute(
                            """
                            UPDATE myaso.products
                            SET embedding = $1::vector
                            WHERE id = $2
                            """,
                            vector_str,
                            product_dict["id"],
                        )

                        processed += 1

                        if processed % EMBEDDING_BATCH_SIZE == 0:
                            logger.info(f"Обработано: {processed}/{total}")

                        if delay > 0:
                            await asyncio.sleep(delay)

                    except Exception as e:
                        errors += 1
                        logger.error(
                            f"Ошибка при обработке товара ID={product_dict.get('id', 'N/A')}: {e}"
                        )
                        continue

                logger.info(
                    f"Готово! Обработано: {processed}, Ошибок: {errors}, Всего: {total}"
                )
                return {"processed": processed, "errors": errors, "total": total}

        except Exception as e:
            logger.error(
                f"Критическая ошибка при создании embeddings: {e}", exc_info=True
            )
            raise
