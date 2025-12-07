from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

import asyncpg
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from openai import OpenAI

from src.utils.rules import (
    get_rule_as_float,
    get_rule_as_int,
)
from src.config.settings import settings
from src.database import get_pool

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
        k: int | None = None,
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
            k: Количество документов для возврата (если None, используется значение из __init__).
               Если k >= 100000, возвращаются все товары без ограничения.

        Returns:
            Список Document объектов с найденными товарами
        """
        if k is None:
            if self._k is None:
                try:
                    k = await get_rule_as_int("DEFAULT_VECTOR_SEARCH_K")
                except Exception as e:
                    logger.warning(f"[vector_retrievers] Не удалось загрузить DEFAULT_VECTOR_SEARCH_K из БД, используем 10: {e}")
                    k = 10
            else:
                k = self._k
        return await self._get_relevant_documents(query, k=k)

    async def _get_relevant_documents(self, query: str, k: int) -> List[Document]:
        """Внутренняя реализация получения документов.
        
        Args:
            query: Текстовый запрос для поиска
            k: Количество документов для возврата. Если k >= 100000, возвращаются все товары.
        """
        vector = await self._embed(query)

        use_limit = k < 100000

        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                vector_str = "[" + ",".join(map(str, vector)) + "]"

                if use_limit:
                    rows: Sequence[asyncpg.Record] = await conn.fetch(
                        """
                        SELECT
                          id,
                          title,
                          supplier_name,
                          from_region,
                          photo,
                          order_price_kg,
                          embedding <-> ($1::vector) AS distance
                        FROM myaso.products
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <-> ($1::vector)
                        LIMIT $2
                        """,
                        vector_str,
                        k,
                    )
                else:
                    rows: Sequence[asyncpg.Record] = await conn.fetch(
                        """
                        SELECT
                          id,
                          title,
                          supplier_name,
                          from_region,
                          photo,
                          order_price_kg,
                          embedding <-> ($1::vector) AS distance
                        FROM myaso.products
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <-> ($1::vector)
                        """,
                        vector_str,
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
                f"Cooled/Frozen: {row_dict.get('cooled_or_frozen', '')}",
                f"Ready-made: {row_dict.get('ready_made', '')}",
            ]
            page_content = "; ".join([p for p in content_parts if p])

            metadata = {**row_dict}
            metadata.pop("embedding", None)

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents
