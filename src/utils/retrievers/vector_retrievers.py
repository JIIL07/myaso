from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import asyncpg
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from openai import AsyncOpenAI

from src.services.database.database import get_pool
from src.config.settings import settings
from src.constants import DEFAULT_VECTOR_SEARCH_K

logger = logging.getLogger(__name__)


class PostgresVectorRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        db_dsn: str | None = None,
        k: int | None = None,
    ) -> None:
        super().__init__()
        self._embedder = AsyncOpenAI(
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

    async def _embed(self, text: str) -> list[float]:
        completion = await self._embedder.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        data = completion.model_dump()
        return data["data"][0]["embedding"]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        return await self._get_relevant_documents(query, k=self._k)

    async def get_relevant_documents(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        if k is None:
            if self._k is None:
                k = DEFAULT_VECTOR_SEARCH_K
            else:
                k = self._k
        return await self._get_relevant_documents(query, k=k)

    async def _get_relevant_documents(self, query: str, k: int) -> list[Document]:
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
                "Database connection error: %s: %s", error_type, error_str, exc_info=True
            )

            raise RuntimeError("Ошибка подключения к базе данных") from e

        documents: list[Document] = []
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
