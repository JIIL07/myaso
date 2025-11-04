from __future__ import annotations

from typing import Any, List, Sequence
import os

import asyncpg
from langchain_core.documents import Document

from src.config.settings import settings
from openai import OpenAI


class SupabaseVectorRetriever:
    """Ретривер для семантического поиска по товарам (pgvector)."""

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        db_dsn: str | None = None,
    ) -> None:
        """Инициализация ретривера.

        - embedding_model: модель эмбеддингов
        - db_dsn: DSN Postgres
        """
        self._embedder = OpenAI(
            api_key=settings.alibaba.alibaba_key,
            base_url=settings.alibaba.base_alibaba_url,
        )
        self._embedding_model = (
            embedding_model or settings.alibaba.embedding_model_id or "text-embedding-v4"
        )

        self._db_dsn = db_dsn or os.getenv("POSTGRES_DSN")

    async def _embed(self, text: str) -> List[float]:
        """Создаёт эмбеддинг текста (список float)."""
        completion = self._embedder.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        data = completion.model_dump()
        return data["data"][0]["embedding"]

    async def get_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        """Возвращает top-k документов по близости (LangChain Document)."""
        vector = await self._embed(query)

        if not self._db_dsn:
            raise RuntimeError(
                "POSTGRES_DSN is not set. Provide db_dsn at construction or set POSTGRES_DSN in .env"
            )

        conn: asyncpg.Connection | None = None
        try:
            conn = await asyncpg.connect(dsn=self._db_dsn)
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
                  embedding <-> $1::vector AS distance
                FROM myaso.products
                ORDER BY embedding <-> $1::vector
                LIMIT $2
                """,
                vector,
                k,
            )
        finally:
            if conn is not None:
                await conn.close()

        documents: List[Document] = []
        for row in rows:
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

            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )

        return documents
