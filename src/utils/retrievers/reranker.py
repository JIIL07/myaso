"""Модуль для переранжирования результатов поиска."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.utils.rules import get_rule_as_bool

logger = logging.getLogger(__name__)


async def rerank_documents(
    documents: List[Document],
    query: str,
    context: Dict[str, Any],
) -> List[Document]:
    """Переранжирует документы с учетом контекста.

    Args:
        documents: Список документов для переранжирования
        query: Исходный запрос
        context: Контекст клиента (предпочтения, настройки)

    Returns:
        Переранжированный список документов
    """
    try:
        enable_reranking = await get_rule_as_bool("ENABLE_RERANKING")
    except Exception as e:
        logger.warning(f"[reranker] Не удалось загрузить ENABLE_RERANKING из БД, используем False: {e}")
        enable_reranking = False
    
    if not enable_reranking:
        return documents

    try:
        require_photo = context.get("require_photo", False)
        preferences = context.get("preferences", {})
        
        scored_documents = []
        for doc in documents:
            score = 0.0
            metadata = doc.metadata
            
            if require_photo and metadata.get("photo"):
                score += 1.0
            
            if preferences.get("regions"):
                doc_region = metadata.get("from_region", "").lower()
                for preferred_region in preferences["regions"]:
                    if preferred_region.lower() in doc_region:
                        score += 0.5
                        break
            
            if preferences.get("cooled_or_frozen"):
                doc_cooled = metadata.get("cooled_or_frozen", "").lower()
                if preferences["cooled_or_frozen"].lower() in doc_cooled:
                    score += 0.3
            
            scored_documents.append((score, doc))
        
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        
        reranked = [doc for _, doc in scored_documents]
        
        logger.info(f"[reranker] Переранжировано {len(reranked)} документов")
        return reranked

    except Exception as e:
        logger.error(f"[reranker] Ошибка при переранжировании: {e}", exc_info=True)
        return documents

