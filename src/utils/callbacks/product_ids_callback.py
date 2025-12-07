"""Callback handler для сохранения product_ids из artifacts инструментов поиска."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from src.agents.tools.context_tools import save_product_ids_to_context

logger = logging.getLogger(__name__)


class ProductIdsCallbackHandler(BaseCallbackHandler):
    """Callback handler для сохранения product_ids из artifacts инструментов поиска.
    
    Сохраняет product_ids в контекст агента сразу после выполнения инструмента поиска,
    чтобы они были доступны для других инструментов (например, show_product_photos)
    во время выполнения агента.
    """
    
    # Инструменты поиска товаров, которые возвращают product_ids как artifacts
    PRODUCT_SEARCH_TOOLS = {
        'vector_search',
        'execute_sql_query',
        'get_random_products',
        'find_similar_products',
        'get_product_by_title',
    }
    
    def __init__(self, client_phone: str):
        """Инициализация ProductIdsCallbackHandler.
        
        Args:
            client_phone: Номер телефона клиента для сохранения контекста
        """
        super().__init__()
        self.client_phone = client_phone
        self._accumulated_product_ids: List[int] = []
    
    def on_tool_end(
        self,
        output: Any,
        *,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        """Вызывается когда инструмент завершает выполнение.
        
        Args:
            output: Результат выполнения инструмента
            name: Имя инструмента
            **kwargs: Дополнительные параметры
        """
        # Проверяем, является ли это инструментом поиска товаров
        if not name or name not in self.PRODUCT_SEARCH_TOOLS:
            return
        
        # Извлекаем artifact из output
        # Когда инструмент использует response_format="content_and_artifact",
        # LangChain возвращает кортеж (content, artifact) или ToolMessage с artifact
        artifact = None
        
        if isinstance(output, tuple) and len(output) == 2:
            # Инструмент вернул кортеж (content, artifact)
            _, artifact = output
        elif isinstance(output, ToolMessage) and hasattr(output, 'artifact'):
            # Инструмент вернул ToolMessage с artifact
            artifact = output.artifact
        elif hasattr(output, 'artifact'):
            # У output есть атрибут artifact
            artifact = output.artifact
        
        if artifact is None:
            logger.debug(
                f"[ProductIdsCallbackHandler] Инструмент {name} не вернул artifact"
            )
            return
        
        # Извлекаем product_ids из artifact
        product_ids = self._extract_product_ids_from_artifact(artifact)
        
        if product_ids:
            self._accumulated_product_ids.extend(product_ids)
            logger.debug(
                f"[ProductIdsCallbackHandler] Извлечено {len(product_ids)} product_ids "
                f"из инструмента {name}"
            )
    
    def _extract_product_ids_from_artifact(self, artifact: Any) -> List[int]:
        """Извлекает product_ids из artifact.
        
        Args:
            artifact: Artifact из инструмента (может быть списком, int, dict)
            
        Returns:
            Список product_ids
        """
        product_ids = []
        
        try:
            if isinstance(artifact, list):
                # Если artifact - список, извлекаем все ID
                for item in artifact:
                    if isinstance(item, (int, str)):
                        product_id = int(item)
                        if product_id > 0:
                            product_ids.append(product_id)
                    elif isinstance(item, dict) and 'id' in item:
                        product_id = int(item['id'])
                        if product_id > 0:
                            product_ids.append(product_id)
            elif isinstance(artifact, (int, str)):
                # Если artifact - одиночное значение
                product_id = int(artifact)
                if product_id > 0:
                    product_ids.append(product_id)
            elif isinstance(artifact, dict):
                # Если artifact - словарь
                if 'id' in artifact:
                    product_id = int(artifact['id'])
                    if product_id > 0:
                        product_ids.append(product_id)
                elif 'product_ids' in artifact:
                    # Если artifact содержит список product_ids
                    ids_list = artifact['product_ids']
                    if isinstance(ids_list, list):
                        for item in ids_list:
                            product_id = int(item)
                            if product_id > 0:
                                product_ids.append(product_id)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"[ProductIdsCallbackHandler] Ошибка извлечения product_ids из artifact: {e}. "
                f"Artifact type: {type(artifact)}, value: {artifact}"
            )
        
        return product_ids
    
    async def save_accumulated_product_ids(self) -> None:
        """Сохраняет накопленные product_ids в контекст агента.
        
        Должно вызываться периодически или в конце выполнения агента.
        """
        if not self._accumulated_product_ids:
            return
        
        try:
            # Удаляем дубликаты, сохраняя порядок
            unique_ids = list(dict.fromkeys(self._accumulated_product_ids))
            
            # Сохраняем в контекст
            await save_product_ids_to_context(self.client_phone, unique_ids)
            
            logger.info(
                f"[ProductIdsCallbackHandler] Сохранено {len(unique_ids)} product_ids "
                f"в контекст для {self.client_phone}"
            )
            
            # Очищаем накопленные ID после сохранения
            self._accumulated_product_ids.clear()
        except Exception as e:
            logger.error(
                f"[ProductIdsCallbackHandler] Ошибка сохранения product_ids для {self.client_phone}: {e}",
                exc_info=True
            )
