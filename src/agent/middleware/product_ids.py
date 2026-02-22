from __future__ import annotations

import logging
from typing import Any, Awaitable

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

PRODUCT_SEARCH_TOOLS = {
    "vector_search",
    "execute_sql_query",
    "get_random_products",
    "get_product_by_title",
}


def _extract_product_ids_from_result(result: Any) -> list[int]:
    product_ids: list[int] = []

    try:
        if hasattr(result, "artifact"):
            artifact = result.artifact
        elif isinstance(result, tuple) and len(result) == 2:
            _, artifact = result
        else:
            return product_ids

        if isinstance(artifact, list):
            for item in artifact:
                if isinstance(item, (int, str)):
                    product_id = int(item)
                    if product_id > 0:
                        product_ids.append(product_id)
        elif isinstance(artifact, dict):
            raw_product_ids = artifact.get("product_ids", [])
            if isinstance(raw_product_ids, list):
                for item in raw_product_ids:
                    if isinstance(item, (int, str)):
                        product_id = int(item)
                        if product_id > 0:
                            product_ids.append(product_id)
        elif isinstance(artifact, (int, str)):
            product_id = int(artifact)
            if product_id > 0:
                product_ids.append(product_id)
    except (ValueError, TypeError) as e:
        logger.debug("[ProductIds] Error extracting product_ids: %s", e)

    return product_ids


@wrap_tool_call
async def save_product_ids_middleware(request: Any, handler: Any) -> Any:
    if callable(handler):
        result = handler(request)
        if isinstance(result, Awaitable):
            result = await result
    else:
        result = await handler if isinstance(handler, Awaitable) else handler

    tool_name = None
    tool_call_id = None
    if hasattr(request, "tool_call"):
        if isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name")
            tool_call_id = request.tool_call.get("id")
        elif hasattr(request.tool_call, "name"):
            tool_name = request.tool_call.name
            tool_call_id = getattr(request.tool_call, "id", None)

    if tool_name and tool_name in PRODUCT_SEARCH_TOOLS:
        try:
            product_ids = _extract_product_ids_from_result(result)

            if product_ids:
                current_product_ids: list[int] = []
                if hasattr(request, "runtime") and request.runtime:
                    current_product_ids = request.runtime.state.get("product_ids", [])

                all_ids = current_product_ids + product_ids
                unique_ids = list(dict.fromkeys(all_ids))

                logger.debug(
                    "[ProductIds] +%d from %s, total: %d",
                    len(product_ids), tool_name, len(unique_ids),
                )

                if isinstance(result, ToolMessage):
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [result],
                        }
                    )
                elif isinstance(result, tuple) and len(result) == 2:
                    content, artifact = result
                    tool_message = ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id or "",
                        artifact=artifact,
                    )
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [tool_message],
                        }
                    )
        except Exception as e:
            logger.warning("[ProductIds] Error saving from %s: %s", tool_name, e, exc_info=True)

    return result

