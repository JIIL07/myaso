from fastapi import APIRouter, BackgroundTasks
from src.schemas import UserMessageRequest
from agents.factory import AgentFactory
from src.utils import remove_markdown_symbols
import requests
from langfuse import Langfuse

router = APIRouter(prefix="/ai/v2")

# LangChain version - Process conversation
async def process_conversation_v2_background(request: UserMessageRequest):
    langfuse = None
    trace = None
    try:
        try:
            langfuse = Langfuse()
            trace = langfuse.trace(
                name="processConversation_v2",
                user_id=request.client_phone,
                input={
                    "client_phone": request.client_phone,
                    "message": request.message,
                },
                tags=["langchain", "agent", "v2"],
            )
        except Exception as _:
            langfuse = None
            trace = None

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={})

        # Run agent
        response_text = await agent.run(user_input=request.message, client_phone=request.client_phone)

        try:
            requests.post(
                "http://51.250.42.45:2026/send-message",
                json={
                    "recipient": request.client_phone,
                    "message": remove_markdown_symbols(response_text),
                },
            )
        except Exception as _:
            pass

        if trace is not None:
            try:
                trace.update(output={"response": response_text})
            except Exception:
                pass

        return {"success": True}

    except Exception as e:
        try:
            requests.post(
                "http://51.250.42.45:2026/send-message",
                json={
                    "recipient": request.client_phone,
                    "message": "Что-то барахлит вотсап 😞. Пожалуйста, отправьте сообщение ещё раз",
                },
            )
        except Exception:
            pass

        if trace is not None:
            try:
                trace.update(output={"error": str(e)})
            except Exception:
                pass

        return {"success": False}


@router.post("/processConversation", status_code=200)
async def process_conversation_v2(request: UserMessageRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_conversation_v2_background, request)
    return {"success": True}


# LangChain version - Init conversation
from src.schemas import InitConverastionRequest
from src.utils.langchain_memory import SupabaseConversationMemory
from agents.tools import get_client_profile, enhance_user_product_query
from src.services.orders_service import OrderService


async def init_conversation_v2_background(request: InitConverastionRequest):
    langfuse = None
    trace = None

    try:
        try:
            langfuse = Langfuse()
            trace = langfuse.trace(
                name="initConversation_v2",
                user_id=request.client_phone,
                input={
                    "client_phone": request.client_phone,
                    "topic": request.topic,
                },
                tags=["langchain", "agent", "v2", "init"],
            )
        except Exception:
            langfuse = None
            trace = None

        memory = await SupabaseConversationMemory(request.client_phone)
        try:
            await memory.clear()
        except Exception:
            pass

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        profile_text = ""
        try:
            profile_text = await get_client_profile.ainvoke({"phone": request.client_phone})
        except Exception as e:
            profile_text = "Профиль клиента не найден."

        products_text = ""
        try:
            seed_query = request.topic or "Ассортимент товаров"
            rag_text = await enhance_user_product_query.ainvoke({"query": seed_query})
            if rag_text and "не найдены" not in rag_text.lower():
                products_text = rag_text
            else:
                raise ValueError("RAG empty")
        except Exception:
            try:
                order_service = await OrderService()
                random_products = await order_service.get_random_products(limit=10)
                if random_products:
                    lines = []
                    for p in random_products[:10]:
                        title = p.get("title", "")
                        supplier = p.get("supplier_name", "")
                        price = p.get("order_price_kg", "")
                        lines.append(f"- {title} (поставщик: {supplier}, цена/кг: {price})")
                    products_text = "Топ-товары:\n" + "\n".join(lines)
                else:
                    products_text = "Ассортимент будет обновлён позже."
            except Exception:
                products_text = "Ассортимент будет обновлён позже."

        welcome_input = (
            "Сформируй короткое приветствие для клиента, учитывая его профиль и ассортимент.\n"
            f"Тема диалога: {request.topic}\n\n"
            f"Профиль клиента:\n{profile_text}\n\n"
            f"Ассортимент/подборка:\n{products_text}\n\n"
            "Поприветствуй дружелюбно, предложи помощь и ненавязчиво уточни запрос."
        )

        response_text = await agent.run(user_input=welcome_input, client_phone=request.client_phone)

        try:
            requests.post(
                "http://51.250.42.45:2026/send-message",
                json={
                    "recipient": request.client_phone,
                    "message": remove_markdown_symbols(response_text),
                },
            )
        except Exception:
            pass

        if trace is not None:
            try:
                trace.update(
                    output={
                        "response": response_text,
                        "profile": profile_text,
                        "products": products_text,
                    }
                )
            except Exception:
                pass

        return {"success": True}

    except Exception as e:
        try:
            requests.post(
                "http://51.250.42.45:2026/send-message",
                json={
                    "recipient": request.client_phone,
                    "message": "Что-то барахлит вотсап 😞. Пожалуйста, отправьте сообщение ещё раз",
                },
            )
        except Exception:
            pass

        if trace is not None:
            try:
                trace.update(output={"error": str(e)})
            except Exception:
                pass

        return {"success": False}


@router.post("/initConversation", status_code=200)
async def init_conversation_v2(request: InitConverastionRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(init_conversation_v2_background, request)
    return {"success": True}


# LangChain version - Get profile
from pydantic import BaseModel
from typing import Optional, Dict, Any
from agents.tools import get_client_profile as lc_get_client_profile
from src.services.history_service import HistoryService
from src.services.orders_service import OrderService as OrdersService


class ClientProfileResponse(BaseModel):
    client_phone: str
    profile: str
    message_count: int
    last_order: Optional[Dict[str, Any]] = None
    status: str


@router.get("/getProfile", response_model=ClientProfileResponse, status_code=200)
async def get_profile_v2(client_phone: str):
    try:
        profile_text = await lc_get_client_profile.ainvoke({"phone": client_phone})
    except Exception:
        profile_text = "Профиль клиента не найден в базе данных."

    message_count = 0
    try:
        history_service = await HistoryService()
        history_resp = await history_service.get_history(client_phone=client_phone)
        data = getattr(history_resp, "data", [])
        message_count = len(data)
    except Exception:
        message_count = 0

    last_order: Optional[Dict[str, Any]] = None
    try:
        orders_service = await OrdersService()
        orders = await orders_service.get_all_orders_by_client_phone(client_phone=client_phone)
        if orders:
            def _created_at(o: Dict[str, Any]):
                return o.get("created_at") or ""
            orders_sorted = sorted(orders, key=_created_at, reverse=True)
            o = orders_sorted[0]
            last_order = {
                "title": o.get("title"),
                "created_at": o.get("created_at"),
                "destination": o.get("destination"),
                "price_out": o.get("price_out"),
                "weight_kg": o.get("weight_kg"),
            }
    except Exception:
        last_order = None

    status = "active" if (message_count > 0 or last_order is not None) else "new"

    return ClientProfileResponse(
        client_phone=client_phone,
        profile=profile_text,
        message_count=message_count,
        last_order=last_order,
        status=status,
    )
