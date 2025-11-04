from typing import List
import mistune
import re
from mistune.plugins.formatting import strikethrough
from mistune_telegram import TelegramHTMLRenderer
import json

from typing import Any, TypeVar

T = TypeVar("T", bound="AsyncMixin")

# https://dev.to/akarshan/asynchronous-python-magic-how-to-create-awaitable-constructors-with-asyncmixin-18j5
# https://web.archive.org/web/20230915163459/https://dev.to/akarshan/asynchronous-python-magic-how-to-create-awaitable-constructors-with-asyncmixin-18j5
class AsyncMixin:
    """
    Асинхронный миксин для поддержки асинхронной инициализации объектов.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __initobj(self: T) -> T:
        assert not self.async_initialized
        self.async_initialized = True
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()

def get_paths_from_map(paths: List[str], path_map):
    real_paths = []
    for path in paths:
        mapped_path = path_map.get(path, None)
        if mapped_path is not None:
            real_paths.append(mapped_path)
    
    return real_paths

def read_file_content(file_path):
    if file_path is None:
        return None

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tansform_files_to_context(file_paths: List[str] = [], path_map={}):
    real_paths = get_paths_from_map(file_paths, path_map)

    system_prompt = f"""
    Ты помощник с доступом к файлам. Ниже предоставлено содержимое файлов. используй их как контекст, для помощи ответа на запрос пользователя
    Контекст:
    
    """
    file_counter = 1
    for file_path in real_paths:
        content=read_file_content(file_path)
        
        if content is None:
            continue

        system_prompt+=f"""
        Файл #{file_counter}:
        {content}
        
        """    

        file_counter += 1

    system_prompt += "В следующем сообщении будет запрос пользователя: "
    return system_prompt

async def transorm_history_to_llm_format(history):
    llm_history = []
    i = 0
    
    while i < len(history.data):
        record = history.data[i]
        
        if record['role'] == 'tool':
            message_content = record['message']
            if message_content.startswith("Tool call:"):
                # This is a tool call message
                try:
                    # Extract tool name and arguments
                    # Format: "Tool call: ShowProductPhotos with args: {json}"
                    parts = message_content.split(" with args: ")
                    if len(parts) == 2:
                        tool_name = parts[0].replace("Tool call: ", "").strip()
                        args_json = parts[1]
                        
                        # Parse the arguments JSON
                        args_data = json.loads(args_json)
                        
                        # Extract the actual function arguments
                        function_args = args_data.get('tool_call', {}).get('function', {}).get('arguments', '{}')
                        if isinstance(function_args, str):
                            # If it's already a string, parse it to get the dict, then serialize it back to string
                            function_args_dict = json.loads(function_args)
                            function_args_str = function_args  # Keep the string version
                        else:
                            # If it's a dict, serialize it to string
                            function_args_dict = function_args
                            function_args_str = json.dumps(function_args)
                        
                        tool_call_id = args_data.get('tool_call', {}).get('id', '')
                        
                        # Create structured tool call format
                        # IMPORTANT: The arguments field must be a string for the LLM API
                        tool_call_entry = {
                            'role': 'assistant',
                            'tool_calls': [{
                                'id': tool_call_id,
                                'function': {
                                    'name': tool_name,
                                    'arguments': function_args_str  # This must be a string
                                },
                                'type': 'function'
                            }]
                        }
                        llm_history.append(tool_call_entry)
                        
                        # Look for the corresponding tool response in the next record
                        if i + 1 < len(history.data):
                            next_record = history.data[i + 1]
                            if next_record['role'] == 'tool' and not next_record['message'].startswith("Tool call:"):
                                # This is the tool response
                                tool_response_content = next_record['message']
                                # If it starts with "Tool ", remove that prefix
                                if tool_response_content.startswith("Tool "):
                                    tool_response_content = tool_response_content[5:]  # Remove "Tool "
                                
                                tool_response_entry = {
                                    'role': 'tool',
                                    'content': tool_response_content,
                                    'tool_call_id': tool_call_id,
                                    'name': tool_name
                                }
                                llm_history.append(tool_response_entry)
                                
                                # Skip the next record since we've already processed it
                                i += 2
                                continue
                            else:
                                # No tool response found, increment by 1
                                i += 1
                        else:
                            # No next record, increment by 1
                            i += 1
                    else:
                        # Parsing failed, add as simple message
                        llm_history.append({'role': record['role'], 'content': record['message']})
                        i += 1
                except Exception as e:
                    # If parsing fails, add as a simple message
                    llm_history.append({'role': record['role'], 'content': record['message']})
                    i += 1
            elif message_content.startswith("Tool "):
                # This is a tool response message
                # Format: "Tool Фотографии успешно отправлены."
                tool_response_content = message_content[5:]  # Remove "Tool "
                
                # Add as tool response (without pairing since we don't have the tool call)
                llm_history.append({
                    'role': 'tool',
                    'content': tool_response_content,
                    'tool_call_id': '',
                    'name': 'unknown_tool'
                })
                i += 1
            else:
                # Regular tool message
                llm_history.append({'role': record['role'], 'content': record['message']})
                i += 1
        else:
            # Regular message (user, assistant, system)
            llm_history.append({'role': record['role'], 'content': record['message']})
            i += 1
    
    return llm_history

def remove_markdown_symbols(text: str) -> str:
    """
    Удаляет markdown символы из текста, оставляя только чистый текст.
    
    Args:
        text: Текст с markdown форматированием
        
    Returns:
        Текст без markdown символов
    """
    if not text:
        return text
    
    # Удаляем блоки кода (``код```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Удаляем inline код (`код`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Удаляем изображения ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Удаляем ссылки [текст](url), оставляя только текст
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Удаляем жирный текст **текст** или __текст__, оставляя содержимое
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Удаляем курсив *текст* или _текст_, оставляя содержимое
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Удаляем зачеркнутый текст ~~текст~~, оставляя содержимое
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Удаляем заголовки (# ## ### и т.д.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем горизонтальные линии (---, ___, ***)
    text = re.sub(r'^[\-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Удаляем маркеры неупорядоченных списков
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем маркеры упорядоченных списков
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем цитаты (>)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем лишние пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Удаляем пробелы в начале и конце
    text = text.strip()
    
    return text

def parse_sql_result(response_text):
    
    if response_text.find("```sql") == -1:
        raise ValueError("No SQL query found in the response [No ```sql block]")

    sql_start = response_text.find("```sql")
    response_text = response_text[sql_start+6:]
    sql_end = response_text.rfind("```")
    
    
    if sql_end != -1:
        response_text = response_text[:sql_end].strip()

    return response_text

def records_to_json(records):
    """
    Преобразует список asyncpg Record объектов в JSON-совместимый список словарей.
    
    Args:
        records: Список Record объектов из asyncpg
        
    Returns:
        Список словарей, готовых для JSON сериализации
    """
    if not records:
        return []
    
    json_result = []
    for record in records:
        # Преобразуем каждый Record в словарь
        record_dict = dict(record)
        json_result.append(record_dict)
    
    return json_result
