#!/usr/bin/env python3
"""
Скрипт для выполнения большого SQL файла в локальном Supabase.

Использование:
    python scripts/run_sql_file.py <путь_к_sql_файлу>
    
Или с кастомными параметрами:
    python scripts/run_sql_file.py <путь_к_sql_файлу> \\
        --host localhost \\
        --port 54322 \\
        --database postgres \\
        --user postgres \\
        --password postgres
"""

import asyncio
import argparse
import sys
from pathlib import Path
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()


async def execute_sql_file(
    file_path: str,
    host: str = "localhost",
    port: int = 54322,
    database: str = "postgres",
    user: str = "postgres",
    password: str = "postgres",
    stop_on_error: bool = False,
):
    """Выполняет SQL файл в базе данных."""
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        print(f"Файл не найден: {file_path}")
        sys.exit(1)
    
    print(f"Читаю файл: {file_path}")
    sql_content = file_path_obj.read_text(encoding='utf-8')
    
    commands = [
        cmd.strip() 
        for cmd in sql_content.split(';') 
        if cmd.strip() and not cmd.strip().startswith('--')
    ]
    
    print(f"Найдено команд: {len(commands)}")
    print(f"Подключаюсь к БД: {host}:{port}/{database}")
    
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        
        print("Подключение успешно")
        
        success_count = 0
        error_count = 0
        
        for i, command in enumerate(commands, 1):
            if not command:
                continue
                
            try:
                if len(commands) > 10:
                    print(f"Выполняю команду {i}/{len(commands)}...", end='\r')
                
                await conn.execute(command)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nОшибка в команде {i}:")
                print(f"   {str(e)}")
                print(f"   Команда: {command[:100]}...")
                if stop_on_error:
                    await conn.close()
                    print(f"\nОстановка выполнения из-за ошибки (команда {i})")
                    sys.exit(1)
        
        if len(commands) > 10:
            print()
        
        print(f"\nВыполнено успешно: {success_count}")
        if error_count > 0:
            print(f"Ошибок: {error_count}")
            await conn.close()
            sys.exit(1)
        
        await conn.close()
        print("Готово!")
        
    except asyncpg.exceptions.InvalidPasswordError:
        print("Неверный пароль")
        sys.exit(1)
    except asyncpg.exceptions.ConnectionRefusedError:
        print(f"Не удалось подключиться к {host}:{port}")
        print("   Убедитесь, что Supabase запущен локально")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Выполняет SQL файл в локальной базе данных Supabase"
    )
    parser.add_argument(
        "file",
        type=str,
        help="Путь к SQL файлу"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("POSTGRES_HOST", "localhost"),
        help="Хост БД (по умолчанию: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("POSTGRES_PORT", "54322")),
        help="Порт БД (по умолчанию: 54322 для локального Supabase)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("POSTGRES_DB", "postgres"),
        help="Имя базы данных (по умолчанию: postgres)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=os.getenv("POSTGRES_USER", "postgres"),
        help="Пользователь БД (по умолчанию: postgres)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("POSTGRES_PASSWORD", "postgres"),
        help="Пароль БД (по умолчанию: postgres)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Остановить выполнение при первой ошибке"
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        execute_sql_file(
            file_path=args.file,
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.user,
            password=args.password,
            stop_on_error=args.stop_on_error,
        )
    )


if __name__ == "__main__":
    main()

