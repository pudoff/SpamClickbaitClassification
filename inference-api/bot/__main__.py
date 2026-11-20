"""Запуск Telegram-бота через long polling."""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from aiohttp import ClientSession, ClientTimeout

from bot.handlers import create_router


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN в .env")
    api_url = os.getenv("INFERENCE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    async with Bot(token) as bot, ClientSession(base_url=api_url, timeout=ClientTimeout(total=150, connect=5)) as api:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Как отправить сообщение на проверку"),
                BotCommand(command="help", description="Поддерживаемые форматы и ограничения"),
            ]
        )
        await dispatcher.start_polling(bot, api=api, tasks_concurrency_limit=2, close_bot_session=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
