"""Приём сообщений Telegram, запрос в inference-api и ответ в чате."""

from pathlib import PurePath

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import Audio, BufferedInputFile, Document, Message, Voice
from aiohttp import ClientError, ClientSession, FormData

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 300
MAX_TEXT_LENGTH = 20_000
MAX_TEXT_BYTES = MAX_TEXT_LENGTH * 4 + 3  # UTF-8 и необязательный BOM.
AUDIO_SUFFIXES = {".ogg", ".opus", ".m4a", ".mp3", ".wav"}
HELP = (
    "Отправьте одно из четырёх:\n"
    "1. Запишите голосовое кнопкой микрофона Telegram.\n"
    "2. Прикрепите аудиофайл OGG/Opus, M4A/AAC, MP3 или WAV/PCM.\n"
    "3. Напишите или вставьте текст в сообщение.\n"
    "4. Прикрепите файл .txt в кодировке UTF-8.\n\n"
    "Аудио — до 20 МиБ и 5 минут. Текст — до 20 000 символов.\n"
    "Бот вернёт результат классификации и расшифровку для аудио."
)
LABELS = {"spam": "🚫 Спам", "clickbait": "🎣 Кликбейт", "normal": "✅ Обычное сообщение"}
REASONS = {
    "no_speech": "Речь не обнаружена.",
    "speech_too_short": "Речь слишком короткая. Запишите хотя бы одну секунду речи.",
    "unsupported_language": "Сейчас бот распознаёт русский язык.",
}
API_ERRORS = {
    "payload_too_large": "Файл больше 20 МиБ.",
    "unsupported_audio_format": "Поддерживаются OGG/Opus, M4A/AAC, MP3 и WAV/PCM.",
    "audio_decode_failed": "Не удалось прочитать аудиофайл. Попробуйте отправить другой файл.",
    "audio_too_long": "Аудио должно быть не длиннее 5 минут.",
    "overloaded": "Распознавание занято. Попробуйте через несколько секунд.",
}
UNAVAILABLE = "Сервис проверки временно недоступен. Попробуйте позже."
INVALID_TEXT = "Отправьте текст с буквами длиной до 20 000 символов."


async def send_result(message: Message, response: dict) -> None:
    match response.get("status"):
        case "classified":
            summary = LABELS.get(response.get("label"), UNAVAILABLE)
        case "uncertain":
            summary = "🤔 Не уверен, к какому классу относится сообщение."
        case "not_classified":
            summary = REASONS.get(response.get("reason"), "Распознать речь не удалось.")
        case _:
            summary = UNAVAILABLE
    confidence = response.get("confidence")
    if isinstance(confidence, (int, float)):
        summary += f"\nУверенность модели — {confidence:.0%}."

    transcript = response.get("transcript")
    if not transcript:
        await message.answer(summary)
        return
    reply = f"{summary}\n\nРасшифровка\n{transcript}"
    if len(reply.encode("utf-16-le")) <= 8192:
        await message.answer(reply)
    else:
        await message.answer_document(BufferedInputFile(transcript.encode(), "transcript.txt"), caption=summary)


async def check_message(
    message: Message, api: ClientSession, *, text: str | None = None, audio: FormData | None = None
) -> None:
    if text is not None:
        if not text.strip() or len(text) > MAX_TEXT_LENGTH:
            await message.answer(INVALID_TEXT)
            return
        path, payload = "/predict", {"json": {"text": text}}
    else:
        path, payload = "/v1/classifications", {"data": audio}

    try:
        async with api.post(path, **payload) as reply:
            body = await reply.json()
            if not isinstance(body, dict):
                raise ValueError("ожидался JSON-объект")
            code = reply.status
    except (ClientError, TimeoutError, ValueError):
        await message.answer(UNAVAILABLE)
        return

    if code == 200:
        await send_result(message, body)
    elif text is not None and code == 422:
        await message.answer(INVALID_TEXT)
    else:
        detail = body.get("detail")
        await message.answer(API_ERRORS.get(detail, UNAVAILABLE) if isinstance(detail, str) else UNAVAILABLE)


async def check_file(
    message: Message,
    bot: Bot,
    api: ClientSession,
    media: Voice | Audio | Document,
    *,
    is_text: bool = False,
) -> None:
    limit = MAX_TEXT_BYTES if is_text else MAX_AUDIO_BYTES
    too_large = INVALID_TEXT if is_text else API_ERRORS["payload_too_large"]
    if media.file_size is not None and media.file_size > limit:
        await message.answer(too_large)
        return
    if isinstance(media, (Voice, Audio)) and media.duration > MAX_AUDIO_SECONDS:
        await message.answer(API_ERRORS["audio_too_long"])
        return
    try:
        download = await bot.download(media, timeout=30)
    except (ClientError, TelegramAPIError, TimeoutError):
        await message.answer("Не удалось скачать файл из Telegram. Попробуйте отправить его ещё раз.")
        return
    if download is None:
        await message.answer("Не удалось скачать файл из Telegram.")
        return
    with download:
        data = download.read(limit + 1)
    if len(data) > limit:
        await message.answer(too_large)
        return
    if is_text:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            await message.answer("Сохраните .txt в кодировке UTF-8 и отправьте снова.")
            return
        await check_message(message, api, text=text)
    else:
        form = FormData()
        form.add_field("audio", data, filename="audio", content_type="application/octet-stream")
        source = "forwarded" if message.forward_origin else "own" if isinstance(media, Voice) else "file"
        form.add_field("source_kind", source)
        await check_message(message, api, audio=form)


def create_router() -> Router:
    router = Router()

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def help_message(message: Message) -> None:
        await message.answer(HELP)

    @router.message(F.voice)
    async def voice_message(message: Message, bot: Bot, api: ClientSession) -> None:
        await check_file(message, bot, api, message.voice)

    @router.message(F.audio)
    async def audio_message(message: Message, bot: Bot, api: ClientSession) -> None:
        await check_file(message, bot, api, message.audio)

    @router.message(F.document)
    async def document_message(message: Message, bot: Bot, api: ClientSession) -> None:
        document = message.document
        suffix = PurePath(document.file_name or "").suffix.lower()
        mime = document.mime_type or ""
        if suffix == ".txt" or (not suffix and mime == "text/plain"):
            await check_file(message, bot, api, document, is_text=True)
        elif suffix in AUDIO_SUFFIXES or mime.startswith("audio/"):
            await check_file(message, bot, api, document)
        else:
            await message.answer("Прикрепите аудио или файл .txt. Поддерживаемые форматы — в /help.")

    @router.message(F.text)
    async def text_message(message: Message, api: ClientSession) -> None:
        await check_message(message, api, text=message.text)

    @router.message()
    async def other_message(message: Message) -> None:
        await message.answer(HELP)

    return router
