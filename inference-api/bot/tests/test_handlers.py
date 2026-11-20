"""Проверки диспетчера Telegram и HTTP-обмена бота без внешних сервисов."""

from io import BytesIO
from unittest import IsolatedAsyncioTestCase

from aiogram import Bot, Dispatcher
from aiogram.methods import SendDocument, SendMessage
from aiogram.types import Update
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from bot.handlers import MAX_AUDIO_BYTES, MAX_TEXT_BYTES, create_router

MESSAGE_TEXT = "Буду через десять минут, возьми хлеба."
AUDIO_CONTENT = b"fixture audio"


class TelegramStub(Bot):
    def __init__(self):
        super().__init__("123456:fixture")
        self.replies: list[SendMessage | SendDocument] = []
        self.downloads = []
        self.contents = {
            "audio": AUDIO_CONTENT,
            "text": MESSAGE_TEXT.encode("utf-8-sig"),
            "bad-text": b"\xff",
            "large-text": b"x" * (MAX_TEXT_BYTES + 1),
            "timeout": TimeoutError(),
        }

    async def download(self, file, destination: BytesIO, **kwargs):
        self.downloads.append(file.file_id)
        data = self.contents[file.file_id]
        if isinstance(data, Exception):
            raise data
        for start in range(0, len(data), 64 * 1024):
            destination.write(data[start : start + 64 * 1024])
        destination.seek(0)
        return destination

    async def __call__(self, method, **kwargs):
        self.replies.append(method)


def media(file_id: str, **fields) -> dict:
    return {"file_id": file_id, "file_unique_id": file_id, **fields}


class HandlerTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api_calls = []
        self.result = {"status": "classified", "label": "normal", "confidence": 0.83}
        self.status_code = 200
        api = web.Application()
        api.router.add_post("/predict", self.inference)
        api.router.add_post("/v1/classifications", self.inference)
        self.server = TestServer(api)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.api = ClientSession(base_url=self.server.make_url("/"))
        self.addAsyncCleanup(self.api.close)
        self.bot = TelegramStub()
        self.addAsyncCleanup(self.bot.session.close)
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router())

    async def inference(self, request: web.Request) -> web.Response:
        result = self.result.copy()
        if request.path == "/predict":
            payload = await request.json()
        else:
            reader = await request.multipart()
            parts = {}
            async for part in reader:
                parts[part.name] = await part.read()
            payload = parts
            if result.get("status") == "classified":
                result.setdefault("transcript", MESSAGE_TEXT)
        self.api_calls.append((request.path, payload))
        return web.json_response(result, status=self.status_code)

    async def send(self, **message):
        update_id = len(self.bot.replies) + 1
        update = Update.model_validate(
            {
                "update_id": update_id,
                "message": {
                    "message_id": update_id,
                    "date": 1795187746,
                    "chat": {"type": "private", "id": 117},
                    **message,
                },
            }
        )
        await self.dispatcher.feed_update(self.bot, update, api=self.api)
        assert len(self.bot.replies) == update_id
        reply = self.bot.replies[-1]
        assert reply.chat_id == 117
        return reply

    async def test_start_and_help_describe_inputs_without_calling_api(self):
        for command in ("/start", "/help"):
            with self.subTest(command=command):
                reply = await self.send(text=command)
                for option in ("голосовое", "аудиофайл", "сообщение", ".txt"):
                    self.assertIn(option, reply.text)
        self.assertEqual(self.api_calls, [])
        self.assertEqual(self.bot.downloads, [])

    async def test_message_and_utf8_text_file_send_same_json(self):
        await self.send(text=MESSAGE_TEXT)
        reply = await self.send(document=media("text", file_name="Текст.TXT"))
        self.assertEqual(self.api_calls, [("/predict", {"text": MESSAGE_TEXT})] * 2)
        self.assertEqual(self.bot.downloads, ["text"])
        self.assertIn("Обычное сообщение", reply.text)
        self.assertIn("83%", reply.text)

    async def test_voice_audio_and_document_send_multipart(self):
        cases = [
            ({"voice": media("audio", duration=5)}, b"own"),
            ({"audio": media("audio", duration=5)}, b"file"),
            ({"document": media("audio", file_name="Голос.M4A")}, b"file"),
        ]
        for fields, source in cases:
            with self.subTest(fields=fields):
                reply = await self.send(**fields)
                self.assertEqual(
                    self.api_calls[-1],
                    ("/v1/classifications", {"audio": AUDIO_CONTENT, "source_kind": source}),
                )
                self.assertIn("Расшифровка", reply.text)
                self.assertIn(MESSAGE_TEXT, reply.text)
        self.assertEqual(self.bot.downloads, ["audio"] * 3)

    async def test_file_metadata_is_checked_before_download(self):
        for fields, message in (
            ({"voice": media("audio", duration=301)}, "5 минут"),
            ({"audio": media("audio", duration=2, file_size=MAX_AUDIO_BYTES + 1)}, "20 МиБ"),
            ({"document": media("text", file_name="text.txt", file_size=MAX_TEXT_BYTES + 1)}, "20 000"),
            ({"document": media("text", file_name="book.pdf")}, "/help"),
        ):
            with self.subTest(fields=fields):
                self.assertIn(message, (await self.send(**fields)).text)
        self.assertEqual(self.bot.downloads, [])
        self.assertEqual(self.api_calls, [])

    async def test_bad_encoding_and_overlimit_download_are_rejected_locally(self):
        for file_id, message in (("bad-text", "UTF-8"), ("large-text", "20 000")):
            with self.subTest(file_id=file_id):
                reply = await self.send(document=media(file_id, file_name="text.txt"))
                self.assertIn(message, reply.text)
        self.assertEqual(self.api_calls, [])

    async def test_telegram_download_timeout_gets_reply(self):
        reply = await self.send(voice=media("timeout", duration=5))
        self.assertIn("Не удалось скачать", reply.text)
        self.assertEqual(self.api_calls, [])

    async def test_inference_refusals_and_errors_get_user_facing_replies(self):
        cases = [
            (200, {"status": "uncertain", "confidence": 0.4}, "Не уверен"),
            (200, {"status": "not_classified", "reason": "no_speech"}, "Речь не обнаружена"),
            (415, {"detail": "unsupported_audio_format"}, "OGG/Opus"),
            (503, {"detail": "overloaded"}, "Распознавание занято"),
        ]
        for code, body, message in cases:
            with self.subTest(code=code, body=body):
                self.status_code, self.result = code, body
                reply = await self.send(voice=media("audio", duration=5))
                self.assertIn(message, reply.text)
                self.assertNotIn("Обычное сообщение", reply.text)

    async def test_api_connection_error_gets_reply(self):
        await self.server.close()
        reply = await self.send(text=MESSAGE_TEXT)
        self.assertIn("временно недоступен", reply.text)

    async def test_long_transcript_is_sent_as_utf8_document(self):
        transcript = "🚀" * 3000
        self.result["transcript"] = transcript
        reply = await self.send(voice=media("audio", duration=5))
        self.assertIsInstance(reply, SendDocument)
        self.assertEqual(reply.document.filename, "transcript.txt")
        self.assertEqual(reply.document.data.decode("utf-8"), transcript)
        self.assertIn("Обычное сообщение", reply.caption)
