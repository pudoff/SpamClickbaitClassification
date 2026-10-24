"""Pydantic-модели запросов и ответов (шаги 3 и 6 задания)."""

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Label = Literal["spam", "clickbait", "normal"]
# Вероятность или порог уверенности
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

_LETTER_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ]")


class PredictRequest(BaseModel):
    """Текст для классификации. В будущем сюда попадёт расшифровка GigaAM."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"text": "Вам одобрен кредит на 300 000 рублей! Перезвоните по номеру 8-800..."},
                {"text": "Ты не поверишь, что случилось с ценами! Срочно перешли всем", "threshold": 0.5},
                {"text": "Буду через десять минут, возьми хлеба"},
            ]
        },
    )

    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="Сырой текст сообщения (до 20 000 символов, как transcript в контракте).",
    )
    threshold: Probability | None = Field(
        default=None,
        description="Порог уверенности для этого запроса. Если не задан — политика сервиса (TAU / TAU_SPAM).",
    )

    @field_validator("text")
    @classmethod
    def must_contain_letters(cls, value: str) -> str:
        value = value.strip()
        if not _LETTER_RE.search(value):
            raise ValueError("текст должен содержать хотя бы одну букву")
        return value


class Scores(BaseModel):
    """Вероятности классов; сумма равна 1 с точностью до округления."""

    spam: Probability
    clickbait: Probability
    normal: Probability


class PredictResponse(BaseModel):
    status: Literal["classified", "uncertain"] = Field(
        description="classified — метка выставлена; uncertain — уверенность ниже порога, метки нет."
    )
    label: Label | None = Field(description="Метка; заполнена только при status = classified.")
    label_id: Literal[0, 1, 2] | None = Field(description="Метка в кодировке датасета: 0 normal, 1 spam, 2 clickbait.")
    reason: Literal["low_confidence"] | None
    confidence: Probability = Field(description="Вероятность наиболее вероятного класса.")
    threshold: Probability = Field(description="Порог, с которым сравнивалась уверенность.")
    scores: Scores
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class Metrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    f1_macro: Probability | None = None
    accuracy: Probability | None = None


class Policy(BaseModel):
    tau: Probability
    tau_spam: Probability


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    model_version: str = Field(description="Идентификатор вида имя@версия, как в ModelVersions контракта.")
    trained_at: datetime | None = Field(description="Дата обучения (UTC).")
    framework: str
    sklearn_version: str | None
    artifact: str
    artifact_sha256: str
    classes: dict[str, Label]
    metrics: Metrics | None = Field(
        description="Метрики test из экспорта; протокол и ограничения указаны в dataset/config."
    )
    dataset: dict | None = None
    config: dict | None = None
    policy: Policy
