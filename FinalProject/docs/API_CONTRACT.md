# API / контракт сервиса

Два уровня контракта: **реализованный текстовый** (`inference-api`) и **целевой аудио**
(концепт v2, `04-service-contract.openapi.yaml`, валидирован `tools/validate_openapi.py`).

## 1. Реализованный текстовый контракт (inference-api)

Схемы: `inference-api/app/schemas.py` (pydantic v2, `extra="forbid"`).
Снимок OpenAPI зафиксирован characterization-тестом: `inference-api/tests/snapshots/openapi.json`
(изменение контракта требует осознанного обновления снимка `UPDATE_SNAPSHOTS=1 pytest`).
Swagger: `http://localhost:8000/docs`, ReDoc: `/redoc`, спецификация: `/openapi.json`.

### POST /predict

Запрос:

```json
{"text": "Вам одобрен кредит на 300 000 рублей! Перезвоните по номеру 8-800", "threshold": 0.65}
```

| Поле | Тип | Ограничения |
|---|---|---|
| `text` | string | 1–20 000 символов, обязательна хотя бы одна буква (иначе 422) |
| `threshold` | float \| null | [0, 1]; если не задан — политика сервиса `TAU`/`TAU_SPAM` |

Ответ 200:

```json
{
  "status": "classified",
  "label": "spam",
  "label_id": 1,
  "reason": null,
  "confidence": 0.91,
  "threshold": 0.65,
  "scores": {"spam": 0.91, "clickbait": 0.05, "normal": 0.04},
  "model_version": "rnn-cnn-lr@ensemble-20261018"
}
```

Инварианты (проверяются тестами):
- `status="classified"` ⇔ `label ≠ null`; при `uncertain` — `label=null`, `reason="low_confidence"`;
- сумма `scores` = 1 ± 1e-6; `confidence` = max(scores);
- порог для класса `spam` повышенный (`TAU_SPAM=0.65` против `TAU=0.55`);
- `model_version` вида `имя@версия` в каждом ответе.

Ошибки: **422** — невалидное тело (важнее 503: валидация выполняется до проверки модели);
**503** — модель не загружена/несовместима (причина в `detail` и логах); молчаливого
переключения на другую модель нет.

### GET /healthcheck

`{"status": "ok", "model_loaded": true, "model_version": "..."}`; при проблемах — 503
с `detail`. Используется Docker healthcheck.

### GET /model-info

Версия, дата обучения, framework, `artifact_sha256`, классы, метрики test из манифеста,
`dataset` (включая `split_id`), `config` (веса, нормализатор), `policy` (τ, τ_spam).

## 2. Аудиоконтракт

Учебный сервис принимает файлы в `POST /v1/transcriptions` и
`POST /v1/classifications`. Для кандидата голосового эксперимента
`/v1/classifications` использует пороги из `config.voice_policy` в метаданных.
[Действующий контракт и примеры запросов](../../inference-api/README.md#аудио-и-распознавание-речи)

Полный целевой контракт описан в
`ConceptAndArchitecture/после ревью/ConceptAndArchitecture/v2/04-service-contract.openapi.yaml`.

- `POST /v1/classifications` — `multipart/form-data`, поле `audio` (ogg/opus, m4a, mp3, wav;
  ≤ 20 МБ, ≤ 300 с), заголовки `X-API-Key`, `X-Request-Id`;
- ответ: `status` (`classified` / `uncertain` / `not_classified`), `label|null`, `reason`
  (`no_speech`, `speech_too_short`, `unsupported_language`, `low_confidence`, ...),
  `scores|null`, `transcript`, `models` (`asr_model`, `lid_model`, `classifier_model`, `decision_policy`);
- `GET /v1/health`, `GET /v1/service-info` (лимиты и поддерживаемые форматы);
- ошибки: 401 (ключ), 413 (величина), 415 (формат), 422 (декодирование/длина), 503 (перегрузка/таймаут, `Retry-After`).

После ASR используется тот же классификатор, что и в `/predict`.
Метки и семантика отказа совпадают, а голосовой кандидат может иметь собственные пороги.

## 3. Версионирование контракта

- OpenAPI-снимок в тестах защищает от случайного изменения полей;
- версии моделей — в манифесте и ответах (`имя@версия`); версия нормализации — в `config.normalizer`;
- целевой контракт версионируется в `04-service-contract.openapi.yaml` (v2 после ревью);
- ломающие изменения требуют обновления снимков, документации и согласования команды.
