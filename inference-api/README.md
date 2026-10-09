# inference-api

FastAPI-сервис текстового классификатора **spam / clickbait / normal**. Это текстовая часть `inference-api` из архитектуры v2 (`ConceptAndArchitecture/после ревью/.../03-final-architecture.md`). Аудиоконвейер (VAD → ASR GigaAM → классификатор) будет подключён позже к этому же классификатору.

## Быстрый старт (из корня репозитория)

```bash
cp .env.example .env                                  # по желанию: порт, пороги, конфигурация обучения
docker compose --profile train run --rm trainer       # 1. обучить модель -> inference-api/models/
docker compose up -d --build                          # 2. поднять сервис
```

Swagger UI: **http://localhost:8000/docs** (ReDoc: `/redoc`).

Готовый `model.pkl` из ноутбука можно использовать без тренера: положите его в `inference-api/models/` и задайте `MODEL_FILE=model.pkl` в `.env`.

## Эндпоинты

| Метод и путь | Назначение |
|---|---|
| `POST /predict` | Инференс: `{"text": "...", "threshold": 0.6}` → `status`, `label`, `confidence`, `scores` |
| `GET /healthcheck` | `200 {"status": "ok"}`, если модель загружена; иначе `503` с причиной |
| `GET /model-info` | Версия, дата обучения, метрики, конфигурация, хэши модели и датасета, пороги |

Пример:

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "Вам одобрен кредит на 300 000 рублей! Перезвоните по номеру 8-800-555-35-35"}'
```

Ответ модели `exp4` (значения округлены):

```json
{"status": "classified", "label": "spam", "label_id": 1, "reason": null,
 "confidence": 0.956, "threshold": 0.65,
 "scores": {"spam": 0.956, "clickbait": 0.001, "normal": 0.043},
 "model_version": "tfidf-logreg@1"}
```

> Windows / Git Bash: curl может передать кириллицу из аргументов не в UTF-8, тогда сервис отвечает `400 There was an error parsing the body`. Сохраните JSON в файл в UTF-8 и отправьте его через `--data-binary @body.json` или пользуйтесь Swagger UI.

**Политика решения** (концепт §6): если вероятность лучшего класса ниже порога (`TAU=0.55`, для спама `TAU_SPAM=0.65`), ответ будет `status: "uncertain"`, `label: null`. Низкая уверенность никогда не превращается в `normal`.

**Валидация** (pydantic): `threshold` — `Field(ge=0, le=1)`; `text` — от 1 до 20 000 символов и хотя бы одна буква; лишние поля запрещены. Нарушение любого правила даёт ответ `422`.

## Обучение

`training/train.py` повторяет конфигурации из `Datasets/model_mlflow.ipynb`: `baseline` и `exp1`…`exp5`. По умолчанию берётся `exp4` — лучший прогон в `mlflow.db`. Смена конфигурации и версии: `TRAIN_CONFIG` и `MODEL_VERSION` в `.env`. Быстрый пробный прогон:

```bash
docker compose --profile train run --rm trainer python -m training.train \
    --data=/data/df.csv --out=/models --config=baseline --max-rows=20000
```

Метрики в `model_meta.json` получены на случайном сплите `df.csv` и **завышены** (ревью, R-36). Это не качество на голосовых.

## Тесты и линтер

```bash
docker compose run --rm inference-api pytest -q      # TestClient, игрушечная модель
docker compose run --rm inference-api ruff check .
```

Запуск из Jupyter/Colab через uvicorn + `nest_asyncio` показан в `notebooks/serve_in_notebook.ipynb`. Локально пакеты ставятся только в `.venv` в корне проекта.
