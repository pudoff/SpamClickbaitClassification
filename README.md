# Классификация спама и кликбейта в голосовых сообщениях Telegram

**Великолепная четвёрка**
Цель: Telegram-бот принимает голосовое, распознаёт речь и отвечает «🚫 Спам», «🎣 Кликбейт», «✅ Обычное» или «🤔 Не уверен».
Решения и план описаны в концепте v2: `ConceptAndArchitecture/после ревью/ConceptAndArchitecture/v2/` (начинать с `07-executive-summary.md`).

## Что где лежит

```
├── README.md                      ← этот файл: запуск, структура, статус
├── compose.yaml                   ← Docker Compose: inference-api + trainer (профиль train)
├── .env.example                   ← шаблон настроек (порт, пороги, конфиг обучения) → скопировать в .env
├── inference-api/                 ← FastAPI-сервис классификатора (описание в inference-api/README.md)
│   ├── app/                       ←   main.py (эндпоинты) · schemas.py (pydantic) · model.py · text.py · config.py
│   ├── training/train.py          ←   обучение модели из df.csv → models/model.joblib + model_meta.json
│   ├── tests/                     ←   тесты API через TestClient
│   ├── notebooks/                 ←   запуск из Jupyter/Colab (uvicorn + nest_asyncio)
│   ├── models/                    ←   артефакты модели (не в git)
│   └── Dockerfile
├── Datasets/
│   ├── df.csv                     ← текстовый датасет (Git LFS): text, is_spam1_or_clickbait2 (0 обычное, 1 спам, 2 кликбейт)
│   ├── model_mlflow.ipynb         ← EDA, «Модель 2» (TF-IDF word+char → LogReg), эксперименты MLflow
│   ├── mlflow.db                  ← журнал экспериментов MLflow (SQLite)
│   └── test_model.py              ← pytest-проверки качества модели из ноутбука
├── ConceptAndArchitecture/
│   ├── до ревью/                  ← версия 1 (история): docx, pptx, прототип audio_module
│   └── после ревью/               ← версия 2 — актуальный концепт, архитектура, OpenAPI-контракт
└── CasesAndTeams/                 ← описание кейсов и состав команд курса
```

## Как запустить

Нужен только Docker Desktop. Все команды выполняются из корня репозитория.

```bash
cp .env.example .env                                  # необязательно: свои порт, пороги, конфигурация
docker compose --profile train run --rm trainer       # 1. обучить модель (конфигурация exp4, ~15 минут, ~6 ГБ RAM)
docker compose up -d --build                          # 2. поднять API
```

Откройте **http://localhost:8000/docs** — Swagger UI, где можно вызвать все эндпоинты.

| Эндпоинт | Что делает |
|---|---|
| `POST /predict` | `{"text": "...", "threshold": 0.6}` → `status`, `label`, `confidence`, `scores` |
| `GET /healthcheck` | Сервис жив и модель загружена (иначе ответ 503 с причиной) |
| `GET /model-info` | Версия модели, дата обучения, метрики, конфигурация, пороги |

Другие команды:

```bash
docker compose ps                                      # статус (healthy = модель загружена)
docker compose logs -f inference-api                   # логи
docker compose run --rm inference-api pytest -q        # тесты
docker compose run --rm inference-api ruff check .     # линтер
docker compose down                                    # остановить
```

Быстрое обучение на подвыборке (для проверки, около минуты):

```bash
docker compose --profile train run --rm trainer python -m training.train \
    --data=/data/df.csv --out=/models --config=baseline --max-rows=20000
```

**Локально без Docker** пакеты ставятся только в `.venv` проекта, в системный Python ничего не ставим:

```bash
python -m venv .venv
.venv/Scripts/pip install -r inference-api/requirements-dev.txt     # Windows (Linux/macOS: .venv/bin/pip)
cd inference-api && ../.venv/Scripts/python -m pytest -q
```

## Что сделано на данный момент

| Часть | Статус |
|---|---|
| Концепт и архитектура v2 после перекрёстного ревью | ✅ готово, ждёт согласования командой |
| Текстовый датасет `df.csv` (Git LFS, ~380 МБ: 642 тыс. строк, после удаления дублей 456 тыс.) | ✅ последняя версия от 07.10.2026 |
| «Модель 2»: TF-IDF word+char → LogisticRegression, эксперименты в MLflow | ✅ ноутбук и pytest-проверки (`Datasets/test_model.py`) |
| Сервис инференса `inference-api` (FastAPI, Swagger, pydantic-валидация) | ✅ `/predict`, `/healthcheck`, `/model-info`, тесты, Docker Compose; модель `exp4`: macro-F1 0.804, F1 кликбейта 0.643 |
| Воспроизводимое обучение в контейнере (`trainer`) | ✅ конфигурации из ноутбука: baseline, exp1–exp5 |
| ASR (GigaAM-v3 + Silero VAD + определение языка) | ⏳ прототип `audio_module` в «до ревью», в сервис не подключён |
| Аудио-API `/v1/classifications` по контракту `04-service-contract.openapi.yaml` | ⏳ не начато |
| Telegram-бот (aiogram) | ⏳ не начато |
| TTS-обогащение (Piper), калибровка, holdout реальных голосовых | ⏳ не начато |

Важно: метрики модели на `df.csv` **завышены** (классы взяты из разных источников, сплит случайный; ревью, R-36). Итоговое качество проверяется на реальных голосовых (концепт v2, `05-mlops-and-validation.md`).
