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
│   ├── training/export_ensemble.py ←  экспорт готового ансамбля → models/ensemble/manifest.json
│   ├── tests/                     ←   тесты API через TestClient
│   ├── notebooks/                 ←   запуск из Jupyter/Colab (uvicorn + nest_asyncio)
│   ├── models/                    ←   артефакты модели (не в git)
│   └── Dockerfile
├── Datasets/
│   ├── df.csv                     ← текстовый датасет (Git LFS): text, is_spam1_or_clickbait2 (0 обычное, 1 спам, 2 кликбейт)
│   ├── dataset_description.md/html ← описание данных и результаты моделей
│   ├── task1_report.md/html       ← короткий отчёт для преподавателя
│   ├── notebooks/                ← model.ipynb, model_mlflow.ipynb, model_rnn.ipynb
│   ├── scripts/                  ← EDA, загрузчик, перевод, рендер отчёта, requirements.txt
│   └── outputs/                  ← графики, модели, CSV/NPY, HTML-профили, mlflow.db, test_model.py, eda/
├── ConceptAndArchitecture/
│   ├── до ревью/                  ← версия 1 (история): docx, pptx, прототип audio_module
│   └── после ревью/               ← версия 2 — актуальный концепт, архитектура, OpenAPI-контракт
└── CasesAndTeams/                 ← описание кейсов и состав команд курса
```

## Как запустить

Нужны Docker Desktop и комплект ансамбля в `inference-api/models/ensemble/`. Все команды выполняются из корня репозитория. В этой рабочей копии комплект подготовлен; он исключён из Git. На другом компьютере передайте его отдельно или выполните экспорт по [инструкции сервиса](inference-api/README.md).

```bash
# В .env: MODEL_FILE=ensemble/manifest.json
docker compose up -d --build inference-api
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

Отдельный LR-тренер для экспериментов (не создаёт ансамбль и не переключает API):

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

Учебные документы: [Model Card](MODEL_CARD.md) · [BPMN retraining-цикла — просмотр](Reports/2026-10-18/bpmn/retraining_cycle.html) · [BPMN — исходник](Reports/2026-10-18/bpmn/retraining_cycle.bpmn) · [Poetry, линтеры и pre-commit — самостоятельная инструкция](Reports/2026-10-18/homework_production_manual.md).

| Часть | Статус |
|---|---|
| Концепт и архитектура v2 после перекрёстного ревью | ✅ готово, ждёт согласования командой |
| Текстовый датасет `df.csv` (Git LFS, ~380 МБ: 642 тыс. строк, после удаления дублей 456 тыс.) | ✅ последняя версия от 07.10.2026 |
| «Модель 2»: TF-IDF word+char → LogisticRegression, эксперименты в MLflow | ✅ ноутбук и pytest-проверки (`Datasets/outputs/test_model.py`) |
| Сервис инференса `inference-api` (FastAPI, Swagger, pydantic-валидация) | ✅ RNN/CNN из второго ноутбука + сохранённая лучшая LR из первого; веса заново по dev; фактические метрики и состав через `/model-info` |
| Воспроизводимое обучение в контейнере (`trainer`) | ✅ конфигурации из ноутбука: baseline, exp1–exp5 |
| ASR (GigaAM-v3 + Silero VAD + определение языка) | ⏳ прототип `audio_module` в «до ревью», в сервис не подключён |
| Аудио-API `/v1/classifications` по контракту `04-service-contract.openapi.yaml` | ⏳ не начато |
| Telegram-бот (aiogram) | ⏳ не начато |
| TTS-обогащение (Piper), калибровка, holdout реальных голосовых | ⏳ не начато |

Важно: метрики модели на `df.csv` **завышены** (классы взяты из разных источников, сплит случайный; ревью, R-36). Итоговое качество проверяется на реальных голосовых (концепт v2, `05-mlops-and-validation.md`).
