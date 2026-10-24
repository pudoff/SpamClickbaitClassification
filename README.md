# Классификация спама и кликбейта в голосовых сообщениях Telegram

**Великолепная четвёрка**
Цель: Telegram-бот принимает голосовое, распознаёт речь и отвечает «🚫 Спам», «🎣 Кликбейт», «✅ Обычное» или «🤔 Не уверен».
Решения и план описаны в концепте v2: `ConceptAndArchitecture/после ревью/ConceptAndArchitecture/v2/` (начинать с `07-executive-summary.md`).

## Что где лежит

```
├── README.md                      ← этот файл: запуск, структура, статус
├── MODEL_CARD.md                  ← карточка модели rnn-cnn-lr: метрики, ограничения, риски
├── compose.yaml                   ← Docker Compose: inference-api + trainer (профиль train)
├── .env.example                   ← шаблон настроек (порт, модель, пороги, конфиг LR) → скопировать в .env
├── pyproject.toml, poetry.lock    ← Poetry (venv в .venv), настройки ruff и pytest
├── .pre-commit-config.yaml        ← ruff check + ruff format для inference-api/ и Datasets/scripts/
├── inference-api/                 ← FastAPI-сервис классификатора (описание в inference-api/README.md)
│   ├── app/                       ←   main.py (эндпоинты) · schemas.py (pydantic) · model.py (загрузка, политика)
│   │                              ←   ensemble.py (RNN/CNN, смесь, формат комплекта) · labels.py (классы)
│   │                              ←   artifacts.py (sha256, загрузка LR) · config.py · logs.py · text.py
│   ├── training/train.py          ←   отдельная LR из df.csv → models/model.joblib + model_meta.json
│   ├── training/export_ensemble.py ←  экспорт готового ансамбля → models/ensemble/manifest.json
│   ├── training/data.py, metrics.py ← чтение df.csv и метрики, общие для обоих скриптов
│   ├── tests/                     ←   тесты на игрушечных моделях; снимки контракта в tests/snapshots/
│   ├── notebooks/                 ←   запуск из Jupyter/Colab (uvicorn в фоне)
│   ├── models/                    ←   артефакты модели (не в git)
│   └── Dockerfile
├── Datasets/
│   ├── df.csv                     ← текстовый датасет (Git LFS): text, is_spam1_or_clickbait2 (0 обычное, 1 спам, 2 кликбейт)
│   ├── 1_EDA-TFIDF_LogisticRegression.ipynb ← EDA и лучшая TF-IDF/LR (входит в ансамбль)
│   ├── 2_RNN_CNN_LR.ipynb         ← RNN, char-CNN и ансамбль
│   ├── dataset_description.md     ← описание данных и результаты моделей
│   ├── Homework1_report.md        ← отчёт по первому заданию
│   ├── notebooks/                 ← ранние ноутбуки: model.ipynb, model_mlflow.ipynb, model_rnn.ipynb, model_rnn_2.ipynb
│   ├── scripts/                   ← загрузчик, фильтрация, grouped-сплит, EDA, перевод, тесты, requirements.txt
│   └── outputs/                   ← графики, CSV/NPY, HTML-профили, mlflow.db, test_model.py, eda/
├── Reports/                       ← снимок манифеста ансамбля для Model Card, BPMN-диаграмма
├── Переобучение.bpmn              ← BPMN процесса переобучения
├── ConceptAndArchitecture/
│   ├── до ревью/                  ← версия 1 (история): docx, pptx, прототип audio_module
│   └── после ревью/               ← версия 2 — актуальный концепт, архитектура, OpenAPI-контракт
└── CasesAndTeams/                 ← описание кейсов и состав команд курса
```

## Как запустить

Всё запускается в Docker из корня репозитория. Нужны **Docker Desktop** (должен быть запущен) и **Git LFS**. В системный Python ничего ставить не нужно.

### 1. Подготовка

```bash
git clone https://github.com/pudoff/SpamClickbaitClassification.git && cd SpamClickbaitClassification
git lfs pull                 # Datasets/df.csv (~380 МБ): нужен тренеру и экспорту, для самого API не обязателен
cp .env.example .env         # порт, модель, пороги; поменяйте при необходимости
```

### 2. Модель

Сервис по умолчанию грузит **ансамбль** из `inference-api/models/ensemble/` (`manifest.json`, `rnn.pt`, `char.pt`, `lr.joblib`, `vocab.json`). В Git его нет: возьмите каталог у команды или выполните экспорт по [инструкции сервиса](inference-api/README.md#1-подготовить-комплект-ансамбля).

Если комплекта нет, можно запустить сервис на отдельной LR. Обучите её тренером, затем в `.env` укажите `MODEL_FILE=model.joblib`:

```bash
docker compose --profile train run --rm trainer          # конфиг exp4 на всём df.csv → inference-api/models/model.joblib
# быстрая проверка (~1 мин): ... trainer python -m training.train --data=/data/df.csv --out=/models --config=baseline --max-rows=20000
```

LR, обученная другой версией scikit-learn (не 1.8.0, как в образе), не загрузится: сервис ответит 503, переобучите её тренером.

### 3. Запуск API

```bash
docker compose up -d --build inference-api
docker compose ps            # через 10–20 с (загрузка torch): Up … (healthy)
```

Откройте **http://localhost:8000/docs** — Swagger UI, где можно вызвать все эндпоинты (порт задаётся `API_PORT` в `.env`).

| Эндпоинт | Что делает |
|---|---|
| `POST /predict` | `{"text": "...", "threshold": 0.6}` → `status`, `label`, `confidence`, `scores` |
| `GET /healthcheck` | Сервис жив и модель загружена (иначе ответ 503 с причиной) |
| `GET /model-info` | Версия модели, дата обучения, метрики, конфигурация, пороги |

Если модели нет или комплект неполный, сервис всё равно стартует, но `/healthcheck` и `/predict` отвечают **503**, а причина видна в `detail` и в `docker compose logs inference-api`. Молча перейти на другую модель сервис не может.

```bash
docker compose logs -f inference-api                   # логи
docker compose down                                    # остановить
```

### 4. Проверки (тесты и линтеры)

```bash
docker compose build inference-api                                    # после правок кода: код копируется в образ
docker compose run --rm inference-api pytest -q -p no:cacheprovider   # тесты сервиса: 68 passed, 2 skipped
docker compose run --rm inference-api ruff check .                    # линтер
docker compose run --rm inference-api ruff format --check .           # формат
docker compose run --rm -v ./Datasets:/data:ro -w /data/scripts inference-api python -m pytest -q -p no:cacheprovider   # тесты Datasets/scripts
```

Два пропущенных теста сверяют структуру манифеста с эталоном в `Reports/`, которого нет внутри образа. С примонтированным репозиторием они выполняются: `docker compose run --rm -v .:/repo -w /repo inference-api python -m pytest -q -p no:cacheprovider` (70 passed).

Pre-commit (ruff check + format для `inference-api/` и `Datasets/scripts/`) настроен через Poetry, окружение Python 3.12 как в образе: `poetry install --with dev`, затем `poetry run pre-commit install`.

## Что сделано на данный момент

Учебные документы: [Model Card](MODEL_CARD.md) · [BPMN переобучения](Переобучение.bpmn) · [BPMN-диаграмма](Reports/diagram.bpmn) · [снимок манифеста ансамбля](Reports/2026-10-18/model_card/model_manifest_snapshot.json).

| Часть | Статус |
|---|---|
| Концепт и архитектура v2 после перекрёстного ревью | ✅ готово, ждёт согласования командой |
| Текстовый датасет `df.csv` (Git LFS, ~380 МБ: 642 тыс. строк, после удаления дублей 456 тыс.) | ✅ последняя версия от 07.10.2026 |
| «Модель 2»: TF-IDF word+char → LogisticRegression, эксперименты в MLflow | ✅ ноутбук и pytest-проверки (`Datasets/outputs/test_model.py`) |
| Сервис инференса `inference-api` (FastAPI, Swagger, pydantic-валидация) | ✅ RNN/CNN из второго ноутбука + сохранённая лучшая LR из первого; веса заново по dev; фактические метрики и состав через `/model-info` |
| Воспроизводимое обучение в контейнере (`trainer`) | ✅ конфигурации из ноутбука: baseline, exp1–exp5 |
| Рефакторинг `inference-api` без изменения поведения | ✅ общие модули (`labels`, `artifacts`, `logs`, `training/data`, `metrics`), экспорт по шагам; снимки контракта и вероятностей в тестах |
| ASR (GigaAM-v3 + Silero VAD + определение языка) | ⏳ прототип `audio_module` в «до ревью», в сервис не подключён |
| Аудио-API `/v1/classifications` по контракту `04-service-contract.openapi.yaml` | ⏳ не начато |
| Telegram-бот (aiogram) | ⏳ не начато |
| TTS-обогащение (Piper), калибровка, holdout реальных голосовых | ⏳ не начато |

Важно: метрики модели на `df.csv` **завышены** (классы взяты из разных источников, сплит случайный; ревью, R-36). Итоговое качество проверяется на реальных голосовых (концепт v2, `05-mlops-and-validation.md`).
