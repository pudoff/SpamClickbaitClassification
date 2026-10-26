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
│   ├── notebooks/                 ←   демонстрация API в Jupyter (uvicorn в потоке)
│   ├── pyproject.toml / poetry.lock ← отдельное окружение для работы API
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

Нужны Docker с Compose и комплект ансамбля в `inference-api/models/ensemble/`.
Все Docker-команды выполняются из корня репозитория. Комплект исключён из Git
и после клонирования отсутствует: получите его у команды или выполните экспорт
по [инструкции сервиса](inference-api/README.md#если-api-возвращает-503).
Без модели Swagger работает, но `/healthcheck` и корректные запросы к `/predict`
возвращают 503; после периода ожидания контейнер становится `unhealthy`.

Dockerfile пока использует **корневые** `pyproject.toml` и `poetry.lock`,
устанавливая группы `main,training,dev`. Отдельный `inference-api/poetry.lock`
предназначен для локального окружения API и в Docker-сборке пока не используется.
Poetry устанавливается только в стадии сборки, на компьютере нужен лишь Docker.

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

## Локальный запуск без Docker

Можно использовать **Poetry** или **обычный venv с pip и requirements**.
Для обоих вариантов нужен Python 3.12. Выберите один способ для своей `.venv`:
Poetry и pip не следует использовать для параллельного управления одним окружением.
Все команды ниже начинаются из корня репозитория.

### Вариант 1: Poetry

Для запуска только API используйте отдельный проект в `inference-api`.
Нужен установленный Poetry 2.5.1. Из корня перейдите в папку сервиса и настройте
локальное окружение:

```text
cd inference-api
poetry config virtualenvs.in-project true --local
```

Выберите Python: `poetry env use python3.12` в Linux/macOS или
`poetry env use 3.12` в Windows. Затем, оставаясь в `inference-api`:

```text
poetry sync
poetry run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Окружение создаётся в `inference-api/.venv`, версии берутся из
`inference-api/poetry.lock`. Здесь только зависимости работающего API:
pytest, Ruff, pandas и Jupyter в этот набор не входят.

### Вариант 2: venv + requirements

Poetry не требуется. Сначала создаётся `.venv`, затем устанавливается CPU-версия
PyTorch и зависимости сервиса. Используется **`inference-api/requirements-dev.txt`**;
корневой `requirements.txt` — исторический список другого окружения.

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r inference-api/requirements-dev.txt
.venv/bin/python -m pytest -q
cd inference-api
../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r inference-api/requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
cd inference-api
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Активация окружения не нужна: команды явно используют Python из `.venv`.
В этом варианте pip использует requirements, а `poetry.lock` не применяется:
версии некоторых инструментов и транзитивных пакетов могут отличаться от Poetry.

В обоих вариантах Swagger доступен на **http://127.0.0.1:8000/docs**, остановка —
`Ctrl+C`. Для предсказаний нужен комплект в `inference-api/models/ensemble/`;
без него сервер работает, но возвращает 503 при обращении к модели.
Локальный запуск читает `MODEL_PATH` (по умолчанию `models/ensemble/manifest.json`
относительно `inference-api`), а `MODEL_FILE` из `.env` используется Compose.
Если Docker API уже занимает порт 8000, остановите его командой
`docker compose stop inference-api` из корня или укажите локальному uvicorn `--port 8001`.

Вариант с `requirements-dev.txt` также покрывает тесты и экспорт ансамбля.
Для тех же задач через Poetry используйте корневое окружение ниже.
Полное обучение в исследовательских ноутбуках требует дополнительных библиотек;
см. [описание ноутбуков и скриптов](DEVELOPMENT.md).

### Корневое окружение для тестов и экспорта

Корневой Poetry-проект сохранён отдельно от минимального окружения API.
Из корня репозитория выберите Python 3.12 командой `poetry env use python3.12`
(Windows: `poetry env use 3.12`), затем выполните:

```text
poetry sync --only main,training,dev
poetry run python -m pytest -q
poetry run python -m pytest -q Datasets/scripts/test_dataset_filtering.py Datasets/scripts/test_dataset_splitting.py
```

Окружение располагается в корневой `.venv`, зависимости берутся из корневого
`poetry.lock`. Для экспорта из папки `inference-api` используйте
`poetry -P .. run python -m training.export_ensemble --threads 2`:
`-P ..` явно выбирает корневой проект с pandas.

## Что сделано на данный момент

Документы: [Model Card](MODEL_CARD.md) · [BPMN — исходник](Reports/diagram.bpmn) · [Окружение, ноутбуки и скрипты](DEVELOPMENT.md).

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

Ограничение метрик: классы в `df.csv` взяты из разных источников; даже текущее
разбиение с группировкой по нормализованному тексту не исключает все почти-дубликаты
и пересечения источников. Исторический LR-тренер использует случайный split.
Качество на реальных голосовых и влияние ASR отдельно не проверены
(концепт v2, `05-mlops-and-validation.md`).
