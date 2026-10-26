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

Если комплекта нет, можно запустить сервис на отдельной LR (TF-IDF + LogisticRegression). Обучите её тренером, затем в `.env` укажите `MODEL_FILE=model.joblib`:

```bash
# exp4 на всём df.csv, ~15 мин → inference-api/models/model.joblib + model_meta.json (+ test_sample.pkl для тестов модели)
docker compose --profile train run --rm trainer python -m training.train --data=/data/df.csv --out=/models --config=exp4 --test-sample
# быстрая проверка (~1 мин): ... --config=baseline --max-rows=20000
```

Результат exp4 (24.10.2026, sklearn 1.8.0): accuracy **0.8326**, macro-F1 **0.8038** на случайном отложенном сплите (метрики завышены, см. ниже). LR, обученная другой версией scikit-learn (не 1.8.0, как в образе), работает ненадёжно: переобучите её тренером.

Нормализация текста версионируется: тренер обучает на `clean-text-v2` (ссылки маскируются в любом регистре) и пишет версию в `model_meta.json` → `config.normalizer`; сервис применяет к запросам ту же версию. Выпущенный ансамбль и старые LR без этой записи работают на прежней `notebook-clean-text-v1`.

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
docker compose run --rm inference-api pytest -q -p no:cacheprovider   # тесты сервиса: 82 passed, 2 skipped
docker compose run --rm inference-api ruff check .                    # линтер
docker compose run --rm inference-api ruff format --check .           # формат
docker compose run --rm -v ./Datasets:/data:ro -w /data/scripts inference-api python -m pytest -q -p no:cacheprovider   # тесты Datasets/scripts
# тесты качества «Модели 2» на артефактах тренера (нужен --test-sample при обучении, ~2 мин)
docker compose run --rm -v ./Datasets:/data:ro -v ./inference-api/models:/artifacts:ro -e TEST_MODEL_DIR=/artifacts -w /data/outputs inference-api python -m pytest -q -p no:cacheprovider test_model.py
```

Два пропущенных теста сервиса сверяют структуру манифеста с эталоном в `Reports/`, которого нет внутри образа. С примонтированным репозиторием они выполняются: `docker compose run --rm -v .:/repo -w /repo inference-api python -m pytest -q -p no:cacheprovider` (84 passed).

`test_model.py` на exp4 от 24.10.2026: 3 passed, 3 failed. Падения — найденные слабые места модели, а не ошибки кода: добавленная ссылка меняет 16 % предсказаний (порог 10 %), очень длинный повторённый кликбейт уходит в «обычное», на коротких репликах macro-F1 0.56 (70 % от общего, требуется 85 %).

Pre-commit (ruff check + format для `inference-api/` и `Datasets/scripts/`) настроен через Poetry, окружение Python 3.12 как в образе: `poetry install --with dev`, затем `poetry run pre-commit install`.
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

Учебные документы: [Model Card](MODEL_CARD.md) · [BPMN переобучения](Переобучение.bpmn) · [BPMN-диаграмма](Reports/diagram.bpmn) · [снимок манифеста ансамбля](Reports/2026-10-18/model_card/model_manifest_snapshot.json).

| Часть | Статус |
|---|---|
| Концепт и архитектура v2 после перекрёстного ревью | ✅ готово, ждёт согласования командой |
| Текстовый датасет `df.csv` (Git LFS, ~380 МБ: 642 тыс. строк, после удаления дублей 456 тыс.) | ✅ последняя версия от 07.10.2026 |
| «Модель 2»: TF-IDF word+char → LogisticRegression, эксперименты в MLflow | ✅ ноутбук; переобучена тренером на sklearn 1.8.0 (exp4, нормализация v2); pytest-проверки `Datasets/outputs/test_model.py` запускаются на артефактах тренера |
| Сервис инференса `inference-api` (FastAPI, Swagger, pydantic-валидация) | ✅ RNN/CNN из второго ноутбука + сохранённая лучшая LR из первого; веса заново по dev; фактические метрики и состав через `/model-info` |
| Воспроизводимое обучение в контейнере (`trainer`) | ✅ конфигурации из ноутбука: baseline, exp1–exp5 |
| Рефакторинг `inference-api` без изменения поведения | ✅ общие модули (`labels`, `artifacts`, `logs`, `training/data`, `metrics`), экспорт по шагам; снимки контракта и вероятностей в тестах |
| Исправления после рефакторинга | ✅ маскирование ссылок КАПСОМ (`clean-text-v2`), `httpx2` для TestClient, загрузка моделей через joblib, `.gitignore`/`.dockerignore` |
| ASR (GigaAM-v3 + Silero VAD + определение языка) | ⏳ прототип `audio_module` в «до ревью», в сервис не подключён |
| Аудио-API `/v1/classifications` по контракту `04-service-contract.openapi.yaml` | ⏳ не начато |
| Telegram-бот (aiogram) | ⏳ не начато |
| TTS-обогащение (Piper), калибровка, holdout реальных голосовых | ⏳ не начато |

Важно: метрики модели на `df.csv` **завышены** (классы взяты из разных источников, сплит случайный; ревью, R-36). Итоговое качество проверяется на реальных голосовых (концепт v2, `05-mlops-and-validation.md`).
