# inference-api

FastAPI-сервис классификации текста: **normal / spam / clickbait**. По умолчанию использует RNN и char-CNN из `Datasets/2_RNN_CNN_LR.ipynb` и сохранённую лучшую TF-IDF/LogisticRegression из первого ноутбука. Веса этого варианта выбраны заново по dev macro-F1. Версия, веса и метрики фактически загруженного комплекта видны через `/model-info`.

В исходном ансамбле второго ноутбука были веса 0.4/0.4/0.2, test accuracy **0.8669**, macro-F1 **0.8320**. Его LR не был сохранён; попытка повторного обучения не воспроизвела вероятности, поэтому эти метрики не приписываются новой версии. Вместо неподтверждённой копии используется готовая лучшая LR с тем же `split_id=8fcc58bc...`, новые веса подбираются только на dev, затем test оценивается один раз.

Метрики относятся к очищенному текстовому корпусу: почти-дубликаты/общие источники исключены не полностью, качество ASR и реальных голосовых отдельно не проверено. Старые файлы непосредственно в `Datasets/outputs/` относятся к другому протоколу; они не используются экспортом. Аудиоконвейер VAD → ASR пока не подключён.

Проверенный экспорт **`rnn-cnn-lr@ensemble-20261018`**: веса **RNN 0.5 / CNN 0.2 / LR 0.3**, dev macro-F1 **0.8330175**, test accuracy **0.8693539**, macro-F1 **0.8347251**, weighted-F1 **0.8691651**. Test содержит 86 830 записей. Эти показатели заново рассчитаны для текущего комплекта.

## 1. Подготовить комплект ансамбля

Комплект лежит в `inference-api/models/ensemble/` и в Git не хранится: на другой компьютер передайте каталог целиком или выполните экспорт. Одного `rnn_best.pt` недостаточно — нужны обе сети, словари, LR и манифест.

Для воспроизведения экспорта нужны:

- `Datasets/df.csv`, `Datasets/scripts/dataset_filtering.py`, `dataset_splitting.py`;
- `Datasets/2_RNN_CNN_LR.ipynb` для фиксации происхождения архитектур;
- `Datasets/outputs/validation/1_EDA-TFIDF_LogisticRegression/model.pkl` и его `split/summary.json` — сохранённая лучшая LR;
- `Datasets/outputs/validation/2_RNN_CNN_LR/`: `rnn_best.pt`, `char_best.pt`, `ensemble_weights.csv`, `model_comparison.csv`, `split/summary.json`, вероятности `proba_rnn_dev/test.npy`, `proba_char_dev/test.npy`.

Из корня репозитория в Docker Desktop:

```powershell
docker compose build inference-api
docker compose --profile train run --rm trainer python -m training.export_ensemble --data=/data/df.csv --source=/data/outputs/validation/2_RNN_CNN_LR --out=/models/ensemble --version=ensemble-20261018
```

Или локально: экспорт требует pandas и зависимостей из
[корневого Poetry-окружения](../README.md#корневое-окружение-для-тестов-и-экспорта)
либо `inference-api/requirements-dev.txt`. Минимальный Poetry-проект внутри
`inference-api` pandas не содержит. Пример после установки корневого окружения,
начиная из корня репозитория:

```powershell
Push-Location inference-api
poetry -P .. run python -m training.export_ensemble --threads 2
Pop-Location
```

При использовании venv выполните из `inference-api` ту же команду через
`../.venv/bin/python -m training.export_ensemble --threads 2` в Linux/macOS
или `..\.venv\Scripts\python.exe -m training.export_ensemble --threads 2` в Windows.

Экспорт восстанавливает фильтрацию и исходный fit/dev/test, проверяет совпадение `split_id` всех моделей, строит словари **только на fit** с прежним порядком ID. Если доступен прежний журнал очистки, его повторное использование защищено SHA-256 исходного CSV и отпечатком подготовленной таблицы. Модели повторно не обучаются. Все готовые компоненты остаются замороженными; перебираются только веса ансамбля на dev с шагом 0.1. Лучший macro-F1 выбирается до обращения к test-меткам. Операция занимает несколько минут и использует несколько гигабайт RAM.

Проверяется совпадение вероятностей RNN/CNN на 64 фиксированных примерах dev и test. Для расчёта dev/test-метрик используются сохранённые вероятности проверенных нейросетей и новые предсказания загруженной LR. При расхождении нейросетей экспорт останавливается. Историческая смесь не проверяется на совпадение, поскольку LR теперь другая; новая версия имеет собственные измеренные метрики. После изменения данных или правил очистки старые чекпойнты повторно использовать нельзя.

Комплект содержит `manifest.json`, `vocab.json`, `rnn.pt`, `char.pt`, `lr.joblib`. Оптимизаторы в него не входят. Манифест хранит версии, веса, протокол, происхождение, метрики и SHA-256 каждого компонента. sklearn для этого экспорта — **1.8.0**, PyTorch — **2.10.0**. Загрузчик проверяет версию sklearn и контрольные суммы; несовместимый или неполный комплект даёт `503`, без перехода на другую модель.

### Если API возвращает 503

HTTP 503 (`Service Unavailable`) здесь означает, что сервис не готов выполнять
предсказания: при старте не удалось загрузить модель. Сервер продолжает работать,
поэтому `/docs` открывается, но `/healthcheck`, `/model-info` и корректные запросы
к `/predict` получают 503. Docker после периода ожидания отмечает контейнер как
`unhealthy`. Точная причина записывается в ответ `/healthcheck` и в логи:

```bash
docker compose logs --tail=50 inference-api
```

Например, `FileNotFoundError: ... '/models/ensemble/manifest.json'` означает,
что манифеста нет по настроенному пути. Compose подключает локальный каталог
`inference-api/models` к `/models` внутри контейнера. Файлы модели не входят
в образ и не появляются при его пересборке; в `.gitignore` они исключены из Git.

**Вариант 1 — получить готовый комплект у команды.** Скопируйте целиком каталог
ансамбля с компьютера, где эта версия модели уже работает. Ожидаемая структура:

```text
inference-api/models/ensemble/
├── manifest.json
├── vocab.json
├── rnn.pt
├── char.pt
└── lr.joblib
```

Все файлы должны относиться к одному экспорту: манифест содержит их контрольные
суммы. Одного манифеста, одиночного `rnn_best.pt` или файлов из разных запусков
недостаточно. В `.env` проверьте строку `MODEL_FILE=ensemble/manifest.json`.
После копирования комплекта выполните из корня:

```bash
docker compose up -d --force-recreate inference-api
```

Команда применяет настройки `.env` и заново загружает модель. Пересборка образа
для замены весов не требуется. Если меняются только файлы модели, достаточно
`docker compose restart inference-api`: на лету сервис их не перечитывает.

**Вариант 2 — экспортировать комплект из результатов обучения.** Используйте
команду экспорта выше, если уже есть все перечисленные исходные артефакты
в `Datasets/outputs/validation/` и настоящий CSV, полученный через Git LFS.
Экспорт не обучает отсутствующие модели. Если есть только исходники репозитория,
сначала нужно получить результаты обучения у команды либо выполнить ноутбуки
и подготовить совместимый набор артефактов. После успешного экспорта перезапустите API.

**Вариант 3 — запустить только LR для отдельного эксперимента.** Если у вас уже
есть подходящая трёхклассовая `model.joblib`, положите её в `inference-api/models/`,
задайте `MODEL_FILE=model.joblib` в `.env` и пересоздайте контейнер командой выше.
Это отдельная LR-модель: метрики и поведение ансамбля к ней не относятся.

Проверка результата: откройте `http://localhost:8000/healthcheck` — ожидаются
HTTP 200 и `model_loaded=true`. Затем проверьте версию через `/model-info`.
Если 503 остаётся, смотрите новую причину: загрузчик также отклоняет комплект
при несовпадении версии sklearn, повреждении файлов или неверных контрольных суммах.

При локальном запуске Python действует тот же принцип: задайте `MODEL_PATH`
к реальному `manifest.json` (например, `models/ensemble/manifest.json` при запуске
из `inference-api`) и перезапустите uvicorn. Переменная `MODEL_FILE` используется
Compose; локальный Python сам её в `MODEL_PATH` не преобразует.

## 2. Запустить API и Swagger

В `.env` укажите:

```dotenv
MODEL_FILE=ensemble/manifest.json
API_PORT=8000
TAU=0.55
TAU_SPAM=0.65
```

Не копируйте `.env.example` поверх существующих личных настроек без просмотра. Из корня выполните:

```powershell
docker compose up -d --build inference-api
docker compose logs --tail=50 inference-api
```

Swagger UI: **http://localhost:8000/docs**, OpenAPI: `/openapi.json`, ReDoc: `/redoc`.

Локальный запуск без Docker поддерживает [два способа установки: Poetry и venv + requirements](../README.md#локальный-запуск-без-docker).
Команды установки для Windows и Linux/macOS приведены в корневом README.
Для самого API используйте отдельные `inference-api/pyproject.toml` и
`inference-api/poetry.lock`. После установки этого окружения, начиная из корня:

```powershell
Push-Location inference-api
$env:MODEL_PATH = 'models/ensemble/manifest.json'
poetry run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Остановка: Ctrl+C; затем Pop-Location
```

Для venv замените команду запуска на
`../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
в Linux/macOS или
`..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
в Windows. Команды выполняются из `inference-api`.

`TORCH_NUM_THREADS` по умолчанию равен 2; значение читается в `Settings` и проверяется при старте (меньше 1 — сервис не запустится, как и при `TAU` вне [0, 1]). Это CPU-инференс; CUDA и GPU не требуются. Модели загружаются один раз при старте, переводятся в eval, предсказания выполняются без градиентов.

## 3. Проверить результат

В Swagger последовательно откройте endpoint, нажмите **Try it out → Execute**:

1. `GET /healthcheck`: ожидается HTTP 200, `model_loaded=true`, `model_version="rnn-cnn-lr@ensemble-20261018"`.
2. `GET /model-info`: проверьте `framework="PyTorch + scikit-learn"`, `config.weights`, `config.selection_split="dev"`, версию, метрики и `dataset.split_id`. Сумма весов равна 1; фактические значения записаны экспортом в manifest.json.
3. `POST /predict`, например:

```json
{"text": "Вам одобрен кредит на 300 000 рублей! Перезвоните по номеру 8-800", "threshold": 0.65}
```

Ответ должен содержать `model_version="rnn-cnn-lr@ensemble-20261018"` и вероятности `normal`, `spam`, `clickbait` с суммой около 1. Конкретную метку для произвольного текста заранее не гарантируем.

4. Для проверки валидации отправьте `{"text":"123 !!!"}` — ожидается `422`.

Смотрите фактический ответ в блоке **Server response**. Нижние блоки **Responses / Example Value / Schema** описывают возможные ответы и не являются дополнительными результатами запроса.

PowerShell-проверки:

```powershell
Invoke-RestMethod http://localhost:8000/healthcheck
Invoke-RestMethod http://localhost:8000/model-info | ConvertTo-Json -Depth 10
$body = @{ text = 'Буду через десять минут, возьми хлеба'; threshold = 0.65 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Политика API сохранена: ниже порога → `status="uncertain"`, `label=null`, `reason="low_confidence"`; низкая уверенность не превращается в normal. Правила маршрутизации на ручную проверку из конца ноутбука в этот текстовый контракт не переносились. Клиент может задать `threshold`; иначе используются `TAU/TAU_SPAM`.

## Тесты

В отдельном окружении API pytest и httpx не установлены. Для тестов подготовьте
[корневое Poetry-окружение](../README.md#корневое-окружение-для-тестов-и-экспорта),
затем из корня репозитория:

```powershell
Push-Location inference-api
poetry -P .. run python -m pytest -q -p no:cacheprovider
Pop-Location
```

Альтернатива — venv с `requirements-dev.txt`: из `inference-api` выполните
`../.venv/bin/python -m pytest -q` в Linux/macOS или
`..\.venv\Scripts\python.exe -m pytest -q` в Windows.

Либо из корня выполните `docker compose run --rm inference-api pytest -q -p no:cacheprovider`:
Dockerfile пока устанавливает корневые группы `main,training,dev` и использует
корневой lock-файл. Тесты не требуют настоящих весов: создают небольшие временные
модели и проверяют обе ветки загрузки, взвешивание, сериализацию, API,
Swagger/OpenAPI, шаги экспорта на синтетическом макете каталогов и отказ при
повреждении компонентов. Реальная эквивалентность
сохранённому ноутбуку проверяется экспортом отдельно.

Характеризационные тесты (`tests/test_characterization.py`, снимки в `tests/snapshots/`) фиксируют контракт: `/openapi.json` побайтно, таблицу решений `decide()`, словарь метрик `train`, структуру манифеста и вероятности игрушечного ансамбля. Снимки перезаписываются только осознанно, при изменении контракта: `UPDATE_SNAPSHOTS=1 pytest`. Два теста сверяют структуру с эталоном `Reports/…/model_manifest_snapshot.json`; в образе его нет, поэтому там они пропускаются (`2 skipped`), а с примонтированным репозиторием выполняются.

## Структура кода

| Модуль | Что внутри |
|---|---|
| `app/main.py` | `create_app`, эндпоинты, `get_model` (503, пока модель не загружена; вызывается после валидации тела, поэтому 422 важнее 503), `build_model_info` |
| `app/model.py` | `load_model` (ансамбль или LR), `LoadedModel` (`name`, `version`, `model_version`, `trained_at`), политика `decide()` → `Decision` |
| `app/ensemble.py` | RNN/char-CNN, `encode_texts`, `blend`, `EnsemblePredictor`, формат комплекта (`FORMAT`, `COMPONENTS`, `BUNDLE_FILES`, `save_bundle`, `load_ensemble`) |
| `app/labels.py` | `LABELS`, `CLASS_IDS`, `LABEL_IDS`, `N_CLASSES` — единственное место с кодировкой классов |
| `app/artifacts.py` | `sha256_of`, `load_artifact` (`.joblib` / `.pkl`) |
| `app/config.py` | `Settings` из окружения: `MODEL_PATH`, `MODEL_META_PATH`, `TAU`, `TAU_SPAM`, `TORCH_NUM_THREADS` |
| `app/schemas.py` | pydantic-схемы контракта, тип `Probability` |
| `app/logs.py` | `configure_logging()` — один формат логов для сервиса и скриптов |
| `app/text.py` | нормализация по версиям: `clean_text` (`notebook-clean-text-v1`, побайтно как в ноутбуках, не менять) и `clean_text_v2` (`clean-text-v2`); `normalizer_of(meta)` выбирает версию модели |
| `training/train.py` | отдельная LR: конфиги, `build_pipeline`, CLI (`--normalizer`, `--test-sample`) |
| `training/export_ensemble.py` | экспорт ансамбля по шагам: `load_sources` → `restore_models` → `verify_parity` → `select_weights` → `evaluate_test` → `build_meta` |
| `training/data.py` | `load_fixed_csv`, `deduplicate`, `import_dataset_scripts` |
| `training/metrics.py` | `evaluate()` — метрики для `train` и экспорта |
| `tests/conftest.py` | `TOY_DATA`, игрушечные LR и ансамбль, фабрика `make_settings` |

## LogisticRegression для отдельных экспериментов

Поддержка прежних `model.joblib`/`model.pkl` сохранена явно: задайте `MODEL_FILE=model.joblib` в `.env` или `MODEL_PATH=models/model.joblib` при локальном запуске. Это переключение на LR, а не ансамбль.

`python -m training.train` по-прежнему обучает **только LR**, а его исторический случайный split отличается от нового протокола ноутбуков. Этот тренер не создаёт ансамбль и не заменяет описанный экспорт. По умолчанию Compose/API ожидают `ensemble/manifest.json`.

Тренер обучает на нормализации `clean-text-v2` (флаг `--normalizer`) и записывает её в `model_meta.json` → `config.normalizer`; сервис берёт версию оттуда, неизвестная версия даёт 503. Флаг `--test-sample` дополнительно сохраняет `test_sample.pkl` (91 110 строк: `text_raw`, `text_clean`, `label`, `n_words`) для `Datasets/outputs/test_model.py`.

## Нормализация текста

| Версия | Функция | Где используется |
|---|---|---|
| `notebook-clean-text-v1` | `clean_text` | ноутбуки, выпущенный ансамбль, LR без записи `config.normalizer` |
| `clean-text-v2` | `clean_text_v2` | тренер LR по умолчанию |

v2 приводит текст к нижнему регистру до масок, поэтому ссылки КАПСОМ (`HTTP://…`, `WWW.…`) тоже становятся `url`. На всём `df.csv` обе версии дают одинаковый результат (ссылок КАПСОМ там нет), так что переобучение на v2 не меняет модель; разница проявляется только на реальных входящих сообщениях. Ансамбль перейдёт на v2 после переобучения ноутбуков: экспорт принимает `--normalizer=clean-text-v2`.
