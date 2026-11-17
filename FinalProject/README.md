# FinalProject — итоговые материалы курса «Интеграция ИИ в разработку»

Новые материалы итоговой защиты проекта **SpamClickbaitClassification**
(команда «Великолепная четвёрка»). Существующие файлы проекта не изменялись;
всё содержимое этой папки — новые файлы.

## Состав

| Путь | Что это |
|---|---|
| `README.md` | этот файл: навигация и инструкции запуска |
| `demo/` | Python-пакет демонстрационного пайплайна (данные, валидация, сплит, модели, политика, дрейф) |
| `data/demo_sample.csv` | стратифицированная выборка 15 000 строк из `Datasets/df.csv` для ноутбука/Colab |
| `experiments/run_experiments.py` | скрипт серии экспериментов E0–E4 + дрейф-тест, трекинг в MLflow |
| `experiments/EXPERIMENTS.md` | протокол и результаты экспериментов с выводами |
| `experiments/results/` | JSON/CSV/PNG результаты последнего запуска |
| `experiments/mlflow.db` | локальное хранилище MLflow (sqlite, как в Datasets/outputs/mlflow.db) |
| `notebooks/demo_solution.ipynb` | самодостаточный демонстрационный ноутбук (работает локально и в Colab), выполнен с outputs |
| `tests/test_pipeline.py` | pytest-тесты демонстрационного пайплайна |
| `docs/ARCHITECTURE.md` | архитектура: реализовано сейчас + целевая схема, обоснование решений |
| `docs/API_CONTRACT.md` | контракт текущего текстового API и целевого аудио-API |
| `docs/DATA_VALIDATION.md` | валидация данных, найденные проблемы и риски |
| `docs/TEST_PLAN.md` | тест-план: уровни, сценарии, ворота качества, найденные проблемы |
| `docs/DEPLOYMENT.md` | деплой и контейнеризация + честный учебный маршрут запуска |
| `docs/MONITORING.md` | план мониторинга: метрики, ≥3 аллерта, действия при отклонениях |
| `report/Отчёт_итоговый_проект.docx` | отчёт 5–7 страниц |
| `presentation/Защита_SpamClickbaitClassification.pptx` | презентация 14 слайдов |
| `speech/Выступление_5_минут.md` | текст выступления на 5 минут |

Опорные материалы проекта (не изменялись): корневой `README.md`, `MODEL_CARD.md`,
`inference-api/` (FastAPI-сервис, Dockerfile, 84 теста), `Datasets/` (ноутбуки, EDA,
скрипты очистки/сплитов), `ConceptAndArchitecture/после ревью/.../v2/` (концепт,
архитектура, OpenAPI-контракт, MLOps-профиль).

## Быстрый запуск (локально, Windows)

Используется готовое окружение проекта `venv` (Python 3.11, scikit-learn, MLflow,
FastAPI, pytest уже установлены). Из корня репозитория:

```powershell
# 1. Тесты демонстрационного пайплайна (~10 с)
.\venv\Scripts\python.exe -m pytest FinalProject\tests -q

# 2. Серия экспериментов E0–E4 + дрейф (~10–15 мин, CPU, 60 тыс. строк)
.\venv\Scripts\python.exe FinalProject\experiments\run_experiments.py

# 3. Ноутбук-демонстрация (уже выполнен с outputs; для перезапуска):
.\venv\Scripts\python.exe -m jupyter lab FinalProject\notebooks\demo_solution.ipynb
```

Результаты экспериментов: `FinalProject/experiments/results/` и MLflow UI:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///FinalProject/experiments/mlflow.db --port 5000
# http://localhost:5000
```

## Запуск в Colab

1. Откройте `notebooks/demo_solution.ipynb` в Google Colab.
2. Загрузите `data/demo_sample.csv` (ячейка загрузки предлагает `files.upload()`),
   либо ноутбук автоматически найдёт его в репозитории при локальном запуске.
3. Выполните все ячейки: установка зависимостей в Colab выполняется первой ячейкой.

Полный датасет `Datasets/df.csv` (~380 МБ, Git LFS) для Colab не требуется:
демо работает на репрезентативной выборке 15 000 строк.

## Демонстрация сервиса (существующая часть проекта)

```powershell
docker compose up -d --build inference-api   # нужен комплект моделей, см. inference-api/README.md
# Swagger: http://localhost:8000/docs
```

Без Docker и без весов ансамбля минимальный сценарий «обучение → API → запрос»
показан в ноутбуке (FastAPI + TestClient, по образцу практики В12 курса).