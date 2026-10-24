"""Единая настройка логов для сервиса и скриптов обучения. Тексты запросов в логи не пишем (концепт §8)."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)
