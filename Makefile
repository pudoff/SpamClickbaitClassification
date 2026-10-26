.DEFAULT_GOAL := local
.PHONY: local lf

COMPOSE ?= docker compose
UP_OPTIONS := up --detach --build --force-recreate --remove-orphans

local: lf
	$(COMPOSE) $(UP_OPTIONS) inference-api

lf:
	@echo "Нормализация окончаний файлов (CRLF -> LF)"
	find . -type f \( -name "*.sh" -o -name "*.bash" \) \
		-not -path '*/node_modules/*' \
		-not -path '*/venv/*' \
		-not -path '*/.*' \
		-exec sed -i 's/\r$$//' {} +
