"""Проверка OpenAPI-контракта сервиса инференса.

1. Спецификация валидна по OpenAPI 3.1 (openapi-spec-validator).
2. Каждый пример запроса и ответа соответствует своей схеме.
3. Заведомо неверные ответы отклоняются схемой `Classification`.

Запуск из корня репозитория:
    pip install openapi-spec-validator pyyaml
    python tools/validate_openapi.py [путь к спецификации]
"""

import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from openapi_spec_validator import validate

DEFAULT_SPEC = (
    Path(__file__).resolve().parents[1]
    / "ConceptAndArchitecture"
    / "v2"
    / "04-service-contract.openapi.yaml"
)


def main() -> int:
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPEC
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    validate(spec)
    print(f"openapi-spec-validator: OK ({spec_path.name}, OpenAPI {spec['openapi']})")

    components = spec["components"]

    def resolve(obj):
        while isinstance(obj, dict) and "$ref" in obj:
            node = spec
            for key in obj["$ref"].lstrip("#/").split("/"):
                node = node[key]
            obj = node
        return obj

    def errors(schema, value):
        validator = Draft202012Validator(
            {**schema, "components": components}, format_checker=FormatChecker()
        )
        return [e.message for e in validator.iter_errors(value)]

    ok = True
    checked = 0

    def check_content(content, where):
        nonlocal ok, checked
        for media_type, media in content.items():
            for name, example in (media.get("examples") or {}).items():
                checked += 1
                errs = errors(media.get("schema", {}), resolve(example)["value"])
                print(("OK  " if not errs else "FAIL"), where, media_type, name)
                for message in errs:
                    print("     ", message)
                ok &= not errs

    for path, item in spec["paths"].items():
        for method, operation in item.items():
            body = resolve(operation.get("requestBody", {}))
            if body.get("content"):
                check_content(body["content"], f"{method.upper()} {path} request")
            for code, response in operation["responses"].items():
                response = resolve(response)
                if response.get("content"):
                    check_content(response["content"], f"{method.upper()} {path} {code}")

    classification = {"$ref": "#/components/schemas/Classification"}
    good = spec["paths"]["/v1/classifications"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["examples"]["classifiedSpam"]["value"]
    negatives = {
        "classified без label": dict(good, label=None),
        "not_classified с label=normal": dict(
            good, status="not_classified", label="normal", reason="no_speech"
        ),
        "label=fraud": dict(good, label="fraud"),
        "uncertain с reason=no_speech": dict(
            good, status="uncertain", label=None, reason="no_speech"
        ),
        "not_classified со scores/confidence": dict(
            good, status="not_classified", label=None, reason="no_speech"
        ),
    }
    for name, value in negatives.items():
        rejected = bool(errors(classification, value))
        print(("OK  " if rejected else "FAIL"), "негативный пример отклонён:", name)
        ok &= rejected

    print(f"примеров проверено: {checked}, негативных: {len(negatives)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
