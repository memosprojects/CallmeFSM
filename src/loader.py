import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptItem


def load_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in '{path}' at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read file '{path}': {exc}") from exc


def _format_pydantic_error(prefix: str, exc: ValidationError) -> str:
    details = [
        f"{' -> '.join(str(p) for p in e['loc'])}: {e['msg']}"
        for e in exc.errors()
    ]
    return f"{prefix}: {'; '.join(details)}"


def _ensure_json_array(raw_data: Any, path: Path, label: str) -> list[Any]:
    if not isinstance(raw_data, list):
        raise ValueError(f"{label} must be a JSON array in file: {path}")
    return raw_data


def _ensure_unique_function_names(functions: list[FunctionDefinition]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fn in functions:
        if fn.name in seen:
            duplicates.add(fn.name)
        seen.add(fn.name)
    if duplicates:
        raise ValueError(f"Duplicate function names found: {', '.join(sorted(duplicates))}")


def parse_function_definitions(path: Path) -> list[FunctionDefinition]:
    raw_data = load_json_file(path)
    raw_items = _ensure_json_array(raw_data, path, "Function definitions")
    try:
        functions = [FunctionDefinition.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise ValueError(
            _format_pydantic_error(f"Invalid function definition in '{path}'", exc)
        ) from exc
    _ensure_unique_function_names(functions)
    return functions


def parse_prompt_items(path: Path) -> list[PromptItem]:
    raw_data = load_json_file(path)
    raw_items = _ensure_json_array(raw_data, path, "Prompt input")
    try:
        return [PromptItem.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise ValueError(
            _format_pydantic_error(f"Invalid prompt input in '{path}'", exc)
        ) from exc
