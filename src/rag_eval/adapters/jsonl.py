import json
from pathlib import Path

from pydantic import ValidationError

from rag_eval.models import DatasetItem, Record, RunItem


class DataFileError(ValueError):
    pass


def read_dataset(path: Path) -> tuple[DatasetItem, ...]:
    return _read(path, DatasetItem)


def read_run(path: Path) -> tuple[RunItem, ...]:
    return _read(path, RunItem)


def _read[M: Record](path: Path, model: type[M]) -> tuple[M, ...]:
    items: list[M] = []
    seen: dict[str, int] = {}

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataFileError(f"{where}: invalid JSON ({e.msg})") from e
            if not isinstance(raw, dict):
                raise DataFileError(f"{where}: expected a JSON object")

            item_id = raw.get("id")
            label = f"{where} (id={item_id})" if isinstance(item_id, str) else where

            try:
                item = model.model_validate(raw)
            except ValidationError as e:
                raise DataFileError(f"{label}: {_describe(e)}") from e

            if item.id in seen:
                raise DataFileError(f"{label}: duplicate id, first seen on line {seen[item.id]}")
            seen[item.id] = line_no
            items.append(item)

    if not items:
        raise DataFileError(f"{path}: file has no records")
    return tuple(items)


def _describe(error: ValidationError) -> str:
    parts = []
    for err in error.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "record"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
