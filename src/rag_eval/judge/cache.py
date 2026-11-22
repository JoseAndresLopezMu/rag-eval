import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError

from rag_eval.judge.base import Judge, Judgement, Prompt


class CachedJudge:
    """Stores each judgement on disk, keyed by model, prompt and output schema.

    Re-running an evaluation then costs nothing and returns identical scores,
    even if the remote model would answer differently today.
    """

    def __init__(self, inner: Judge, directory: Path) -> None:
        self._inner = inner
        self._dir = directory
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    def ask[J: Judgement](self, prompt: Prompt, schema: type[J]) -> J:
        path = self._path(prompt, schema)
        if path.exists():
            try:
                cached = schema.model_validate_json(path.read_text(encoding="utf-8"))
            except ValidationError:
                pass  # unreadable entry: ask again and overwrite it
            else:
                self.hits += 1
                return cached

        self.misses += 1
        result = self._inner.ask(prompt, schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)
        return result

    def _path(self, prompt: Prompt, schema: type[Judgement]) -> Path:
        key_data = {
            "model": self._inner.model,
            "name": prompt.name,
            "version": prompt.version,
            "system": prompt.system,
            "user": prompt.user,
            "schema": schema.__name__,
        }
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return self._dir / key[:2] / f"{key}.json"
