from collections.abc import Callable, Mapping
from typing import Any

from rag_eval.judge.base import JudgeError, Judgement, Prompt

Handler = Callable[[Prompt], Mapping[str, Any]]


class FakeJudge:
    """Judge for tests: answers each prompt with a handler chosen by prompt name."""

    def __init__(self, handlers: Mapping[str, Handler], model: str = "fake") -> None:
        self._handlers = handlers
        self._model = model
        self.calls: list[Prompt] = []

    @property
    def model(self) -> str:
        return self._model

    def ask[J: Judgement](self, prompt: Prompt, schema: type[J]) -> J:
        self.calls.append(prompt)
        handler = self._handlers.get(prompt.name)
        if handler is None:
            raise JudgeError(f"no fake handler for prompt {prompt.name!r}")
        return schema.model_validate(handler(prompt))
