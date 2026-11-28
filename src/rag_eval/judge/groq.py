import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from rag_eval.judge.base import JudgeError, Judgement, Prompt

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

_RETRY_STATUS = {429, 500, 502, 503, 504}


class GroqJudge:
    """LLM judge over Groq's OpenAI-compatible chat completions API.

    Temperature 0 and JSON mode. If the model returns JSON that does not match
    the schema, it gets one chance to fix it, with the validation error shown.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise JudgeError("GROQ_API_KEY is empty")
        if not api_key.isascii() or any(c.isspace() or not c.isprintable() for c in api_key):
            # Never echo the key: it may be real and only slightly mangled.
            raise JudgeError(
                "GROQ_API_KEY contains spaces or non-ASCII characters; "
                "check that the whole key was pasted and nothing else"
            )
        self._model = model
        self._client = client or httpx.Client(timeout=60)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_retries = max_retries
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def ask[J: Judgement](self, prompt: Prompt, schema: type[J]) -> J:
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        content = self._complete(messages)
        try:
            return schema.model_validate_json(content)
        except ValidationError as first_error:
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That JSON is invalid: {_short(first_error)}. "
                    "Reply again with only the corrected JSON object.",
                },
            ]
            content = self._complete(messages)
            try:
                return schema.model_validate_json(content)
            except ValidationError as e:
                raise JudgeError(
                    f"{prompt.name}: model returned invalid JSON twice: {_short(e)}"
                ) from e

    def _complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(GROQ_URL, json=body, headers=self._headers)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise JudgeError(f"Groq request failed: {e}") from e
                self._sleep(_backoff(attempt))
                continue

            if response.status_code in _RETRY_STATUS and attempt < self._max_retries:
                self._sleep(_retry_after(response) or _backoff(attempt))
                continue
            if response.status_code != 200:
                raise JudgeError(f"Groq returned {response.status_code}: {response.text[:300]}")
            return _message_content(response.json())

        raise AssertionError("unreachable")


def _message_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise JudgeError(f"unexpected Groq response: {json.dumps(payload)[:300]}") from e
    if not isinstance(content, str):
        raise JudgeError("Groq response has no text content")
    return content


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 30))


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _short(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in error.errors()
    )[:300]
