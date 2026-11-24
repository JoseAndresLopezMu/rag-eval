import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from rag_eval.models import Context, DatasetItem, RunItem

_RETRY_STATUS = {429, 500, 502, 503, 504}


class AdapterError(RuntimeError):
    pass


class ChatbotRagClient:
    """Collects answers from chatbot-rag's /api/chat.

    Requests `include_contexts`, so the response carries the chunks the LLM
    actually saw (SPEC 7, option B). Without them faithfulness cannot be judged,
    so a response missing `contexts` is an error rather than an empty list.
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat"
        self._client = client or httpx.Client(timeout=120)
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    def answer(self, item: DatasetItem) -> RunItem:
        start = self._clock()
        payload = self._post(item.id, {"input": item.question, "include_contexts": True})
        latency_ms = (self._clock() - start) * 1000
        return _to_run_item(item.id, payload, latency_ms)

    def collect(
        self,
        items: Iterable[DatasetItem],
        on_progress: Callable[[RunItem], None] | None = None,
    ) -> list[RunItem]:
        results = []
        for item in items:
            run_item = self.answer(item)
            results.append(run_item)
            if on_progress is not None:
                on_progress(run_item)
        return results

    def _post(self, item_id: str, body: dict[str, Any]) -> Any:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self._url, json=body)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise AdapterError(f"{item_id}: request to {self._url} failed: {e}") from e
                self._sleep(2.0**attempt)
                continue
            if response.status_code in _RETRY_STATUS and attempt < self._max_retries:
                self._sleep(2.0**attempt)
                continue
            if response.status_code != 200:
                raise AdapterError(
                    f"{item_id}: chatbot-rag returned {response.status_code}: {response.text[:300]}"
                )
            return response.json()
        raise AssertionError("unreachable")


def _to_run_item(item_id: str, payload: Any, latency_ms: float) -> RunItem:
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        raise AdapterError(f"{item_id}: response has no 'answer' string")
    raw_contexts = payload.get("contexts")
    if not isinstance(raw_contexts, list):
        raise AdapterError(
            f"{item_id}: response has no 'contexts' list; "
            "is the include_contexts option deployed in chatbot-rag?"
        )
    try:
        contexts = tuple(
            Context(
                rank=rank,
                source=c["fileName"],
                page=c.get("pageNumber"),
                text=c["content"],
            )
            for rank, c in enumerate(raw_contexts, start=1)
        )
    except (KeyError, TypeError, ValueError) as e:
        raise AdapterError(f"{item_id}: malformed context in response: {e}") from e
    return RunItem(id=item_id, answer=payload["answer"], contexts=contexts, latency_ms=latency_ms)
