import json
from collections.abc import Iterator

import httpx
import pytest

from rag_eval.adapters.chatbot_rag import AdapterError, ChatbotRagClient
from rag_eval.models import DatasetItem, RunItem

ITEM = DatasetItem(
    id="q1",
    question="¿Incluye la recuperación de datos?",
    reference_answer="Sí.",
    answerable=False,
)
CONTEXTS = [
    {"content": "Servicio de recuperación de datos.", "fileName": "pliego.pdf", "pageNumber": 7},
    {"content": "Soporte in situ.", "fileName": "oferta.docx", "pageNumber": None},
]


def ok(contexts: object = CONTEXTS) -> httpx.Response:
    return httpx.Response(200, json={"answer": "Sí.", "sources": [], "contexts": contexts})


def make_client(
    responses: list[httpx.Response | Exception],
) -> tuple[ChatbotRagClient, list[httpx.Request], list[float]]:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    ticks: Iterator[float] = iter([10.0, 12.5] * 10)
    client = ChatbotRagClient(
        "https://bot.example/api/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        clock=lambda: next(ticks),
    )
    return client, requests, sleeps


def test_asks_for_contexts_and_maps_the_response() -> None:
    client, requests, _ = make_client([ok()])
    run = client.answer(ITEM)

    (request,) = requests
    assert str(request.url) == "https://bot.example/api/chat"
    assert json.loads(request.content) == {"input": ITEM.question, "include_contexts": True}

    assert run.id == "q1"
    assert run.answer == "Sí."
    assert [(c.rank, c.source, c.page) for c in run.contexts] == [
        (1, "pliego.pdf", 7),
        (2, "oferta.docx", None),
    ]
    assert run.latency_ms == 2500


def test_missing_contexts_points_to_the_deployment() -> None:
    client, _, _ = make_client([httpx.Response(200, json={"answer": "Sí.", "sources": []})])
    with pytest.raises(AdapterError, match="include_contexts"):
        client.answer(ITEM)


def test_empty_contexts_are_valid() -> None:
    client, _, _ = make_client([ok(contexts=[])])
    assert client.answer(ITEM).contexts == ()


def test_malformed_context_is_reported() -> None:
    client, _, _ = make_client([ok(contexts=[{"content": "x"}])])
    with pytest.raises(AdapterError, match="q1: malformed context"):
        client.answer(ITEM)


def test_server_errors_are_retried() -> None:
    client, requests, sleeps = make_client([httpx.Response(503), httpx.ConnectError("down"), ok()])
    client.answer(ITEM)
    assert len(requests) == 3
    assert sleeps == [1.0, 2.0]


def test_application_errors_include_the_question_id_and_body() -> None:
    body = {"error": "(ENOTFOUND) tenant/user postgres.x not found"}
    client, requests, _ = make_client([httpx.Response(500, json=body)] * 3)
    with pytest.raises(AdapterError, match=r"q1: chatbot-rag returned 500: .*ENOTFOUND"):
        client.answer(ITEM)
    assert len(requests) == 3


def test_collect_keeps_order_and_reports_progress() -> None:
    items = [ITEM.model_copy(update={"id": f"q{i}"}) for i in range(3)]
    client, _, _ = make_client([ok(), ok(), ok()])
    seen: list[RunItem] = []
    runs = client.collect(items, on_progress=seen.append)
    assert [r.id for r in runs] == ["q0", "q1", "q2"]
    assert seen == runs
