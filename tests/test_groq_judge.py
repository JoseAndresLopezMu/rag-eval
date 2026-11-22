import json
from collections.abc import Callable

import httpx
import pytest

from rag_eval.judge import prompts
from rag_eval.judge.base import CorrectnessJudgement, JudgeError
from rag_eval.judge.groq import GROQ_URL, GroqJudge

PROMPT = prompts.correctness("q", "a", "r")
VALID = '{"rationale": "ok", "correct": true}'


def completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def make_judge(
    responses: list[httpx.Response | Exception],
) -> tuple[GroqJudge, list[httpx.Request], list[float]]:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = GroqJudge("secret", model="m", client=client, max_retries=2, sleep=sleeps.append)
    return judge, requests, sleeps


def body(request: httpx.Request) -> dict[str, object]:
    data = json.loads(request.content)
    assert isinstance(data, dict)
    return data


def test_sends_prompt_with_deterministic_json_settings() -> None:
    judge, requests, _ = make_judge([completion(VALID)])
    assert judge.ask(PROMPT, CorrectnessJudgement).correct is True

    (request,) = requests
    assert str(request.url) == GROQ_URL
    assert request.headers["authorization"] == "Bearer secret"
    sent = body(request)
    assert sent["model"] == "m"
    assert sent["temperature"] == 0
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"] == [
        {"role": "system", "content": PROMPT.system},
        {"role": "user", "content": PROMPT.user},
    ]


def test_invalid_json_gets_one_repair_attempt() -> None:
    judge, requests, _ = make_judge([completion('{"correct": "maybe"}'), completion(VALID)])
    assert judge.ask(PROMPT, CorrectnessJudgement).correct is True

    repair = body(requests[1])["messages"]
    assert isinstance(repair, list)
    assert repair[2] == {"role": "assistant", "content": '{"correct": "maybe"}'}
    assert "invalid" in repair[3]["content"]


def test_invalid_json_twice_fails() -> None:
    judge, _, _ = make_judge([completion("not json"), completion("still not json")])
    with pytest.raises(JudgeError, match="invalid JSON twice"):
        judge.ask(PROMPT, CorrectnessJudgement)


def test_rate_limit_is_retried_honouring_retry_after() -> None:
    judge, requests, sleeps = make_judge(
        [httpx.Response(429, headers={"retry-after": "7"}), completion(VALID)]
    )
    judge.ask(PROMPT, CorrectnessJudgement)
    assert len(requests) == 2
    assert sleeps == [7.0]


def test_server_and_network_errors_back_off_exponentially() -> None:
    judge, _, sleeps = make_judge(
        [httpx.Response(503), httpx.ConnectError("down"), completion(VALID)]
    )
    judge.ask(PROMPT, CorrectnessJudgement)
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries() -> None:
    judge, requests, _ = make_judge([httpx.Response(503)] * 3)
    with pytest.raises(JudgeError, match="503"):
        judge.ask(PROMPT, CorrectnessJudgement)
    assert len(requests) == 3


@pytest.mark.parametrize("status", [400, 401])
def test_client_errors_are_not_retried(status: int) -> None:
    judge, requests, _ = make_judge([httpx.Response(status, text="bad key")])
    with pytest.raises(JudgeError, match=f"{status}: bad key"):
        judge.ask(PROMPT, CorrectnessJudgement)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "make_payload",
    [lambda: {"choices": []}, lambda: {"choices": [{"message": {"content": None}}]}],
)
def test_unexpected_response_shape(make_payload: Callable[[], object]) -> None:
    judge, _, _ = make_judge([httpx.Response(200, json=make_payload())])
    with pytest.raises(JudgeError):
        judge.ask(PROMPT, CorrectnessJudgement)


def test_empty_api_key_is_rejected() -> None:
    with pytest.raises(JudgeError, match="GROQ_API_KEY"):
        GroqJudge("")
