from collections.abc import Sequence

from rag_eval.judge.base import Prompt

# Bump when any prompt text changes: the version is part of the cache key and
# is recorded in reports, so scores from different prompts are never mixed.
PROMPT_VERSION = "1"

_JSON_ONLY = "Respond with a single JSON object and nothing else."


def faithfulness(answer: str, contexts: Sequence[str]) -> Prompt:
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return Prompt(
        name="faithfulness",
        version=PROMPT_VERSION,
        system=(
            "You check whether an answer is supported by retrieved context. "
            "Split the answer into atomic factual claims, keeping the answer's language. "
            "A claim is supported only if the context states it or it follows directly "
            "from the context; general knowledge does not count. "
            "Statements that the information is unavailable are not claims. "
            f"{_JSON_ONLY} Format: "
            '{"claims": [{"claim": "...", "supported": true}]}'
        ),
        user=f"CONTEXT:\n{numbered or '(no context)'}\n\nANSWER:\n{answer}",
    )


def correctness(question: str, answer: str, reference: str) -> Prompt:
    return Prompt(
        name="correctness",
        version=PROMPT_VERSION,
        system=(
            "You grade an answer against a reference answer written by a domain expert. "
            "The answer is correct if it contains the essential information of the "
            "reference and does not contradict it. Extra details are fine unless they "
            "contradict the reference. Wording and length do not matter. "
            f"{_JSON_ONLY} Format: "
            '{"rationale": "one or two sentences", "correct": true}'
        ),
        user=f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER:\n{answer}",
    )


def abstention(question: str, answer: str) -> Prompt:
    return Prompt(
        name="abstention",
        version=PROMPT_VERSION,
        system=(
            "The question cannot be answered from the available documents. Decide whether "
            "the answer abstains: it says the information is not available, without "
            "providing a substantive answer anyway. "
            f"{_JSON_ONLY} Format: "
            '{"rationale": "one sentence", "abstained": true}'
        ),
        user=f"QUESTION:\n{question}\n\nANSWER:\n{answer}",
    )
