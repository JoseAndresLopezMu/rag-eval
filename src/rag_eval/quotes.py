from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rag_eval.models import DatasetItem
from rag_eval.relevance import normalize, quote_coverage

MIN_QUOTE_WORDS = 8


class Issue(StrEnum):
    MISSING_SOURCE = "missing source"
    NOT_VERBATIM = "not verbatim"
    TOO_SHORT = "too short"


@dataclass(frozen=True)
class QuoteProblem:
    item_id: str
    source: str
    quote: str
    issue: Issue
    detail: str


def check_quotes(
    dataset: Sequence[DatasetItem],
    sources: Mapping[str, str],
    min_words: int = MIN_QUOTE_WORDS,
) -> list[QuoteProblem]:
    """Find evidence quotes that would make retrieval scores wrong.

    A quote that is not verbatim in its source can never be covered, so the
    system gets 0 through no fault of its own. Short quotes match unrelated
    chunks too easily (SPEC 4.3).
    """
    normalized = {name: f" {' '.join(normalize(text))} " for name, text in sources.items()}
    problems = []
    for item in dataset:
        for ev in item.evidence:
            words = normalize(ev.quote)
            found: list[tuple[Issue, str]] = []
            if ev.source not in sources:
                found.append((Issue.MISSING_SOURCE, "no text loaded for this source"))
            else:
                if f" {' '.join(words)} " not in normalized[ev.source]:
                    best = quote_coverage(ev.quote, sources[ev.source])
                    found.append((Issue.NOT_VERBATIM, f"longest matching run covers {best:.0%}"))
                if len(words) < min_words:
                    found.append((Issue.TOO_SHORT, f"{len(words)} words, minimum {min_words}"))
            problems += [QuoteProblem(item.id, ev.source, ev.quote, i, d) for i, d in found]
    return problems
