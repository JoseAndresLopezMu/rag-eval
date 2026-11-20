import re
import unicodedata

from rag_eval.models import Context, Evidence

DEFAULT_THRESHOLD = 0.6

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", without_accents).split()


def quote_coverage(quote: str, text: str) -> float:
    """Share of the quote found in `text` as one contiguous run of tokens.

    Contiguity matters: counting loose tokens lets any chunk that shares common
    words ("de", "la", "servicio") look relevant. See SPEC 4.3.
    """
    q = normalize(quote)
    if not q:
        raise ValueError(f"quote has no words after normalization: {quote!r}")
    return _longest_common_run(q, normalize(text)) / len(q)


def covers(context: Context, evidence: Evidence, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if context.source != evidence.source:
        return False
    return quote_coverage(evidence.quote, context.text) >= threshold


def _longest_common_run(a: list[str], b: list[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            if token == other:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best
