from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Model(BaseModel):
    # extra="forbid" turns a typo such as "refrence_answer" into a load error
    # instead of a silently missing field.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Evidence(_Model):
    source: NonEmptyStr
    page: Annotated[int, Field(ge=1)] | None = None
    quote: NonEmptyStr


class DatasetItem(_Model):
    id: NonEmptyStr
    question: NonEmptyStr
    reference_answer: NonEmptyStr
    answerable: bool = True
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _evidence_matches_answerable(self) -> Self:
        if self.answerable and not self.evidence:
            raise ValueError("answerable questions need at least one evidence item")
        if not self.answerable and self.evidence:
            raise ValueError("unanswerable questions must not have evidence")
        return self


class Context(_Model):
    rank: Annotated[int, Field(ge=1)]
    source: NonEmptyStr
    page: Annotated[int, Field(ge=1)] | None = None
    text: str


class RunItem(_Model):
    id: NonEmptyStr
    answer: str
    contexts: tuple[Context, ...] = ()
    latency_ms: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _ranks_are_consecutive(self) -> Self:
        ranks = [c.rank for c in self.contexts]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"context ranks must be 1..n in order, got {ranks}")
        return self
