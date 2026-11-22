from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class JudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user: str


class Judgement(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class ClaimVerdict(Judgement):
    claim: str
    supported: bool


class FaithfulnessJudgement(Judgement):
    claims: list[ClaimVerdict]


class CorrectnessJudgement(Judgement):
    correct: bool
    rationale: str


class AbstentionJudgement(Judgement):
    abstained: bool
    rationale: str


class Judge(Protocol):
    @property
    def model(self) -> str: ...

    def ask[J: Judgement](self, prompt: Prompt, schema: type[J]) -> J: ...
