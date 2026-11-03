"""LLM client with structured output.

Structured output (JSON schema enforced by the API) means we never regex-parse free text:
we get {answer, citations, abstained} and can validate each field in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from ragkit.config import Settings
from ragkit.generation.prompts import ABSTAIN_MESSAGE, PromptBundle
from ragkit.observability import cost


class LLMAnswer(BaseModel):
    answer: str = Field(description="The answer in plain text, at most 150 words.")
    citations: list[str] = Field(description='Refs of documents used, e.g. ["S1","S3"].')
    abstained: bool = Field(description="True if the documents do not answer the question.")


@dataclass
class LLMResult:
    parsed: LLMAnswer | None
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    error: str | None = None


class LLM(Protocol):
    def generate(self, bundle: PromptBundle) -> LLMResult: ...


class OpenAILLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set (or use LLM_PROVIDER=fake)")
        self.s = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(), max_retries=2, timeout=60
        )

    def generate(self, bundle: PromptBundle) -> LLMResult:
        cost.check_budget(self.s)
        kwargs: dict = {}
        # Reasoning models (gpt-5*, o*) accept an effort setting; others reject it.
        if self.s.llm_reasoning_effort and self.s.llm_model.startswith(("gpt-5", "o")):
            kwargs["reasoning"] = {"effort": self.s.llm_reasoning_effort}
        resp = self.client.responses.parse(
            model=self.s.llm_model,
            instructions=bundle.instructions,
            input=[{"role": "user", "content": bundle.user_message}],
            text_format=LLMAnswer,
            max_output_tokens=self.s.max_output_tokens,
            store=False,  # don't retain prompts (which contain internal docs) on OpenAI's side
            **kwargs,
        )
        usage = resp.usage
        in_t = usage.input_tokens if usage else 0
        out_t = usage.output_tokens if usage else 0
        usd = cost.record(self.s, "llm", self.s.llm_model, in_t, out_t)
        parsed = resp.output_parsed
        err = None if parsed else f"no parsed output (status={resp.status})"
        return LLMResult(parsed, in_t, out_t, usd, err)


class FakeLLM:
    """Offline stand-in: 'answers' with the first sentence of the top document."""

    def generate(self, bundle: PromptBundle) -> LLMResult:
        if not bundle.ref_map:
            return LLMResult(LLMAnswer(answer=ABSTAIN_MESSAGE, citations=[], abstained=True))
        ref, chunk = next(iter(bundle.ref_map.items()))
        body = chunk.text.split("\n\n", 1)[-1]
        first = body.split(". ")[0].strip()[:300]
        return LLMResult(LLMAnswer(answer=first, citations=[ref], abstained=False))


def build_llm(settings: Settings) -> LLM:
    return OpenAILLM(settings) if settings.llm_provider == "openai" else FakeLLM()
