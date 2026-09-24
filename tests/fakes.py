"""Test doubles for the LLM, embedding model and cross-encoder."""

from __future__ import annotations

import math
import re
from typing import Any, Callable

# Words in the same group embed to the same axis, so "income" is semantically close to "revenue".
_CONCEPTS = [
    {"revenue", "income", "sales", "earnings", "turnover"},
    {"policy", "approval", "approve", "rule", "governance"},
    {"hire", "hiring", "recruit", "engineers", "headcount"},
    {"risk", "risks", "threat", "delay", "delays"},
    {"launch", "roadmap", "release", "version"},
]


class FakeEmbedder:
    dimension = len(_CONCEPTS) + 1

    def __init__(self) -> None:
        self.calls = 0
        self.last_error = ""

    def available(self) -> bool:
        return True

    def embed(self, text: str) -> list[float] | None:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else None

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z]+", text.lower())
        vector = [float(sum(1 for token in tokens if token in concept)) for concept in _CONCEPTS] + [0.1]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeLLM:
    """Scripted stand-in for ``LLMService``: ``responder(system, user, run_name)`` returns the reply text."""

    def __init__(self, responder: Callable[[str, str, str], str | None] | None = None, vision: str | None = None) -> None:
        self.responder = responder or (lambda system, user, run_name: None)
        self.vision_reply = vision
        self.calls: list[dict[str, Any]] = []
        self.model_name = "fake"

    def available(self) -> bool:
        return True

    def status(self) -> str:
        return "available (fake)"

    def complete(self, system: str, user: str, *, run_name: str = "llm", tags: list[str] | None = None, history: list[dict[str, str]] | None = None) -> str | None:
        self.calls.append({"run_name": run_name, "system": system, "user": user, "history": history})
        return self.responder(system, user, run_name)

    def complete_json(self, system: str, user: str, *, run_name: str = "llm-json") -> dict[str, Any] | None:
        from app.graph.llm import parse_json_object

        reply = self.complete(system, user, run_name=run_name)
        return parse_json_object(reply) if reply else None

    def describe_images(self, prompt: str, image_paths: list[str], *, run_name: str = "vision") -> str | None:
        self.calls.append({"run_name": run_name, "images": image_paths, "prompt": prompt})
        return self.vision_reply


class FakeCrossEncoder:
    """Scores a (query, passage) pair by how many query words appear in the passage, reversed for testing order."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [next((score for key, score in self.scores.items() if key in passage), 0.0) for _, passage in pairs]
