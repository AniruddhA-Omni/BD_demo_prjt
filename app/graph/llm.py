from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import Settings, load_settings

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_AVAILABILITY_TTL_SECONDS = 30.0


def build_llm(model: str | None = None) -> ChatOllama:
    settings = Settings()

    llm_model = model or os.getenv("LLM_MODEL", settings.ollama_llm_model or "gemma4:e4b")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", str(settings.llm_temperature)))

    logger.info("LLM model initialised: %s (temperature=%s, base_url=%s)", llm_model, llm_temperature, settings.ollama_base_url)
    return ChatOllama(
        model=llm_model,
        temperature=llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=settings.llm_num_ctx,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models such as deepseek-r1."""
    return _THINK_BLOCK.sub("", text).strip()


def visible_stream_text(text: str) -> str:
    """Like ``strip_reasoning`` but also hides a <think> block that is still streaming (no closing tag yet)."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE).lstrip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model reply (models often wrap JSON in prose or code fences)."""
    text = strip_reasoning(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


class LLMService:
    """Single entry point for LLM calls with graceful degradation.

    Every method returns ``None`` when the LLM is disabled, Ollama is unreachable or a call fails, and callers fall
    back to deterministic behaviour. A failed call marks the service unavailable for a short TTL so a missing Ollama
    doesn't add a timeout to every step of every query.
    """

    def __init__(
        self,
        chat_model: Any | None = None,
        vision_model: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._chat_model = chat_model
        self._vision_model = vision_model
        self._injected = chat_model is not None
        self._available: bool | None = True if self._injected else None
        self._checked_at = 0.0
        self.last_error = ""
        self._reported: bool | None = None  # last availability that was logged, so only changes are logged

    @property
    def model_name(self) -> str:
        return os.getenv("LLM_MODEL", self.settings.ollama_llm_model)

    def available(self) -> bool:
        if self._injected:
            return bool(self._available)
        if not self.settings.enable_llm:
            self.last_error = "disabled in settings"
            return False
        if self._available is not None and time.monotonic() - self._checked_at < _AVAILABILITY_TTL_SECONDS:
            return self._available
        self._available = self._ping()
        self._checked_at = time.monotonic()
        self._report_availability()
        return self._available

    def _report_availability(self) -> None:
        if self._available == self._reported:
            return
        self._reported = self._available
        if self._available:
            logger.info("LLM available: %s at %s", self.model_name, self.settings.ollama_base_url)
        else:
            logger.warning("LLM unavailable, agents will use deterministic fallbacks: %s", self.last_error)

    def status(self) -> str:
        if self.available():
            return f"available ({self.model_name})"
        return f"unavailable ({self.last_error or 'unknown error'})"

    def complete(
        self,
        system: str,
        user: str,
        *,
        run_name: str = "llm",
        tags: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str | None:
        if not self.available():
            return None
        messages: list[BaseMessage] = [SystemMessage(content=system)]
        for turn in history or []:
            content = str(turn.get("content", ""))
            messages.append(AIMessage(content=content) if turn.get("role") == "assistant" else HumanMessage(content=content))
        messages.append(HumanMessage(content=user))
        try:
            reply = self._chat().invoke(messages, config={"run_name": run_name, "tags": tags or [run_name]})
        except Exception as exc:
            self._mark_failed(exc)
            return None
        return strip_reasoning(str(reply.content))

    def complete_json(self, system: str, user: str, *, run_name: str = "llm-json") -> dict[str, Any] | None:
        reply = self.complete(system + "\nRespond with a single JSON object and nothing else.", user, run_name=run_name)
        return parse_json_object(reply) if reply else None

    def describe_images(self, prompt: str, image_paths: list[str], *, run_name: str = "vision") -> str | None:
        """Ask the multimodal model about one or more local images."""
        if not self.available() or not image_paths:
            return None
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            path = Path(image_path)
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError as exc:
                logger.warning("Could not read image %s: %s", path, exc)
                continue
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        if len(content) == 1:
            return None
        try:
            reply = self._vision().invoke([HumanMessage(content=content)], config={"run_name": run_name, "tags": [run_name]})  # type: ignore[arg-type]
        except Exception as exc:
            self._mark_failed(exc)
            return None
        return strip_reasoning(str(reply.content))

    def _chat(self) -> Any:
        if self._chat_model is None:
            self._chat_model = build_llm()
        return self._chat_model

    def _vision(self) -> Any:
        if self._vision_model is None:
            vision_name = self.settings.ollama_vision_model
            self._vision_model = build_llm(vision_name) if vision_name else self._chat()
        return self._vision_model

    def _ping(self) -> bool:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                models = {item.get("name", "") for item in json.loads(response.read()).get("models", [])}
        except Exception as exc:
            self.last_error = f"Ollama not reachable at {self.settings.ollama_base_url}: {exc}"
            return False
        wanted = self.model_name
        if not any(name == wanted or name.split(":")[0] == wanted for name in models):
            self.last_error = f"model '{wanted}' not pulled (run `ollama pull {wanted}`)"
            return False
        self.last_error = ""
        return True

    def _mark_failed(self, exc: Exception) -> None:
        logger.warning("LLM call failed, falling back to deterministic behaviour: %s", exc)
        self.last_error = str(exc)
        self._available = False
        self._checked_at = time.monotonic()
