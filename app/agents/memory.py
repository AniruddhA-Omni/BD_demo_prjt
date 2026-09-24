"""Conversation memory: follow-up rewriting and a bounded running summary (plan §9).

Memory is session-only. The graph never sees the full conversation: it gets the last ``GENERAL_HISTORY_TURNS``
messages plus, once a conversation outgrows that window, a short LLM-written summary of the older turns.

Follow-up questions ("What drove it?", "and in Q3?") are rewritten into standalone questions before routing, so
retrieval, the data agent and synthesis all work on a question that makes sense on its own. With the LLM the rewrite is
a real reformulation; without it the previous question is appended as context, which is enough for keyword retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app import services
from app.agents.router import is_social_or_meta
from app.logging_config import content

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 1200
_MAX_STANDALONE_CHARS = 400

# Words that point back at something said earlier. Kept to references that rarely start a standalone question.
_REFERENCE_WORDS = {"it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those", "he", "she", "one", "ones", "there", "same"}
_CONNECTOR_START = re.compile(r"^\s*(and|also|what about|how about|and what about|what else|why|how come|then|so)\b", re.IGNORECASE)

_REWRITE_SYSTEM = """You rewrite follow-up questions from a chat about the user's uploaded files into standalone
questions. Use the conversation to replace pronouns and fill in the missing subject, keep every detail of the
follow-up, and keep it to one short question. Do not answer it. If the question is already standalone, return it
unchanged. Reply with the question only."""

_SUMMARY_SYSTEM = f"""You maintain a running summary of a chat about the user's uploaded files. Merge the new turns into
the existing summary: keep the topics, files, figures and conclusions the user may refer back to, drop pleasantries.
Stay under {SUMMARY_MAX_CHARS} characters. Reply with the summary only."""


@dataclass
class Rewrite:
    question: str
    method: str  # "unchanged" | "llm" | "heuristic"

    @property
    def rewritten(self) -> bool:
        return self.method != "unchanged"


def looks_like_follow_up(question: str, history: list[dict[str, str]]) -> bool:
    """A question that only makes sense with the conversation: pronoun references, leading connectors, or very short."""
    if not any(turn.get("role") == "user" for turn in history) or is_social_or_meta(question):
        return False
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return False
    return bool(_REFERENCE_WORDS & set(words)) or bool(_CONNECTOR_START.match(question)) or len(words) <= 3


class QuestionRewriter:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm if llm is not None else services.get_llm()

    def rewrite(self, question: str, history: list[dict[str, str]], summary: str = "") -> Rewrite:
        if not looks_like_follow_up(question, history):
            return Rewrite(question, "unchanged")

        if self.llm.available():
            conversation = _format_turns(history)
            prompt = (f"Summary of earlier conversation:\n{summary}\n\n" if summary else "") + (
                f"Recent conversation:\n{conversation}\n\nFollow-up question: {question}"
            )
            reply = self.llm.complete(_REWRITE_SYSTEM, prompt, run_name="rewrite_question")
            standalone = _clean_question(reply)
            if standalone:
                logger.info("Follow-up rewritten by the LLM: %s", content(standalone))
                return Rewrite(standalone, "llm" if standalone != question else "unchanged")

        previous = next((turn["content"] for turn in reversed(history) if turn.get("role") == "user"), "")
        if not previous:
            return Rewrite(question, "unchanged")
        standalone = f"{question} (regarding: {previous})"
        logger.info("Follow-up expanded with the previous question (no LLM): %s", content(standalone))
        return Rewrite(standalone, "heuristic")


def recent_turns(messages: list[dict[str, Any]], turns: int) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": str(message["content"])}
        for message in messages[-turns:]
        if message.get("role") in {"user", "assistant"}
    ]


def update_summary(summary: str, older: list[dict[str, Any]], llm: Any | None = None) -> str | None:
    """Fold turns that fell out of the recent window into the running summary. Returns None without an LLM."""
    llm = llm if llm is not None else services.get_llm()
    if not older or not llm.available():
        return None
    prompt = f"Existing summary:\n{summary or '(none)'}\n\nNew turns:\n{_format_turns(older)}"
    reply = llm.complete(_SUMMARY_SYSTEM, prompt, run_name="summarize_conversation")
    if not reply:
        return None
    reply = reply.strip()
    return reply if len(reply) <= SUMMARY_MAX_CHARS else reply[: SUMMARY_MAX_CHARS - 1] + "…"


def _format_turns(turns: list[dict[str, Any]], limit: int = 600) -> str:
    lines = []
    for turn in turns:
        text = re.sub(r"\s*\[source:[^\]]*\]", "", str(turn.get("content", ""))).strip()
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        lines.append(f"{'User' if turn.get('role') == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


def _clean_question(reply: str | None) -> str | None:
    if not reply:
        return None
    text = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    text = re.sub(r"^(standalone question|question)\s*:\s*", "", text, flags=re.IGNORECASE).strip().strip('"').strip("'").strip()
    if not text or len(text) > _MAX_STANDALONE_CHARS:
        return None
    return text
