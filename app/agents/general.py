from __future__ import annotations

import re
from typing import Any

from app import services
from app.agents.router import is_social_or_meta

GENERAL_NOTICE = "General answer — not based on your uploaded files."

_SYSTEM = """You are the assistant inside a local document-intelligence app. Users upload PDFs, Word documents,
slides, spreadsheets, images and Python files and ask questions about them; those questions are answered by other
agents with citations.
You handle everything else: greetings, small talk, questions about what you can do, and general-knowledge questions.
- Answer briefly and helpfully from general knowledge.
- Never claim to know what is in the user's files and never invent facts about their company, data or documents.
  If the question seems to be about their files, suggest uploading them or asking about them directly.
- If you are not sure of a fact, say so."""

CAPABILITIES = (
    "I can answer questions about the files you upload — PDFs, Word documents, slides, text/Markdown, spreadsheets, "
    "images and Python code — with citations to the exact page, section, sheet range or lines. I can do spreadsheet "
    "calculations, read charts and diagrams, explain code, and I'll tell you when your files don't contain the answer."
)


class GeneralAgent:
    """Answers questions that need no sources. Output is never presented as grounded in the uploaded files."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm if llm is not None else services.get_llm()

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> tuple[str, bool]:
        """Return ``(answer, used_llm)``."""
        if self.llm.available():
            reply = self.llm.complete(_SYSTEM, question, run_name="general_answer", tags=["general"], history=history)
            if reply:
                return reply, True
        return self._fallback(question), False

    def _fallback(self, question: str) -> str:
        normalized = question.lower().strip()
        if re.match(r"^\s*(thanks|thank you|thx|cheers)\b", normalized):
            return "You're welcome! Ask me anything about your uploaded files."
        if re.match(r"^\s*(bye|goodbye|see you)\b", normalized):
            return "Goodbye! Your session's files stay available until you close the app."
        if is_social_or_meta(question):
            return f"Hello! {CAPABILITIES}"
        return (
            "General questions need the local LLM (Ollama), which isn't available right now. "
            "I can still answer questions about your uploaded files."
        )
