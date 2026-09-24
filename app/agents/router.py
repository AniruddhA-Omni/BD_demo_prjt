from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app import services
from app.config import load_settings

ROUTES = ("retrieval", "data", "vision", "code", "general", "multi_agent")
AGENTS = ("retrieval", "data", "vision", "code", "general")

# Which uploaded evidence each specialist agent needs; without it the agent is dropped from the plan.
AGENT_SOURCE_TYPES = {
    "data": {"table"},
    "vision": {"image", "vision"},
    "code": {"code"},
}

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|hiya|yo|good (morning|afternoon|evening)|thanks|thank you|thx|cheers|ok|okay|cool|great|"
    r"nice|bye|goodbye|see you)\b[\s!.,?]*(there|so much|a lot|again|you)?[\s!.,?:)]*$"
)
_META_PHRASES = (
    "who are you", "what are you", "what can you do", "how do you work", "what is your name", "what's your name",
    "how are you", "help me get started", "how do i use", "what files can i upload",
)

# Generic verbs that appear in document questions *and* general-knowledge questions ("explain what RAG is").
# On their own they are a weak signal, so the LLM gets to decide between retrieval and a general answer.
_WEAK_DOCUMENT_SIGNALS = ["what happened", "what does", "describe", "explain", "summary"]
_STRONG_DOCUMENT_SIGNALS = [
    "pdf", "document", "contract", "report", "notes", "clause", "agreement", "policy", "process", "procedure",
    "terms", "slide", "presentation", "file", "upload", "attachment", "according to",
]

# Data questions need *numeric intent*: an explicit spreadsheet reference, or an aggregate word together with a
# metric. A metric on its own ("sales policy", "revenue recognition") is only a topic, i.e. a weak signal.
_SPREADSHEET_SOURCES = ["excel", "csv", "sheet", "spreadsheet", "workbook", "dataframe", "pivot", "table", "row", "column"]
_METRIC_WORDS = ["revenue", "sales", "cost", "budget", "price", "profit", "expense", "spend", "units", "amount"]
_AGGREGATE_WORDS = [
    "sum", "total", "average", "mean", "median", "highest", "lowest", "maximum", "minimum", "max", "min", "count",
    "aggregate", "group by", "top", "bottom", "most", "least",
]
_CHANGE_WORDS = ["growth", "change", "increase", "decrease", "difference"]

_ROUTER_SYSTEM = """You route questions in a document-chat app to specialist agents.
Agents: retrieval (text in the user's documents/slides/notes), data (calculations over the user's spreadsheets/CSV
tables), vision (the user's images, charts, diagrams, screenshots), code (explaining or documenting the user's source
code){general}.
Pick every agent needed; pick more than one only when the question genuinely spans source types.
Return JSON: {{"agents": ["retrieval"], "reason": "..."}} using only these agent names: {names}."""

_GENERAL_AGENT_DESCRIPTION = (
    ", general (greetings, small talk, questions about the assistant, or general-knowledge questions that are not "
    "about the user's files; never combine general with other agents)"
)


@dataclass
class RouterDecision:
    route: str
    selected_agents: list[str] = field(default_factory=list)
    needs_parallel: bool = False
    reason: str = ""
    method: str = "rules"
    confident: bool = True


def _mentions(text: str, keywords: Iterable[str]) -> bool:
    """Whole-word (optionally plural) keyword match; keywords starting with "." (file extensions) match anywhere."""
    for keyword in keywords:
        if keyword.startswith("."):
            if keyword in text:
                return True
        elif re.search(rf"(?<![a-z0-9]){re.escape(keyword)}s?(?![a-z0-9])", text):
            return True
    return False


def _decision(agents: list[str], reason: str, method: str) -> RouterDecision:
    agents = list(dict.fromkeys(agents)) or ["retrieval"]
    route = agents[0] if len(agents) == 1 else "multi_agent"
    return RouterDecision(route=route, selected_agents=agents, needs_parallel=len(agents) > 1, reason=reason, method=method)


def is_social_or_meta(query: str) -> bool:
    normalized = query.lower().strip()
    if _GREETING.match(normalized):
        return True
    return len(normalized.split()) <= 8 and any(phrase in normalized for phrase in _META_PHRASES)


class RouterAgent:
    """Hybrid router: deterministic keyword rules first, LLM classification when no rule (or only a weak one) fires.

    ``general`` covers greetings, questions about the assistant and general knowledge. It is chosen by rules only for
    social/meta messages, or when nothing matched and no files are uploaded; otherwise only the LLM can pick it, so
    questions about the user's files keep going to grounded agents. When the available source types are known,
    agents whose evidence type isn't uploaded are dropped.
    """

    def __init__(self, llm: Any | None = None, allow_general: bool | None = None) -> None:
        self.llm = llm if llm is not None else services.get_llm()
        self.allow_general = load_settings().allow_general_answers if allow_general is None else allow_general

    def route_query(self, query: str, available_types: Iterable[str] | None = None) -> RouterDecision:
        available = set(available_types) if available_types is not None else None
        decision = self._rule_decision(query)

        if decision is None and self.allow_general and available is not None and not available:
            # Nothing uploaded and no document signal: grounded agents have nothing to search, so don't ask the LLM.
            decision = RouterDecision(route="general", selected_agents=["general"], reason="No files uploaded and no document signal.")
        elif decision is None or not decision.confident:
            llm_decision = self._llm_decision(query, available)
            if llm_decision is not None:
                decision = llm_decision
            elif decision is None:
                decision = RouterDecision(route="retrieval", selected_agents=["retrieval"], reason="Default to retrieval for document question.")

        if decision.route == "general" and not self.allow_general:
            decision = RouterDecision(route="retrieval", selected_agents=["retrieval"], reason="General answers are disabled.")
        return self._restrict_to_available(decision, available)

    def _rule_decision(self, query: str) -> RouterDecision | None:
        normalized = query.lower()

        if self.allow_general and is_social_or_meta(query):
            return RouterDecision(route="general", selected_agents=["general"], reason="Greeting or question about the assistant.")

        if _mentions(normalized, ["diagram", "screenshot", "image", "chart", "architecture", "visual", "picture", "photo"]):
            return RouterDecision(route="vision", selected_agents=["vision"], reason="Visual question detected.")

        code_signals = [
            ".py", "python", "code", "function", "method", "docstring", "script", "module",
            "generate documentation", "write documentation", "document this", "implementation",
        ]
        # snake_case identifiers ("load_documents") are a strong code signal too.
        if _mentions(normalized, code_signals) or re.search(r"\b[a-z]+_[a-z_]+\b", normalized):
            return RouterDecision(route="code", selected_agents=["code"], reason="Code question detected.")

        has_metric = _mentions(normalized, _METRIC_WORDS)
        # "growth"/"change" only count as a calculation when a range is given ("from Q3 to Q4").
        has_aggregate = _mentions(normalized, _AGGREGATE_WORDS) or bool(
            _mentions(normalized, _CHANGE_WORDS) and re.search(r"\bfrom\s+\S+\s+to\s+\S+", normalized)
        )
        strong_data = _mentions(normalized, _SPREADSHEET_SOURCES) or (has_aggregate and has_metric)
        has_strong_document_signal = _mentions(normalized, _STRONG_DOCUMENT_SIGNALS)
        has_document_signal = has_strong_document_signal or _mentions(normalized, _WEAK_DOCUMENT_SIGNALS)

        if strong_data and has_document_signal:
            return RouterDecision(
                route="multi_agent",
                selected_agents=["retrieval", "data"],
                needs_parallel=True,
                reason="Cross-source question requires document and spreadsheet reasoning.",
            )
        if strong_data:
            return RouterDecision(route="data", selected_agents=["data"], reason="Structured-data question detected.")
        if has_document_signal:
            return RouterDecision(
                route="retrieval",
                selected_agents=["retrieval"],
                reason="Document-grounded lookup detected.",
                confident=has_strong_document_signal,
            )
        if has_metric:
            # Topic only ("what were sales like?"): lean towards the data agent but let the LLM decide.
            return RouterDecision(route="data", selected_agents=["data"], reason="Metric mentioned without a calculation.", confident=False)
        return None

    def _llm_decision(self, query: str, available: set[str] | None) -> RouterDecision | None:
        if not self.llm.available():
            return None
        names = [agent for agent in AGENTS if agent != "general" or self.allow_general]
        system = _ROUTER_SYSTEM.format(general=_GENERAL_AGENT_DESCRIPTION if self.allow_general else "", names=", ".join(names))
        uploaded = ", ".join(sorted(available)) if available else "none"
        reply = self.llm.complete_json(system, f"Uploaded content types: {uploaded}\nQuestion: {query}", run_name="router_classify")
        if not reply:
            return None
        agents = [agent for agent in reply.get("agents", []) if agent in names]
        if len(agents) > 1 and "general" in agents:
            agents.remove("general")
        if not agents:
            return None
        return _decision(agents, str(reply.get("reason") or "LLM classification."), "llm")

    def _restrict_to_available(self, decision: RouterDecision, available: set[str] | None) -> RouterDecision:
        if not available:
            return decision
        kept = [
            agent
            for agent in decision.selected_agents
            if agent not in AGENT_SOURCE_TYPES or AGENT_SOURCE_TYPES[agent] & available
        ]
        if kept == decision.selected_agents:
            return decision
        dropped = sorted(set(decision.selected_agents) - set(kept))
        return _decision(kept, f"{decision.reason} Dropped {', '.join(dropped)}: no matching files uploaded.", decision.method)
