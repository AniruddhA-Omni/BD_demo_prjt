from __future__ import annotations

import logging
import operator
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Annotated, Any, Callable, Iterable, Iterator, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from app import services
from app.agents.code import CodeAgent
from app.agents.data import DataAgent
from app.agents.general import GENERAL_NOTICE, GeneralAgent
from app.agents.memory import QuestionRewriter
from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.agents.synthesis import ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE, SynthesisAgent, format_citation
from app.agents.verification import VerificationAgent
from app.agents.vision import VisionAgent
from app.config import load_settings
from app.ingestion.models import Evidence
from app.logging_config import content, current_request_id, log_context, new_request_id
from app.observability import record_step

logger = logging.getLogger(__name__)

AGENT_NODES = {
    "retrieval": "retrieval_agent",
    "data": "data_agent",
    "vision": "vision_agent",
    "code": "code_agent",
    "general": "general_agent",
}
# Nodes whose LLM tokens are streamed to the UI.
STREAMED_NODES = {"synthesize", "general_agent"}


class GraphState(TypedDict, total=False):
    question: str
    evidence: list[Evidence]
    session_id: str | None
    history: list[dict[str, str]]
    summary: str
    # Follow-ups are rewritten into standalone questions before routing; agents use this, the UI shows the original.
    standalone_question: str
    rewrite_method: str
    route: str
    selected_agents: list[str]
    router_reason: str
    router_method: str
    # Parallel agent branches each append their result; the reducer merges them at the fan-in.
    agent_results: Annotated[list[dict[str, Any]], operator.add]
    retrieved: list[Evidence]
    answer: str
    used_llm: bool
    verification: dict[str, Any]
    attempts: int
    regenerate: bool
    feedback: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def build_orchestration_graph() -> Any:
    """contextualize → route → parallel specialist agents → synthesize → verify → (synthesize → verify)* → finalize.

    The general agent (greetings, general knowledge) goes straight from route to finalize.
    """

    workflow = StateGraph(GraphState)
    workflow.add_node("contextualize", _timed("contextualize", _contextualize_step))
    workflow.add_node("route", _timed("route", _route_step))
    workflow.add_node("retrieval_agent", _timed("retrieval_agent", _retrieval_step))
    workflow.add_node("data_agent", _timed("data_agent", _data_step))
    workflow.add_node("vision_agent", _timed("vision_agent", _vision_step))
    workflow.add_node("code_agent", _timed("code_agent", _code_step))
    workflow.add_node("general_agent", _timed("general_agent", _general_step))
    workflow.add_node("synthesize", _timed("synthesize", _synthesize_step))
    workflow.add_node("verify", _timed("verify", _verify_step))
    workflow.add_node("finalize", _timed("finalize", _finalize_step))

    workflow.set_entry_point("contextualize")
    workflow.add_edge("contextualize", "route")
    # Returning several node names fans out: the selected agents run in parallel in one superstep.
    workflow.add_conditional_edges(
        "route",
        lambda state: [AGENT_NODES[agent] for agent in state["selected_agents"]],
        list(AGENT_NODES.values()),
    )
    for agent, node in AGENT_NODES.items():
        # General answers need no evidence, so they skip synthesis and verification.
        workflow.add_edge(node, "finalize" if agent == "general" else "synthesize")
    workflow.add_edge("synthesize", "verify")
    workflow.add_conditional_edges(
        "verify",
        lambda state: "regenerate" if state.get("regenerate") else "finalize",
        {"regenerate": "synthesize", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def _compiled_graph() -> Any:
    started = time.perf_counter()
    graph = build_orchestration_graph()
    nodes = [name for name in graph.get_graph().nodes if not name.startswith("__")]
    logger.info("Orchestration graph compiled: %d nodes (%s) in %.0f ms", len(nodes), ", ".join(nodes), (time.perf_counter() - started) * 1000)
    return graph


@contextmanager
def _request_scope(question: str, evidence: list[Evidence], history: list[dict[str, str]] | None):
    """Give the question a request id (unless the caller set one) and log its start; yields a dict for the result."""
    request_id = current_request_id() if current_request_id() != "-" else new_request_id()
    outcome: dict[str, Any] = {}
    with log_context(request_id=request_id):
        started = time.perf_counter()
        files = {doc.document_id for doc in evidence}
        logger.info(
            "Question received: %s (%d chunks from %d files, %d history turns)", content(question), len(evidence), len(files), len(history or [])
        )
        try:
            yield outcome
        except Exception:
            logger.exception("Query failed after %.0f ms", (time.perf_counter() - started) * 1000)
            raise
        state = outcome.get("state") or {}
        final = next((entry for entry in reversed(state.get("trace", [])) if entry.get("node") == "finalize"), {})
        logger.info(
            "Answer completed: route=%s outcome=%s llm=%s supported=%s in %.0f ms",
            state.get("route"),
            final.get("outcome"),
            state.get("used_llm"),
            (state.get("verification") or {}).get("supported"),
            (time.perf_counter() - started) * 1000,
        )


def run_query(
    question: str,
    evidence: Iterable[Evidence],
    session_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    summary: str = "",
) -> dict[str, Any]:
    docs = list(evidence)
    with _request_scope(question, docs, history) as outcome:
        outcome["state"] = _compiled_graph().invoke(_inputs(question, docs, session_id, history, summary))
    return outcome["state"]


def stream_query(
    question: str,
    evidence: Iterable[Evidence],
    session_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    summary: str = "",
) -> Iterator[tuple[str, Any]]:
    """Stream a query for the UI. Yields, in order of occurrence:

    - ``("step", {"node": ..., **trace})`` when a graph node finishes (routing decision, retrieval, verification…),
    - ``("citations", [label, ...])`` just before synthesis, so ``[n]`` markers can be rendered while streaming,
    - ``("token", text)`` for LLM tokens from synthesis or the general agent,
    - ``("reset", None)`` when verification triggers a rewrite and a new draft starts streaming,
    - ``("final", state)`` once, at the end.
    """
    docs = list(evidence)
    with _request_scope(question, docs, history) as outcome:
        yield from _stream_events(question, docs, session_id, history, summary, outcome)


def _stream_events(
    question: str,
    docs: list[Evidence],
    session_id: str | None,
    history: list[dict[str, str]] | None,
    summary: str,
    outcome: dict[str, Any],
) -> Iterator[tuple[str, Any]]:
    final: dict[str, Any] = {}
    current_message: str | None = None
    stream = _compiled_graph().stream(
        _inputs(question, docs, session_id, history, summary), stream_mode=["messages", "updates", "custom", "values"]
    )
    for mode, payload in stream:
        if mode == "values":
            final = payload
            continue
        if mode == "updates":
            for node, update in (payload or {}).items():
                for entry in (update or {}).get("trace", []):
                    yield ("step", entry)
            continue
        if mode == "custom":
            if isinstance(payload, dict) and "citations" in payload:
                yield ("citations", payload["citations"])
            continue
        message, metadata = payload
        if metadata.get("langgraph_node") not in STREAMED_NODES:
            continue
        text = message.content if isinstance(message.content, str) else ""
        if not text:
            continue
        if message.id != current_message:
            if current_message is not None:
                yield ("reset", None)
            current_message = message.id
        yield ("token", text)
    outcome["state"] = final
    yield ("final", final)


def _inputs(
    question: str,
    evidence: Iterable[Evidence],
    session_id: str | None,
    history: list[dict[str, str]] | None = None,
    summary: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "evidence": list(evidence),
        "session_id": session_id,
        "history": list(history or []),
        "summary": summary,
        "attempts": 0,
    }


def _question(state: GraphState) -> str:
    """The question agents should work on: the standalone rewrite of a follow-up, else the original."""
    return state.get("standalone_question") or state["question"]


def _timed(name: str, step: Callable[[GraphState], dict[str, Any]]) -> Callable[[GraphState], dict[str, Any]]:
    def run(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        update = step(state)
        entry = {"node": name, "duration_ms": round((time.perf_counter() - started) * 1000, 1), **update.pop("_trace", {})}
        update["trace"] = [entry]
        logger.info("Step %s finished in %.0f ms%s", name, entry["duration_ms"], _log_summary(entry))
        record_step(name, step_summary(entry))
        return update

    return run


def _contextualize_step(state: GraphState) -> dict[str, Any]:
    rewrite = QuestionRewriter().rewrite(state["question"], state.get("history") or [], state.get("summary") or "")
    return {
        "standalone_question": rewrite.question,
        "rewrite_method": rewrite.method,
        "_trace": {"rewritten": rewrite.rewritten, "method": rewrite.method},
    }


def _route_step(state: GraphState) -> dict[str, Any]:
    available = {doc.source_type for doc in state.get("evidence", [])}
    decision = RouterAgent().route_query(_question(state), available_types=available)
    return {
        "route": decision.route,
        "selected_agents": decision.selected_agents,
        "router_reason": decision.reason,
        "router_method": decision.method,
        "_trace": {"route": decision.route, "agents": decision.selected_agents, "method": decision.method, "reason": decision.reason},
    }


def _retrieval_step(state: GraphState) -> dict[str, Any]:
    docs = state.get("evidence", [])
    retrieved: list[Evidence] = []
    if docs:
        retrieved = RetrievalAgent().retrieve(
            _question(state),
            docs,
            top_k=load_settings().retrieval_top_k,
            session_id=state.get("session_id"),
            # With the data agent running alongside, spreadsheets are its job; keep retrieval on text evidence.
            text_only="data" in state.get("selected_agents", []),
        )
    return {"agent_results": [{"agent": "retrieval", "evidence": retrieved}], "_trace": {"retrieved": [_trace_doc(d) for d in retrieved]}}


def _data_step(state: GraphState) -> dict[str, Any]:
    result = DataAgent().answer_from_evidence(_question(state), state.get("evidence", []))
    return {
        "agent_results": [{"agent": "data", "evidence": result.get("evidence", []), "answer": result["answer"], "value": result["value"]}],
        "_trace": {"plan": result.get("plan"), "answer": result["answer"], "cell_range": result.get("cell_range")},
    }


def _vision_step(state: GraphState) -> dict[str, Any]:
    docs = state.get("evidence", [])
    found = VisionAgent().run(_question(state), docs, top_k=load_settings().retrieval_top_k, session_id=state.get("session_id")) if docs else []
    return {
        "agent_results": [{"agent": "vision", "evidence": found}],
        "_trace": {"retrieved": [_trace_doc(d) for d in found], "vision_model_used": any(d.section == "vision model reading" for d in found)},
    }


def _code_step(state: GraphState) -> dict[str, Any]:
    docs = state.get("evidence", [])
    found = CodeAgent().run(_question(state), docs, top_k=load_settings().retrieval_top_k + 1, session_id=state.get("session_id")) if docs else []
    return {"agent_results": [{"agent": "code", "evidence": found}], "_trace": {"retrieved": [_trace_doc(d) for d in found]}}


def _general_step(state: GraphState) -> dict[str, Any]:
    turns = load_settings().general_history_turns
    history = [
        {"role": turn["role"], "content": str(turn["content"])}
        for turn in (state.get("history") or [])[-turns:]
        if turn.get("role") in {"user", "assistant"}
    ]
    summary = state.get("summary") or ""
    if summary:
        history = [{"role": "user", "content": f"(Summary of our earlier conversation: {summary})"}, *history]
    answer, used_llm = GeneralAgent().answer(state["question"], history)
    return {
        "answer": answer,
        "used_llm": used_llm,
        "retrieved": [],
        "verification": {"supported": None, "general": True, "reason": GENERAL_NOTICE, "claims": []},
        "_trace": {"used_llm": used_llm, "history_turns": len(history)},
    }


def _synthesize_step(state: GraphState) -> dict[str, Any]:
    results = state.get("agent_results", [])
    evidence = _merge_evidence(results)
    attempt = state.get("attempts", 0)
    if not evidence:
        return {"retrieved": [], "answer": ABSTAIN_MESSAGE, "used_llm": False, "_trace": {"used_llm": False, "attempt": attempt}}

    data_result = next((r for r in results if r["agent"] == "data" and r.get("value") is not None), None)
    style = _style(state.get("selected_agents", []))
    synthesis = SynthesisAgent()

    if synthesis.llm.available():
        # Lets the UI turn streamed [n] markers into readable citations before the final answer arrives.
        get_stream_writer()({"citations": [format_citation(doc) for doc in evidence]})
        composed = synthesis.compose(_question(state), evidence, style=style, feedback=state.get("feedback") if attempt else None)
        answer, used_llm = composed.answer, composed.used_llm
    elif data_result is not None and state.get("selected_agents") == ["data"]:
        # Deterministic path: the computed value *is* the answer; its evidence carries the sheet/range citation.
        answer, used_llm = str(data_result["answer"]), False
    elif data_result is not None:
        computed = [doc for doc in evidence if doc.source_type == "computed"]
        text_evidence = [doc for doc in evidence if doc.source_type not in {"table", "computed"}]
        parts = [f"{doc.content} [source: {format_citation(doc)}]" for doc in computed]
        if text_evidence:
            text = synthesis.compose(_question(state), text_evidence, style=style).answer
            if text not in {ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE}:
                parts.append(text)
        answer, used_llm = " ".join(parts), False
    else:
        text_evidence = [doc for doc in evidence if doc.source_type != "table"] or evidence
        answer, used_llm = synthesis.compose(_question(state), text_evidence, style=style).answer, False

    return {"retrieved": evidence, "answer": answer, "used_llm": used_llm, "_trace": {"used_llm": used_llm, "style": style, "attempt": attempt}}


def _verify_step(state: GraphState) -> dict[str, Any]:
    evidence = state.get("retrieved", [])
    if not evidence:
        verification: dict[str, Any] = {"supported": False, "reason": "No evidence available.", "claims": [], "support_ratio": 0.0}
    else:
        verification = VerificationAgent().verify_answer(state.get("answer", ""), evidence)
    unsupported = [claim["text"] for claim in verification.get("claims", []) if not claim.get("supported")]
    attempts = state.get("attempts", 0)
    regenerate = bool(
        unsupported
        and state.get("used_llm")
        and attempts < load_settings().max_regenerations
        and services.get_llm().available()
    )
    update: dict[str, Any] = {
        "verification": verification,
        "regenerate": regenerate,
        "_trace": {"supported": verification["supported"], "support_ratio": verification.get("support_ratio"), "unsupported": unsupported, "regenerate": regenerate},
    }
    if regenerate:
        update["attempts"] = attempts + 1
        update["feedback"] = (
            "These statements were NOT supported by the evidence: "
            + "; ".join(f'"{text}"' for text in unsupported)
            + ". Rewrite the answer using only the evidence: remove or correct them and keep citations on every claim."
        )
    return update


def _finalize_step(state: GraphState) -> dict[str, Any]:
    """Downgrade answers the verifier couldn't fully support: keep verified claims and name the unverified ones."""
    verification = state.get("verification") or {}
    if verification.get("general"):
        return {"_trace": {"outcome": "general"}}
    answer = state.get("answer", "")
    claims = verification.get("claims") or []
    if verification.get("supported") or not claims or answer in {ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE}:
        return {"_trace": {"outcome": "pass" if verification.get("supported") else "abstain"}}

    supported = [claim for claim in claims if claim.get("supported")]
    unsupported = [claim for claim in claims if not claim.get("supported")]
    if not supported:
        return {"answer": ABSTAIN_MESSAGE, "_trace": {"outcome": "abstain", "dropped": len(unsupported)}}

    kept = " ".join(
        claim["text"]
        + "".join(f" [source: {label}]" for label in claim.get("sources", []) if label not in claim.get("invalid_citations", []))
        for claim in supported
    )
    doubtful = "; ".join(_trim(claim["text"]) for claim in unsupported)
    return {
        "answer": f"{kept}\n\nI could not verify the following against the uploaded files: {doubtful}",
        "_trace": {"outcome": "partial", "kept": len(supported), "dropped": len(unsupported)},
    }


def _merge_evidence(results: list[dict[str, Any]]) -> list[Evidence]:
    # Computed/visual answers first: they are the most specific evidence for the question.
    priority = {"data": 0, "vision": 1, "code": 2, "retrieval": 3}
    merged: list[Evidence] = []
    seen: set[str] = set()
    for result in sorted(results, key=lambda r: priority.get(r["agent"], 4)):
        for doc in result.get("evidence", []):
            if doc.key not in seen:
                seen.add(doc.key)
                merged.append(doc)
    return merged


def _style(agents: list[str]) -> str:
    if agents == ["code"]:
        return "code"
    if agents == ["vision"]:
        return "vision"
    if "data" in agents:
        return "data"
    return "default"


def _trace_doc(doc: Evidence) -> dict[str, Any]:
    return {
        "file": doc.file_name,
        "section": doc.section,
        "page": doc.page,
        "source_type": doc.source_type,
        "hybrid": doc.metadata.get("hybrid_score"),
        "dense": doc.metadata.get("dense_score"),
        "rerank": doc.metadata.get("rerank_score"),
    }


def step_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Content-free summary of a trace entry: route, file names, counts, scores and timings, never document or answer
    text. Shared by the step logs and the LangSmith step metadata."""
    summary: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "node" or value in (None, [], {}):
            continue
        if key == "retrieved":
            summary["retrieved"] = len(value)
            summary["top_sources"] = [f"{doc['file']} ({doc.get('hybrid')})" for doc in value[:3]]
        elif key == "unsupported":
            summary["unsupported"] = len(value)
        elif key == "plan":
            summary["plan"] = f"{value.get('operation')}({value.get('column')}) by={value.get('group_by')} via={value.get('method')}"
        elif key == "answer":
            summary["answer_chars"] = len(str(value))
        else:
            summary[key] = value
    return summary


def _log_summary(entry: dict[str, Any]) -> str:
    """One-line log rendering of ``step_summary``; the answer itself is shown only when LOG_CONTENT is on."""
    summary = step_summary(entry)
    summary.pop("duration_ms", None)
    if "answer_chars" in summary:
        summary.pop("answer_chars")
        summary["answer"] = content(entry["answer"])
    parts = [
        f"retrieved={value} [{', '.join(summary.get('top_sources', []))}]" if key == "retrieved" else f"{key}={value}"
        for key, value in summary.items()
        if key != "top_sources"
    ]
    return ": " + "; ".join(parts) if parts else ""


def _trim(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
