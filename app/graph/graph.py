from __future__ import annotations

from io import StringIO
from typing import Any, Iterable

import pandas as pd
from langgraph.graph import END, StateGraph

from app.agents.data import DataAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.verification import VerificationAgent
from app.ingestion.models import Evidence


def build_orchestration_graph() -> Any:
    """Create a small LangGraph workflow that routes a question through the right agent path."""

    workflow = StateGraph(dict)

    workflow.add_node("route", _route_step)
    workflow.add_node("retrieve", _retrieve_step)
    workflow.add_node("data", _data_step)
    workflow.add_node("multi_agent_worker", _multi_agent_step)
    workflow.add_node("verify", _verify_step)
    workflow.set_entry_point("route")

    workflow.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "retrieval": "retrieve",
            "vision": "retrieve",
            "data": "data",
            "multi_agent": "multi_agent_worker",
        },
    )
    workflow.add_edge("retrieve", "verify")
    workflow.add_edge("data", "verify")
    workflow.add_edge("multi_agent_worker", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()


def run_query(question: str, evidence: Iterable[Evidence]) -> dict[str, object]:
    docs = list(evidence)
    graph = build_orchestration_graph()
    result = graph.invoke({"question": question, "evidence": docs})
    return result


def _route_step(state: dict[str, Any]) -> dict[str, Any]:
    route = RouterAgent().route_query(state["question"])
    state["route"] = route.route
    state["selected_agents"] = route.selected_agents
    return state


def _retrieve_step(state: dict[str, Any]) -> dict[str, Any]:
    docs = list(state.get("evidence", []))
    question = state["question"]
    if not docs:
        state["retrieved"] = []
        state["answer"] = "I couldn't find sufficient evidence in the uploaded files to answer this reliably."
        state["verification"] = {"supported": False, "reason": "No evidence available."}
        return state

    retrieved = RetrievalAgent().retrieve(question, docs, top_k=3)
    answer = SynthesisAgent().synthesize(question, retrieved)
    verification = VerificationAgent().verify(answer, retrieved)
    state["retrieved"] = retrieved
    state["answer"] = answer
    state["verification"] = verification
    return state


def _data_step(state: dict[str, Any]) -> dict[str, Any]:
    docs = list(state.get("evidence", []))
    question = state["question"]
    if not docs:
        state["answer"] = "I couldn't find sufficient evidence in the uploaded files to answer this reliably."
        state["verification"] = {"supported": False, "reason": "No evidence available."}
        return state

    table = _coerce_table_data(docs)
    result = DataAgent().answer_question(table, question)
    answer = str(result["answer"])
    verification = VerificationAgent().verify(answer, docs)
    state["retrieved"] = docs
    state["answer"] = answer
    state["verification"] = verification
    return state


def _multi_agent_step(state: dict[str, Any]) -> dict[str, Any]:
    docs = list(state.get("evidence", []))
    question = state["question"]
    if not docs:
        state["answer"] = "I couldn't find sufficient evidence in the uploaded files to answer this reliably."
        state["verification"] = {"supported": False, "reason": "No evidence available."}
        return state

    retrieved = RetrievalAgent().retrieve(question, docs, top_k=3)
    data_answer = DataAgent().answer_question(_coerce_table_data(docs), question)
    answer = str(data_answer["answer"]) if data_answer["value"] is not None else SynthesisAgent().synthesize(question, retrieved)
    verification = VerificationAgent().verify(answer, retrieved or docs)
    state["retrieved"] = retrieved
    state["answer"] = answer
    state["verification"] = verification
    state["selected_agents"] = ["retrieval", "data"]
    return state


def _verify_step(state: dict[str, Any]) -> dict[str, Any]:
    restored = dict(state)
    if "verification" not in restored:
        restored["verification"] = {"supported": False, "reason": "No evidence was available to verify this answer."}
    return restored


def _coerce_table_data(evidence: Iterable[Evidence]) -> pd.DataFrame:
    docs = list(evidence)
    csv_payloads = [doc.content for doc in docs if doc.file_type == "csv" or doc.source_type == "table"]
    if not csv_payloads:
        return pd.DataFrame()

    payload = csv_payloads[0]
    try:
        return pd.read_csv(StringIO(payload))
    except Exception:
        return pd.DataFrame()
