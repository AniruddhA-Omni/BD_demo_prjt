from __future__ import annotations

from typing import Iterable

from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.verification import VerificationAgent
from app.ingestion.models import Evidence


def run_query(question: str, evidence: Iterable[Evidence]) -> dict[str, object]:
    route = RouterAgent().route_query(question)
    docs = list(evidence)

    if docs:
        retrieved = RetrievalAgent().retrieve(question, docs, top_k=3)
        answer = SynthesisAgent().synthesize(question, retrieved)
        verification = VerificationAgent().verify(answer, retrieved)
    else:
        retrieved = []
        answer = "I couldn't find sufficient evidence in the uploaded files to answer this reliably."
        verification = {"supported": False, "reason": "No evidence available."}

    return {
        "route": route.route,
        "selected_agents": route.selected_agents,
        "retrieved": retrieved,
        "answer": answer,
        "verification": verification,
    }
