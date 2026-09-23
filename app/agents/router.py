from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouterDecision:
    route: str
    selected_agents: list[str] = field(default_factory=list)
    needs_parallel: bool = False
    reason: str = ""


class RouterAgent:
    """Simple hybrid router with deterministic rules and a clear task routing contract."""

    def route_query(self, query: str) -> RouterDecision:
        normalized = query.lower()

        if any(keyword in normalized for keyword in ["diagram", "screenshot", "image", "chart", "architecture", "visual"]):
            return RouterDecision(
                route="vision",
                selected_agents=["vision"],
                reason="Visual question detected.",
            )

        spreadsheet_signals = [
            "excel",
            "csv",
            "sheet",
            "spreadsheet",
            "dataframe",
            "pivot",
            "sum",
            "average",
            "mean",
            "highest revenue",
            "lowest revenue",
            "total revenue",
            "group by",
            "aggregate",
        ]
        document_signals = ["pdf", "document", "contract", "report", "notes", "what happened", "what does", "clause", "agreement"]

        if any(keyword in normalized for keyword in spreadsheet_signals):
            if any(keyword in normalized for keyword in document_signals):
                return RouterDecision(
                    route="multi_agent",
                    selected_agents=["retrieval", "data"],
                    needs_parallel=True,
                    reason="Cross-source question requires document and spreadsheet reasoning.",
                )
            return RouterDecision(
                route="data",
                selected_agents=["data"],
                reason="Structured-data question detected.",
            )

        if any(keyword in normalized for keyword in ["contract", "report", "document", "policy", "note", "terms", "clause", "agreement"]):
            return RouterDecision(
                route="retrieval",
                selected_agents=["retrieval"],
                reason="Document-grounded lookup detected.",
            )

        if any(keyword in normalized for keyword in ["compare", "summarize", "across", "and", "vs", "between"]):
            return RouterDecision(
                route="multi_agent",
                selected_agents=["retrieval", "data"],
                needs_parallel=True,
                reason="Comparison across sources requires multiple evidence channels.",
            )

        return RouterDecision(
            route="retrieval",
            selected_agents=["retrieval"],
            reason="Default to retrieval for document question.",
        )
