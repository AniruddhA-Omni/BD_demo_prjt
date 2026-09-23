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
            "revenue",
            "sales",
            "totals",
            "table",
            "row",
            "column",
        ]
        document_signals = [
            "pdf",
            "document",
            "contract",
            "report",
            "notes",
            "what happened",
            "what does",
            "clause",
            "agreement",
            "policy",
            "process",
            "procedure",
            "terms",
            "describe",
            "explain",
            "summary",
        ]

        has_spreadsheet_signal = any(keyword in normalized for keyword in spreadsheet_signals)
        has_document_signal = any(keyword in normalized for keyword in document_signals)
        wants_comparison = any(keyword in normalized for keyword in ["compare", "summarize", "across", "vs", "between", "trend"])
        explicit_cross_source = (has_spreadsheet_signal and has_document_signal) or (
            wants_comparison and has_spreadsheet_signal
        )

        if explicit_cross_source:
            return RouterDecision(
                route="multi_agent",
                selected_agents=["retrieval", "data"],
                needs_parallel=True,
                reason="Cross-source question requires document and spreadsheet reasoning.",
            )

        if has_spreadsheet_signal:
            return RouterDecision(
                route="data",
                selected_agents=["data"],
                reason="Structured-data question detected.",
            )

        if has_document_signal:
            return RouterDecision(
                route="retrieval",
                selected_agents=["retrieval"],
                reason="Document-grounded lookup detected.",
            )

        return RouterDecision(
            route="retrieval",
            selected_agents=["retrieval"],
            reason="Default to retrieval for document question.",
        )
