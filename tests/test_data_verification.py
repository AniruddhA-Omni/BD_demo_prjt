import pandas as pd

from app.agents.data import DataAgent, SpreadsheetRegistry
from app.agents.verification import VerificationAgent
from app.graph.graph import build_orchestration_graph, run_query
from app.ingestion.models import Evidence


def test_data_agent_computes_highest_revenue_region():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "West"],
            "revenue": [120, 90, 150],
        }
    )

    result = DataAgent().answer_question(df, "Which region had the highest revenue?")

    assert result["answer"] == "West"
    assert result["value"] == 150


def test_verification_agent_accepts_supported_claims():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="sales.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4.",
        )
    ]

    verdict = VerificationAgent().verify("Revenue increased by 12% in Q4.", evidence)

    assert verdict["supported"] is True


def test_verification_agent_flags_unsupported_claims():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="sales.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4.",
        )
    ]

    verdict = VerificationAgent().verify("Revenue increased by 30% in Q1.", evidence)

    assert verdict["supported"] is False
    assert "not supported" in verdict["reason"].lower() # type: ignore


def test_run_query_answers_structured_data_questions():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="revenue.csv",
            file_type="csv",
            content="region,revenue\nNorth,120\nWest,150\n",
            source_type="table",
            metadata={"source_name": "revenue.csv"},
        )
    ]

    result = run_query("Which region had the highest revenue?", evidence)

    assert result["route"] == "data"
    assert result["answer"] == "West"


def test_data_agent_supports_aggregate_queries_and_registry_summary():
    df = pd.DataFrame(
        {
            "region": ["North", "West", "South"],
            "revenue": [120, 150, 90],
        }
    )

    registry = SpreadsheetRegistry()
    summary = registry.register("revenue.csv", df)

    assert summary["row_count"] == 3
    assert summary["column_names"] == ["region", "revenue"]

    result = DataAgent().answer_question(df, "What is the average revenue?")
    assert result["answer"] == "120.0"


def test_build_orchestration_graph_handles_multi_agent_flow():
    graph = build_orchestration_graph()
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="quarterly_report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4. This was the strongest quarter on record.",
            metadata={"page": 3},
        ),
        Evidence(
            document_id="doc-2",
            file_name="revenue.csv",
            file_type="csv",
            content="region,revenue\nNorth,120\nWest,150\n",
            source_type="table",
            metadata={"source_name": "revenue.csv"},
        ),
    ]

    result = graph.invoke({"question": "Compare the contract revenue with the Excel totals and summarize the trend.", "evidence": evidence})

    assert result["route"] == "multi_agent"
    assert "data" in result["selected_agents"]
    assert "retrieval" in result["selected_agents"]
