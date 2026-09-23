from app.graph.graph import run_query
from app.ingestion.models import Evidence


def test_run_query_orchestrates_retrieval_and_verification():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="quarterly_report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4. This was the strongest quarter on record.",
            metadata={"page": 3},
        )
    ]

    result = run_query("What happened in Q4?", evidence)

    assert result["route"] == "retrieval"
    assert len(result["retrieved"]) >= 1
    assert "12%" in result["answer"]
    assert result["verification"]["supported"] is True


def test_run_query_handles_missing_evidence():
    result = run_query("What happened in Q4?", [])

    assert result["route"] == "retrieval"
    assert "I couldn't find sufficient evidence" in result["answer"]
