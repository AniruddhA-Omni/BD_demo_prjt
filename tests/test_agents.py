from app.agents.retrieval import RetrievalAgent
from app.agents.synthesis import SynthesisAgent
from app.ingestion.models import Evidence


def test_retrieval_agent_returns_top_evidence():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="quarterly_report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4. This was the strongest quarter on record.",
        ),
        Evidence(
            document_id="doc-2",
            file_name="sales_notes.txt",
            file_type="txt",
            content="The launch campaign reached more customers in February.",
        ),
    ]

    results = RetrievalAgent().retrieve("Q4 revenue growth", evidence, top_k=2)

    assert results[0].file_name == "quarterly_report.txt"
    assert "12%" in results[0].content


def test_synthesis_agent_uses_grounded_evidence_and_citations():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="quarterly_report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4. This was the strongest quarter on record.",
            metadata={"page": 3},
        ),
    ]

    answer = SynthesisAgent().synthesize("What happened in Q4?", evidence)

    assert "12%" in answer
    assert "quarterly_report.txt" in answer
    assert "Q4" in answer


def test_synthesis_agent_abstains_when_evidence_is_unrelated():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="marketing_notes.txt",
            file_type="txt",
            content="The campaign focused on brand awareness and launch timing.",
        ),
    ]

    answer = SynthesisAgent().synthesize("What was the Q4 revenue growth?", evidence)

    assert "insufficient evidence" in answer.lower()
