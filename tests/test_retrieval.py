from app.ingestion.models import Evidence
from app.retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_prioritizes_relevant_document():
    evidence = [
        Evidence(
            document_id="a",
            file_name="sales_report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4 and the region grew faster than expected.",
        ),
        Evidence(
            document_id="b",
            file_name="marketing_notes.txt",
            file_type="txt",
            content="The marketing plan focuses on launch timing and brand awareness across channels.",
        ),
    ]

    results = HybridRetriever().search("Q4 revenue growth", evidence, top_k=2)

    assert results[0].file_name == "sales_report.txt"
    assert results[1].file_name == "marketing_notes.txt"


def test_hybrid_retriever_handles_empty_query():
    evidence = [
        Evidence(document_id="a", file_name="notes.txt", file_type="txt", content="Alpha beta gamma."),
    ]

    results = HybridRetriever().search("", evidence)

    assert results[0].file_name == "notes.txt"
