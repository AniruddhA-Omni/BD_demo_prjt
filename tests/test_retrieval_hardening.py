from app.ingestion.models import Evidence
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


def test_reranker_prefers_relevant_policy_docs_over_q4_boost():
    docs = [
        Evidence(
            document_id="1",
            file_name="revenue.txt",
            file_type="txt",
            source_type="text",
            content="Revenue increased by 12% in Q4.",
        ),
        Evidence(
            document_id="2",
            file_name="policy.txt",
            file_type="txt",
            source_type="text",
            content="The process and policy require manager approval before changes.",
        ),
    ]

    ranked = Reranker().rerank("Explain the policy and process", docs)

    assert ranked[0].file_name == "policy.txt"


def test_hybrid_retriever_ignores_generic_stopwords_in_query_matching():
    docs = [
        Evidence(
            document_id="1",
            file_name="quarterly.txt",
            file_type="txt",
            source_type="text",
            content="Revenue increased by 12% in Q4 and sales grew quickly.",
        ),
        Evidence(
            document_id="2",
            file_name="policy.txt",
            file_type="txt",
            source_type="text",
            content="The process and policy require manager approval before changes.",
        ),
    ]

    ranked = HybridRetriever().search("Explain the policy and process", docs, top_k=1)

    assert ranked[0].file_name == "policy.txt"
