from app.ingestion.models import Evidence
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


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


def test_bm25_retriever_prefer_exact_keyword_match():
    evidence = [
        Evidence(document_id="a", file_name="other.txt", file_type="txt", content="The team launched a new brand campaign."),
        Evidence(document_id="b", file_name="revenue.txt", file_type="txt", content="Revenue increased by 12% in Q4 and the quarterly growth was strong."),
    ]

    results = BM25Retriever().search("Q4 revenue growth", evidence, top_k=2)

    assert results[0].file_name == "revenue.txt"


def test_reranker_promotes_best_quality_match():
    evidence = [
        Evidence(document_id="a", file_name="weak.txt", file_type="txt", content="The launch plan was a success."),
        Evidence(document_id="b", file_name="strong.txt", file_type="txt", content="Revenue increased by 12% in Q4 with strong growth."),
    ]

    ranked = Reranker().rerank("Q4 revenue growth", evidence)

    assert ranked[0].file_name == "strong.txt"


def test_hybrid_retriever_keeps_revenue_recall_for_clear_signal():
    evidence = [
        Evidence(document_id="a", file_name="policy.txt", file_type="txt", content="The process and policy require manager approval before changes."),
        Evidence(document_id="b", file_name="revenue.txt", file_type="txt", content="Revenue increased by 12% in Q4 and growth accelerated across the region."),
    ]

    ranked = HybridRetriever().search("Q4 revenue growth", evidence, top_k=2)

    assert ranked[0].file_name == "revenue.txt"


def test_hybrid_retriever_avoids_false_positive_for_generic_policy_query():
    evidence = [
        Evidence(document_id="a", file_name="revenue.txt", file_type="txt", content="Revenue increased by 12% in Q4."),
        Evidence(document_id="b", file_name="policy.txt", file_type="txt", content="The process and policy require manager approval before changes."),
    ]

    ranked = HybridRetriever().search("Explain the policy and process", evidence, top_k=2)

    assert ranked[0].file_name == "policy.txt"


def test_hybrid_retriever_prefers_qdrant_dense_results_when_enabled(monkeypatch):
    class FakePoint:
        def __init__(self, doc_id: str, payload: dict) -> None:
            self.id = doc_id
            self.payload = payload

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def get_collection(self, collection_name: str):
            raise RuntimeError("missing")

        def create_collection(self, **kwargs):
            self.calls.append(kwargs)

        def query_points(self, **kwargs):
            return type("Result", (), {"points": [FakePoint("doc-2", {"document_id": "doc-2", "file_name": "dense.txt", "file_type": "txt", "source_type": "text", "content": "Revenue surged in Q4.", "metadata": {}})]})( )

    retriever = HybridRetriever()
    retriever.qdrant.enabled = True
    retriever.qdrant.client = FakeClient()
    retriever.qdrant._embed = lambda text: [1.0, 0.0] if "revenue" in text.lower() else [0.0, 1.0]

    evidence = [
        Evidence(document_id="doc-1", file_name="policy.txt", file_type="txt", content="The process and policy require manager approval."),
        Evidence(document_id="doc-2", file_name="dense.txt", file_type="txt", content="Revenue surged in Q4."),
    ]

    ranked = retriever.search("revenue q4", evidence, top_k=2)

    assert ranked[0].file_name == "dense.txt"


def test_hybrid_retriever_uses_weighted_hybrid_scoring():
    retriever = HybridRetriever()

    strong = Evidence(
        document_id="strong",
        file_name="strong.txt",
        file_type="txt",
        content="Revenue increased by 12% in Q4 with strong growth.",
    )
    weak = Evidence(
        document_id="weak",
        file_name="weak.txt",
        file_type="txt",
        content="Revenue summary only.",
    )

    strong_score = retriever._hybrid_score("Q4 revenue growth", strong, dense=0.4, bm25=0.8, lexical=1.0)
    weak_score = retriever._hybrid_score("Q4 revenue growth", weak, dense=0.9, bm25=0.6, lexical=0.2)

    assert strong_score > weak_score


def test_qdrant_store_uses_live_client_from_settings(monkeypatch):
    import qdrant_client

    class FakeClient:
        def __init__(self, url: str, timeout: float | int = 10):
            self.url = url
            self.timeout = timeout
            self.collected = []

        def get_collection(self, collection_name: str):
            raise RuntimeError("missing")

        def create_collection(self, **kwargs):
            self.collected.append(kwargs)

    fake_factory = FakeClient
    monkeypatch.setattr(qdrant_client, "QdrantClient", fake_factory)

    store = __import__("app.retrieval.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore(
        collection_name="demo_collection",
        enabled=True,
        url="http://localhost:6333",
        timeout=7,
    )

    assert isinstance(store.client, FakeClient)
    assert store.client.url == "http://localhost:6333"
    assert store.client.timeout == 7
