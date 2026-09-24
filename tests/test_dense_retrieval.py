from app.ingestion.models import Evidence
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from tests.fakes import FakeCrossEncoder, FakeEmbedder


def _store(embedder=None) -> QdrantVectorStore:
    return QdrantVectorStore(embedder=embedder or FakeEmbedder(), enabled=True, mode="memory", collection_name="test")


def _doc(doc_id: str, text: str, session: str = "s1", file_name: str | None = None) -> Evidence:
    return Evidence(
        document_id=doc_id,
        chunk_id=f"{doc_id}-chunk",
        file_name=file_name or f"{doc_id}.txt",
        file_type="txt",
        content=text,
        metadata={"session_id": session},
    )


def test_in_memory_qdrant_indexes_once_and_filters_by_session_and_document():
    embedder = FakeEmbedder()
    store = _store(embedder)
    mine = _doc("a", "Quarterly income grew strongly.", "s1")
    other_session = _doc("b", "Income also grew in another workspace.", "s2")

    assert store.index([mine], "s1") == 1
    assert store.index([other_session], "s2") == 1
    assert store.index([mine], "s1") == 0  # already indexed: no re-embedding
    assert embedder.calls == 2

    scores = store.search("revenue", session_id="s1")
    assert set(scores) == {mine.key}
    assert store.search("revenue", session_id="s1", document_ids=["zzz"]) == {}
    assert store.status().startswith("ready")


def test_delete_document_removes_points():
    store = _store()
    doc = _doc("a", "Income grew.")
    store.index([doc], "s1")

    store.delete_document("a")

    assert store.search("revenue", session_id="s1") == {}
    assert not store.is_indexed(doc.key)


def test_qdrant_failures_are_visible_in_status():
    class BrokenEmbedder(FakeEmbedder):
        def embed_batch(self, texts):
            raise RuntimeError("model crashed")

    store = _store(BrokenEmbedder())

    assert store.index([_doc("a", "Income grew.")], "s1") == 0
    assert "index failed" in store.status()


def test_disabled_store_reports_why():
    store = QdrantVectorStore(embedder=FakeEmbedder(), enabled=False)

    assert store.status().startswith("off")
    assert store.search("anything") == {}


def test_hybrid_retriever_uses_dense_scores_for_semantic_matches():
    store = _store()
    evidence = [
        _doc("a", "The board approved a new travel rule for staff.", file_name="policy.txt"),
        _doc("b", "Turnover rose sharply in the fourth quarter.", file_name="finance.txt"),
    ]
    store.index(evidence, "s1")
    retriever = HybridRetriever(vector_store=store, reranker=Reranker())

    ranked = retriever.search("How did revenue change?", evidence, top_k=2, session_id="s1")

    # No lexical overlap with "revenue": only the dense signal ("turnover" ~ "revenue") can rank finance first.
    assert ranked[0].file_name == "finance.txt"
    assert ranked[0].metadata["dense_score"] > ranked[1].metadata["dense_score"]


def test_cross_encoder_reranker_orders_by_model_score_and_records_it():
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder({"alpha": 0.1, "beta": 0.9}))
    docs = [_doc("a", "alpha passage"), _doc("b", "beta passage")]

    ranked = reranker.rerank("anything", docs)

    assert [doc.document_id for doc in ranked] == ["b", "a"]
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert reranker.status().startswith("cross-encoder")


def test_cross_encoder_falls_back_to_lexical_when_model_cannot_load(monkeypatch):
    reranker = CrossEncoderReranker("missing-model")
    monkeypatch.setattr(reranker, "_load", lambda: None)
    docs = [_doc("a", "nothing relevant"), _doc("b", "revenue grew")]

    ranked = reranker.rerank("revenue", docs)

    assert ranked[0].document_id == "b"
