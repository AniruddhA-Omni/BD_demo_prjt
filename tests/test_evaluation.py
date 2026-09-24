from app.evaluation.metrics import recall_at_k, reciprocal_rank


def test_recall_and_rr_metrics_measure_retrieval_quality():
    relevant = ["doc-2", "doc-5"]
    retrieved = ["doc-9", "doc-2", "doc-7", "doc-5"]

    assert recall_at_k(relevant, retrieved, 5) == 1.0
    assert recall_at_k(relevant, retrieved, 2) == 0.5
    assert reciprocal_rank(relevant, retrieved) == 0.5


def test_metrics_handle_empty_and_missing_results():
    assert recall_at_k([], ["a"]) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0
    assert recall_at_k(["a"], []) == 0.0
    assert reciprocal_rank(["a"], ["b", "c"]) == 0.0
