from app.evaluation.metrics import (
    DemoEvaluationSet,
    evaluation_report,
    ragas_evaluation_report,
    retrieval_evaluation_report,
    recall_at_k,
    reciprocal_rank,
)


def test_demo_evaluation_set_has_expected_questions():
    dataset = DemoEvaluationSet()
    rows = dataset.rows()

    assert len(rows) >= 3
    assert any("highest revenue" in row["question"].lower() for row in rows)
    assert any("q4" in row["question"].lower() for row in rows)


def test_evaluation_report_scores_supported_outputs():
    dataset = DemoEvaluationSet()
    report = evaluation_report(dataset)

    assert report["total_questions"] >= 3
    assert report["supported_ratio"] >= 0.0
    assert "average_score" in report


def test_recall_and_rr_metrics_measure_retrieval_quality():
    relevant = ["doc-2", "doc-5"]
    retrieved = ["doc-9", "doc-2", "doc-7", "doc-5"]

    assert recall_at_k(relevant, retrieved, 5) == 1.0
    assert reciprocal_rank(relevant, retrieved) == 0.5


def test_retrieval_evaluation_report_aggregates_metrics():
    dataset = DemoEvaluationSet()
    retrieval_rows = dataset.retrieval_rows()

    report = retrieval_evaluation_report(retrieval_rows)

    assert "average_recall_at_3" in report
    assert "average_mrr" in report
    assert "false_positive_rate" in report
    assert 0.0 <= report["false_positive_rate"] <= 1.0


def test_ragas_evaluation_report_runs_with_local_fallback():
    rows = [
        {
            "question": "What happened in Q4?",
            "answer": "Revenue increased by 12% in Q4.",
            "context": ["Revenue increased by 12% in Q4."],
            "ground_truth": "Revenue increased by 12% in Q4.",
        }
    ]

    report = ragas_evaluation_report(rows)

    assert "metrics" in report
    assert "backend" in report
    assert "rows" in report
