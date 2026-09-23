from app.evaluation.metrics import DemoEvaluationSet, evaluation_report


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
