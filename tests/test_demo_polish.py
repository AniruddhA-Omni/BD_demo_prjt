from app.main import build_demo_summary


def test_build_demo_summary_includes_route_and_sources():
    summary = build_demo_summary(
        route="multi_agent",
        selected_agents=["retrieval", "data"],
        evidence=[
            {"file_name": "report.txt", "source_type": "text"},
            {"file_name": "revenue.csv", "source_type": "table"},
        ],
    )

    assert summary["route"] == "multi_agent"
    assert "retrieval" in summary["selected_agents"]
    assert "report.txt" in summary["sources"]
    assert "revenue.csv" in summary["sources"]
    assert "Agent trace" in summary["trace_text"]
