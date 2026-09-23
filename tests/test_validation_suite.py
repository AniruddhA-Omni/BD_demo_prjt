from pathlib import Path

import pandas as pd

from app.agents.data import DataAgent
from app.agents.router import RouterAgent
from app.graph.graph import run_query
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter


def test_end_to_end_document_retrieval_flow():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="report.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4. Operating profit also rose.",
            metadata={"page": 5},
        )
    ]

    result = run_query("What happened in Q4?", evidence)

    assert result["route"] == "retrieval"
    assert result["verification"]["supported"] is True
    assert "12%" in result["answer"]


def test_structured_data_question_flow():
    evidence = [
        Evidence(
            document_id="doc-2",
            file_name="revenue.csv",
            file_type="csv",
            content="region,revenue\nNorth,120\nWest,150\n",
            source_type="table",
            metadata={"source_name": "revenue.csv"},
        )
    ]

    result = run_query("Which region had the highest revenue?", evidence)

    assert result["route"] == "data"
    assert result["answer"] == "West"


def test_multi_agent_cross_source_route():
    decision = RouterAgent().route_query("Compare the PDF revenue with the Excel Q4 totals and summarize the trend.")

    assert decision.route == "multi_agent"
    assert decision.needs_parallel is True
    assert set(["retrieval", "data"]).issubset(set(decision.selected_agents))


def test_python_code_ingestion_keeps_function_and_class_context(tmp_path: Path):
    py_path = tmp_path / "example.py"
    py_path.write_text(
        "def greet(name):\n    return f'Hello {name}'\n\nclass Demo:\n    pass\n",
        encoding="utf-8",
    )

    evidence = IngestionRouter().ingest_file(py_path)

    assert evidence[0].source_type == "code"
    assert "def greet" in evidence[0].content
    assert "class Demo" in evidence[0].content


def test_data_agent_handles_average_and_total_queries():
    df = pd.DataFrame({"region": ["North", "West", "South"], "revenue": [120, 150, 90]})

    highest = DataAgent().answer_question(df, "Which region had the highest revenue?")
    average = DataAgent().answer_question(df, "What is the average revenue?")
    total = DataAgent().answer_question(df, "What is the total revenue?")

    assert highest["answer"] == "West"
    assert average["answer"] == "120.0"
    assert total["answer"] == "360"
    assert total["value"] == 360.0
