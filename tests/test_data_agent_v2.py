import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

import app.main as main_module
from app.agents.data import DataAgent, DataPlan, TableRef, tables_from_evidence, unmatched_constraints, validate_plan
from app.graph.graph import run_query
from app.ingestion.router import IngestionRouter
from app.ingestion.tables import normalise_grid
from tests.fakes import FakeLLM

REGIONS = pd.DataFrame({"region": ["North", "South", "East", "West"], "revenue": [120, 90, 110, 150], "quarter": ["Q4"] * 4})
QUARTERS = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4"], "revenue": [380, 410, 400, 470]})


def _xlsx(path: Path, rows: list[list], title: str = "Sheet1") -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = title
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


# ---------------------------------------------------------------- header detection


def test_header_detection_skips_title_rows_and_records_position(tmp_path: Path):
    path = _xlsx(tmp_path / "budget.xlsx", [["FY2026 Budget"], [], ["department", "budget"], ["Eng", 500], ["Sales", 250]], "Budget")

    [evidence] = IngestionRouter().ingest_file(path)

    assert evidence.content.splitlines()[0] == "department,budget"
    assert evidence.cell_range == "A3:B5"
    assert evidence.section == "FY2026 Budget"
    assert (evidence.metadata["header_row"], evidence.metadata["first_col"]) == (3, 1)


def test_header_detection_handles_offset_columns_and_gaps():
    grid = [
        [None, None, None],
        [None, "item", "amount"],
        [None, "a", "1,200"],
        [None, None, None],  # blank row inside the data
        [None, "b", "300"],
    ]

    section = normalise_grid(grid)

    assert list(section.frame.columns) == ["item", "amount"]
    assert section.frame["amount"].tolist() == [1200, 300]  # "1,200" parsed as a number
    assert (section.header_row, section.first_col) == (2, 2)
    assert section.row_numbers == [3, 5]


def test_csv_with_a_title_row_is_normalised(tmp_path: Path):
    path = tmp_path / "sales.csv"
    path.write_text("Quarterly sales export\nregion,revenue\nNorth,120\nWest,150\n", encoding="utf-8")

    [evidence] = IngestionRouter().ingest_file(path)

    assert evidence.content.splitlines() == ["region,revenue", "North,120", "West,150"]
    assert evidence.cell_range == "A2:B4"


# ---------------------------------------------------------------- operations


def test_top_n_and_bottom_n():
    top = DataAgent().answer_question(REGIONS, "What are the top 2 regions by revenue?")
    bottom = DataAgent().answer_question(REGIONS, "Which are the bottom 1 regions by revenue?")

    assert top["answer"] == "West (150), North (120)"
    assert [row["region"] for row in top["rows"]] == ["West", "North"]
    assert bottom["answer"] == "South (90)"


def test_percentage_change_and_difference_between_two_values():
    growth = DataAgent().answer_question(QUARTERS, "What was the revenue growth from Q3 to Q4?")
    difference = DataAgent().answer_question(QUARTERS, "What is the revenue difference from Q1 to Q4?")

    assert growth["answer"] == "+17.5%"
    assert "400 (Q3) to 470 (Q4)" in growth["explanation"]
    assert difference["answer"] == "+90"


def test_numeric_filters_and_counts():
    result = DataAgent().answer_question(REGIONS, "How many regions have revenue above 100?")

    assert result["value"] == 3
    assert "revenue > 100" in result["explanation"]


def test_grouped_sum_returns_a_table():
    frame = pd.DataFrame({"region": ["North", "North", "West"], "revenue": [100, 20, 150]})

    result = DataAgent().answer_question(frame, "What is the total revenue by region?")

    assert result["answer"] == "North: 120; West: 150"
    assert result["rows"] == [{"region": "North", "revenue": 120}, {"region": "West", "revenue": 150}]


def test_average_per_unique_group_is_an_average_across_groups():
    frame = pd.DataFrame({"department": ["Eng", "Sales", "Support"], "cost": [300, 180, 120]})

    result = DataAgent().answer_question(frame, "What is the average cost per department?")

    assert result["answer"] == "200.0"


def test_unmatched_year_or_quarter_is_refused_instead_of_summing_everything():
    assert unmatched_constraints("total revenue in 2019", REGIONS) == ["2019"]
    assert unmatched_constraints("revenue in Q4", REGIONS) == []

    result = DataAgent().answer_question(REGIONS, "What was the total revenue in 2019?")

    assert result["value"] is None and "2019" in result["answer"]


def test_cell_ranges_follow_the_header_offset(tmp_path: Path):
    path = _xlsx(tmp_path / "budget.xlsx", [["Title"], [], ["department", "budget"], ["Eng", 500], ["Sales", 250]])
    evidence = IngestionRouter().ingest_file(path)

    result = DataAgent().answer_from_evidence("Which department has the highest budget?", evidence)

    assert result["answer"] == "Eng"
    assert result["cell_range"] == "A4:B4"  # row 4 in the sheet, not row 2
    assert result["evidence"][0].metadata["rows"] == [{"department": "Eng", "budget": 500}]


# ---------------------------------------------------------------- plan validation


def test_validate_plan_reports_every_problem():
    table = TableRef(name="t", frame=REGIONS)
    plan = DataPlan(table="t", operation="sum", column="region", group_by="missing", filters=[{"column": "nope", "op": "like"}])

    errors = validate_plan(plan, table)

    assert any("not a numeric column" in e for e in errors)
    assert any("group_by" in e for e in errors)
    assert any("filter column" in e for e in errors)
    assert any("unknown filter op" in e for e in errors)
    assert validate_plan(DataPlan(table="t", operation="max", column="revenue", label_column="region"), table) == []


def test_invalid_llm_plan_falls_back_to_rules(tmp_path: Path):
    llm = FakeLLM(lambda s, u, r: json.dumps({"table": "sales.csv", "operation": "sum", "column": "region"}))
    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nNorth,120\nWest,150\n", encoding="utf-8")

    result = DataAgent(llm=llm).answer_from_evidence("What is the total revenue?", IngestionRouter().ingest_file(path))

    assert result["answer"] == "270"
    assert result["plan"]["method"] == "rules"


def test_llm_plan_with_top_n_and_operator_filters(tmp_path: Path):
    llm = FakeLLM(lambda s, u, r: json.dumps({
        "table": "sales.csv", "operation": "top_n", "column": "revenue", "label_column": "region", "limit": 1,
        "filters": [{"column": "revenue", "op": "lt", "value": 140}],
    }))
    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nNorth,120\nWest,150\nSouth,90\n", encoding="utf-8")

    result = DataAgent(llm=llm).answer_from_evidence("Best region below 140?", IngestionRouter().ingest_file(path))

    assert result["answer"] == "North (120)"
    assert result["plan"]["method"] == "llm"


def test_tables_from_evidence_carry_sheet_position(tmp_path: Path):
    path = _xlsx(tmp_path / "b.xlsx", [["Title"], ["a", "b"], ["x", 1]])

    [table] = tables_from_evidence(IngestionRouter().ingest_file(path))

    assert (table.header_row, table.first_col) == (2, 1)
    assert table.excel_row(0) == 3


# ---------------------------------------------------------------- end to end + UI


def test_result_rows_reach_the_ui_details(tmp_path: Path):
    path = tmp_path / "sales.csv"
    path.write_text("region,revenue\nNorth,120\nWest,150\nSouth,90\n", encoding="utf-8")

    result = run_query("What are the top 2 regions by revenue?", IngestionRouter().ingest_file(path))
    details = main_module._answer_details(result)

    assert result["answer"] == "West (150), North (120)"
    assert details["tables"][0]["rows"] == [{"region": "West", "revenue": 150}, {"region": "North", "revenue": 120}]
