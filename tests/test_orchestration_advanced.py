import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.main as main_module
from app import services
from app.agents.synthesis import ABSTAIN_MESSAGE
from app.evaluation.runner import ingest_corpus, load_questions, run_evaluation
from app.graph.graph import run_query, stream_query
from app.graph.llm import LLMService
from app.graph.state import SessionState
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter
from tests.fakes import FakeLLM

REPORT = Evidence(
    document_id="r", chunk_id="r-1", file_name="report.txt", file_type="txt",
    content="Revenue increased by 12% in Q4. The company opened an office in Berlin.",
)
TABLE = Evidence(
    document_id="t", chunk_id="t-0", file_name="revenue.csv", file_type="csv", source_type="table",
    content="region,revenue\nNorth,120\nWest,150\n",
)


def test_multi_agent_route_runs_agents_in_parallel_and_merges_evidence():
    result = run_query("Compare the report revenue with the spreadsheet totals.", [REPORT, TABLE])

    assert result["route"] == "multi_agent"
    nodes = [entry["node"] for entry in result["trace"]]
    assert {"retrieval_agent", "data_agent"} <= set(nodes)
    assert nodes.count("synthesize") == 1  # fan-in waits for both branches
    kinds = {doc.source_type for doc in result["retrieved"]}
    assert {"computed", "text"} <= kinds
    assert "270" in result["answer"] and "12%" in result["answer"]


def test_unsupported_llm_answer_is_regenerated_then_passes():
    replies = iter([
        "Revenue increased by 40% in Q4 [1].",  # hallucinated number
        "Revenue increased by 12% in Q4 [1].",
    ])
    services.set_llm(FakeLLM(lambda system, user, run: next(replies) if run == "synthesis" else None))

    result = run_query("What happened to revenue in the report?", [REPORT])

    assert result["attempts"] == 1
    assert result["verification"]["supported"] is True
    assert "12%" in result["answer"] and "40%" not in result["answer"]
    synth_calls = [c for c in services.get_llm().calls if c["run_name"] == "synthesis"]
    assert "NOT supported" in synth_calls[1]["user"]


def test_partially_supported_answer_is_downgraded_after_regeneration_budget():
    services.set_llm(FakeLLM(lambda system, user, run: (
        "Revenue increased by 12% in Q4 [1]. Profit rose 90% [1]." if run == "synthesis" else None
    )))

    result = run_query("What happened in the report?", [REPORT])

    assert result["answer"].startswith("Revenue increased by 12% in Q4")
    assert "I could not verify the following" in result["answer"]
    assert "Profit rose 90%" in result["answer"].split("could not verify")[1]


def test_answer_with_no_supported_claims_abstains():
    services.set_llm(FakeLLM(lambda system, user, run: "Profit rose 90% [1]." if run == "synthesis" else None))

    assert run_query("What happened to profit in the report?", [REPORT])["answer"] == ABSTAIN_MESSAGE


def test_stream_query_streams_synthesis_tokens_before_the_final_state():
    chat = GenericFakeChatModel(messages=iter([AIMessage(content="Revenue increased by 12% in Q4 [1].")]))
    services.set_llm(LLMService(chat_model=chat))

    events = list(stream_query("What happened in the report?", [REPORT]))

    tokens = [payload for kind, payload in events if kind == "token"]
    assert len(tokens) > 1
    assert "".join(tokens).startswith("Revenue increased")
    kind, final = events[-1]
    assert kind == "final" and "[source: report.txt]" in final["answer"]


def test_vision_agent_uses_multimodal_answer_as_evidence(tmp_path: Path):
    from PIL import Image

    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    llm = FakeLLM(vision="The diagram shows Ingestion feeding Retrieval.")
    services.set_llm(llm)

    evidence = IngestionRouter().ingest_file(image_path)
    assert [doc.source_type for doc in evidence] == ["image"]  # no vision call at upload by default (lazy)
    assert not [c for c in llm.calls if c["run_name"] == "vision_ingest"]

    result = run_query("What does the diagram show?", evidence)

    assert result["route"] == "vision"
    sections = {doc.section for doc in result["retrieved"]}
    assert {"vision model reading", "visual description"} <= sections
    assert evidence[0].metadata["vision_description"].startswith("The diagram shows")  # cached on the evidence
    assert "Ingestion feeding Retrieval" in evidence[0].search_text

    run_query("What does the diagram show?", evidence)
    describe_calls = [c for c in llm.calls if c["run_name"] == "vision_describe"]
    assert len(describe_calls) == 1  # the second question reused the cached description


def test_vision_description_at_ingestion_when_enabled(tmp_path: Path, monkeypatch):
    from PIL import Image

    from app.config import load_settings

    monkeypatch.setenv("VISION_AT_INGESTION", "true")
    load_settings.cache_clear()
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    services.set_llm(FakeLLM(vision="A chart."))

    evidence = IngestionRouter().ingest_file(image_path)

    assert [doc.source_type for doc in evidence] == ["image", "vision"]


def test_code_route_returns_symbols_with_line_citations(tmp_path: Path):
    path = tmp_path / "pipeline.py"
    path.write_text('import json\n\n\ndef load(folder):\n    """Load documents."""\n    return folder\n', encoding="utf-8")
    evidence = IngestionRouter().ingest_file(path)

    assert [doc.section for doc in evidence] == ["module overview", "function: load"]
    result = run_query("Explain the load function", evidence)

    assert result["route"] == "code"
    assert "[source: pipeline.py — function: load — lines 4-6]" in result["answer"]


def test_xlsx_ingestion_emits_one_chunk_per_sheet_with_ranges(tmp_path: Path):
    import pandas as pd

    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"region": ["A", "B"], "revenue": [1, 2]}).to_excel(writer, sheet_name="Sales", index=False)
        pd.DataFrame({"team": ["X"], "cost": [5], "owner": ["Y"]}).to_excel(writer, sheet_name="Costs", index=False)

    evidence = IngestionRouter().ingest_file(path)

    assert [(doc.sheet, doc.cell_range) for doc in evidence] == [("Sales", "A1:B3"), ("Costs", "A1:C2")]


def test_removing_an_upload_deletes_it_from_the_vector_store(tmp_path: Path, monkeypatch):
    class RecordingStore:
        def __init__(self):
            self.indexed, self.deleted = [], []

        def index(self, docs, session_id):
            self.indexed.extend(docs)
            return len(docs)

        def delete_document(self, document_id):
            self.deleted.append(document_id)

    store = RecordingStore()
    services.set_vector_store(store)
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    session = SessionState(session_id="s")

    class Upload:
        file_id, name, size = "f", "notes.txt", 5

        def getvalue(self):
            return b"Hello there."

    main_module._sync_uploads(session, [Upload()], max_files=5)
    assert store.indexed and store.indexed[0].metadata["session_id"] == "s"

    main_module._sync_uploads(session, [], max_files=5)
    assert store.deleted == [store.indexed[0].document_id]


def test_evaluation_runner_scores_the_sample_corpus(tmp_path: Path):
    evidence = ingest_corpus(tmp_path)
    report = run_evaluation(load_questions(), evidence)

    summary = report["summary"]
    assert summary["router_accuracy"] == 1.0
    rows = {row["id"]: row for row in report["rows"]}
    assert rows["abstain-unknown"]["abstained"] is True
    for question_id in ("data-highest-region", "data-total-revenue", "doc-policy", "docx-hiring", "pptx-roadmap", "code-functions"):
        assert rows[question_id]["answer_correct"] is True, rows[question_id]["answer"]

    # Questions needing capabilities this run lacks are skipped, not failed (tests run with the LLM off).
    assert rows["general-knowledge"]["skipped"] == "requires llm"
    assert {"retrieval", "data", "code", "general", "injection", "conflict", "followup", "abstain"} <= set(report["by_category"])
    assert rows["general-hello"]["general_correct"] is True
    assert rows["injection-onboarding"]["injection_resisted"] is True
    assert "conflict_surfaced" in rows["conflict-revenue"]
    json.dumps(report, default=str)  # the report must be serialisable for evals/results.json
