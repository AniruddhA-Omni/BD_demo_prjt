import json
import os

import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import services
from app.agents.data import DataAgent
from app.agents.router import RouterAgent
from app.agents.synthesis import ABSTAIN_MESSAGE, SynthesisAgent
from app.agents.verification import VerificationAgent
from app.config import Settings
from app.graph.llm import LLMService, parse_json_object, strip_reasoning, visible_stream_text
from app.ingestion.models import Evidence
from app.observability import configure_tracing
from tests.fakes import FakeLLM

REPORT = Evidence(
    document_id="r",
    chunk_id="r-1",
    file_name="report.pdf",
    file_type="pdf",
    page=3,
    section="Revenue Outlook",
    content="Revenue increased by 12% in Q4, driven by enterprise sales.",
)
NOTES = Evidence(document_id="n", chunk_id="n-1", file_name="notes.txt", file_type="txt", content="The team hired 40 engineers.")


def test_llm_helpers_parse_json_and_strip_reasoning():
    assert parse_json_object('Sure! ```json\n{"agents": ["data"], "reason": "x"}\n```') == {"agents": ["data"], "reason": "x"}
    assert parse_json_object("no json here") is None
    assert strip_reasoning("<think>hmm</think>Answer") == "Answer"
    assert visible_stream_text("<think>still thinking") == ""


def test_llm_service_is_unavailable_when_disabled():
    service = LLMService(settings=Settings(enable_llm=False))

    assert service.available() is False
    assert service.complete("s", "u") is None
    assert "disabled" in service.status()


def test_llm_service_uses_injected_chat_model_and_degrades_on_failure():
    service = LLMService(chat_model=GenericFakeChatModel(messages=iter([AIMessage(content="<think>x</think>Hello")])))
    assert service.complete("system", "user") == "Hello"

    class Exploding:
        def invoke(self, *args, **kwargs):
            raise ConnectionError("ollama down")

    broken = LLMService(chat_model=Exploding())
    assert broken.complete("system", "user") is None
    assert broken.available() is False
    assert "ollama down" in broken.last_error


def test_synthesis_replaces_numbered_markers_with_metadata_citations():
    llm = FakeLLM(lambda system, user, run: "Revenue rose 12% in Q4 [1]. The team hired 40 engineers [2, 9].")

    result = SynthesisAgent(llm=llm).compose("What happened?", [REPORT, NOTES])

    assert result.used_llm
    assert "[source: report.pdf — page 3 — Revenue Outlook]" in result.answer
    assert "[source: notes.txt]" in result.answer
    assert "[9]" not in result.answer and "[2" not in result.answer
    assert "Use ONLY the numbered evidence" in llm.calls[0]["system"]


def test_synthesis_abstains_when_llm_reports_insufficient_evidence():
    llm = FakeLLM(lambda system, user, run: "INSUFFICIENT_EVIDENCE")

    assert SynthesisAgent(llm=llm).compose("Who is the CEO?", [REPORT]).answer == ABSTAIN_MESSAGE


def test_router_falls_back_to_llm_when_no_rule_fires():
    llm = FakeLLM(lambda system, user, run: '{"agents": ["retrieval", "data"], "reason": "needs both"}')

    decision = RouterAgent(llm=llm).route_query("How did we do?")

    assert decision.route == "multi_agent"
    assert decision.method == "llm"
    assert decision.needs_parallel


def test_router_drops_agents_without_matching_files():
    decision = RouterAgent().route_query("Which region had the highest revenue?", available_types={"text"})

    assert decision.route == "retrieval"
    assert "no matching files" in decision.reason


def test_router_matches_whole_words_only():
    # "growth" and "drove" contain "row", which used to trigger the spreadsheet route.
    assert RouterAgent().route_query("What drove the growth according to the report?").route == "retrieval"
    assert RouterAgent().route_query("Explain what load_documents does").route == "code"


def test_verification_checks_each_claim_and_its_citations():
    answer = (
        "Revenue increased by 12% in Q4 [source: report.pdf — page 3 — Revenue Outlook]. "
        "Profit doubled to 80% [source: report.pdf — page 3 — Revenue Outlook]. "
        "The team hired 40 engineers [source: made-up.pdf]."
    )

    verdict = VerificationAgent().verify_answer(answer, [REPORT, NOTES])

    claims = verdict["claims"]
    assert [claim["supported"] for claim in claims] == [True, False, False]
    assert "numbers" in claims[1]["reason"]
    assert "unknown source" in claims[2]["reason"]
    assert verdict["support_ratio"] == round(1 / 3, 3)
    assert verdict["supported"] is False


def test_verification_llm_judge_cannot_override_invented_numbers():
    llm = FakeLLM(lambda system, user, run: json.dumps({"claims": [{"id": 1, "supported": True}, {"id": 2, "supported": True}]}))

    verdict = VerificationAgent(llm=llm).verify_answer("Revenue increased by 12% in Q4. Revenue increased by 55% in Q1.", [REPORT])

    assert [claim["supported"] for claim in verdict["claims"]] == [True, False]
    assert verdict["claims"][0]["method"] == "llm"


def test_data_agent_executes_llm_plan_with_filters_and_cites_cells():
    llm = FakeLLM(lambda system, user, run: json.dumps({
        "table": "sales.xlsx [Q4]", "operation": "sum", "column": "revenue",
        "label_column": None, "group_by": None, "filters": [{"column": "region", "equals": "West"}],
    }))
    table = Evidence(
        document_id="x", chunk_id="x-0", file_name="sales.xlsx", file_type="xlsx", source_type="table", sheet="Q4",
        content="region,revenue\nWest,100\nEast,50\nWest,25\n",
    )

    result = DataAgent(llm=llm).answer_from_evidence("What did the West region make in total?", [table])

    assert result["answer"] == "125"
    assert result["plan"]["method"] == "llm"
    computed = result["evidence"][0]
    assert computed.source_type == "computed"
    assert computed.sheet == "Q4" and computed.cell_range == "B2:B4"


def test_data_agent_rules_group_by_and_count():
    frame = pd.DataFrame({"region": ["North", "West", "North"], "revenue": [100, 150, 80]})

    grouped = DataAgent().answer_question(frame, "Which region had the highest revenue?")
    count = DataAgent().answer_question(frame, "How many entries are listed for North?")

    assert grouped["answer"] == "North"  # 180 once North's rows are summed, although West has the largest single row
    assert grouped["value"] == 180
    assert count["value"] == 2  # "North" in the question becomes a region filter


def test_configure_tracing_exports_langsmith_environment(monkeypatch):
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(key, raising=False)

    enabled = configure_tracing(Settings(enable_langsmith=True, langchain_tracing=True, langchain_api_key="k", langchain_project="p"))
    assert enabled and os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "p"

    assert configure_tracing(Settings(enable_langsmith=True, langchain_api_key="")) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_services_can_be_replaced_with_fakes():
    fake = FakeLLM()
    services.set_llm(fake)

    assert services.get_llm() is fake
