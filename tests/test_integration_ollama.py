"""Integration tests against a real local Ollama (plan item 0.2).

Excluded from the default run (see ``[tool.pytest.ini_options]`` in pyproject.toml). Run with:

    uv run python -m pytest -m integration

They skip with a clear reason when Ollama isn't reachable or the configured model (``OLLAMA_LLM_MODEL`` /
``LLM_MODEL``) isn't pulled. Small local models word things differently from run to run, so assertions check
behaviour (routing, citations, verification, streaming, rewriting) rather than exact text.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

from app import services
from app.agents.memory import QuestionRewriter
from app.agents.router import RouterAgent
from app.agents.verification import VerificationAgent
from app.config import Settings, load_settings
from app.graph.graph import run_query, stream_query
from app.graph.llm import LLMService
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter

pytestmark = pytest.mark.integration

REPORT = Evidence(
    document_id="r", chunk_id="r-1", file_name="annual_report.md", file_type="md", section="Revenue Outlook",
    content="Revenue increased by 12% in Q4 2025, driven by enterprise sales and the launch of the analytics add-on.",
)
POLICY = Evidence(
    document_id="p", chunk_id="p-1", file_name="policy.txt", file_type="txt",
    content="All production changes require manager approval before deployment.",
)
_CITATION = re.compile(r"\[source:\s*([^\]]+)\]")


@functools.cache
def _ollama_status() -> tuple[bool, str]:
    """Checked once per test session, so an absent Ollama costs one connection timeout rather than one per test."""
    service = LLMService(settings=Settings(enable_llm=True))
    return service.available(), service.status()


@pytest.fixture(autouse=True)
def live_llm(monkeypatch):
    """Undo the offline pins from tests/conftest.py for the LLM only, and skip unless Ollama is really usable."""
    ready, status = _ollama_status()
    if not ready:
        pytest.skip(f"Ollama not usable: {status}")
    monkeypatch.setenv("ENABLE_LLM", "true")
    load_settings.cache_clear()
    services.reset()
    return services.get_llm()


def test_grounded_answer_cites_the_evidence():
    result = run_query("What drove revenue growth in Q4 according to the report?", [REPORT, POLICY])

    assert result["route"] == "retrieval"
    assert result["used_llm"] is True
    assert "annual_report.md" in " ".join(_CITATION.findall(result["answer"]))
    assert "enterprise" in result["answer"].lower()
    assert result["verification"]["support_ratio"] > 0


def test_unanswerable_question_abstains_instead_of_guessing():
    result = run_query("What is the name of the CEO's dog according to the report?", [REPORT, POLICY])

    answer = result["answer"].lower()
    assert "couldn't find sufficient evidence" in answer or "insufficient evidence" in answer or "could not verify" in answer


def test_general_question_is_answered_without_sources():
    result = run_query("What is a vector database?", [])

    assert result["route"] == "general"
    assert result["used_llm"] is True
    assert result["answer"].strip() and not _CITATION.findall(result["answer"])
    assert result["verification"].get("general") is True


def test_data_question_reports_the_computed_value_with_a_citation(tmp_path: Path):
    path = tmp_path / "revenue.csv"
    path.write_text("region,revenue\nNorth,120\nWest,150\nSouth,90\n", encoding="utf-8")

    result = run_query("Which region had the highest revenue?", IngestionRouter().ingest_file(path))

    assert result["route"] == "data"
    assert "west" in result["answer"].lower()
    assert any(doc.source_type == "computed" for doc in result["retrieved"])


def test_regeneration_round_trip_removes_an_invented_number():
    class HallucinateOnce:
        """Wraps the real service; the first synthesis draft is replaced by one with an invented figure."""

        def __init__(self, real):
            self.real, self.injected, self.runs = real, False, []

        def __getattr__(self, name):
            return getattr(self.real, name)

        def complete(self, system, user, *, run_name="llm", tags=None, history=None):
            self.runs.append(run_name)
            if run_name == "synthesis" and not self.injected:
                self.injected = True
                return "Revenue increased by 45% in Q4 [1]."
            return self.real.complete(system, user, run_name=run_name, tags=tags, history=history)

    wrapper = HallucinateOnce(services.get_llm())
    services.set_llm(wrapper)

    result = run_query("How much did revenue grow in Q4 according to the report?", [REPORT])

    assert result["attempts"] == 1  # verification rejected the draft and asked for a rewrite
    assert wrapper.runs.count("synthesis") == 2
    assert "45%" not in result["answer"]


def test_stream_query_streams_real_tokens():
    events = list(stream_query("What does the policy require before deployment?", [REPORT, POLICY]))

    tokens = [payload for kind, payload in events if kind == "token"]
    assert len(tokens) > 3
    assert events[-1][0] == "final" and "approval" in events[-1][1]["answer"].lower()


def test_follow_up_is_rewritten_into_a_standalone_question():
    history = [
        {"role": "user", "content": "How did revenue change in Q4 according to the report?"},
        {"role": "assistant", "content": "Revenue increased by 12% in Q4 2025."},
    ]

    rewrite = QuestionRewriter().rewrite("What drove it?", history)
    result = run_query("What drove it?", [REPORT, POLICY], history=history)

    assert rewrite.method == "llm"
    assert "revenue" in rewrite.question.lower()
    assert "enterprise" in result["answer"].lower()


def test_router_uses_the_llm_when_no_rule_fires():
    decision = RouterAgent().route_query("How are we doing overall?", available_types={"text", "table"})

    assert decision.method == "llm"
    assert set(decision.selected_agents) <= {"retrieval", "data", "vision", "code", "general"}


def test_llm_judge_rejects_an_unsupported_claim():
    verdict = VerificationAgent().verify_answer("The CEO resigned after the Q4 results.", [REPORT])

    assert verdict["supported"] is False


def test_vision_model_reads_an_image(tmp_path: Path):
    from PIL import Image, ImageDraw

    path = tmp_path / "diagram.png"
    image = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(["Ingestion", "Retrieval", "Synthesis"]):
        draw.rectangle([20 + index * 155, 50, 145 + index * 155, 110], outline="black", width=3)
        draw.text((40 + index * 155, 72), label, fill="black")
    image.save(path)

    description = services.get_llm().describe_images("List the words written in the boxes.", [str(path)])
    if description is None:
        pytest.skip("The configured model did not return an image description (it may not support vision).")
    assert description.strip()
