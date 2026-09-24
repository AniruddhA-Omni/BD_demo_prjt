import app.main as main_module
from app import services
from app.agents.memory import SUMMARY_MAX_CHARS, QuestionRewriter, looks_like_follow_up, update_summary
from app.graph.graph import run_query
from app.graph.state import SessionState
from app.ingestion.models import Evidence
from tests.fakes import FakeLLM

HISTORY = [
    {"role": "user", "content": "How did revenue change in Q4 according to the report?"},
    {"role": "assistant", "content": "Revenue increased by 12% in Q4. [source: report.md — Revenue Outlook]"},
]
REPORT = Evidence(
    document_id="r", chunk_id="r-1", file_name="report.md", file_type="md", section="Revenue Outlook",
    content="Revenue increased by 12% in Q4, driven by enterprise sales.",
)
POLICY = Evidence(document_id="p", chunk_id="p-1", file_name="policy.txt", file_type="txt", content="Changes need manager approval.")


def test_follow_up_detection():
    assert looks_like_follow_up("What drove it?", HISTORY)
    assert looks_like_follow_up("And in Q3?", HISTORY)
    assert looks_like_follow_up("Why?", HISTORY)
    assert not looks_like_follow_up("What does the policy require before deployment?", HISTORY)
    assert not looks_like_follow_up("thanks!", HISTORY)  # social messages go to the general agent untouched
    assert not looks_like_follow_up("What drove it?", [])  # no conversation yet


def test_standalone_questions_are_left_alone():
    llm = FakeLLM(lambda s, u, r: "should not be called")

    rewrite = QuestionRewriter(llm=llm).rewrite("What does the policy require before deployment?", HISTORY)

    assert rewrite.method == "unchanged" and not llm.calls


def test_llm_rewrite_is_cleaned_and_used():
    llm = FakeLLM(lambda s, u, r: 'Standalone question: "What drove the Q4 revenue increase in the report?"\nExtra text')

    rewrite = QuestionRewriter(llm=llm).rewrite("What drove it?", HISTORY, summary="User asked about revenue.")

    assert rewrite.question == "What drove the Q4 revenue increase in the report?"
    assert rewrite.method == "llm"
    assert "Summary of earlier conversation" in llm.calls[0]["user"]
    assert "[source:" not in llm.calls[0]["user"]  # citations are stripped from the conversation shown to the model


def test_without_llm_the_previous_question_is_added_as_context():
    rewrite = QuestionRewriter().rewrite("What drove it?", HISTORY)

    assert rewrite.method == "heuristic"
    assert rewrite.question == "What drove it? (regarding: How did revenue change in Q4 according to the report?)"


def test_unusable_llm_rewrite_falls_back_to_the_heuristic():
    llm = FakeLLM(lambda s, u, r: "x" * 1000)

    assert QuestionRewriter(llm=llm).rewrite("What drove it?", HISTORY).method == "heuristic"


def test_graph_uses_the_rewritten_question_but_keeps_the_original():
    services.set_llm(FakeLLM(lambda s, u, r: {
        "rewrite_question": "What drove the Q4 revenue increase according to the report?",
    }.get(r)))

    result = run_query("What drove it?", [REPORT, POLICY], history=HISTORY)

    assert result["question"] == "What drove it?"
    assert result["standalone_question"] == "What drove the Q4 revenue increase according to the report?"
    assert result["trace"][0] == {**result["trace"][0], "node": "contextualize", "rewritten": True, "method": "llm"}
    assert result["retrieved"][0].file_name == "report.md"


def test_follow_up_is_answered_without_an_llm():
    result = run_query("What drove it?", [REPORT, POLICY], history=HISTORY)

    assert result["rewrite_method"] == "heuristic"
    assert "enterprise sales" in result["answer"]


def test_summary_is_bounded_and_needs_the_llm():
    older = [{"role": "user", "content": "What are the risks?"}, {"role": "assistant", "content": "Supply chain delays."}]

    assert update_summary("", older) is None  # LLM disabled in tests
    summary = update_summary("", older, llm=FakeLLM(lambda s, u, r: "S" * (SUMMARY_MAX_CHARS + 500)))
    assert len(summary) == SUMMARY_MAX_CHARS


def test_ui_folds_old_turns_into_the_summary(monkeypatch):
    monkeypatch.setenv("GENERAL_HISTORY_TURNS", "2")
    from app.config import load_settings

    load_settings.cache_clear()
    session = SessionState(session_id="s")
    session.conversation_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(6)
    ]

    assert main_module.update_memory(session) is False  # no LLM: only the recent window is used

    llm = FakeLLM(lambda s, u, r: "Discussed turns 0-3." if r == "summarize_conversation" else None)
    services.set_llm(llm)
    assert main_module.update_memory(session) is True
    assert session.summary == "Discussed turns 0-3." and session.summarized_upto == 4
    assert "turn 3" in llm.calls[0]["user"] and "turn 4" not in llm.calls[0]["user"]
    assert main_module.update_memory(session) is False  # nothing new left the window


def test_summary_reaches_the_general_agent():
    llm = FakeLLM(lambda s, u, r: "Sure." if r == "general_answer" else None)
    services.set_llm(llm)

    run_query("thanks!", [], history=[], summary="We compared Q3 and Q4 revenue.")

    call = next(c for c in llm.calls if c["run_name"] == "general_answer")
    assert "We compared Q3 and Q4 revenue." in call["history"][0]["content"]


def test_progress_line_for_contextualize():
    assert main_module.describe_step({"node": "contextualize", "rewritten": True, "method": "llm", "duration_ms": 2}).startswith(
        "Understanding the question → rewritten using the conversation (llm)"
    )
