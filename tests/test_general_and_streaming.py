from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.main as main_module
from app import services
from app.agents.general import GENERAL_NOTICE
from app.agents.router import RouterAgent
from app.agents.synthesis import replace_citation_markers
from app.graph.graph import run_query, stream_query
from app.graph.llm import LLMService
from app.graph.state import SessionState
from app.ingestion.models import Evidence
from tests.fakes import FakeLLM

REPORT = Evidence(
    document_id="r", chunk_id="r-1", file_name="report.txt", file_type="txt", section="Results",
    content="Revenue increased by 12% in Q4.",
)


def _router_llm(agents: list[str]) -> FakeLLM:
    return FakeLLM(lambda system, user, run: '{"agents": %s, "reason": "llm"}' % str(agents).replace("'", '"'))


def test_greetings_and_meta_questions_route_to_general_without_llm():
    router = RouterAgent()

    for message in ["hi", "Hello there!", "thanks a lot", "What can you do?", "who are you"]:
        assert router.route_query(message, available_types={"text"}).route == "general", message


def test_no_files_and_no_document_signal_routes_to_general():
    assert RouterAgent().route_query("What is the capital of France?", available_types=set()).route == "general"


def test_document_questions_never_default_to_general():
    router = RouterAgent()
    # With files uploaded and no LLM, an unmatched question still goes to grounded retrieval.
    assert router.route_query("What is the capital of France?", available_types={"text"}).route == "retrieval"
    # A document-flavoured question with no files abstains via retrieval rather than answering from memory.
    assert router.route_query("What happened in Q4?", available_types=set()).route == "retrieval"
    assert router.route_query("What does the contract say?", available_types=set()).route == "retrieval"


def test_weak_document_signal_lets_the_llm_choose_general():
    decision = RouterAgent(llm=_router_llm(["general"])).route_query("Explain what a vector database is", available_types={"text"})

    assert decision.route == "general" and decision.method == "llm"


def test_strong_document_signal_is_not_overridden_by_the_llm():
    llm = _router_llm(["general"])

    decision = RouterAgent(llm=llm).route_query("Explain the travel policy", available_types={"text"})

    assert decision.route == "retrieval"
    assert not llm.calls


def test_llm_cannot_mix_general_with_grounded_agents():
    decision = RouterAgent(llm=_router_llm(["general", "retrieval"])).route_query("How are we doing?", available_types={"text"})

    assert decision.selected_agents == ["retrieval"]


def test_general_answers_can_be_disabled():
    decision = RouterAgent(allow_general=False).route_query("hello", available_types=set())

    assert decision.route == "retrieval"


def test_general_route_answers_without_sources_or_verification():
    services.set_llm(FakeLLM(lambda system, user, run: "Paris is the capital of France." if run == "general_answer" else None))
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]

    result = run_query("What is the capital of France?", [], history=history)

    assert result["route"] == "general"
    assert result["answer"] == "Paris is the capital of France."
    assert result["retrieved"] == []
    assert result["verification"] == {"supported": None, "general": True, "reason": GENERAL_NOTICE, "claims": []}
    assert [entry["node"] for entry in result["trace"]] == ["contextualize", "route", "general_agent", "finalize"]
    general_call = next(call for call in services.get_llm().calls if call["run_name"] == "general_answer")
    assert general_call["history"] == history


def test_general_route_without_llm_uses_built_in_replies():
    assert "I can answer questions about the files" in run_query("hello", [REPORT])["answer"]
    assert run_query("thanks!", [REPORT])["answer"].startswith("You're welcome")
    assert "need the local LLM" in run_query("Who won the 1998 World Cup?", [])["answer"]


def test_stream_query_streams_general_answer_tokens_and_steps():
    services.set_llm(LLMService(chat_model=GenericFakeChatModel(messages=iter([AIMessage(content="Hello! Ask me about your files.")]))))

    events = list(stream_query("hi", []))

    kinds = [kind for kind, _ in events]
    assert kinds.count("token") > 1
    assert [payload["node"] for kind, payload in events if kind == "step"] == ["contextualize", "route", "general_agent", "finalize"]
    assert events[-1][0] == "final" and events[-1][1]["answer"] == "Hello! Ask me about your files."


def test_stream_query_sends_citation_labels_before_synthesis_tokens():
    chat = GenericFakeChatModel(messages=iter([AIMessage(content="Revenue increased by 12% in Q4 [1].")]))
    services.set_llm(LLMService(chat_model=chat))

    events = list(stream_query("What happened in the report?", [REPORT]))

    kinds = [kind for kind, _ in events]
    assert kinds.index("citations") < kinds.index("token")
    labels = next(payload for kind, payload in events if kind == "citations")
    streamed = "".join(payload for kind, payload in events if kind == "token")
    assert replace_citation_markers(streamed, labels).endswith("[source: report.txt — Results].")


def test_replace_citation_markers_handles_partial_and_invalid_markers():
    labels = ["a.pdf — page 1", "b.txt"]

    assert replace_citation_markers("x [1, 2] y [7]", labels) == "x [source: a.pdf — page 1] [source: b.txt] y "
    assert replace_citation_markers("still streaming [2", labels) == "still streaming [2"


def test_ui_helpers_describe_steps_and_exclude_current_question_from_history():
    assert main_module.describe_step({"node": "route", "agents": ["general"], "method": "rules", "duration_ms": 3}).startswith("Routing → general (rules)")
    assert "rewriting" in main_module.describe_step({"node": "verify", "support_ratio": 0.5, "regenerate": True, "duration_ms": 1})

    session = SessionState(session_id="s")
    session.conversation_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!", "details": {}},
        {"role": "user", "content": "current question"},
    ]
    assert main_module._history(session) == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]


def test_no_files_goes_straight_to_general_without_asking_the_llm():
    llm = _router_llm(["retrieval"])

    decision = RouterAgent(llm=llm).route_query("What is a vector database?", available_types=set())

    assert decision.route == "general"
    assert not llm.calls  # with nothing uploaded, grounded agents can't answer, so the LLM isn't consulted
