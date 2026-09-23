from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.agents.synthesis import SynthesisAgent
from app.config import get_settings
from app.graph.state import SessionState
from app.ingestion.router import IngestionRouter


def build_demo_summary(route: str, selected_agents: list[str], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    source_names = [item.get("file_name") or item.get("name") or "unknown" for item in evidence]
    return {
        "route": route,
        "selected_agents": selected_agents,
        "sources": source_names,
        "trace_text": (
            f"Agent trace: route={route}; agents={', '.join(selected_agents) if selected_agents else 'default'}; "
            f"sources={', '.join(source_names) if source_names else 'none'}"
        ),
    }


def _init_session_state() -> None:
    if "session_state" not in st.session_state:
        st.session_state.session_state = SessionState(session_id=str(uuid.uuid4()))

    session = st.session_state.session_state
    if "messages" not in st.session_state:
        st.session_state.messages = session.conversation_messages

    if not session.conversation_messages and st.session_state.messages:
        session.conversation_messages = list(st.session_state.messages)

    if session.conversation_messages != st.session_state.messages:
        st.session_state.messages = list(session.conversation_messages)


def _render_sidebar(settings: dict[str, str | bool | int]) -> None:
    with st.sidebar:
        st.title("Controls")
        st.caption(settings["app_description"])

        uploaded = st.file_uploader(
            "Upload files",
            type=[
                "pdf",
                "docx",
                "txt",
                "md",
                "pptx",
                "csv",
                "xlsx",
                "png",
                "jpg",
                "jpeg",
                "py",
            ],
            accept_multiple_files=True,
        )

        if uploaded:
            file_names = [file.name for file in uploaded]
            st.session_state.session_state.uploaded_files = file_names
            st.success(f"{len(file_names)} file(s) selected")
            st.dataframe({"Files": file_names}, use_container_width=True)

        st.markdown("---")
        st.subheader("Session")
        st.text(f"Session ID: {st.session_state.session_state.session_id}")
        st.text(f"Max files: {settings['max_files_per_session']}")
        st.text(f"LangSmith: {'enabled' if settings['enable_langsmith'] else 'disabled'}")


def _render_chat() -> None:
    st.subheader("Chat")
    session = st.session_state.session_state

    for message in session.conversation_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a question about your uploaded documents")
    if prompt:
        user_message = {"role": "user", "content": prompt}
        session.conversation_messages.append(user_message)
        st.session_state.messages = list(session.conversation_messages)

        with st.chat_message("user"):
            st.write(prompt)

        router = RouterAgent().route_query(prompt)
        answer = f"Routing decision: {router.route} via {', '.join(router.selected_agents) or 'default'}"
        summary: dict[str, Any] | None = None

        if session.uploaded_files:
            evidence = []
            for file_name in session.uploaded_files:
                evidence.extend(IngestionRouter().ingest_file(file_name))

            if evidence:
                relevant = RetrievalAgent().retrieve(prompt, evidence, top_k=3)
                answer = SynthesisAgent().synthesize(prompt, relevant)
                evidence_labels = ", ".join(doc.file_name for doc in relevant[:3])
                summary = build_demo_summary(
                    router.route,
                    router.selected_agents,
                    [{"file_name": doc.file_name, "source_type": doc.source_type} for doc in relevant[:3]],
                )
                answer += f"\n\nRouter: {router.route} | Agents: {', '.join(router.selected_agents)} | Sources: {evidence_labels or 'none'}"
                answer += f"\n\n{summary['trace_text']}"

        if summary is None:
            summary = build_demo_summary(router.route, router.selected_agents, [])

        session.last_question = prompt
        session.last_answer = answer
        session.conversation_messages.append({"role": "assistant", "content": answer})
        st.session_state.messages = list(session.conversation_messages)

        with st.chat_message("assistant"):
            st.write(answer)

        if summary:
            with st.expander("Agent trace"):
                st.write(summary["trace_text"])


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title=settings["page_title"], page_icon="📄")  # type: ignore[arg-type]
    _init_session_state()

    st.title(settings["app_name"])
    st.caption(settings["app_description"])

    _render_sidebar(settings)
    _render_chat()


if __name__ == "__main__":
    main()
