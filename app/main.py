from __future__ import annotations

import functools
import logging
import platform
import shutil
import tempfile
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import streamlit as st

from app import services
from app.agents.memory import update_summary
from app.agents.synthesis import format_citation, replace_citation_markers
from app.config import Settings, load_settings
from app.graph.graph import stream_query
from app.graph.llm import visible_stream_text
from app.graph.state import IngestedFile, SessionState
from app.ingestion.background import IngestionManager
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter
from app.logging_config import configure_logging, recent_logs, set_session
from app.observability import configure_tracing, tracing_enabled

# Explicit name: Streamlit executes this file as "__main__", which would fall outside the configured "app" logger.
logger = logging.getLogger("app.main")

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "bd_demo_uploads"
MAX_FILE_BYTES = 50 * 1024 * 1024
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "pptx", "csv", "xlsx", "png", "jpg", "jpeg", "py"]
# Spreadsheets are answered by the data agent from their DataFrames, so they aren't embedded as text.
_UNINDEXED_SOURCE_TYPES = {"table", "error", "empty", "unknown"}


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


def ingest_upload(session: SessionState, file_id: str, name: str, data: bytes) -> IngestedFile:
    """Persist an uploaded file under the session's upload folder, ingest it once and index its chunks."""
    target = UPLOAD_ROOT / session.session_id / file_id / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    logger.info("Upload received: %s (%.1f KB)", target.name, len(data) / 1024)

    evidence = IngestionRouter().ingest_file(target)
    for item in evidence:
        item.metadata["session_id"] = session.session_id
    statuses = {str(item.metadata.get("status", "")) for item in evidence}
    status = "parsed" if "parsed" in statuses else (next(iter(statuses)) if statuses else "empty")
    if status == "parsed":
        store = services.get_vector_store()
        started = time.perf_counter()
        indexed = store.index(
            [item for item in evidence if item.source_type not in _UNINDEXED_SOURCE_TYPES and not item.metadata.get("placeholder")],
            session.session_id,
        )
        if indexed:
            logger.info("Indexed %d chunks of %s in %.0f ms", indexed, target.name, (time.perf_counter() - started) * 1000)
        else:
            logger.info("Dense index skipped for %s: %s", target.name, store.status())
    return IngestedFile(file_id=file_id, name=target.name, size_bytes=len(data), path=str(target), status=status, evidence=evidence)


def discard_ingested(session: SessionState, ingested: IngestedFile) -> None:
    """Clean up a file whose upload was removed while it was still being ingested in the background."""
    for document_id in {item.document_id for item in ingested.evidence}:
        services.get_vector_store().delete_document(document_id)
    shutil.rmtree(UPLOAD_ROOT / session.session_id / ingested.file_id, ignore_errors=True)


def ingestion_manager(session: SessionState, settings: Settings) -> IngestionManager | None:
    """The session's background ingestion pool (one per browser session), or None when disabled."""
    if not settings.background_ingestion:
        return None
    manager = st.session_state.get("ingestion_manager")
    if manager is None:
        manager = IngestionManager(
            lambda file_id, name, data: ingest_upload(session, file_id, name, data),
            max_workers=settings.ingest_workers,
            on_discard=lambda ingested: discard_ingested(session, ingested),
        )
        st.session_state.ingestion_manager = manager
    return manager


def collect_ingested(session: SessionState, manager: IngestionManager | None) -> int:
    """Move files finished in the background into the session; returns how many arrived."""
    if manager is None:
        return 0
    finished = manager.collect()
    for ingested in finished:
        session.files[ingested.file_id] = ingested
    if finished:
        session.uploaded_files = [file.name for file in session.files.values()]
    return len(finished)


def remove_upload(session: SessionState, file_id: str) -> None:
    removed = session.files.pop(file_id, None)
    if removed is not None:
        for document_id in {item.document_id for item in removed.evidence}:
            services.get_vector_store().delete_document(document_id)
        logger.info("File removed: %s (%d chunks and their vectors deleted)", removed.name, len(removed.evidence))
    shutil.rmtree(UPLOAD_ROOT / session.session_id / file_id, ignore_errors=True)


def _init_session_state() -> SessionState:
    if "session_state" not in st.session_state:
        st.session_state.session_state = SessionState(session_id=str(uuid.uuid4()))
    return st.session_state.session_state


def _sync_uploads(
    session: SessionState, uploaded: list[Any], max_files: int, manager: IngestionManager | None = None
) -> list[str]:
    """Ingest newly added uploads and drop removed ones. Streamlit reruns this on every interaction.

    With a ``manager`` new files are queued for background ingestion and picked up later by ``collect_ingested``;
    without one they are ingested inline (tests, and BACKGROUND_INGESTION=false).
    """
    warnings: list[str] = []
    current_ids = {file.file_id for file in uploaded}

    for file_id in list(session.files):
        if file_id not in current_ids:
            remove_upload(session, file_id)
    if manager is not None:
        for job in manager.jobs():
            if job.file_id not in current_ids and job.status != "cancelled":
                manager.cancel(job.file_id)
        collect_ingested(session, manager)
    pending = {job.file_id for job in manager.pending()} if manager is not None else set()

    for file in uploaded:
        if file.file_id in session.files or (manager is not None and manager.has_job(file.file_id)):
            continue
        if file.size > MAX_FILE_BYTES:
            warnings.append(f"{file.name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB and was skipped.")
            logger.warning("Upload rejected: %s is %.1f MB (limit %d MB)", file.name, file.size / 1024 / 1024, MAX_FILE_BYTES // (1024 * 1024))
            continue
        if len(session.files) + len(pending) >= max_files:
            warnings.append(f"Session limit of {max_files} files reached; {file.name} was skipped.")
            logger.warning("Upload rejected: %s (session limit of %d files reached)", file.name, max_files)
            continue
        if manager is not None:
            manager.submit(file.file_id, file.name, file.getvalue())
            pending.add(file.file_id)
            continue
        with st.spinner(f"Ingesting {file.name}…"):
            session.files[file.file_id] = ingest_upload(session, file.file_id, file.name, file.getvalue())

    session.uploaded_files = [file.name for file in session.files.values()]
    return warnings


def system_status() -> dict[str, str]:
    return {
        "LLM": services.get_llm().status(),
        "Dense retrieval": services.get_vector_store().status(),
        "Reranker": services.get_reranker().status(),
        "LangSmith": "tracing" if tracing_enabled() else "off",
    }


def _render_sidebar(session: SessionState, settings: Settings) -> None:
    with st.sidebar:
        st.title("Controls")
        st.caption(settings.app_description)

        manager = ingestion_manager(session, settings)
        uploaded = st.file_uploader("Upload files", type=SUPPORTED_TYPES, accept_multiple_files=True) or []
        for warning in _sync_uploads(session, uploaded, settings.max_files_per_session, manager):
            st.warning(warning)

        # While files are ingesting in the background, only this panel re-runs (every second) to show progress.
        polling = manager is not None and bool(manager.pending())
        st.fragment(_render_files, run_every=1.0 if polling else None)(session, settings, manager)

        st.markdown("---")
        st.subheader("Session")
        st.text(f"Session ID: {session.session_id}")
        with st.expander("System status"):
            for name, value in system_status().items():
                st.markdown(f"**{name}:** {_md(value)}")
        if settings.log_level.upper() == "DEBUG":
            with st.expander("Logs"):
                st.code("\n".join(recent_logs(session.session_id)[-100:]) or "No log records yet.", language="text")


def file_rows(session: SessionState, manager: IngestionManager | None) -> list[dict[str, Any]]:
    """Rows for the sidebar file table: ingested files plus queued, processing and failed background jobs."""
    rows = [
        {"File": f.name, "Size (KB)": round(f.size_bytes / 1024, 1), "Chunks": len(f.evidence), "Status": f.status}
        for f in session.files.values()
    ]
    if manager is not None:
        for job in manager.jobs():
            if job.status in {"queued", "processing", "failed"} and job.file_id not in session.files:
                status = job.status if job.status != "failed" else f"failed: {job.error}"
                rows.append({"File": job.name, "Size (KB)": round(job.size_bytes / 1024, 1), "Chunks": None, "Status": status})
    return rows


def _render_files(session: SessionState, settings: Settings, manager: IngestionManager | None) -> None:
    was_pending = manager is not None and bool(manager.pending())
    collect_ingested(session, manager)
    pending = manager.pending() if manager is not None else []
    rows = file_rows(session, manager)
    if pending:
        done = len(session.files)
        st.progress(done / max(done + len(pending), 1), text=f"Ingesting {len(pending)} file(s)… you can already ask about the others.")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    st.text(f"Files: {len(session.files) + len(pending)} / {settings.max_files_per_session}")
    if was_pending and not pending:
        st.rerun()  # everything finished: refresh the whole app once, which also stops this panel's polling


def _answer_details(result: dict[str, Any]) -> dict[str, Any]:
    retrieved: list[Evidence] = list(result.get("retrieved") or [])
    summary = build_demo_summary(
        str(result.get("route", "")),
        list(result.get("selected_agents") or []),
        [{"file_name": doc.file_name, "source_type": doc.source_type} for doc in retrieved],
    )
    return {
        "trace_text": summary["trace_text"],
        "router": {"reason": result.get("router_reason", ""), "method": result.get("router_method", "")},
        "interpreted_as": result.get("standalone_question") if result.get("rewrite_method") not in (None, "unchanged") else None,
        "rewrite_method": result.get("rewrite_method"),
        "verification": result.get("verification") or {},
        "used_llm": bool(result.get("used_llm")),
        "nodes": [
            {"node": entry["node"], "ms": entry.get("duration_ms")}
            for entry in result.get("trace", [])
        ],
        "tables": [
            {"title": format_citation(doc), "rows": doc.metadata["rows"]}
            for doc in retrieved
            if doc.source_type == "computed" and doc.metadata.get("rows")
        ],
        "sources": [
            {
                "citation": format_citation(doc),
                "content": doc.content,
                "source_type": doc.source_type,
                "score": doc.metadata.get("hybrid_score"),
                "rerank": doc.metadata.get("rerank_score"),
            }
            for doc in retrieved
        ],
    }


def _render_details(details: dict[str, Any]) -> None:
    verification = details.get("verification") or {}
    if verification.get("general"):
        st.caption(f"ℹ️ {verification.get('reason', '')}")
    elif verification:
        ratio = verification.get("support_ratio")
        label = "Supported by evidence" if verification.get("supported") else "Not fully verified"
        suffix = f" · {round(ratio * 100)}% of claims supported" if ratio is not None and verification.get("claims") else ""
        st.caption(f"{label}{suffix}: {verification.get('reason', '')}")

    for table in details.get("tables") or []:
        st.caption(f"Computed from {_md(table['title'])}")
        st.dataframe(table["rows"], hide_index=True, width="stretch")

    sources = details.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                scores = []
                if source.get("score") is not None:
                    scores.append(f"hybrid {source['score']}")
                if source.get("rerank") is not None:
                    scores.append(f"rerank {source['rerank']}")
                suffix = f" · {' · '.join(scores)}" if scores else ""
                st.markdown(f"**{_md(source['citation'])}** ({source.get('source_type', '')}){suffix}")
                st.text(source["content"])

    claims = verification.get("claims") or []
    if claims:
        with st.expander(f"Claim check ({sum(1 for c in claims if c.get('supported'))}/{len(claims)} supported)"):
            for claim in claims:
                mark = "✅" if claim.get("supported") else "⚠️"
                st.markdown(f"{mark} {_md(claim['text'])}  \n*{_md(claim.get('reason', ''))}* ({claim.get('method', 'rules')})")

    with st.expander("Agent trace"):
        st.write(details.get("trace_text", ""))
        router = details.get("router") or {}
        if details.get("interpreted_as"):
            st.caption(f"Interpreted as ({details.get('rewrite_method')}): {_md(details['interpreted_as'])}")
        if router.get("reason"):
            st.caption(f"Router ({router.get('method', 'rules')}): {router['reason']}")
        st.caption(f"Answer written by: {'LLM' if details.get('used_llm') else 'deterministic fallback'}")
        if details.get("nodes"):
            st.dataframe(details["nodes"], hide_index=True, width="stretch")


def _md(text: str) -> str:
    """Escape characters Streamlit markdown would interpret (``$`` starts LaTeX)."""
    return str(text).replace("$", "\\$")


_STEP_LABELS = {
    "contextualize": "Understanding the question",
    "route": "Routing",
    "retrieval_agent": "Searching documents",
    "data_agent": "Computing with pandas",
    "vision_agent": "Reading images",
    "code_agent": "Analysing code",
    "general_agent": "Answering directly",
    "synthesize": "Writing the answer",
    "verify": "Checking claims against the evidence",
    "finalize": "Finalizing",
}


def describe_step(entry: dict[str, Any]) -> str:
    """One human-readable progress line for a finished graph node (see ``_timed`` in app/graph/graph.py)."""
    node = entry.get("node", "")
    label = _STEP_LABELS.get(node, node)
    detail = ""
    if node == "contextualize":
        detail = f"→ rewritten using the conversation ({entry.get('method')})" if entry.get("rewritten") else "→ standalone"
    elif node == "route":
        detail = f"→ {', '.join(entry.get('agents', []))} ({entry.get('method', 'rules')})"
    elif node in {"retrieval_agent", "vision_agent", "code_agent"}:
        detail = f"→ {len(entry.get('retrieved', []))} chunk(s)"
    elif node == "data_agent":
        detail = f"→ {entry.get('answer', '')}"
    elif node == "synthesize":
        detail = f"({'LLM' if entry.get('used_llm') else 'extractive fallback'}{', rewrite' if entry.get('attempt') else ''})"
    elif node == "verify":
        ratio = entry.get("support_ratio")
        detail = f"→ {round(ratio * 100)}% supported" if ratio is not None else ""
        if entry.get("regenerate"):
            detail += ", rewriting unsupported claims"
    elif node == "general_agent":
        detail = f"({'LLM' if entry.get('used_llm') else 'built-in reply'})"
    elif node == "finalize":
        detail = f"→ {entry.get('outcome', '')}"
    return f"{label} {detail} · {entry.get('duration_ms', 0):.0f} ms".replace("  ", " ")


def _history(session: SessionState) -> list[dict[str, str]]:
    """Recent turns (excluding the question being asked) for agents that use conversational context."""
    turns = load_settings().general_history_turns
    previous = session.conversation_messages[:-1][-turns:]
    return [{"role": m["role"], "content": str(m["content"])} for m in previous if m.get("role") in {"user", "assistant"}]


def update_memory(session: SessionState, max_batch: int = 20) -> bool:
    """Fold turns that left the recent-history window into the running summary (needs the LLM; bounded batches).

    Without the LLM nothing is summarised and only the recent window is used, so context stays bounded either way.
    Returns True when the summary was updated.
    """
    boundary = len(session.conversation_messages) - load_settings().general_history_turns
    if boundary <= session.summarized_upto:
        return False
    start = max(session.summarized_upto, boundary - max_batch)
    older = [
        {"role": m["role"], "content": str(m["content"])}
        for m in session.conversation_messages[start:boundary]
        if m.get("role") in {"user", "assistant"}
    ]
    summary = update_summary(session.summary, older)
    if summary is None:
        return False
    session.summary, session.summarized_upto = summary, boundary
    logger.info("Conversation summary updated (%d turns folded in, %d chars)", len(older), len(summary))
    return True


def _stream_answer(session: SessionState, prompt: str) -> dict[str, Any]:
    """Run the graph with live progress and token streaming; returns the final graph state."""
    status = st.status("Working…", expanded=False)
    placeholder = st.empty()
    buffer = ""
    labels: list[str] = []
    result: dict[str, Any] = {}

    for kind, payload in stream_query(prompt, session.queryable_evidence(), session.session_id, _history(session), session.summary):
        if kind == "step":
            status.write(describe_step(payload))
            status.update(label=f"{_STEP_LABELS.get(payload.get('node', ''), 'Working')}…")
        elif kind == "citations":
            labels = payload
        elif kind == "token":
            buffer += payload
            text = visible_stream_text(buffer)
            if labels:
                text = replace_citation_markers(text, labels)
            placeholder.markdown(_md(text) + " ▌")
        elif kind == "reset":
            buffer = ""
            placeholder.caption("Revising the answer after verification…")
        elif kind == "final":
            result = payload

    total_ms = sum(entry.get("duration_ms", 0) for entry in result.get("trace", []))
    status.update(label=f"Done in {total_ms / 1000:.1f} s", state="complete")
    placeholder.markdown(_md(str(result.get("answer", ""))))
    return result


def _render_chat(session: SessionState) -> None:
    st.subheader("Chat")

    for message in session.conversation_messages:
        with st.chat_message(message["role"]):
            st.markdown(_md(message["content"]))
            if message.get("details"):
                _render_details(message["details"])

    prompt = st.chat_input("Ask about your files, or anything else")
    if not prompt:
        return

    session.conversation_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(_md(prompt))

    with st.chat_message("assistant"):
        manager = st.session_state.get("ingestion_manager")
        collect_ingested(session, manager)
        still_ingesting = manager.pending() if manager is not None else []
        if still_ingesting:
            st.caption(f"{len(still_ingesting)} file(s) still processing; answering from the files that are ready.")
        result = _stream_answer(session, prompt)
        answer = str(result.get("answer", ""))
        details = _answer_details(result)
        _render_details(details)

    session.last_question = prompt
    session.last_answer = answer
    session.conversation_messages.append({"role": "assistant", "content": answer, "details": details})
    update_memory(session)


@functools.cache
def log_startup() -> None:
    """Once per process: log the configuration and whether every component came up (Streamlit reruns skip this)."""
    settings = load_settings()
    started = time.perf_counter()
    try:
        version = metadata.version("bd-demo-prjt")
    except metadata.PackageNotFoundError:
        version = "dev"
    logger.info(
        "App started: %s %s (Python %s); llm=%s qdrant=%s reranker=%s general_answers=%s",
        settings.app_name,
        version,
        platform.python_version(),
        services.get_llm().model_name if settings.enable_llm else "disabled",
        settings.qdrant_mode if settings.qdrant_enabled else "disabled",
        settings.reranker,
        settings.allow_general_answers,
    )
    from app.graph.graph import _compiled_graph

    _compiled_graph()
    status = system_status()
    tracing = status.pop("LangSmith")  # off is a deliberate setting, not a degraded component
    logger.info("Component LangSmith: %s", tracing)
    healthy = {"LLM": "available", "Dense retrieval": "ready", "Reranker": "cross-encoder"}
    ready = [name for name, value in status.items() if value.startswith(healthy[name])]
    for name, value in status.items():
        (logger.info if name in ready else logger.warning)("Component %s: %s", name, value)
    summary = "all components ready" if len(ready) == len(status) else f"{len(ready)}/{len(status)} components fully available"
    logger.info("Startup complete in %.1f s: %s", time.perf_counter() - started, summary)


def main() -> None:
    settings = load_settings()
    st.set_page_config(page_title=settings.page_title, page_icon="📄")  # type: ignore[arg-type]
    configure_logging(settings)
    configure_tracing()
    session = _init_session_state()
    set_session(session.session_id)
    log_startup()

    st.title(settings.app_name)
    st.caption(settings.app_description)

    _render_sidebar(session, settings)
    _render_chat(session)


if __name__ == "__main__":
    main()
