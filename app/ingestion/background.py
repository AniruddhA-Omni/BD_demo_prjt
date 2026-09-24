"""Background ingestion for the Streamlit UI.

Parsing, embedding and indexing 50 files blocks a Streamlit rerun for a long time. ``IngestionManager`` runs
ingestion on a small thread pool instead: uploads are queued immediately, the UI polls ``collect()`` for finished
files, and questions can be asked meanwhile using the files that are already ready.

Worker threads never touch Streamlit APIs or the session object; they only run ``ingest_fn`` and hand back an
``IngestedFile``. The logging context (session id) is copied into each worker so its log lines stay correlated.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

IngestFn = Callable[[str, str, bytes], Any]


@dataclass
class IngestJob:
    file_id: str
    name: str
    size_bytes: int
    status: str = "queued"  # queued | processing | done | failed | cancelled
    error: str = ""
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    collected: bool = False
    future: Future | None = field(default=None, repr=False)

    @property
    def pending(self) -> bool:
        return self.status in {"queued", "processing"}


class IngestionManager:
    """Queues uploads onto a thread pool and hands finished files back to the UI thread."""

    def __init__(self, ingest_fn: IngestFn, max_workers: int = 2, on_discard: Callable[[Any], None] | None = None) -> None:
        self._ingest_fn = ingest_fn
        self._on_discard = on_discard
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="ingest")
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def submit(self, file_id: str, name: str, data: bytes) -> IngestJob:
        job = IngestJob(file_id=file_id, name=name, size_bytes=len(data))
        with self._lock:
            self._jobs[file_id] = job
        context = contextvars.copy_context()
        job.future = self._executor.submit(context.run, self._run, job, data)
        logger.info("Ingestion queued: %s (%d file(s) pending)", name, len(self.pending()))
        return job

    def has_job(self, file_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(file_id)
            return job is not None and job.status != "cancelled"

    def cancel(self, file_id: str) -> None:
        """Forget a job whose upload was removed; a result that arrives later is discarded (and cleaned up)."""
        with self._lock:
            job = self._jobs.get(file_id)
            if job is None:
                return
            finished = job.status in {"done", "failed"}
            job.status = "cancelled"
        if job.future is not None:
            job.future.cancel()  # only succeeds while still queued
        if finished and job.result is not None and not job.collected and self._on_discard:
            self._on_discard(job.result)
        logger.info("Ingestion cancelled: %s", job.name)

    def collect(self) -> list[Any]:
        """Finished results not handed out yet (each result is returned once)."""
        results = []
        with self._lock:
            for job in self._jobs.values():
                if job.status == "done" and not job.collected:
                    job.collected = True
                    results.append(job.result)
        return results

    def pending(self) -> list[IngestJob]:
        with self._lock:
            return [job for job in self._jobs.values() if job.pending]

    def failed(self) -> list[IngestJob]:
        with self._lock:
            return [job for job in self._jobs.values() if job.status == "failed"]

    def jobs(self) -> list[IngestJob]:
        with self._lock:
            return list(self._jobs.values())

    def forget(self, file_id: str) -> None:
        with self._lock:
            self._jobs.pop(file_id, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: IngestJob, data: bytes) -> None:
        with self._lock:
            if job.status == "cancelled":
                return
            job.status, job.started_at = "processing", time.monotonic()
        try:
            result = self._ingest_fn(job.file_id, job.name, data)
        except Exception as exc:
            logger.exception("Ingestion failed: %s", job.name)
            with self._lock:
                if job.status != "cancelled":
                    job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                job.finished_at = time.monotonic()
            return

        with self._lock:
            cancelled = job.status == "cancelled"
            job.result, job.finished_at = result, time.monotonic()
            if not cancelled:
                job.status = "done"
        elapsed = job.finished_at - (job.started_at or job.finished_at)
        if cancelled:
            logger.info("Discarding %s: its upload was removed while ingesting", job.name)
            if self._on_discard:
                self._on_discard(result)
        else:
            logger.info("Ingestion finished: %s in %.1f s", job.name, elapsed)
