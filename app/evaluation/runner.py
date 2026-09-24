"""End-to-end evaluation over the sample corpus.

    uv run python -m app.evaluation.runner                 # evaluate, print a summary, write evals/results.json
    uv run python -m app.evaluation.runner --sweep         # also grid-search the hybrid retrieval weights

Each question in ``evals/questions.json`` has a ``category``; results are reported overall and per category so a
regression in one category isn't hidden by the average.

Metrics: router accuracy, Recall@K / MRR over retrieved chunks, answer correctness (expected substrings),
citation precision/recall (files cited in the answer vs relevant files), groundedness (verifier support ratio),
abstention accuracy, general-answer correctness, conflict surfacing, injection resistance and latency. Uses whatever
LLM / Qdrant / reranker the current settings provide, so the same run measures the deterministic fallback or the full
local stack. Questions whose ``requires`` (``llm``, ``pdf``, ``vision``) aren't met are reported as skipped.

Question fields: ``id``, ``category``, ``question``, ``expected_route``, ``relevant`` ([{file, section?}]),
``expected_answer_contains``; optional ``history``, ``requires``, ``expect_abstain``, ``expect_general``,
``expect_conflict``, ``forbidden_answer_contains``, ``reference_answer`` (for LLM-judged metrics, plan 1.2), ``note``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from app import services
from app.agents.synthesis import ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE
from app.config import load_settings
from app.evaluation.metrics import recall_at_k, reciprocal_rank
from app.evaluation.sample_corpus import build_sample_corpus
from app.graph.graph import run_query
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter
from app.logging_config import configure_logging
from app.observability import configure_tracing
from app.retrieval.hybrid import HybridRetriever

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "evals" / "questions.json"
EVAL_SESSION = "evaluation"
_CITATION = re.compile(r"\[source:\s*([^\]]+)\]")
_UNINDEXED = {"table", "error", "empty", "unknown"}

# (metric key in a row, label) reported for every category where at least one row has the metric.
_METRICS = [
    ("route_correct", "router_accuracy"),
    ("recall_at_k", "recall_at_k"),
    ("mrr", "mrr"),
    ("answer_correct", "answer_correctness"),
    ("citation_precision", "citation_precision"),
    ("citation_recall", "citation_recall"),
    ("groundedness", "groundedness"),
    ("abstain_correct", "abstention_accuracy"),
    ("general_correct", "general_correctness"),
    ("conflict_surfaced", "conflict_surfaced"),
    ("injection_resisted", "injection_resistance"),
]


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ingest_corpus(folder: str | Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    router = IngestionRouter()
    for path in build_sample_corpus(folder):
        for item in router.ingest_file(path):
            if item.metadata.get("status") == "parsed":
                item.metadata["session_id"] = EVAL_SESSION
                evidence.append(item)
    services.get_vector_store().index(
        [doc for doc in evidence if doc.source_type not in _UNINDEXED and not doc.metadata.get("placeholder")], EVAL_SESSION
    )
    return evidence


def unmet_requirements(item: dict[str, Any], evidence: list[Evidence]) -> list[str]:
    """Capabilities a question needs that this run lacks: ``llm``, ``pdf`` (PyMuPDF loaded), ``vision`` (OCR or LLM)."""
    available = {
        "llm": services.get_llm().available(),
        "pdf": any(doc.file_type == "pdf" for doc in evidence),
        "vision": services.get_llm().available()
        or any(doc.source_type in {"image", "vision"} and not doc.metadata.get("placeholder") for doc in evidence),
    }
    return [requirement for requirement in item.get("requires", []) if not available.get(requirement, False)]


def run_evaluation(questions: list[dict[str, Any]], evidence: list[Evidence]) -> dict[str, Any]:
    rows = []
    for item in questions:
        missing = unmet_requirements(item, evidence)
        if missing:
            rows.append({"id": item["id"], "category": item.get("category", "other"), "skipped": f"requires {', '.join(missing)}"})
            continue
        started = time.perf_counter()
        result = run_query(item["question"], evidence, session_id=EVAL_SESSION, history=item.get("history"))
        rows.append(_score_row(item, result, time.perf_counter() - started))

    scored = [row for row in rows if "skipped" not in row]
    categories = sorted({row["category"] for row in scored})
    return {
        "summary": _summarize(scored),
        "by_category": {category: _summarize([r for r in scored if r["category"] == category]) for category in categories},
        "skipped": [{"id": row["id"], "reason": row["skipped"]} for row in rows if "skipped" in row],
        "rows": rows,
        "stack": _stack(),
    }


def sweep_weights(questions: list[dict[str, Any]], evidence: list[Evidence], step: float = 0.25) -> list[dict[str, Any]]:
    """Grid-search (dense, bm25, lexical) weights on single-turn retrieval questions; returns results best-first."""
    retrieval_questions = [
        q for q in questions
        if q.get("relevant") and q["expected_route"] in {"retrieval", "code", "vision"} and not q.get("history") and not unmet_requirements(q, evidence)
    ]
    top_k = HybridRetriever().settings.retrieval_top_k
    steps = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    text_docs = [doc for doc in evidence if doc.source_type != "table"]
    results = []
    for dense, bm25 in itertools.product(steps, steps):
        lexical = round(1 - dense - bm25, 2)
        if lexical < 0:
            continue
        retriever = HybridRetriever(weights=(dense, bm25, lexical))
        recalls, mrrs = [], []
        for item in retrieval_questions:
            retrieved = retriever.search(item["question"], text_docs, top_k=top_k, session_id=EVAL_SESSION)
            keys, relevant = _relevance_keys(item, retrieved)
            recalls.append(recall_at_k(relevant, keys, top_k))
            mrrs.append(reciprocal_rank(relevant, keys))
        results.append({
            "weights": {"dense": dense, "bm25": bm25, "lexical": lexical},
            "recall_at_k": round(statistics.mean(recalls), 3) if recalls else 0.0,
            "mrr": round(statistics.mean(mrrs), 3) if mrrs else 0.0,
        })
    results.sort(key=lambda r: (r["mrr"], r["recall_at_k"]), reverse=True)
    return results


def _score_row(item: dict[str, Any], result: dict[str, Any], latency: float) -> dict[str, Any]:
    answer = str(result.get("answer", ""))
    lowered = answer.lower()
    retrieved: list[Evidence] = list(result.get("retrieved") or [])
    keys, relevant = _relevance_keys(item, retrieved)
    relevant_files = {r["file"] for r in item.get("relevant", [])}
    cited_files = {label.split(" — ")[0].strip() for label in _CITATION.findall(answer)}
    abstained = answer.strip() in {ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE}
    expected = item.get("expected_answer_contains", [])
    verification = result.get("verification") or {}

    row: dict[str, Any] = {
        "id": item["id"],
        "category": item.get("category", "other"),
        "question": item["question"],
        "route": result.get("route"),
        "expected_route": item["expected_route"],
        "route_correct": result.get("route") == item["expected_route"],
        "answer": answer,
        "abstained": abstained,
        "abstain_correct": abstained == bool(item.get("expect_abstain")),
        "answer_correct": all(text.lower() in lowered for text in expected) if expected else None,
        "groundedness": verification.get("support_ratio"),
        "latency_s": round(latency, 3),
    }
    if relevant:
        row["recall_at_k"] = recall_at_k(relevant, keys, len(keys) or 1)
        row["mrr"] = reciprocal_rank(relevant, keys)
    if cited_files:
        row["citation_precision"] = len(cited_files & relevant_files) / len(cited_files) if relevant_files else 0.0
    if relevant_files and not abstained:
        row["citation_recall"] = len(cited_files & relevant_files) / len(relevant_files)
    if item.get("expect_general"):
        row["general_correct"] = result.get("route") == "general" and not cited_files and bool(answer.strip())
    if item.get("expect_conflict"):
        # Both disagreeing values stated, each backed by a citation from a different source file.
        row["conflict_surfaced"] = bool(expected) and all(t.lower() in lowered for t in expected) and len(cited_files & relevant_files) >= 2
    if item.get("forbidden_answer_contains"):
        row["injection_resisted"] = not any(text.lower() in lowered for text in item["forbidden_answer_contains"])
    if item.get("note"):
        row["note"] = item["note"]
    return row


def _relevance_keys(item: dict[str, Any], retrieved: list[Evidence]) -> tuple[list[str], list[str]]:
    """Map retrieved chunks and gold items onto comparable ids: ``file`` or ``file#section``."""
    relevant = [f"{r['file']}#{r['section']}" if r.get("section") else r["file"] for r in item.get("relevant", [])]
    keys = []
    for doc in retrieved:
        with_section = f"{doc.file_name}#{doc.section}"
        keys.append(with_section if with_section in relevant else doc.file_name)
    return keys, relevant


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(statistics.mean(float(v) for v in values), 3) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(row["latency_s"] for row in rows)
    summary: dict[str, Any] = {"questions": len(rows)}
    for key, label in _METRICS:
        value = _mean(rows, key)
        if value is not None:
            summary[label] = value
    if latencies:
        summary["latency_p50_s"] = round(statistics.median(latencies), 3)
        summary["latency_p95_s"] = round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3)
    return summary


def _stack() -> dict[str, str]:
    return {
        "llm": services.get_llm().status(),
        "dense_retrieval": services.get_vector_store().status(),
        "reranker": services.get_reranker().status(),
    }


def _row_flags(row: dict[str, Any]) -> str:
    if "skipped" in row:
        return f"skipped ({row['skipped']})"
    flags = ["route ok" if row["route_correct"] else f"route {row['route']} != {row['expected_route']}"]
    flags.append({True: "answer ok", False: "answer WRONG", None: "answer n/a"}[row["answer_correct"]])
    flags.append("abstain ok" if row["abstain_correct"] else "abstain WRONG")
    for key, label in (("general_correct", "general"), ("conflict_surfaced", "conflict"), ("injection_resisted", "injection")):
        if key in row:
            flags.append(f"{label} {'ok' if row[key] else 'WRONG'}")
    return "; ".join(flags) + f"; {row['latency_s']}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output", default="evals/results.json")
    parser.add_argument("--sweep", action="store_true", help="grid-search hybrid retrieval weights")
    parser.add_argument("--log-level", default="WARNING", help="application log level (INFO shows every step)")
    args = parser.parse_args()

    configure_logging(load_settings().model_copy(update={"log_level": args.log_level}))
    configure_tracing()
    questions = load_questions(args.questions)
    with tempfile.TemporaryDirectory() as folder:
        evidence = ingest_corpus(folder)
        report = run_evaluation(questions, evidence)
        if args.sweep:
            report["weight_sweep"] = sweep_weights(questions, evidence)

    print("Stack:", json.dumps(report["stack"], indent=2))
    print("Summary:", json.dumps(report["summary"], indent=2))
    print("By category:")
    for category, summary in report["by_category"].items():
        metrics = ", ".join(f"{key}={value}" for key, value in summary.items() if key not in {"latency_p50_s", "latency_p95_s"})
        print(f"  {category}: {metrics}")
    for row in report["rows"]:
        print(f"- [{row['category']}] {row['id']}: {_row_flags(row)}")
    if args.sweep:
        print("Best hybrid weights:", json.dumps(report["weight_sweep"][:3], indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
