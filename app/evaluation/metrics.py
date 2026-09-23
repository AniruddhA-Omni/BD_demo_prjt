from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationRow:
    question: str
    answer: str
    expected_answer: str | None = None
    supported: bool = True
    source_count: int = 1
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DemoEvaluationSet:
    """Small in-memory evaluation set for demo-level quality checks."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "question": "What happened in Q4?",
                "answer": "Revenue increased by 12% in Q4.",
                "expected_answer": "Revenue increased by 12% in Q4.",
                "supported": True,
                "source_count": 1,
                "score": 1.0,
            },
            {
                "question": "Which region had the highest revenue?",
                "answer": "West",
                "expected_answer": "West",
                "supported": True,
                "source_count": 1,
                "score": 1.0,
            },
            {
                "question": "Compare the PDF revenue with the Excel totals and summarize the trend.",
                "answer": "The revenue trend increased across the source files.",
                "expected_answer": "The revenue trend increased across the source files.",
                "supported": True,
                "source_count": 2,
                "score": 0.9,
            },
        ]

    def retrieval_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "question": "Q4 revenue growth",
                "relevant_documents": ["doc-2", "doc-5"],
                "retrieved_documents": ["doc-9", "doc-2", "doc-7", "doc-5"],
                "k": 3,
            },
            {
                "question": "Policy approval process",
                "relevant_documents": ["policy-1"],
                "retrieved_documents": ["policy-1", "revenue-4"],
                "k": 3,
            },
            {
                "question": "What caused the increase in sales?",
                "relevant_documents": ["sales-1"],
                "retrieved_documents": ["marketing-2", "sales-1", "ops-4"],
                "k": 3,
            },
        ]


def recall_at_k(relevant_documents: list[str], retrieved_documents: list[str], k: int | None = None) -> float:
    if not relevant_documents:
        return 0.0
    limit = k if k is not None else len(retrieved_documents)
    retrieved_top = retrieved_documents[:limit]
    hits = sum(1 for doc in relevant_documents if doc in retrieved_top)
    return hits / len(relevant_documents)


def reciprocal_rank(relevant_documents: list[str], retrieved_documents: list[str]) -> float:
    if not relevant_documents:
        return 0.0
    for index, doc in enumerate(retrieved_documents, start=1):
        if doc in relevant_documents:
            return 1.0 / index
    return 0.0


def retrieval_evaluation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "average_recall_at_3": 0.0,
            "average_mrr": 0.0,
            "false_positive_rate": 0.0,
            "total_queries": 0,
            "rows": [],
        }

    recall_values = [
        recall_at_k(row.get("relevant_documents", []), row.get("retrieved_documents", []), row.get("k", 3))
        for row in rows
    ]
    mrr_values = [
        reciprocal_rank(row.get("relevant_documents", []), row.get("retrieved_documents", []))
        for row in rows
    ]

    false_positives = []
    for row in rows:
        relevant = set(row.get("relevant_documents", []))
        retrieved = row.get("retrieved_documents", [])
        false_positives.append(len([doc for doc in retrieved if doc not in relevant]))

    false_positive_rate = sum(false_positives) / max(sum(len(row.get("retrieved_documents", [])) for row in rows), 1)

    return {
        "total_queries": len(rows),
        "average_recall_at_3": sum(recall_values) / len(recall_values),
        "average_mrr": sum(mrr_values) / len(mrr_values),
        "false_positive_rate": false_positive_rate,
        "rows": rows,
    }


def evaluation_report(dataset: DemoEvaluationSet) -> dict[str, Any]:
    rows = dataset.rows()
    supported = sum(1 for row in rows if row["supported"])
    total = max(len(rows), 1)
    average_score = sum(float(row["score"]) for row in rows) / total
    return {
        "total_questions": total,
        "supported_questions": supported,
        "supported_ratio": supported / total,
        "average_score": average_score,
        "rows": rows,
    }


def ragas_evaluation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate RAG rows if ragas is installed, else return a local fallback summary."""
    if not rows:
        return {"backend": "local-fallback", "metrics": {}, "rows": []}

    try:
        import ragas  # type: ignore
        from ragas import evaluate  # type: ignore

        dataset_rows = [
            {
                "user_input": row.get("question", ""),
                "response": row.get("answer", ""),
                "retrieved_contexts": row.get("context", []),
                "reference": row.get("ground_truth") or row.get("expected_answer") or row.get("answer", ""),
            }
            for row in rows
        ]

        result = evaluate(dataset_rows)
        metrics = {
            key: float(value) if isinstance(value, (int, float)) else str(value)
            for key, value in getattr(result, "metrics", {}).items()
        }
        return {
            "backend": "ragas",
            "metrics": metrics,
            "rows": rows,
        }
    except Exception:
        fallback = []
        for row in rows:
            answer = str(row.get("answer", ""))
            context = row.get("context") or []
            expected = row.get("ground_truth") or row.get("expected_answer") or answer
            supported = bool(answer and expected and (expected.lower() in answer.lower() or answer.lower() in expected.lower()))
            fallback.append({
                "question": row.get("question", ""),
                "answer": answer,
                "ground_truth": expected,
                "supported": supported,
                "context_count": len(context) if isinstance(context, list) else 0,
            })

        supported_count = sum(1 for item in fallback if item["supported"])
        total = max(len(fallback), 1)
        return {
            "backend": "local-fallback",
            "metrics": {
                "supported_ratio": supported_count / total,
                "average_context_count": sum(item["context_count"] for item in fallback) / total,
            },
            "rows": fallback,
        }
