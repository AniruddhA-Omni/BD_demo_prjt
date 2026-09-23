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
