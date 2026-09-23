from __future__ import annotations

from typing import Any

import pandas as pd


class DataAgent:
    """Simple spreadsheet agent based on deterministic pandas operations."""

    def answer_question(self, df: pd.DataFrame, question: str) -> dict[str, Any]:
        question_lower = question.lower()

        if "highest" in question_lower and "revenue" in question_lower:
            region = df.loc[df["revenue"].idxmax(), "region"]
            value = int(df["revenue"].max())
            return {"answer": region, "value": value}

        if "sum" in question_lower and "revenue" in question_lower:
            value = int(df["revenue"].sum())
            return {"answer": str(value), "value": value}

        return {"answer": "No structured calculation matched this question.", "value": None}
