from __future__ import annotations

from typing import Any

import pandas as pd


class SpreadsheetRegistry:
    """Keeps a lightweight in-memory registry of loaded spreadsheet frames."""

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}

    def register(self, name: str, df: pd.DataFrame) -> dict[str, Any]:
        frame = df.copy()
        self._frames[name] = frame
        return {
            "name": name,
            "row_count": int(len(frame)),
            "column_names": [str(column) for column in frame.columns],
            "shape": (len(frame), len(frame.columns)),
            "schema": [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        }

    def get(self, name: str) -> pd.DataFrame | None:
        return self._frames.get(name)


class DataAgent:
    """Simple spreadsheet agent based on deterministic pandas operations."""

    def __init__(self, registry: SpreadsheetRegistry | None = None) -> None:
        self.registry = registry or SpreadsheetRegistry()

    def answer_question(self, df: pd.DataFrame, question: str) -> dict[str, Any]:
        if df.empty:
            return {"answer": "No structured data was available to answer this question.", "value": None}

        question_lower = question.lower()
        target_column = self._select_metric_column(df, question_lower)

        if target_column is None:
            return {"answer": "No structured calculation matched this question.", "value": None}

        if "highest" in question_lower or "max" in question_lower:
            if "region" in df.columns.astype(str).str.lower().tolist():
                region = df.loc[df[target_column].idxmax(), "region"]
                value = df[target_column].max()
                return {"answer": str(region), "value": value}

        if "lowest" in question_lower or "min" in question_lower:
            if "region" in df.columns.astype(str).str.lower().tolist():
                region = df.loc[df[target_column].idxmin(), "region"]
                value = df[target_column].min()
                return {"answer": str(region), "value": value}

        if "average" in question_lower or "mean" in question_lower:
            value = df[target_column].mean()
            return {"answer": str(value), "value": float(value)}

        if "sum" in question_lower or "total" in question_lower:
            value = df[target_column].sum()
            return {"answer": str(value), "value": float(value) if pd.notna(value) else None}

        if "highest" in question_lower and "revenue" in question_lower:
            region = df.loc[df["revenue"].idxmax(), "region"]
            value = int(df["revenue"].max())
            return {"answer": region, "value": value}

        if "sum" in question_lower and "revenue" in question_lower:
            value = int(df["revenue"].sum())
            return {"answer": str(value), "value": value}

        return {"answer": "No structured calculation matched this question.", "value": None}

    def _select_metric_column(self, df: pd.DataFrame, question_lower: str) -> str | None:
        candidates = [column for column in df.columns if str(column).lower() in {"revenue", "sales", "amount", "total", "value", "cost"}]
        if not candidates:
            numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
            return str(numeric_columns[0]) if numeric_columns else None
        return str(candidates[0])
