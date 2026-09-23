import pandas as pd

from app.agents.data import DataAgent
from app.agents.verification import VerificationAgent
from app.ingestion.models import Evidence


def test_data_agent_computes_highest_revenue_region():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "West"],
            "revenue": [120, 90, 150],
        }
    )

    result = DataAgent().answer_question(df, "Which region had the highest revenue?")

    assert result["answer"] == "West"
    assert result["value"] == 150


def test_verification_agent_accepts_supported_claims():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="sales.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4.",
        )
    ]

    verdict = VerificationAgent().verify("Revenue increased by 12% in Q4.", evidence)

    assert verdict["supported"] is True


def test_verification_agent_flags_unsupported_claims():
    evidence = [
        Evidence(
            document_id="doc-1",
            file_name="sales.txt",
            file_type="txt",
            content="Revenue increased by 12% in Q4.",
        )
    ]

    verdict = VerificationAgent().verify("Revenue increased by 30% in Q1.", evidence)

    assert verdict["supported"] is False
    assert "not supported" in verdict["reason"].lower() # type: ignore
