from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence


class VerificationAgent:
    """Checks whether a claim is supported by at least one piece of retrieved evidence."""

    def verify(self, claim: str, evidence: Iterable[Evidence]) -> dict[str, object]:
        claim_norm = self._normalize(claim)
        docs = list(evidence)

        if not docs:
            return {"supported": False, "reason": "No evidence was available to verify the claim."}

        for doc in docs:
            content_norm = self._normalize(doc.content)
            if claim_norm == content_norm or content_norm in claim_norm or claim_norm in content_norm:
                return {"supported": True, "reason": "The claim matches the retrieved evidence."}

            if self._is_numeric_mismatch(claim_norm, content_norm):
                continue

            if claim_norm and self._is_strong_match(claim_norm, content_norm):
                return {"supported": True, "reason": "The claim is consistent with the retrieved evidence."}

        return {
            "supported": False,
            "reason": "The claim is not supported by the retrieved evidence and should be treated as unsupported.",
        }

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().replace("[source:", " ").replace("]", " ").split())

    def _is_numeric_mismatch(self, claim: str, content: str) -> bool:
        claim_numbers = [token for token in claim.split() if token.replace("%", "").isdigit()]
        content_numbers = [token for token in content.split() if token.replace("%", "").isdigit()]

        if not claim_numbers:
            return False

        for value in claim_numbers:
            if value not in content_numbers:
                return True
        return False

    def _is_strong_match(self, claim: str, content: str) -> bool:
        claim_terms = [term for term in claim.split() if len(term) > 3]
        if not claim_terms:
            return False

        content_terms = [term for term in content.split() if len(term) > 3]
        overlap = len(set(claim_terms) & set(content_terms))
        return overlap >= max(1, len(set(claim_terms)) // 2)
