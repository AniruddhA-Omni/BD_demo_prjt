from __future__ import annotations

import re
from typing import Any, Iterable

from app import services
from app.agents.synthesis import (
    ABSTAIN_MESSAGE,
    IRRELEVANT_MESSAGE,
    UNTRUSTED_EVIDENCE_RULE,
    evidence_block,
    format_citation,
    split_sentences,
)
from app.ingestion.models import Evidence
from app.observability import traceable

_CITATION = re.compile(r"\[source:\s*([^\]]+)\]")

_JUDGE_SYSTEM = """You are a strict fact-checker. For each numbered claim decide whether the evidence directly
supports it. A claim is supported only if the evidence states it or it follows with no outside knowledge; numbers,
dates and names must match exactly. """ + UNTRUSTED_EVIDENCE_RULE + """ Return JSON: {"claims": [{"id": 1, "supported": true, "reason": "..."}]}"""


class VerificationAgent:
    """Claim-level verification of an answer against evidence.

    ``verify_answer`` splits the answer into claims, validates each claim's citations, checks every claim
    deterministically (number/term matching) and, when an LLM is available, asks it to judge support. A deterministic
    number mismatch always wins over the LLM, since invented numbers are the most damaging hallucination.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm if llm is not None else services.get_llm()

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

    @traceable(name="verify_answer")
    def verify_answer(self, answer: str, evidence: Iterable[Evidence]) -> dict[str, Any]:
        docs = list(evidence)
        if not docs:
            return {"supported": False, "reason": "No evidence available.", "claims": [], "support_ratio": 0.0}
        if answer.strip() in {ABSTAIN_MESSAGE, IRRELEVANT_MESSAGE}:
            return {"supported": False, "reason": "The answer abstained: the evidence was insufficient.", "claims": [], "support_ratio": 0.0}

        by_label: dict[str, list[Evidence]] = {}
        for doc in docs:
            by_label.setdefault(format_citation(doc), []).append(doc)

        claims = []
        for sentence in self._claim_sentences(answer):
            labels = [label.strip() for label in _CITATION.findall(sentence)]
            text = _CITATION.sub("", sentence).strip()
            if not re.search(r"[A-Za-z0-9]", text):  # short answers such as "3" or "West" are still claims
                continue
            cited = [doc for label in labels for doc in by_label.get(label, [])]
            invalid = [label for label in labels if label not in by_label]
            claims.append({"text": text, "sources": labels, "invalid_citations": invalid, "_cited": cited})

        if not claims:
            return {"supported": False, "reason": "The answer contained no checkable claims.", "claims": [], "support_ratio": 0.0}

        for claim in claims:
            scope = claim["_cited"] or docs
            deterministic = self.verify(claim["text"], scope)
            if not deterministic["supported"] and claim["_cited"]:
                deterministic = self.verify(claim["text"], docs)
            claim["supported"] = bool(deterministic["supported"])
            claim["method"] = "rules"
            claim["numbers_ok"] = not self._has_unknown_numbers(claim["text"], docs)
            if claim["invalid_citations"]:
                claim["supported"] = False
                claim["reason"] = f"Cites unknown source(s): {', '.join(claim['invalid_citations'])}"
            elif not claim["numbers_ok"]:
                claim["supported"] = False
                claim["reason"] = "Contains numbers that do not appear in the evidence."

        if self.llm.available():
            self._judge_with_llm(claims, docs)

        for claim in claims:
            claim.pop("_cited", None)
            claim.setdefault("reason", "Supported by the evidence." if claim["supported"] else "Not supported by the evidence.")

        supported = [claim for claim in claims if claim["supported"]]
        ratio = len(supported) / len(claims)
        if ratio == 1.0:
            reason = "All claims are supported by the evidence." if len(claims) > 1 else "The claim matches the retrieved evidence."
        elif supported:
            reason = f"{len(claims) - len(supported)} of {len(claims)} claims could not be verified."
        else:
            reason = "The claims are not supported by the retrieved evidence."
        return {"supported": ratio == 1.0, "reason": reason, "claims": claims, "support_ratio": round(ratio, 3)}

    def _judge_with_llm(self, claims: list[dict[str, Any]], docs: list[Evidence]) -> None:
        pending = [(number, claim) for number, claim in enumerate(claims, start=1) if claim["numbers_ok"] and not claim["invalid_citations"]]
        if not pending:
            return
        evidence_text = "\n\n".join(evidence_block(None, doc) for doc in docs[:12])
        claim_text = "\n".join(f"{number}. {claim['text']}" for number, claim in pending)
        verdict = self.llm.complete_json(_JUDGE_SYSTEM, f"Evidence:\n\n{evidence_text}\n\nClaims:\n{claim_text}", run_name="verification_judge")
        if not verdict:
            return
        results = {int(item.get("id", -1)): item for item in verdict.get("claims", []) if isinstance(item, dict) and str(item.get("id", "")).isdigit()}
        for number, claim in pending:
            item = results.get(number)
            if item is None:
                continue
            claim["supported"] = bool(item.get("supported"))
            claim["method"] = "llm"
            if item.get("reason"):
                claim["reason"] = str(item["reason"])

    def _claim_sentences(self, answer: str) -> list[str]:
        sentences: list[str] = []
        for line in answer.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if not line:
                continue
            # A trailing citation belongs to the sentence before it, so re-attach citation-only fragments.
            for sentence in split_sentences(line):
                if sentences and _CITATION.sub("", sentence).strip() == "":
                    sentences[-1] += " " + sentence
                else:
                    sentences.append(sentence)
        return sentences

    def _has_unknown_numbers(self, claim: str, docs: list[Evidence]) -> bool:
        numbers = re.findall(r"\d+(?:[.,]\d+)*%?", claim)
        if not numbers:
            return False
        corpus = " ".join(doc.content for doc in docs).replace(",", "")
        return any(number.replace(",", "").rstrip("%") not in corpus for number in numbers)

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
