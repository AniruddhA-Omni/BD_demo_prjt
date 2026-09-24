from __future__ import annotations

from typing import Any, Iterable

from app import services
from app.agents.retrieval import RetrievalAgent
from app.ingestion.models import Evidence
from app.ingestion.router import VISION_DESCRIPTION_PROMPT
from app.observability import traceable

IMAGE_SOURCE_TYPES = {"image", "vision"}

_VISION_QUESTION_PROMPT = (
    "Answer the question using only what is visible in the image(s). Read labels, values and text exactly. "
    "If the image does not contain the answer, say so.\n\nQuestion: {question}"
)


class VisionAgent:
    """Image questions: retrieve OCR/visual-description evidence, then ask the multimodal model about the images.

    The model's reading of the image is returned as evidence tied to the image file, so synthesis cites the image
    and verification checks the final answer against it. Without a vision model only OCR/description evidence is used.
    """

    def __init__(self, retrieval: RetrievalAgent | None = None, llm: Any | None = None, max_images: int = 3) -> None:
        self.retrieval = retrieval or RetrievalAgent()
        self.llm = llm if llm is not None else services.get_llm()
        self.max_images = max_images

    @traceable(name="vision_agent")
    def run(self, question: str, evidence: Iterable[Evidence], top_k: int = 4, session_id: str | None = None) -> list[Evidence]:
        docs = list(evidence)
        image_docs = [doc for doc in docs if doc.source_type in IMAGE_SOURCE_TYPES]
        if not image_docs:
            return self.retrieval.retrieve(question, docs, top_k=top_k, session_id=session_id)

        descriptions = self._ensure_descriptions(image_docs)
        ranked = self.retrieval.retrieve(question, image_docs, top_k=top_k, session_id=session_id)
        ranked = [*descriptions, *[doc for doc in ranked if doc.key not in {d.key for d in descriptions}]]
        paths: list[str] = []
        origin: dict[str, Evidence] = {}
        for doc in ranked + image_docs:
            path = str(doc.metadata.get("source_path") or "")
            if path and path not in origin:
                origin[path] = doc
                paths.append(path)
        paths = paths[: self.max_images]

        answer = self.llm.describe_images(_VISION_QUESTION_PROMPT.format(question=question), paths, run_name="vision_answer")
        if not answer:
            return ranked
        source = origin[paths[0]]
        names = ", ".join(origin[path].file_name for path in paths)
        visual = Evidence(
            document_id=source.document_id,
            file_name=source.file_name if len(paths) == 1 else names,
            file_type=source.file_type,
            source_type="vision",
            section="vision model reading",
            chunk_id=f"{source.document_id}:vision-answer",
            content=answer,
            metadata={"source_path": paths[0], "images": paths, "status": "parsed"},
        )
        return [visual, *ranked]

    def _ensure_descriptions(self, image_docs: list[Evidence]) -> list[Evidence]:
        """Describe each image once with the vision model and cache it on the OCR evidence (lazy, not at upload).

        The cached description joins the chunk's ``search_text`` (so later questions can find the image by what it
        shows) and is returned as ``vision`` evidence for this question. Images already described at ingestion
        (VISION_AT_INGESTION=true) are skipped.
        """
        described_at_ingestion = {doc.document_id for doc in image_docs if doc.source_type == "vision"}
        described: list[Evidence] = []
        budget = self.max_images
        for doc in image_docs:
            if doc.source_type != "image" or doc.document_id in described_at_ingestion:
                continue
            description = doc.metadata.get("vision_description")
            path = str(doc.metadata.get("source_path") or "")
            if not description and path and budget > 0:
                budget -= 1
                description = self.llm.describe_images(VISION_DESCRIPTION_PROMPT, [path], run_name="vision_describe")
                if description:
                    doc.metadata["vision_description"] = description
            if description:
                described.append(
                    Evidence(
                        document_id=doc.document_id,
                        file_name=doc.file_name,
                        file_type=doc.file_type,
                        source_type="vision",
                        section="visual description",
                        chunk_id=f"{doc.document_id}:vision-description",
                        content=description,
                        metadata={"source_path": path, "status": "parsed", "lazy": True},
                    )
                )
        return described
