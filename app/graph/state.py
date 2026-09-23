from pydantic import BaseModel, ConfigDict, Field


class SessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(..., description="Unique per-session identifier")
    uploaded_files: list[str] = Field(default_factory=list)
    conversation_messages: list[dict[str, str]] = Field(default_factory=list)
    active_context: str = ""
    last_question: str = ""
    last_answer: str = ""
