from pydantic import BaseModel, Field


class SessionState(BaseModel):
    session_id: str = Field(..., description="Unique per-session identifier")
    uploaded_files: list[str] = Field(default_factory=list)
    conversation_messages: list[str] = Field(default_factory=list)
    active_context: str = ""

    class Config:
        arbitrary_types_allowed = True
