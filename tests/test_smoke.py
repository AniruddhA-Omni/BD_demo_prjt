from app.config import get_settings
from app.graph.state import SessionState


def test_settings_and_state_load():
    settings = get_settings()
    assert settings["app_name"] == "BD Demo Project"
    assert SessionState.model_fields["session_id"].is_required()
