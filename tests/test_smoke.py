from app.config import load_settings
from app.graph.state import SessionState


def test_settings_and_state_load():
    assert load_settings().app_name == "BD Demo Project"
    assert SessionState.model_fields["session_id"].is_required()
