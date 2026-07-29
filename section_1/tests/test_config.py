import pytest

from arabic_voice_agent.config import Settings


def test_missing_required_setting_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GROQ_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        Settings.from_env()
