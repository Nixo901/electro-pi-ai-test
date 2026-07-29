from pathlib import Path

from arabic_voice_agent.pipeline import VoicePipeline


class FakeSpeech:
    async def transcribe(self, path: Path) -> str:
        assert path.name == "input.wav"
        return "أين طلبي؟"

    async def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        assert text == "رد عربي"
        output_path.write_bytes(b"RIFF")
        return output_path


class FakeConversation:
    async def reply(self, transcript: str) -> str:
        assert transcript == "أين طلبي؟"
        return "رد عربي"


async def test_pipeline_orders_stages(tmp_path: Path) -> None:
    result = await VoicePipeline(FakeSpeech(), FakeConversation()).run(
        tmp_path / "input.wav", tmp_path / "response.wav"
    )
    assert result.transcript == "أين طلبي؟"
    assert result.response == "رد عربي"
    assert result.audio_path.read_bytes() == b"RIFF"
