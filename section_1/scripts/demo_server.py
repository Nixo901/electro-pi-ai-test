"""Serve the local web demo and mint short-lived, language-specific JWTs."""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from livekit import api

from arabic_voice_agent.config import Settings

WEB_ROOT = Path(__file__).resolve().parents[1] / "web_demo"


class DemoHandler(SimpleHTTPRequestHandler):
    """Static demo server with a local-only token endpoint."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/token":
            super().do_GET()
            return
        language = parse_qs(parsed.query).get("language", ["english"])[0]
        if language not in {"english", "arabic"}:
            self.send_error(HTTPStatus.BAD_REQUEST, "language must be english or arabic")
            return
        settings = Settings.from_env()
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(f"web-{secrets.token_urlsafe(8)}")
            .with_name("Local Voice Demo User")
            .with_grants(api.VideoGrants(room_join=True, room=f"voice-demo-{language}"))
            .with_room_config(
                api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name="food-support", metadata=language)]
                )
            )
            .to_jwt()
        )
        body = json.dumps({"token": token, "url": settings.livekit_url, "language": language}).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Run only on loopback: this server holds the local development secret."""
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("Open http://127.0.0.1:8000 in your browser")
    server.serve_forever()


if __name__ == "__main__":
    main()
