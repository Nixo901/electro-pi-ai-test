"""Create a development JWT for a browser/mobile LiveKit participant."""

from __future__ import annotations

import argparse

from livekit import api

from arabic_voice_agent.config import Settings


def main() -> None:
    """Print a token; use it only against the local development server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default="arabic-support-demo")
    parser.add_argument("--identity", default="demo-user")
    parser.add_argument("--language", default="english", choices=("english", "arabic"))
    parser.add_argument(
        "--agent-name",
        default="food-support",
        help="Registered LiveKit agent name. Keep the default unless changed in .env.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(args.identity)
        .with_name("Arabic Voice Demo User")
        .with_grants(api.VideoGrants(room_join=True, room=args.room))
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=args.agent_name, metadata=args.language)]
            )
        )
        .to_jwt()
    )
    print(token)


if __name__ == "__main__":
    main()
