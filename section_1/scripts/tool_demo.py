"""Produces a reproducible, offline proof of the order-status business tool."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from arabic_voice_agent.services.order_service import OrderService


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    service = OrderService()
    order_id = "1002"
    result = await service.get_status(order_id)
    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_utterance": "ما حالة طلبي رقم 1002؟",
        "llm_tool_call": {"name": "get_order_status", "arguments": {"order_id": order_id}},
        "tool_result": result,
        "expected_agent_reply": f"{result} هل يمكنني مساعدتك في شيء آخر؟",
        "note": "Run the live agent and speak the same utterance to capture an end-to-end tool-call log.",
    }
    target = Path("artifacts/tool_call_demo.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
