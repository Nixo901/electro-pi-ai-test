"""Safe mocked order lookup used by the agent's function tool."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OrderStatus:
    """Public information returned for one order."""

    order_id: str
    state: str
    eta_minutes: int


class OrderService:
    """Tiny deterministic repository; replace with an authenticated backend in production."""

    _orders = {
        "1001": OrderStatus("1001", "قيد التحضير", 18),
        "1002": OrderStatus("1002", "خرج مع المندوب", 7),
        "1003": OrderStatus("1003", "تم التسليم", 0),
    }

    async def get_status(self, order_id: str) -> str:
        """Validate an opaque numeric identifier and return a privacy-safe status."""
        normalized = order_id.strip()
        if not re.fullmatch(r"\d{4,12}", normalized):
            return "رقم الطلب غير صالح. يرجى إرسال رقم مكوّن من 4 إلى 12 رقمًا."
        status = self._orders.get(normalized)
        if status is None:
            return f"لم أجد طلبًا بالرقم {normalized}. تحقق من الرقم وحاول مرة أخرى."
        if status.state == "تم التسليم":
            return f"الطلب {status.order_id} تم تسليمه بالفعل."
        return f"الطلب {status.order_id} {status.state}. الوقت المتوقع للوصول {status.eta_minutes} دقائق."

    async def get_status_english(self, order_id: str) -> str:
        """Return the same restricted order information in English."""
        normalized = order_id.strip()
        if not re.fullmatch(r"\d{4,12}", normalized):
            return "That order number is invalid. Please provide 4 to 12 digits."
        status = self._orders.get(normalized)
        if status is None:
            return f"I could not find an order with number {normalized}. Please check it and try again."
        states = {
            "قيد التحضير": "is being prepared",
            "خرج مع المندوب": "is out for delivery",
            "تم التسليم": "has already been delivered",
        }
        if status.state == "تم التسليم":
            return f"Order {status.order_id} has already been delivered."
        return f"Order {status.order_id} {states[status.state]}. Estimated arrival is {status.eta_minutes} minutes."
