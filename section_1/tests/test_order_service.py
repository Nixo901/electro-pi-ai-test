from arabic_voice_agent.services.order_service import OrderService


async def test_known_order_returns_mocked_status() -> None:
    result = await OrderService().get_status("1002")
    assert "خرج مع المندوب" in result
    assert "7" in result


async def test_bad_order_id_is_rejected() -> None:
    result = await OrderService().get_status("not-an-id")
    assert "غير صالح" in result
