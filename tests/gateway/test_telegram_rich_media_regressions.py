"""Regression tests for Bot API 10.2 inline media (follow-ups to #64637)."""

import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from telegram.error import NetworkError, TimedOut

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_rich_inline_media import _FakeInputFile


def _adapter(extra=None):
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"rich_messages": True, "rich_inline_media": True, **(extra or {})},
        )
    )
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value={"message_id": 321})
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter


def _build(adapter, content, paths):
    with patch("telegram.InputFile", _FakeInputFile), ExitStack() as stack:
        return adapter._build_rich_inline_media_payload(
            content, [(str(p), False) for p in paths], stack
        )


def _png(tmp_path, name="chart.png"):
    path = tmp_path / name
    path.write_bytes(b"png")
    return path


def test_media_refs_are_not_appended_inside_an_unclosed_code_fence(tmp_path):
    built = _build(_adapter(), "log:\n```\nline1\nline2", [_png(tmp_path)])
    assert built is not None
    markdown = built[0]["markdown"]
    assert markdown.count("```") == 2, markdown
    assert markdown.rindex("```") < markdown.index("tg://photo?id=media_0"), markdown


def test_standalone_image_line_keeps_its_position_in_rich_message():
    adapter = _adapter()
    content = "| a | b |\n|---|---|\n| 1 | 2 |\n\n![chart](https://example.com/c.png)\n\nOutro"
    images, cleaned = adapter.extract_images(content)
    assert images == []
    assert "![chart](https://example.com/c.png)" in cleaned
    markdown = adapter._rich_message_payload(cleaned)["markdown"]
    assert markdown.index("example.com/c.png") < markdown.index("Outro")


def test_standalone_image_line_is_extracted_when_rich_inline_media_is_off():
    adapter = _adapter(extra={"rich_inline_media": False})
    images, cleaned = adapter.extract_images(
        "Intro\n\n![](https://example.com/c.png)\n\nOutro"
    )
    assert images == [("https://example.com/c.png", "")]
    assert "example.com" not in cleaned


@pytest.mark.xfail(strict=True, reason="gap: partial embed (fixed later in this PR)")
@pytest.mark.asyncio
async def test_mixed_attachments_embed_supported_media_and_return_the_rest(tmp_path):
    photo = _png(tmp_path)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    adapter = _adapter()

    with patch("telegram.InputFile", _FakeInputFile):
        result = await adapter.send_rich_media(
            "12345", "## Report", [(str(photo), False), (str(document), False)]
        )

    assert result is not None and result.success
    payload = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert len(payload["rich_message"]["media"]) == 1
    assert result.raw_response["unembedded_media"] == [(str(document), False)]


@pytest.mark.asyncio
async def test_pre_send_network_error_falls_back_to_plain_delivery(tmp_path):
    adapter = _adapter()
    error = NetworkError(
        "httpx.ConnectError: [Errno -3] Temporary failure in name resolution"
    )
    error.__cause__ = httpx.ConnectError("dns")
    adapter._bot.do_api_request = AsyncMock(side_effect=error)

    with patch("telegram.InputFile", _FakeInputFile):
        result = await adapter.send_rich_media(
            "12345", "important text", [(str(_png(tmp_path)), False)]
        )

    # None hands text and files back to the regular send path.
    assert result is None


@pytest.mark.asyncio
async def test_ambiguous_timeout_is_logged_as_dropped_delivery(tmp_path, caplog):
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("Timed out"))

    with patch("telegram.InputFile", _FakeInputFile), caplog.at_level(logging.WARNING):
        result = await adapter.send_rich_media(
            "12345", "important text", [(str(_png(tmp_path)), False)]
        )

    assert result is not None and result.success is False
    assert "response_delivery_dropped" in caplog.text


@pytest.mark.asyncio
async def test_kept_inline_image_goes_native_when_rich_is_rejected():
    from telegram.error import BadRequest

    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("can't parse"))
    adapter.send_image = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="9")
    )

    result = await adapter.send(
        "12345", "| a |\n|---|\n| 1 |\n\n![](https://example.com/c.png)\n\nOutro"
    )

    assert result.success
    sent_text = adapter._bot.send_message.await_args.kwargs["text"]
    assert "example.com" not in sent_text
    adapter.send_image.assert_awaited_once()
    assert adapter.send_image.await_args.args[1] == "https://example.com/c.png"
