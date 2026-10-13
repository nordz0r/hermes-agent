"""``rich_inline_media``: standalone media lines ride a rich message as inline media blocks.

Uses the real ``TelegramAdapter`` (``telegram`` is mocked by ``tests/gateway/conftest.py``) with a
mock Bot API; nothing is sent anywhere.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _rich_separate_media_blocks,
    _rich_standalone_media_urls,
    _rich_strip_standalone_media,
)
from telegram.error import BadRequest

GIF = "https://img.example.com/diagram.gif"
REPLY = f"Here is the flow:\n![flow]({GIF})\nDone."


def _make_adapter(extra=None, *, rich_bot=True):
    config = PlatformConfig(
        enabled=True, token="fake-token",
        extra={"rich_messages": True, "rich_inline_media": True, **(extra or {})})
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    if rich_bot:
        bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    else:
        bot.do_api_request = MagicMock()  # sync callable → no rich transport
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_animation = AsyncMock(return_value=MagicMock(message_id=2))
    bot.send_photo = AsyncMock(return_value=MagicMock(message_id=3))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter


def _rich_markdown(adapter):
    call = adapter._bot.do_api_request.await_args
    assert call.args[0] == "sendRichMessage"
    return call.kwargs["api_kwargs"]["rich_message"]["markdown"]


class TestHelpers:
    def test_standalone_only_outside_code(self):
        text = f"![a]({GIF})\n```\n![b](https://x.example/b.png)\n```\ninline ![c](https://x.example/c.png) text"
        assert _rich_standalone_media_urls(text) == [GIF]

    def test_http_and_non_media_ignored(self):
        assert _rich_standalone_media_urls("![a](http://x.example/a.gif)\n![b](https://x.example/page)") == []

    def test_separate_blocks(self):
        assert _rich_separate_media_blocks(REPLY) == f"Here is the flow:\n\n![flow]({GIF})\n\nDone."

    def test_strip(self):
        assert _rich_strip_standalone_media(REPLY) == "Here is the flow:\nDone."


    def test_invalid_fence_closer_keeps_code(self):
        text = "```text\n```not-a-closer\n![x](https://x.example/demo.png)\n```"
        assert _rich_standalone_media_urls(text) == []
        assert _rich_strip_standalone_media(text) == text


class TestExtractImages:
    def test_placeholder_collision_safe(self):
        content = "literal \x00RICHMEDIA0\x00 token\n" + REPLY
        _, text = _make_adapter().extract_images(content)
        assert text.count(GIF) == 1 and "\x00RICHMEDIA0\x00" in text

    def test_keeps_media_inline_when_rich(self):
        images, text = _make_adapter().extract_images(REPLY)
        assert images == [] and GIF in text

    def test_inline_and_html_images_still_extracted(self):
        content = REPLY + '\nsee <img src="https://x.example/i.png">'
        images, text = _make_adapter().extract_images(content)
        assert images == [("https://x.example/i.png", "")] and GIF in text

    @pytest.mark.parametrize("extra,rich_bot", [({"rich_inline_media": False}, True),
                                                 ({"rich_messages": False}, True), ({}, False)])
    def test_base_behaviour_when_inactive(self, extra, rich_bot):
        images, text = _make_adapter(extra, rich_bot=rich_bot).extract_images(REPLY)
        assert images == [(GIF, "flow")] and GIF not in text

    def test_base_staticmethod_unchanged(self):
        from gateway.platforms.base import BasePlatformAdapter
        assert BasePlatformAdapter.extract_images(REPLY)[0] == [(GIF, "flow")]


class TestSend:
    @pytest.mark.asyncio
    async def test_media_becomes_separate_rich_block(self):
        adapter = _make_adapter()
        result = await adapter.send("12345", REPLY)
        assert result.success and result.message_id == "123"
        assert f"\n\n![flow]({GIF})\n\n" in _rich_markdown(adapter)
        adapter._bot.send_message.assert_not_awaited()
        adapter._bot.send_animation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_off_plain_reply_stays_legacy(self):
        adapter = _make_adapter({"rich_inline_media": False})
        await adapter.send("12345", "plain text")
        adapter._bot.do_api_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rich_rejected_falls_back_to_native_animation(self):
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        result = await adapter.send("12345", REPLY)
        assert result.success
        sent_text = adapter._bot.send_message.await_args.kwargs["text"]
        assert GIF not in sent_text and "Done" in sent_text
        assert adapter._bot.send_animation.await_args.kwargs["animation"] == GIF

    @pytest.mark.asyncio
    async def test_media_only_reply_fallback_sends_photo(self, monkeypatch):
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        result = await adapter.send("12345", "![p](https://x.example/p.png)")
        assert result.success and result.message_id == "3"
        adapter._bot.send_message.assert_not_awaited()
        assert adapter._bot.send_photo.await_args.kwargs["photo"] == "https://x.example/p.png"

    @pytest.mark.asyncio
    async def test_thread_routing_kept_on_rich(self):
        adapter = _make_adapter()
        await adapter.send("-100123", REPLY, metadata={"thread_id": "42"})
        kwargs = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"]
        assert kwargs.get("message_thread_id") == 42

    @pytest.mark.asyncio
    async def test_expect_edits_does_not_force_rich(self):
        adapter = _make_adapter()
        await adapter.send("12345", REPLY, metadata={"expect_edits": True})
        adapter._bot.do_api_request.assert_not_awaited()
        # Editable previews never emit attachments that would outlive the preview.
        adapter._bot.send_animation.assert_not_awaited()


class TestFallbackResults:
    @pytest.mark.asyncio
    async def test_media_only_all_failed_is_failure(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        monkeypatch.setattr(adapter, "send_animation", AsyncMock(
            return_value=__import__("gateway.platforms.base", fromlist=["SendResult"]).SendResult(success=False, error="boom")))
        result = await adapter.send("12345", f"![a]({GIF})")
        assert not result.success

    @pytest.mark.asyncio
    async def test_text_failure_skips_media(self, monkeypatch):
        from gateway.platforms.base import SendResult
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        monkeypatch.setattr(adapter, "_send_chunks", AsyncMock(return_value=SendResult(success=False, error="x")))
        result = await adapter.send("12345", REPLY)
        assert not result.success
        adapter._bot.send_animation.assert_not_awaited()


class TestPartialFallback:
    @pytest.mark.asyncio
    async def test_text_ok_media_failed_reports_partial(self, monkeypatch):
        from gateway.platforms.base import SendResult
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        monkeypatch.setattr(adapter, "send_animation", AsyncMock(return_value=SendResult(success=False, error="boom")))
        result = await adapter.send("12345", REPLY)
        assert not result.success and not result.retryable and result.message_id == "1"

    @pytest.mark.asyncio
    async def test_partial_text_failure_still_sends_media(self, monkeypatch):
        from gateway.platforms.base import SendResult
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        partial = SendResult(success=False, message_id="1", error="tail", retryable=True)
        monkeypatch.setattr(adapter, "_send_chunks", AsyncMock(return_value=partial))
        result = await adapter.send("12345", REPLY)
        assert result is partial
        assert adapter._bot.send_animation.await_count == 1

    @pytest.mark.asyncio
    async def test_media_only_certain_failure_stays_retryable(self, monkeypatch):
        from gateway.platforms.base import SendResult
        adapter = _make_adapter()
        adapter._bot.do_api_request.side_effect = BadRequest("Can't parse rich message")
        monkeypatch.setattr(adapter, "send_animation", AsyncMock(
            return_value=SendResult(success=False, error="flood", retryable=True, retry_after=30)))
        result = await adapter.send("12345", f"![a]({GIF})")
        assert not result.success and result.retryable and result.retry_after == 30

    def test_mp4_not_kept_inline(self):
        assert _rich_standalone_media_urls("![v](https://x.example/v.mp4)") == []

    def test_titled_image_not_kept_inline(self):
        line = f'![a]({GIF} "title")'
        assert _rich_standalone_media_urls(line) == []


class TestGatewayExtraction:
    @pytest.mark.asyncio
    async def test_as_document_uses_plain_extraction(self):
        adapter = _make_adapter()
        event = SimpleNamespace(source=SimpleNamespace(platform=None, chat_id="12345"))
        extracted = await adapter._extract_response_content(
            "[[as_document]]\n" + REPLY, event, "s", is_ephemeral_response=False)
        assert extracted.images and GIF not in extracted.text_content

    @pytest.mark.asyncio
    async def test_base_response_extraction_keeps_inline_media(self):
        adapter = _make_adapter()
        event = SimpleNamespace(source=SimpleNamespace(platform=None, chat_id="12345"))
        try:
            extracted = await adapter._extract_response_content(REPLY, event, "s", is_ephemeral_response=False)
        except Exception as exc:  # pragma: no cover - surfaced loudly
            pytest.fail(f"extraction raised {exc!r}")
        assert not extracted.images
        assert GIF in extracted.text_content
