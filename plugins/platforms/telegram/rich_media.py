"""Bot API 10.2 inline media for Telegram rich messages.

Split out of ``adapter.py``: payload building for ``InputRichMessage.media``,
the send/edit requests with their no-duplicate contract, standalone image
lines kept in place under ``rich_inline_media``, and their legacy fallback.
``TelegramRichMediaMixin`` is mixed into ``TelegramAdapter`` and relies on
its rich helpers (``_rich_message_payload``, ``_content_fits_rich_limits``,
``_bot_supports_rich``, ...).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from typing import Any, Dict, List, Optional

from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

try:
    from telegram.error import TelegramError
except ImportError:  # python-telegram-bot not installed
    TelegramError = OSError  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Errors ``do_api_request`` can surface: PTB wraps transport failures in
# TelegramError subclasses; AttributeError/TypeError/NotImplementedError mean
# the endpoint is unavailable (old PTB), see ``_is_rich_capability_error``.
_RICH_REQUEST_ERRORS = (
    TelegramError,
    OSError,
    AttributeError,
    TypeError,
    ValueError,
    NotImplementedError,
)

_RICH_FENCE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")
_RICH_FENCE_CLOSE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*$")


# Failures raised before the request left this host: Telegram cannot have
# received it, so the plain text + attachment path is safe (no duplicate).
_RICH_PRE_SEND_ERROR_NAMES = frozenset({
    "ConnectError",
    "ConnectTimeout",
    "PoolTimeout",
    "gaierror",
    "ConnectionRefusedError",
})
_RICH_PRE_SEND_MESSAGE_MARKERS = (
    "pool timeout",
    "httpx.connecterror",
    "httpx.connecttimeout",
    "name resolution",
    "name or service not known",
    "connection refused",
)

# ``rich_inline_media``: a standalone ``![alt](https://...)`` line stays in
# place as a rich media block instead of being extracted into a detached
# photo. Mirrors the image set of BasePlatformAdapter.extract_images.
_RICH_MEDIA_LINE_RE = re.compile(r"^[ ]{0,3}!\[[^\]\n]*\]\((https://[^\s)]+)\)[ \t]*$")
_RICH_MEDIA_EXT_RE = re.compile(r"(?i)\.(?:gif|png|jpe?g|webp)$")


def _rich_open_fence(text: str) -> Optional[str]:
    """Return the delimiter of a fenced code block left open at the end of ``text``."""
    fence: Optional[str] = None
    for line in text.split("\n"):
        if fence is not None:
            close = _RICH_FENCE_CLOSE_RE.match(line)
            if (
                close
                and close.group(1)[0] == fence[0]
                and len(close.group(1)) >= len(fence)
            ):
                fence = None
            continue
        opening = _RICH_FENCE_RE.match(line)
        if opening:
            fence = opening.group(1)
    return fence


def _rich_close_open_fence(text: str) -> str:
    """Close a trailing unclosed fence so appended media blocks are not code text."""
    fence = _rich_open_fence(text)
    return f"{text}\n{fence}" if fence else text


def _rich_is_pre_send_error(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) is a connect/DNS/pool failure."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _RICH_PRE_SEND_ERROR_NAMES:
            return True
        text = str(current).lower()
        if any(marker in text for marker in _RICH_PRE_SEND_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _rich_iter_lines(text: str):
    """Yield ``(line, media_url_or_None)``; lines in fenced code are never media."""
    fence: Optional[str] = None
    for line in text.split("\n"):
        if fence is not None:
            close = _RICH_FENCE_CLOSE_RE.match(line)
            if (
                close
                and close.group(1)[0] == fence[0]
                and len(close.group(1)) >= len(fence)
            ):
                fence = None
            yield line, None
            continue
        opening = _RICH_FENCE_RE.match(line)
        if opening:
            fence = opening.group(1)
            yield line, None
            continue
        match = _RICH_MEDIA_LINE_RE.match(line)
        url = match.group(1) if match else None
        if url and not _RICH_MEDIA_EXT_RE.search(url.split("?", 1)[0]):
            url = None
        yield line, url


def _rich_standalone_media_urls(text: str) -> List[str]:
    """HTTPS image URLs that sit alone on a line outside fenced code."""
    return [url for _, url in _rich_iter_lines(text or "") if url]


def _rich_separate_media_blocks(text: str) -> str:
    """Surround each standalone media line with blank lines (own rich block)."""
    if not text or "![" not in text:
        return text
    lines = list(_rich_iter_lines(text))
    out: List[str] = []
    for index, (line, url) in enumerate(lines):
        if not url:
            out.append(line)
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(line.strip())
        if index + 1 < len(lines) and lines[index + 1][0].strip():
            out.append("")
    return "\n".join(out)


def _rich_strip_standalone_media(text: str) -> str:
    """Drop standalone media lines (legacy path sends them as attachments)."""
    kept = [line for line, url in _rich_iter_lines(text or "") if not url]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


class TelegramRichMediaMixin:
    """Inline-media methods of ``TelegramAdapter``."""

    def _rich_inline_media_active(self) -> bool:
        """``rich_inline_media`` opt-in AND a usable rich transport."""
        return bool(
            getattr(self, "_rich_inline_media_enabled", False)
            and self._rich_delivery_enabled()
            and not getattr(self, "_rich_send_disabled", False)
            and self._bot_supports_rich()
        )

    def extract_images(self, content: str) -> tuple[List[tuple[str, str]], str]:
        """Base extraction, but standalone image lines stay in place when the
        reply will ride a rich message (``rich_inline_media``).

        Shadows the base staticmethod on instances only; callers use
        ``adapter.extract_images``.
        """
        base = BasePlatformAdapter.extract_images
        if not (
            content
            and self._rich_inline_media_active()
            and _rich_standalone_media_urls(content)
            and self._rich_eligible(content)
        ):
            return base(content)
        nonce = secrets.token_hex(8)
        while nonce in content:
            nonce = secrets.token_hex(8)
        placeholders: Dict[str, str] = {}
        kept: List[str] = []
        for line, url in _rich_iter_lines(content):
            if url:
                key = f"\x00RICHMEDIA{nonce}:{len(placeholders)}\x00"
                placeholders[key] = line
                kept.append(key)
            else:
                kept.append(line)
        images, cleaned = base("\n".join(kept))
        for key, line in placeholders.items():
            cleaned = cleaned.replace(key, line)
        return images, cleaned

    async def _send_with_standalone_media_fallback(
        self,
        chat_id: str,
        content: str,
        urls: List[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        """Legacy path for kept-inline image lines: text first, then each
        image as a native attachment. Partial delivery is never retryable."""
        text = _rich_strip_standalone_media(content)
        result = (
            await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)
            if text
            else SendResult(success=True, message_id=None)
        )
        if not result.success and result.message_id is None:
            return result
        failure: Optional[SendResult] = None
        first_media: Optional[SendResult] = None
        for url in urls:
            sender = (
                self.send_animation if self._is_animation_url(url) else self.send_image
            )
            media = await sender(chat_id, url, reply_to=reply_to, metadata=metadata)
            if media.success:
                first_media = first_media or media
            else:
                failure = media
                logger.warning(
                    "[%s] inline media fallback failed: %s", self.name, media.error
                )
        if failure is not None and result.success:
            delivered = result.message_id or (
                first_media.message_id if first_media else None
            )
            return SendResult(
                success=False,
                message_id=delivered,
                error=failure.error or "inline_media_failed",
                retryable=delivered is None and bool(failure.retryable),
            )
        if result.message_id is None and first_media is not None:
            return first_media
        return result

    @staticmethod
    def _rich_inline_media_type(path: str, is_voice: bool) -> Optional[tuple[str, str]]:
        """Return ``(InputMedia.type, tg:// scheme)`` for supported local media."""
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            return "photo", "photo"
        if ext == ".gif":
            return "animation", "video"
        if ext == ".mp4":
            return "video", "video"
        if ext in {".mp3", ".m4a"}:
            return "audio", "audio"
        if is_voice and ext in {".ogg", ".opus"}:
            return "voice_note", "audio"
        return None

    def _build_rich_inline_media_payload(
        self,
        content: str,
        media_files: List[tuple],
        stack: Any,
    ) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        """Build InputRichMessage.media plus multipart InputFile parameters.

        Returns ``(rich_message, uploads, unembedded)``; ``unembedded`` lists
        the files that must still go out as regular attachments.
        """
        # Defensive bound: cap simultaneously opened files and multipart parts.
        if not media_files or len(media_files) > 50:
            return None
        from telegram import InputFile

        media_entries: List[Dict[str, Any]] = []
        markdown_blocks: List[str] = []
        uploads: Dict[str, Any] = {}
        unembedded: List[tuple] = []
        for item in media_files:
            try:
                path, is_voice = item
            except (TypeError, ValueError):
                return None
            path = str(path)
            media_type = self._rich_inline_media_type(path, bool(is_voice))
            if media_type is None or not os.path.isfile(path):
                # Embed what Telegram supports; the rest keeps the regular
                # attachment path instead of dragging every file with it.
                unembedded.append(item)
                continue
            index = len(media_entries)
            input_type, scheme = media_type
            file_obj = stack.enter_context(open(path, "rb"))
            input_file = InputFile(
                file_obj,
                filename=os.path.basename(path),
                attach=True,
            )
            media_id = f"media_{index}"
            media_entries.append({
                "id": media_id,
                "media": {
                    "type": input_type,
                    "media": input_file.attach_uri,
                },
            })
            markdown_blocks.append(f"![](tg://{scheme}?id={media_id})")
            # PTB 22.6 cannot recursively discover InputFile inside the new
            # InputRichMessage.media object. Supplying each InputFile as a
            # top-level multipart value makes PTB add the attachment part; the
            # Bot API ignores the extra scalar field and resolves attach://.
            uploads[input_file.attach_name] = input_file

        if not media_entries:
            return None
        rich_message = self._rich_message_payload(content)
        rich_message["markdown"] = (
            _rich_close_open_fence(rich_message["markdown"].rstrip())
            + "\n\n"
            + "\n\n".join(markdown_blocks)
        )
        if not self._content_fits_rich_limits(rich_message["markdown"]):
            return None
        rich_message["media"] = media_entries
        return rich_message, uploads, unembedded

    async def _rich_media_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        content: str,
    ) -> Optional[SendResult]:
        """Execute rich-media send/edit with the existing no-duplicate contract."""
        try:
            raw = await self._bot.do_api_request(endpoint, api_kwargs=payload)
        except _RICH_REQUEST_ERRORS as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                logger.debug(
                    "[%s] %s inline-media rejected (%s) — using separate attachments",
                    self.name,
                    endpoint,
                    exc,
                )
                return None
            safe_error = self._redact_error(exc)
            if _rich_is_pre_send_error(exc):
                logger.warning(
                    "[%s] %s inline-media request not sent (%s) — using plain text + attachments",
                    self.name,
                    endpoint,
                    safe_error,
                )
                return None
            # The request may have reached Telegram: no resend (duplicate risk),
            # but never lose the reply silently.
            logger.warning(
                "[%s] response_delivery_dropped: %s inline-media outcome unknown "
                "(%d chars, %d media), not resent to avoid a duplicate: %s",
                self.name,
                endpoint,
                len(content or ""),
                len(payload.get("rich_message", {}).get("media", [])),
                safe_error,
            )
            return SendResult(success=False, error=safe_error, retryable=True)

        message_id = None
        if isinstance(raw, dict):
            message_id = raw.get("message_id")
            if message_id is None:
                message_id = (raw.get("result") or {}).get("message_id")
        else:
            message_id = getattr(raw, "message_id", None)
        if message_id is not None:
            from gateway import rich_sent_store

            # record() is a no-op on any storage failure.
            rich_sent_store.record(
                str(payload.get("chat_id")), str(message_id), content
            )
        return SendResult(
            success=True,
            message_id=str(message_id) if message_id is not None else None,
        )

    async def send_rich_media(
        self,
        chat_id: str,
        content: str,
        media_files: List[tuple],
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Send one Bot API 10.2 rich report with embedded local media."""
        if self._is_ephemeral_delivery(metadata):
            # The generic caller retains both text and files when this returns
            # None, then routes them through targetable ephemeral send methods.
            return None
        if (
            not getattr(self, "_rich_inline_media_enabled", False)
            or not getattr(self, "_rich_messages_enabled", False)
            or getattr(self, "_rich_send_disabled", False)
            or not self._bot_supports_rich()
        ):
            return None
        from contextlib import ExitStack

        routing = self._compute_single_send_routing(
            chat_id, reply_to, metadata, self._metadata_thread_id(metadata)
        )
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing
        with ExitStack() as stack:
            built = self._build_rich_inline_media_payload(content, media_files, stack)
            if built is None:
                return None
            rich_message, uploads, unembedded = built
            payload: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "rich_message": rich_message,
                **uploads,
            }
            payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
            payload.update(self._notification_kwargs(metadata))
            if getattr(self, "_disable_link_previews", False):
                payload["link_preview_options"] = {"is_disabled": True}
            if reply_to_id is not None:
                payload["reply_parameters"] = {"message_id": reply_to_id}
            result = await self._rich_media_request("sendRichMessage", payload, content)
        return self._with_unembedded_media(result, unembedded)

    async def edit_rich_media(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        media_files: List[tuple],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Upgrade a streamed text preview to rich text with embedded media."""
        if self._is_ephemeral_delivery(metadata):
            # Ephemeral delivery disables streaming, so no public preview is
            # eligible for an in-place rich edit.
            return None
        if (
            not getattr(self, "_rich_inline_media_enabled", False)
            or not getattr(self, "_rich_messages_enabled", False)
            or getattr(self, "_rich_send_disabled", False)
            or not self._bot_supports_rich()
        ):
            return None
        from contextlib import ExitStack

        with ExitStack() as stack:
            built = self._build_rich_inline_media_payload(content, media_files, stack)
            if built is None:
                return None
            rich_message, uploads, unembedded = built
            payload: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "message_id": int(message_id),
                "rich_message": rich_message,
                **uploads,
            }
            if getattr(self, "_disable_link_previews", False):
                payload["link_preview_options"] = {"is_disabled": True}
            result = await self._rich_media_request("editMessageText", payload, content)
        return self._with_unembedded_media(result, unembedded)

    @staticmethod
    def _with_unembedded_media(
        result: Optional[SendResult], unembedded: List[tuple]
    ) -> Optional[SendResult]:
        """Hand files that were not embedded back to the caller.

        Contract: ``raw_response["unembedded_media"]`` lists ``(path, is_voice)``
        items the caller still sends as regular attachments (base.py rich-media
        hook, gateway/run.py post-stream upgrade). Only set on success.
        """
        if result is not None and result.success and unembedded:
            result.raw_response = {"unembedded_media": list(unembedded)}
        return result
