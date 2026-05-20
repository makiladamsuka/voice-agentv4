"""TTS text transforms to strip LLM tool-syntax leaks before speech."""

from __future__ import annotations

import re
from collections.abc import AsyncIterable

# Groq/Llama often emits <function=name {...} </function> without a '>' after the name.
_FUNCTION_TAG_RE = re.compile(
    r"<function[^<]*?</function\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_OPEN_RE = re.compile(r"<function[^<]*", re.IGNORECASE)
_FUNCTION_CLOSE_RE = re.compile(r"</function\s*>", re.IGNORECASE)
# Orphan JSON tool-args fragments sometimes leaked after a partial tag.
_TOOL_JSON_RE = re.compile(r'\{\s*"query"\s*:\s*"[^"]*"\s*\}', re.IGNORECASE)

# Keep enough tail to match tags split across TTS token boundaries.
_TAIL_LEN = 64


def _strip_tool_syntax(text: str) -> str:
    text = _FUNCTION_TAG_RE.sub("", text)
    text = _FUNCTION_OPEN_RE.sub("", text)
    text = _FUNCTION_CLOSE_RE.sub("", text)
    text = _TOOL_JSON_RE.sub("", text)
    return text


async def filter_leaked_tool_syntax(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """Remove leaked tool-call markup from streamed TTS text."""
    buffer = ""
    async for chunk in text:
        buffer += chunk
        if len(buffer) <= _TAIL_LEN:
            continue
        flush_to = len(buffer) - _TAIL_LEN
        out = _strip_tool_syntax(buffer[:flush_to])
        buffer = buffer[flush_to:]
        if out:
            yield out

    if buffer:
        out = _strip_tool_syntax(buffer)
        if out:
            yield out
