"""Extract code blocks from markdown-formatted LLM responses.

Three packages had near-identical implementations:
  - packages/autonomous/dialogue.py  (regex, language=c only, returns None on miss)
  - packages/llm_analysis/agent.py   (string split, cpp/c/python priority, returns content on miss)

Key implementation decisions:
  - Regex approach is used (more robust than string splitting for edge cases like
    adjacent fences or fences without trailing newline).
  - language_hints controls priority order; first matching language wins.
  - fallback_to_content controls behaviour when no fence is found:
      False (default) → return None  (matches dialogue.py)
      True            → return content.strip()  (matches agent.py)
"""

from __future__ import annotations

import re
from typing import List, Optional

# Matches ``` optionally followed by a language tag, then content up to closing ```.
# Group 1: language tag (may be empty).
# Group 2: code content.
_FENCED_BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_code_from_markdown(
    content: str,
    language_hints: Optional[List[str]] = None,
    *,
    fallback_to_content: bool = False,
) -> Optional[str]:
    """Return the first matching code block from a markdown-formatted string.

    Search order:
    1. Fenced blocks whose language tag matches any entry in *language_hints*
       (checked in the order given — first match wins).
    2. Any fenced block (with or without a language tag).
    3. If *fallback_to_content* is ``True``, return ``content.strip()``.
    4. ``None``.

    Args:
        content: Raw LLM response text, possibly containing ``` fences.
        language_hints: Preferred language tags in priority order, e.g.
            ``["cpp", "c", "python"]``.  Case-insensitive.
        fallback_to_content: When ``True`` and no fence is found, return
            ``content.strip()`` instead of ``None``.  Set this for callers
            that always need *some* string back (e.g. patch generation).

    Returns:
        Extracted code string (stripped), or ``None`` / ``content.strip()``
        depending on *fallback_to_content*.
    """
    blocks: List[tuple[str, str]] = _FENCED_BLOCK.findall(content)  # [(lang, code), ...]

    if language_hints:
        for hint in language_hints:
            hint_lower = hint.lower()
            for lang, code in blocks:
                if lang.lower() == hint_lower:
                    return code.strip()

    # Any fenced block
    if blocks:
        return blocks[0][1].strip()

    if fallback_to_content:
        return content.strip()
    return None
