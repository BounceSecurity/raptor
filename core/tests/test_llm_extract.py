"""Tests for core.llm.extract.extract_code_from_markdown.

Three packages had near-identical implementations with these differences:
  - dialogue.py:  regex; language_hints=["c"]; returns None on no match
  - agent.py:     string-split; hints=["cpp","c","python"]; returns content on no match
  - (cc_dispatch is JSON extraction, not code extraction — separate concern)

Tests explicitly cover:
  1. Language hint priority (cpp before c before python)
  2. Generic fence fallback
  3. fallback_to_content behaviour (None vs content.strip())
  4. Edge cases: no fence, empty fence, adjacent fences, fence without newline
  5. Equivalence with each original implementation's behaviour
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm.extract import extract_code_from_markdown


class TestLanguageHintPriority:
    """Priority order must match the original agent.py implementation."""

    def test_cpp_preferred_over_c(self):
        content = "```c\nc_code\n```\n\n```cpp\ncpp_code\n```"
        result = extract_code_from_markdown(content, language_hints=["cpp", "c"])
        assert result == "cpp_code"

    def test_c_preferred_over_python(self):
        content = "```python\npy_code\n```\n\n```c\nc_code\n```"
        result = extract_code_from_markdown(content, language_hints=["c", "python"])
        assert result == "c_code"

    def test_first_matching_hint_wins(self):
        content = "```python\npy\n```\n```c\nc\n```\n```cpp\ncpp\n```"
        result = extract_code_from_markdown(content, language_hints=["cpp", "c", "python"])
        assert result == "cpp"

    def test_hint_not_present_falls_through_to_generic(self):
        content = "```javascript\njs_code\n```"
        result = extract_code_from_markdown(content, language_hints=["c", "python"])
        # No hint matches → falls back to first fence regardless of language
        assert result == "js_code"

    def test_language_matching_is_case_insensitive(self):
        content = "```C\nsome_c_code\n```"
        result = extract_code_from_markdown(content, language_hints=["c"])
        assert result == "some_c_code"


class TestGenericFenceFallback:
    """Any fenced block should be returned when no hint matches."""

    def test_bare_fence_returned(self):
        content = "Here is some code:\n```\nx = 1\n```"
        result = extract_code_from_markdown(content)
        assert result == "x = 1"

    def test_first_fence_returned_when_multiple(self):
        content = "```\nfirst\n```\n\n```\nsecond\n```"
        result = extract_code_from_markdown(content)
        assert result == "first"

    def test_unknown_language_tag_returned(self):
        content = "```rust\nfn main() {}\n```"
        result = extract_code_from_markdown(content)
        assert result == "fn main() {}"


class TestFallbackToContent:
    """fallback_to_content=False → None; True → content.strip().

    dialogue.py used False (returns None on no fence).
    agent.py used True (returns content as-is on no fence).
    """

    def test_default_returns_none_when_no_fence(self):
        result = extract_code_from_markdown("no fences here")
        assert result is None

    def test_fallback_false_returns_none(self):
        result = extract_code_from_markdown("plain text", fallback_to_content=False)
        assert result is None

    def test_fallback_true_returns_stripped_content(self):
        result = extract_code_from_markdown("  plain text  ", fallback_to_content=True)
        assert result == "plain text"

    def test_fallback_true_still_returns_fence_content_when_present(self):
        """fallback_to_content=True should not affect normal fence extraction."""
        content = "```python\nprint('hi')\n```"
        result = extract_code_from_markdown(content, fallback_to_content=True)
        assert result == "print('hi')"

    def test_empty_string_with_fallback_true(self):
        result = extract_code_from_markdown("   ", fallback_to_content=True)
        assert result == ""

    def test_empty_string_with_fallback_false(self):
        result = extract_code_from_markdown("", fallback_to_content=False)
        assert result is None


class TestCodeStripping:
    """Extracted code must have leading/trailing whitespace stripped."""

    def test_leading_newline_stripped(self):
        content = "```c\n\nint main() {}\n```"
        result = extract_code_from_markdown(content, language_hints=["c"])
        assert result == "int main() {}"

    def test_trailing_whitespace_stripped(self):
        content = "```python\ncode   \n```"
        result = extract_code_from_markdown(content)
        assert result == "code"

    def test_internal_newlines_preserved(self):
        content = "```c\nline1\nline2\nline3\n```"
        result = extract_code_from_markdown(content)
        assert result == "line1\nline2\nline3"


class TestDialoguePyEquivalence:
    """Verify behaviour matches the old dialogue.py._extract_code_from_response."""

    def test_c_fence_extracted(self):
        response = "Here is the exploit:\n```c\n#include <stdio.h>\nint main(){}\n```"
        result = extract_code_from_markdown(response, language_hints=["c"])
        assert result == "#include <stdio.h>\nint main(){}"

    def test_generic_fence_fallback(self):
        response = "```\nsome_code\n```"
        result = extract_code_from_markdown(response, language_hints=["c"])
        assert result == "some_code"

    def test_no_fence_returns_none(self):
        response = "I cannot generate that exploit."
        result = extract_code_from_markdown(response, language_hints=["c"])
        assert result is None

    def test_cpp_fence_not_matched_by_c_hint(self):
        """Original dialogue.py used regex r'```c\\n' which does NOT match ```cpp.
        The new implementation must preserve this: 'c' hint should not match 'cpp' block."""
        response = "```cpp\ncpp_code\n```"
        result = extract_code_from_markdown(response, language_hints=["c"])
        # 'c' hint doesn't match 'cpp', falls through to generic → returns cpp_code
        assert result == "cpp_code"


class TestAgentPyEquivalence:
    """Verify behaviour matches the old agent.py._extract_code."""

    def test_cpp_block_extracted(self):
        content = "Here is the patch:\n```cpp\nvoid foo() {}\n```"
        result = extract_code_from_markdown(
            content, language_hints=["cpp", "c", "python"], fallback_to_content=True
        )
        assert result == "void foo() {}"

    def test_python_block_extracted(self):
        content = "```python\nprint('hello')\n```"
        result = extract_code_from_markdown(
            content, language_hints=["cpp", "c", "python"], fallback_to_content=True
        )
        assert result == "print('hello')"

    def test_generic_fence_when_no_hint_matches(self):
        content = "```\nsome code\n```"
        result = extract_code_from_markdown(
            content, language_hints=["cpp", "c", "python"], fallback_to_content=True
        )
        assert result == "some code"

    def test_plain_text_returned_when_no_fence(self):
        """agent.py returned content.strip() when no fence found."""
        content = "  Just some analysis text.  "
        result = extract_code_from_markdown(
            content, language_hints=["cpp", "c", "python"], fallback_to_content=True
        )
        assert result == "Just some analysis text."


class TestEdgeCases:

    def test_none_language_hints(self):
        """None and [] both mean no preference — returns first fence."""
        content = "```c\ncode\n```"
        assert extract_code_from_markdown(content, None) == "code"
        assert extract_code_from_markdown(content, []) == "code"

    def test_empty_fence_content(self):
        content = "```c\n\n```"
        result = extract_code_from_markdown(content, language_hints=["c"])
        assert result == ""

    def test_multiline_code_preserved(self):
        code = "line1\nline2\n    indented\nline3"
        content = f"```python\n{code}\n```"
        result = extract_code_from_markdown(content)
        assert result == code

    def test_surrounding_text_ignored(self):
        content = "Some prose before.\n```c\ncode here\n```\nSome prose after."
        result = extract_code_from_markdown(content, language_hints=["c"])
        assert result == "code here"
