"""Turn Mintlify MDX into plain Markdown suitable for splitting and embedding.

Code fences and inline code are kept verbatim. Outside code: frontmatter is parsed
into metadata, MDX imports/exports and comments are dropped, Mintlify components are
unwrapped (keeping inner text and useful attributes such as `title`), and HTML
(including TypeDoc output in `schema.mdx`) is reduced to text.
"""

import html
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)((?:(?!\n[ \t]*\n).)+?)(?<!`)\1(?!`)", re.DOTALL)

_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_EXPORT_BLOCK = re.compile(r"^export\s+const\s+\w+\s*=.*?^\);?[ \t]*$", re.DOTALL | re.MULTILINE)
_IMPORT_LINE = re.compile(r"^import\s+(?:.+\s+from\s+)?['\"][^'\"]+['\"];?[ \t]*$", re.MULTILINE)
_SVG = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

_TSD_SIGNATURE = re.compile(r'<div class="tsd-signature">(.*?)</div>', re.DOTALL)
_TSD_ANCHOR = re.compile(
    r'<div class="tsd-anchor-link"[^>]*>\s*<span[^>]*>(.*?)</span>.*?</div>', re.DOTALL
)

# Attribute values: "double", 'single' or {expression} (one level of nested braces).
_ATTRS = r"""(?:\s+(?:[^\s"'{}<>/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|\{(?:[^{}]|\{[^{}]*\})*\}))?))*"""
_COMPONENT = re.compile(rf"<(/?)([A-Z][\w.]*)({_ATTRS})\s*(/?)>")
_ATTR_VALUE = re.compile(r"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""")

_CALLOUTS = {"Note", "Warning", "Tip", "Info", "Check"}
_INLINE_REPLACEMENTS = {"CHECK": "Yes"}

_HTML_TAGS = (
    "a|abbr|aside|b|blockquote|br|button|center|code|dd|details|div|dl|dt|em|figcaption|"
    "figure|h[1-6]|hr|i|img|kbd|li|ol|p|pre|section|small|span|strong|sub|summary|sup|"
    "table|tbody|td|th|thead|tr|u|ul|use|video|source|iframe"
)
_HTML_TAG = re.compile(rf"<(/?)({_HTML_TAGS})\b({_ATTRS})\s*/?>", re.IGNORECASE)
_BLOCK_TAGS = {
    "aside", "blockquote", "center", "dd", "details", "div", "dl", "dt", "figcaption",
    "figure", "hr", "li", "ol", "p", "pre", "section", "summary", "table", "tr", "ul",
}  # fmt: skip

_BLANK_RUNS = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass
class CleanedMdx:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the body. Invalid or missing frontmatter yields `{}`."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        data = None
    return (data if isinstance(data, dict) else {}), text[match.end() :]


def _closes(fence: str, marker: str, line: str) -> bool:
    """A closing fence uses the same character, is at least as long, and has no info string."""
    return marker[0] == fence[0] and len(marker) >= len(fence) and not line.strip()[len(marker) :]


def split_code_fences(text: str) -> list[tuple[bool, str]]:
    """Split into `(is_code, chunk)` segments. An unclosed fence runs to the end."""
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: str | None = None

    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence is None and match:
            if buffer:
                segments.append((False, "".join(buffer)))
            buffer, fence = [line], match.group(1)
            continue
        buffer.append(line)
        if fence is not None and match and _closes(fence, match.group(1), line):
            segments.append((True, "".join(buffer)))
            buffer, fence = [], None

    if buffer:
        segments.append((fence is not None, "".join(buffer)))
    return segments


def _attr(attrs: str, name: str) -> str | None:
    match = re.search(_ATTR_VALUE.pattern.format(name=name), attrs)
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _replace_component(match: re.Match[str]) -> str:
    closing, name, attrs, self_closing = match.groups()
    if name in _INLINE_REPLACEMENTS:
        return _INLINE_REPLACEMENTS[name]
    if closing:
        if name == "Tooltip":
            return "\0ENDTIP\0"
        return "\n" if name in _CALLOUTS else ""
    if name in _CALLOUTS:
        return f"\n**{name}:** "

    label = _attr(attrs, "title") or _attr(attrs, "name")
    if name == "Tooltip":
        tip = _attr(attrs, "tip")
        return "" if self_closing else ("\0TIP\0" + tip + "\0" if tip else "")
    if label:
        return f"\n**{label}**\n"
    return ""


def _close_tooltips(text: str) -> str:
    """`<Tooltip tip="x">term</Tooltip>` becomes `term (x)`."""
    text = re.sub(r"\0TIP\0(.*?)\0(.*?)\0ENDTIP\0", r"\2 (\1)", text, flags=re.DOTALL)
    return text.replace("\0ENDTIP\0", "")


def _replace_html(match: re.Match[str]) -> str:
    closing, tag, _attrs = match.groups()
    tag = tag.lower()
    if tag == "br":
        return "\n"
    if tag == "code":
        return "`"
    if tag in {"b", "strong"}:
        return "**"
    if tag == "li" and not closing:
        return "\n- "
    if tag in _BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
        return "\n"
    return ""


def _html_to_text(fragment: str) -> str:
    text = _HTML_TAG.sub(_replace_html, fragment)
    return html.unescape(text).replace("\xa0", " ")


def _typedoc_signature(match: re.Match[str]) -> str:
    code = _html_to_text(match.group(1)).strip("\n")
    return f"\n```typescript\n{code}\n```\n"


def _typedoc_member(match: re.Match[str]) -> str:
    return f"\n\n**{_html_to_text(match.group(1)).strip()}**\n\n"


def clean_prose(text: str) -> str:
    """Clean a non-code segment of MDX. Inline code spans are left untouched."""
    protected: list[str] = []

    def protect(value: str) -> str:
        protected.append(value)
        return f"\0P{len(protected) - 1}\0"

    # TypeDoc signatures become fenced code before inline code is protected.
    text = _TSD_SIGNATURE.sub(lambda m: protect(_typedoc_signature(m)), text)
    text = _INLINE_CODE.sub(lambda m: protect(m.group(0)), text)

    text = _MDX_COMMENT.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = _EXPORT_BLOCK.sub("", text)
    text = _IMPORT_LINE.sub("", text)
    text = _SVG.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _TSD_ANCHOR.sub(_typedoc_member, text)

    text = _COMPONENT.sub(_replace_component, text)
    text = _close_tooltips(text)
    text = _html_to_text(text)

    return re.sub(r"\0P(\d+)\0", lambda m: protected[int(m.group(1))], text)


def _normalise(text: str) -> str:
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip() + "\n"


def clean_mdx(text: str) -> CleanedMdx:
    """Parse frontmatter and clean the body. Code fences pass through unchanged."""
    metadata, body = parse_frontmatter(text)
    parts = [chunk if is_code else clean_prose(chunk) for is_code, chunk in split_code_fences(body)]
    return CleanedMdx(content=_normalise("".join(parts)), metadata=metadata)
