from textwrap import dedent

import pytest

from ingest.clean_mdx import clean_mdx, clean_prose, parse_frontmatter, split_code_fences


def md(text: str) -> str:
    return dedent(text).lstrip("\n")


class TestFrontmatter:
    def test_parses_title_and_description(self) -> None:
        text = '---\ntitle: Lifecycle\ndescription: "How it starts"\n---\nBody'

        meta, body = parse_frontmatter(text)

        assert meta == {"title": "Lifecycle", "description": "How it starts"}
        assert body == "Body"

    def test_missing_frontmatter_returns_text_unchanged(self) -> None:
        assert parse_frontmatter("# Heading\n") == ({}, "# Heading\n")

    @pytest.mark.parametrize("raw", ["title: [unclosed", "- a list"])
    def test_invalid_or_non_mapping_frontmatter_is_ignored(self, raw: str) -> None:
        meta, body = parse_frontmatter(f"---\n{raw}\n---\nBody")

        assert meta == {}
        assert body == "Body"


class TestCodeFences:
    def test_splits_prose_and_code(self) -> None:
        segments = split_code_fences("intro\n```python\nimport os\n```\noutro\n")

        assert segments == [
            (False, "intro\n"),
            (True, "```python\nimport os\n```\n"),
            (False, "outro\n"),
        ]

    def test_indented_and_tilde_fences(self) -> None:
        text = "  ~~~bash\n  echo hi\n  ~~~\n"

        assert split_code_fences(text) == [(True, text)]

    def test_shorter_or_different_marker_does_not_close(self) -> None:
        text = "````md\n```js\nx\n```\n~~~~\n````\n"

        assert split_code_fences(text) == [(True, text)]

    def test_marker_with_info_string_does_not_close(self) -> None:
        text = "```\na\n```python\nb\n```\n"

        assert split_code_fences(text) == [(True, text)]

    def test_unclosed_fence_runs_to_end(self) -> None:
        assert split_code_fences("a\n```\ncode") == [(False, "a\n"), (True, "```\ncode")]


class TestProse:
    def test_drops_mdx_comments_exports_and_imports(self) -> None:
        text = md("""
            {/* prettier-ignore-start */}
            import { Foo } from "./foo";
            import "./styles.css"

            export const CHECK = () => (
              <span className="flex">
                <Icon icon="check" size={18} />
              </span>
            );
            <!-- hidden -->
            Kept text.
        """)

        assert clean_prose(text).strip() == "Kept text."

    @pytest.mark.parametrize("name", ["Note", "Warning", "Tip", "Info"])
    def test_callouts_keep_text_with_a_label(self, name: str) -> None:
        cleaned = clean_prose(f"<{name}>\n\nBe careful.\n\n</{name}>")

        assert f"**{name}:**" in cleaned
        assert "Be careful." in cleaned
        assert "<" not in cleaned

    def test_titled_components_become_bold_labels(self) -> None:
        text = md("""
            <Steps>
            <Step title="Install the server">
            Run the installer.
            </Step>
            </Steps>
            <Card title="Example servers" icon="grid" href="/examples">
            Browse them.
            </Card>
            <Tree.Folder name='src'></Tree.Folder>
        """)

        cleaned = clean_prose(text)

        assert "**Install the server**" in cleaned
        assert "**Example servers**" in cleaned
        assert "**src**" in cleaned
        assert "Run the installer." in cleaned
        assert "<" not in cleaned

    def test_self_closing_and_expression_attributes_are_removed(self) -> None:
        text = '<Frame style={{ textAlign: "center" }}>\nText\n</Frame>\n<Icon size={24} />'

        cleaned = clean_prose(text)

        assert cleaned.strip() == "Text"

    def test_check_component_becomes_yes(self) -> None:
        assert clean_prose("| Claude | <CHECK /> |") == "| Claude | Yes |"

    def test_badge_keeps_inner_text_and_entities(self) -> None:
        assert clean_prose('<Badge color="blue">Tier&nbsp;1</Badge>') == "Tier 1"

    def test_tooltip_appends_tip_in_brackets(self) -> None:
        cleaned = clean_prose('A <Tooltip tip="no shared state">stateless protocol</Tooltip>.')

        assert cleaned == "A stateless protocol (no shared state)."

    def test_tooltip_without_tip_keeps_term_only(self) -> None:
        assert clean_prose("A <Tooltip>term</Tooltip> and <Tooltip />.") == "A term and ."

    def test_html_is_reduced_to_text(self) -> None:
        text = (
            '<div class="x"><p>Use <code>tools/list</code> and <b>never</b> skip.<br/>Next'
            "</p><ul><li>One</li></ul><svg viewBox='0 0 1 1'><use href='#a'></use></svg>"
            "<kbd>Ctrl</kbd> &amp; &lt;done&gt;</div>"
        )

        cleaned = clean_prose(text)

        assert "Use `tools/list` and **never** skip.\nNext" in cleaned
        assert "\n- One" in cleaned
        assert "Ctrl & <done>" in cleaned
        assert "svg" not in cleaned

    def test_images_are_dropped(self) -> None:
        text = '![diagram](/images/a.png)\n<img src="/b.png" alt="b" />\nText'

        assert clean_prose(text).strip() == "Text"

    def test_inline_code_is_left_untouched(self) -> None:
        text = "Send `<protocol-version>` and `<Note>` and `&amp;`, then\n`<Tab\nwrapped>`."

        assert clean_prose(text) == text

    def test_unknown_lowercase_tags_are_kept(self) -> None:
        text = "Use the <protocol-version> header."

        assert clean_prose(text) == text

    def test_typedoc_signature_and_members(self) -> None:
        text = (
            '<div class="tsd-signature"><span class="tsd-signature-keyword">interface</span> '
            "<span>Error</span> <span>&#x7B;</span><br/>&nbsp;&nbsp;<a href='#c'>code</a>"
            "<span>:</span> <span>number</span><span>;</span><br/><span>}</span></div> "
            '<div class="tsd-comment"><p>An error.</p></div>'
            '<div class="tsd-anchor-link" id="error-code"><span>code: number</span>'
            '<a href="#error-code"><svg><use href="#icon"></use></svg></a></div>'
        )

        cleaned = clean_prose(text)

        assert "```typescript\ninterface Error {\n  code: number;\n}\n```" in cleaned
        assert "An error." in cleaned
        assert "**code: number**" in cleaned


def test_clean_mdx_end_to_end() -> None:
    text = md("""
        ---
        title: Tools
        ---

        <Note>
        Tools are model-controlled.
        </Note>



        ```typescript
        import { Client } from "@modelcontextprotocol/sdk";
        <Note>not a component in code</Note>
        ```
        Trailing   \n""")

    result = clean_mdx(text)

    assert result.metadata == {"title": "Tools"}
    assert '```typescript\nimport { Client } from "@modelcontextprotocol/sdk";' in result.content
    assert "<Note>not a component in code</Note>" in result.content
    assert "**Note:**\nTools are model-controlled." in result.content
    assert "\n\n\n" not in result.content
    assert result.content.endswith("Trailing\n")
