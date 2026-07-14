from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prototype_agent1_v4_gfm_normalization.py"
SCRIPTS_DIR = SCRIPT.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("agent1_v4_gfm_normalization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NORMALIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NORMALIZER
SPEC.loader.exec_module(NORMALIZER)

import build_agent1_v4_review_site as SITE  # noqa: E402


def normalize(text: str) -> tuple[str, dict[str, object]]:
    metrics: dict[str, object] = {}
    result = NORMALIZER.normalize_mixed_markup_to_gfm(text, metrics=metrics)
    return result, metrics


def test_preserves_existing_markdown_byte_for_byte_when_no_html_is_present() -> None:
    source = (
        "# Existing heading\n\n"
        "**bold** and *italic* and [link](https://example.org).\n\n"
        "| Existing | Table |\n| --- | ---: |\n| `x` | y \\| z |\n\n"
        "![existing image](asset.webp)\n"
    )

    normalized, metrics = normalize(source)

    assert normalized == source
    assert metrics["tag_counts"] == {}
    renderer, _ = NORMALIZER._markdown_renderer()
    assert NORMALIZER.markdown_token_counts(renderer, normalized) == NORMALIZER.markdown_token_counts(renderer, source)


def test_converts_complex_html_table_to_rectangular_gfm_without_duplicating_spans() -> None:
    source = """Before
<table border="1">
<caption>Measured values</caption>
<thead>
<tr><th rowspan="2">A</th><th colspan="2">B</th></tr>
<tr><th>C</th><th>D</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">x</td><td><b>y</b><br>z|q</td><td><i>w</i></td></tr>
<tr><td>m</td><td>n</td></tr>
</tbody>
</table>
After"""

    normalized, metrics = normalize(source)

    assert "<table" not in normalized
    assert "*Measured values*" in normalized
    assert "| A | B / C | D |" in normalized
    assert "| --- | --- | --- |" in normalized
    assert "| x | **y**; z\\|q | *w* |" in normalized
    assert "|  | m | n |" in normalized
    assert normalized.count("x") == 1
    transformations = metrics["transformations"]
    assert transformations["html_tables_to_gfm"] == 1
    assert transformations["rowspan_cells_expanded"] == 2
    assert transformations["colspan_cells_expanded"] == 1
    assert transformations["multirow_headers_merged"] == 1


def test_table_without_header_uses_empty_header_and_keeps_every_data_row() -> None:
    normalized, metrics = normalize("<table><tr><td>one</td><td>two</td></tr></table>")

    assert "|  |  |" in normalized
    assert "| --- | --- |" in normalized
    assert "| one | two |" in normalized
    assert metrics["transformations"]["synthetic_empty_table_headers"] == 1


def test_flattens_nested_table_with_explicit_row_and_cell_boundaries() -> None:
    source = (
        "<table><tr><th>Outer</th></tr><tr><td>before "
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        " after</td></tr></table>"
    )

    normalized, metrics = normalize(source)

    assert "before A / B; C / D after" in normalized
    assert metrics["transformations"]["html_tables_to_gfm"] == 1
    assert metrics["transformations"]["nested_tables_flattened"] == 1
    assert metrics["transformations"]["nested_table_cells_flattened"] == 4


def test_retains_malformed_orphan_table_cell_content_inline() -> None:
    normalized, metrics = normalize("before <td><b>orphan</b></td> after")

    assert normalized == "before **orphan** after"
    assert metrics["transformations"]["orphan_table_cells_flattened"] == 1


def test_flattens_table_lists_and_converts_math_bold_and_italics() -> None:
    source = (
        "<table><tr><th>Items</th><th>Formula</th></tr>"
        "<tr><td><ul><li><b>first</b></li><li><i>second</i></li></ul></td>"
        "<td><math>F=ma</math></td></tr></table>"
    )

    normalized, _ = normalize(source)

    assert "| • **first**; • *second* | $F=ma$ |" in normalized


def test_removes_unexpressible_elements_but_retains_textual_content_for_flattened_styles() -> None:
    source = (
        "A<sup>2</sup> B<sub>n</sub> <u>underlined</u> "
        '<span style="color:red">red text</span> '
        '<img alt="kept" src="asset.png"> <img alt="decorative"> '
        '<input type="checkbox"> <script>danger()</script>'
    )

    normalized, metrics = normalize(source)

    assert "A2 Bn underlined red text" in normalized
    assert "![kept](asset.png)" in normalized
    assert "decorative" not in normalized
    assert "checkbox" not in normalized
    assert "danger" not in normalized
    assert "<sup>" not in normalized
    assert metrics["transformations"]["source_less_images_removed"] == 1
    assert metrics["transformations"]["inline_inputs_removed"] == 1
    assert metrics["transformations"]["unsupported_elements_removed_with_content"] == 1


def test_flattened_superscript_does_not_break_adjacent_existing_markdown_emphasis() -> None:
    source = "*Title*<sup>11</sup> and <sub>2</sub>*Word*"

    normalized, metrics = normalize(source)
    renderer, _ = NORMALIZER._markdown_renderer()

    assert normalized == "*Title* 11 and 2*Word*"
    assert NORMALIZER.markdown_token_counts(renderer, normalized)["em_open"] == 2
    assert metrics["transformations"]["flattened_style_markdown_boundaries_spaced"] == 1


def test_preserves_gfm_autolinks_and_escapes_non_html_angle_text() -> None:
    source = "<https://example.org/a> <Forbes Diamonds, 2013> <οιδατε οτι> <<εταιρία>>"

    normalized, metrics = normalize(source)

    assert normalized.startswith("<https://example.org/a>")
    assert "&lt;Forbes Diamonds, 2013&gt;" in normalized
    assert "&lt;οιδατε οτι&gt;" in normalized
    assert "&lt;&lt;εταιρία&gt;&gt;" in normalized
    assert metrics["pseudo_tags_escaped"] == 4


def test_keeps_only_explicit_removal_comments() -> None:
    source = "before<!-- Table content goes here -->middle<!-- repeating-text-removed -->after"

    normalized, metrics = normalize(source)

    assert normalized == "beforemiddle<!-- repeating-text-removed -->after"
    assert metrics["comments_removed"] == 1
    assert metrics["comments_preserved"] == 1


def test_relocates_approved_comment_after_gfm_table_instead_of_losing_it() -> None:
    source = (
        "<table><tr><td>one</td><td>two</td></tr>"
        "<!-- repeating-text-removed --></table>"
    )

    normalized, metrics = normalize(source)

    assert normalized.count("<!-- repeating-text-removed -->") == 1
    assert normalized.rstrip().endswith("<!-- repeating-text-removed -->")
    assert metrics["comments_preserved"] == 1
    assert metrics["transformations"]["table_comments_relocated_after_table"] == 1


def test_normalization_presentation_is_lazy_text_safe_and_sandboxed() -> None:
    html = SITE._normalization_html()
    javascript = SITE._normalization_js()

    assert "data/gfm_normalization_audit.json" in javascript
    assert "data/documents/" in javascript
    assert "data/gfm/documents/" in javascript
    assert "Promise.all" in javascript
    assert "synchronizeTextScroll" in javascript
    assert "source.scrollTop/sourceRange" in javascript
    assert "requestAnimationFrame" in javascript
    assert "passive:true" in javascript
    assert "textContent" in javascript
    assert ".innerHTML" not in javascript
    assert "setAttribute('sandbox','')" in javascript
    assert "img-src \\'none\\'" in javascript
    assert "normalized Markdown source" not in html
    assert "raw documents are untouched" in html
    assert "&lt;!-- repeating-text-removed --&gt;" in html
