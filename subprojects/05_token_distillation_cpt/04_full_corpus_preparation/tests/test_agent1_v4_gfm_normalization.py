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
    assert "| A | B |  |" in normalized
    assert "| --- | --- | --- |" in normalized
    assert "|  | **C** | **D** |" in normalized
    assert "| x | **y** z\\|q | *w* |" in normalized
    assert "|  | m | n |" in normalized
    assert normalized.count("x") == 1
    transformations = metrics["transformations"]
    assert transformations["html_tables_to_gfm"] == 1
    assert transformations["rowspan_cells_expanded"] == 2
    assert transformations["colspan_cells_expanded"] == 1
    assert transformations["additional_header_rows_preserved"] == 1


def test_table_without_header_uses_empty_header_and_keeps_every_data_row() -> None:
    normalized, metrics = normalize("<table><tr><td>one</td><td>two</td></tr></table>")

    assert "|  |  |" in normalized
    assert "| --- | --- |" in normalized
    assert "| one | two |" in normalized
    assert metrics["transformations"]["synthetic_empty_table_headers"] == 1


def test_ragged_but_ordered_rows_are_padded_to_valid_gfm() -> None:
    source = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"

    normalized, metrics = normalize(source)

    assert "| A | B |" in normalized
    assert "| C |  |" in normalized
    assert metrics["transformations"]["html_tables_to_gfm"] == 1
    assert "html_tables_fallback_to_text" not in metrics["transformations"]


def test_downgrades_nested_table_to_readable_cell_lines() -> None:
    source = (
        "<table><tr><th>Outer</th></tr><tr><td>before "
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        " after</td></tr></table>"
    )

    normalized, metrics = normalize(source)

    assert "Outer\n\nbefore A\nB\n\nC\nD after" in normalized
    assert "| ---" not in normalized
    assert metrics["transformations"]["html_tables_fallback_to_text"] == 2
    assert metrics["transformations"]["html_table_fallback_cells_preserved"] == 6
    assert metrics["transformations"]["table_fallback_reason_nested_table"] == 1
    event = metrics["table_fallback_events"][0]
    assert event["source_line"] == 1
    assert event["source_column"] == 0


def test_table_fallback_preserves_breaks_as_readable_lines() -> None:
    source = (
        '<table><tr><td rowspan="3">Family</td><td>Apple one<br>Apple two</td></tr>'
        "<tr><td>last</td></tr></table>"
    )

    normalized, metrics = normalize(source)

    assert "Apple one\nApple two" in normalized
    assert "Apple oneApple two" not in normalized
    assert metrics["transformations"]["table_fallback_reason_rowspan_outside_rows"] == 1


def test_retains_malformed_orphan_table_cell_content_inline() -> None:
    normalized, metrics = normalize("before <td><b>orphan</b></td> after")

    assert normalized == "before \n\n**orphan**\n\n after"
    assert metrics["transformations"]["orphan_table_cells_flattened"] == 1


def test_flattens_table_lists_and_converts_math_bold_and_italics() -> None:
    source = (
        "<table><tr><th>Items</th><th>Formula</th></tr>"
        "<tr><td><ul><li><b>first</b></li><li><i>second</i></li></ul></td>"
        "<td><math>F=ma</math></td></tr></table>"
    )

    normalized, _ = normalize(source)

    assert "| **first** *second* | $F=ma$ |" in normalized


def test_generated_image_artifacts_become_provenance_comments_without_losing_descriptions() -> None:
    digest = "9511aaaa0e3447cf006946fe2f6e6aa2"
    source = (
        f"![Readable alt]({digest}_3_img.webp) "
        f"[linked label](assets/{digest}_4_img.png) "
        f"({digest}_5_img.webp) {digest}_6_img.webp "
        f'<img src="{digest}_7_img.webp" alt="HTML alt">'
    )
    metrics: dict[str, object] = {}

    cleaned = NORMALIZER.clean_generated_image_artifacts(source, metrics=metrics)

    assert cleaned == (
        "<!-- description-of-removed-image: Readable alt --> linked label   "
        "<!-- description-of-removed-image: HTML alt -->"
    )
    assert digest not in cleaned
    assert metrics["generated_image_artifact_count"] == 5
    assert metrics["generated_image_rule_counts"] == {
        "bare_generated_image_target_removed": 1,
        "html_generated_image_to_description_comment": 1,
        "markdown_generated_image_link_to_label": 1,
        "markdown_generated_image_to_description_comment": 1,
        "parenthesized_generated_image_target_removed": 1,
    }
    assert metrics["image_description_elements_commented"] == 2
    assert metrics["image_description_comments_emitted"] == 2
    assert metrics["image_description_characters_preserved"] == len("Readable altHTML alt")
    assert metrics["empty_image_description_comments"] == 0


def test_image_description_comments_are_safe_empty_and_never_nested() -> None:
    digest = "9511aaaa0e3447cf006946fe2f6e6aa2"
    source = (
        f"![]({digest}_1_img.webp) "
        f"![before <!-- repeating-text-removed --> after -- value <x>]({digest}_2_img.webp)"
    )
    metrics: dict[str, object] = {}

    cleaned = NORMALIZER.clean_generated_image_artifacts(source, metrics=metrics)
    normalized, markup = normalize(cleaned)

    assert cleaned.startswith("<!-- description-of-removed-image --> ")
    assert "<!-- description-of-removed-image: before -->" in cleaned
    assert "<!-- repeating-text-removed -->" in cleaned
    assert "<!-- description-of-removed-image: after &#45;&#45; value &lt;x&gt; -->" in cleaned
    assert "<!-- description-of-removed-image: before <!--" not in cleaned
    assert normalized == cleaned
    assert markup["comments_preserved"] == 4
    assert metrics["image_description_elements_commented"] == 2
    assert metrics["image_description_comments_emitted"] == 3
    assert metrics["empty_image_description_comments"] == 1
    assert metrics["image_descriptions_split_around_pipeline_markers"] == 1
    assert metrics["pipeline_markers_preserved_inside_image_descriptions"] == 1


def test_follow_up_repetition_expands_partial_cut_to_whole_description_comment() -> None:
    source = "prefix <!-- description-of-removed-image: repeated repeated repeated --> suffix"

    def detector(value: str, *, metrics: dict[str, object]) -> str:
        start = value.find("repeated")
        if start < 0:
            metrics.update(
                {
                    "complex_repetition_replacement_details": [],
                    "complex_repetition_preserved_findings": [],
                }
            )
            return value
        end = value.rfind("repeated") + len("repeated")
        metrics.update(
            {
                "complex_repetition_replacement_details": [
                    {
                        "pass_index": 1,
                        "spans": [
                            {
                                "start_index": start,
                                "end_index": end,
                                "removed_char_count": end - start,
                                "rules": ["repeated_token_block"],
                                "findings": [],
                            }
                        ],
                    }
                ],
                "complex_repetition_preserved_findings": [],
            }
        )
        return value

    cleaned, metrics = NORMALIZER._replace_repetitions_comment_safe(source, detector)

    assert cleaned == "prefix <!-- repeating-text-removed --> suffix"
    span = metrics["complex_repetition_replacement_details"][0]["spans"][0]
    assert span["start_index"] == source.index("<!-- description-of-removed-image")
    assert span["end_index"] == source.index("-->") + 3
    assert span["comment_boundary_expansion"]["original_start_index"] == source.index("repeated")


def test_non_artifact_markdown_and_html_images_remain_expressible() -> None:
    source = "![existing](asset.webp) <img src=photo.png alt=photo>"
    cleaned = NORMALIZER.clean_generated_image_artifacts(source)
    normalized, _ = normalize(cleaned)

    assert normalized == "![existing](asset.webp) ![photo](photo.png)"


def test_source_less_html_image_description_becomes_comment_but_empty_image_is_removed() -> None:
    source = '<img alt="Solid blue line"> <img alt="">'
    metrics: dict[str, object] = {}

    cleaned = NORMALIZER.clean_generated_image_artifacts(source, metrics=metrics)
    normalized, _ = normalize(cleaned)

    assert cleaned == '<!-- description-of-removed-image: Solid blue line --> <img alt="">'
    assert normalized == "<!-- description-of-removed-image: Solid blue line --> "
    assert metrics["generated_image_rule_counts"] == {
        "source_less_html_image_to_description_comment": 1
    }


def test_repetition_damaged_table_falls_back_without_losing_marker() -> None:
    source = (
        "<table><tr><td>one</td><td>two</td></tr>"
        "<tr><!-- repeating-text-removed --></tr></table>"
    )

    normalized, metrics = normalize(source)

    assert "one\ntwo" in normalized
    assert normalized.count("<!-- repeating-text-removed -->") == 1
    assert "| ---" not in normalized
    assert metrics["transformations"]["table_fallback_reason_row_without_cells"] == 1


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
    assert "<!-- description-of-removed-image: decorative -->" in normalized
    assert "checkbox" not in normalized
    assert "danger" not in normalized
    assert "<sup>" not in normalized
    assert metrics["transformations"]["source_less_images_to_description_comment"] == 1
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


def test_unclosed_pseudo_tag_cannot_swallow_later_provenance_comment() -> None:
    source = (
        "Proletarius (<proles-is = descendant): readable etymology.\n\n"
        "<!-- description-of-removed-image: A decorative flourish. -->\n\n"
        "Later footnote<sup>25</sup>."
    )

    normalized, metrics = normalize(source)

    assert "Proletarius (&lt;proles-is = descendant)" in normalized
    assert normalized.count("<!-- description-of-removed-image: A decorative flourish. -->") == 1
    assert "Later footnote25." in normalized
    assert metrics["comments_preserved"] == 1


def test_stale_generated_normalized_documents_are_pruned(tmp_path: Path) -> None:
    output = tmp_path / NORMALIZER.OUTPUT_DOCUMENT_DIR
    output.mkdir(parents=True)
    keep = output / "keep.json"
    stale = output / "stale.json"
    keep.write_text("keep")
    stale.write_text("stale")

    removed = NORMALIZER._remove_stale_normalized_documents(
        tmp_path, [keep.relative_to(tmp_path).as_posix()]
    )

    assert removed == 1
    assert keep.read_text() == "keep"
    assert not stale.exists()


def test_preserves_bare_ampersands_without_inventing_entity_semicolons() -> None:
    source = "<table><tr><td>T.&T. Clark &amp; Co.</td></tr></table>"

    normalized, _ = normalize(source)

    assert "T.&T. Clark &amp; Co." in normalized
    assert "T.&T.;" not in normalized


def test_repairs_ragged_existing_gfm_table_by_padding_only() -> None:
    source = (
        "| Important dates |\n"
        "| --- |\n"
        "| 10 October | First announcement |\n"
        "| 15 February | Submission |\n"
    )

    normalized, metrics = normalize(source)

    assert normalized == (
        "| Important dates |  |\n"
        "| --- | --- |\n"
        "| 10 October | First announcement |\n"
        "| 15 February | Submission |\n"
    )
    assert metrics["transformations"]["existing_gfm_tables_repaired"] == 1
    assert metrics["transformations"]["existing_gfm_table_cells_padded"] == 2


def test_repairs_repetition_marker_as_rectangular_gfm_header() -> None:
    source = (
        "| <!-- repeating-text-removed -->\n"
        "| --- | --- | --- |\n"
        "| first | second | third |\n"
    )

    normalized, metrics = normalize(source)

    assert normalized.startswith(
        "| <!-- repeating-text-removed --> |  |  |\n| --- | --- | --- |"
    )
    assert metrics["transformations"]["repetition_marker_table_headers_repaired"] == 1
    assert metrics["transformations"]["existing_gfm_table_cells_padded"] == 2


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
    assert "synchronizeTextScrollGroup" in javascript
    assert "After deterministic artifact cleaning" in javascript
    assert "normalization-image-descriptions" in html
    assert "imageDescriptionFilter" in javascript
    assert "description-of-removed-image" in html
    assert "data/gfm_luna_validation.json" in javascript
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
