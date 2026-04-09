"""Tests for paper-reading tools."""

import pytest

from src.schemas.experimental import (
    FigureCaption,
    PageText,
    PaperDocument,
    ParsedTable,
    TableColumn,
    TableConditionValue,
    TableRow,
)
from src.agents.tools.paper_tools import (
    PaperDeps,
    get_abstract,
    get_figure_caption,
    get_section,
    get_table,
    list_figures,
    list_sections,
    list_tables,
    search_text,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_paper() -> PaperDocument:
    """A minimal PaperDocument with sections, figures, and tables."""
    pages = [
        PageText(page_num=1, text=(
            "Title: Shock Tube Study of H2/O2\n"
            "Abstract\n"
            "We measured ignition delay times for H2/O2 mixtures at "
            "temperatures of 1000-1400 K and pressures of 1-10 atm."
        )),
        PageText(page_num=2, text=(
            "2. Experimental Methods\n"
            "Experiments were conducted in a stainless steel shock tube "
            "with an internal diameter of 10 cm.\n"
            "The driven section was 3.0 m long.\n"
            "Species measured: H2O, OH, H2O2."
        )),
        PageText(page_num=3, text=(
            "3. Results and Discussion\n"
            "Ignition delay times are plotted in Figure 1.\n"
            "Table 1 summarises the experimental conditions.\n"
            "The H + O2 branching reaction dominates above 1200 K."
        )),
        PageText(page_num=4, text=(
            "4. Conclusions\n"
            "The present measurements extend the validation database.\n"
            "References\n"
            "[1] Smith et al. 2020"
        )),
    ]
    captions = [
        FigureCaption(
            figure_id="Figure 1",
            label_type="Figure",
            page_num=3,
            caption="Ignition delay times as a function of temperature for H2/O2 mixtures at 1 atm.",
        ),
        FigureCaption(
            figure_id="Figure 2",
            label_type="Figure",
            page_num=3,
            caption="Sensitivity analysis of H2/O2 at 1200 K and 1 atm.",
        ),
        FigureCaption(
            figure_id="Table 1",
            label_type="Table",
            page_num=3,
            caption="Summary of experimental conditions.",
        ),
    ]
    tables = [
        ParsedTable(
            table_id="Table 1",
            page_num=3,
            caption="Summary of experimental conditions.",
            columns=[
                TableColumn(header="T (K)", role="temperature", unit="K"),
                TableColumn(header="P (atm)", role="pressure", unit="atm"),
            ],
            rows=[
                TableRow(row_index=0, cells={
                    "T (K)": TableConditionValue(mode="single", raw_text="1000", values=[1000.0]),
                    "P (atm)": TableConditionValue(mode="single", raw_text="1", values=[1.0]),
                }),
                TableRow(row_index=1, cells={
                    "T (K)": TableConditionValue(mode="single", raw_text="1400", values=[1400.0]),
                    "P (atm)": TableConditionValue(mode="single", raw_text="10", values=[10.0]),
                }),
            ],
        ),
    ]
    return PaperDocument(
        pdf_path="/fake/paper.pdf",
        title="Shock Tube Study of H2/O2",
        abstract="We measured ignition delay times for H2/O2 mixtures at temperatures of 1000-1400 K and pressures of 1-10 atm.",
        pages=pages,
        captions=captions,
        tables=tables,
    )


@pytest.fixture
def deps(sample_paper: PaperDocument) -> PaperDeps:
    return PaperDeps(paper=sample_paper, model_species=["H2", "O2", "H2O", "OH"])


# ── get_abstract ─────────────────────────────────────────────────────────────

def test_get_abstract_returns_abstract(deps: PaperDeps):
    result = get_abstract(deps)
    assert "ignition delay times" in result
    assert "1000-1400 K" in result


def test_get_abstract_falls_back_to_page1():
    paper = PaperDocument(
        pdf_path="/fake.pdf",
        pages=[PageText(page_num=1, text="First page text here.")],
        captions=[],
    )
    deps = PaperDeps(paper=paper)
    assert get_abstract(deps) == "First page text here."


def test_get_abstract_empty_paper():
    paper = PaperDocument(pdf_path="/fake.pdf", pages=[], captions=[])
    deps = PaperDeps(paper=paper)
    assert "No abstract" in get_abstract(deps)


# ── get_section ──────────────────────────────────────────────────────────────

def test_get_section_exact_match(deps: PaperDeps):
    result = get_section(deps, "methods")
    assert "shock tube" in result


def test_get_section_partial_match(deps: PaperDeps):
    result = get_section(deps, "meth")
    assert "shock tube" in result


def test_get_section_case_insensitive(deps: PaperDeps):
    result = get_section(deps, "RESULTS")
    assert "Ignition delay" in result


def test_get_section_not_found(deps: PaperDeps):
    result = get_section(deps, "Nonexistent Section")
    assert "Section not found" in result


# ── search_text ──────────────────────────────────────────────────────────────

def test_search_text_single_term(deps: PaperDeps):
    result = search_text(deps, "shock tube")
    assert "Page" in result
    assert "shock tube" in result.lower()


def test_search_text_multi_term(deps: PaperDeps):
    result = search_text(deps, "H2O2")
    assert "Page 2" in result


def test_search_text_no_match(deps: PaperDeps):
    result = search_text(deps, "nonexistent_species_xyz")
    assert "No matches found" in result


def test_search_text_max_5_hits():
    pages = [
        PageText(page_num=i, text=f"Line with keyword match {i}")
        for i in range(1, 20)
    ]
    paper = PaperDocument(pdf_path="/f.pdf", pages=pages, captions=[])
    deps = PaperDeps(paper=paper)
    result = search_text(deps, "keyword")
    assert result.count("Page") == 5


# ── get_figure_caption ───────────────────────────────────────────────────────

def test_get_figure_caption_full_name(deps: PaperDeps):
    result = get_figure_caption(deps, "Figure 1")
    assert "Ignition delay times" in result


def test_get_figure_caption_abbreviated(deps: PaperDeps):
    result = get_figure_caption(deps, "Fig. 1")
    assert "Ignition delay times" in result


def test_get_figure_caption_number_only(deps: PaperDeps):
    result = get_figure_caption(deps, "1")
    assert "Ignition delay times" in result


def test_get_figure_caption_not_found(deps: PaperDeps):
    result = get_figure_caption(deps, "Figure 99")
    assert "not found" in result.lower()


# ── get_table ────────────────────────────────────────────────────────────────

def test_get_table_by_name(deps: PaperDeps):
    result = get_table(deps, "Table 1")
    assert "T (K)" in result
    assert "1000" in result
    assert "1400" in result


def test_get_table_by_number(deps: PaperDeps):
    result = get_table(deps, "1")
    assert "T (K)" in result


def test_get_table_not_found(deps: PaperDeps):
    result = get_table(deps, "99")
    assert "not found" in result.lower()


def test_get_table_falls_back_to_caption_and_page_text():
    """When table isn't in doc.tables but IS in captions, return caption + page text."""
    paper = PaperDocument(
        pdf_path="/f.pdf",
        pages=[
            PageText(page_num=2, text=(
                "Table 3 shows the rate constants used.\n"
                "H + O2 = OH + O  1.2e14  0  16800\n"
                "OH + H2 = H2O + H  1.0e8  1.6  3300"
            )),
        ],
        captions=[
            FigureCaption(
                figure_id="Table 3",
                label_type="Table",
                page_num=2,
                caption="Rate constants for key reactions.",
            ),
        ],
        tables=[],  # no parsed tables — keyword filter excluded it
    )
    deps = PaperDeps(paper=paper)
    result = get_table(deps, "Table 3")
    assert "Rate constants" in result
    assert "H + O2" in result
    assert "Page" not in result  # raw text, not "Table not found"


def test_get_table_prefers_parsed_over_caption(deps: PaperDeps):
    """When table exists in both doc.tables and captions, use structured data."""
    result = get_table(deps, "Table 1")
    # Structured data has pipe-delimited columns
    assert "T (K)" in result
    assert "|" in result


# ── list_figures ─────────────────────────────────────────────────────────────

def test_list_figures(deps: PaperDeps):
    result = list_figures(deps)
    assert "Figure 1" in result
    assert "Figure 2" in result
    # Should NOT include tables.
    assert "Table 1" not in result


def test_list_figures_empty():
    paper = PaperDocument(pdf_path="/f.pdf", pages=[], captions=[])
    deps = PaperDeps(paper=paper)
    assert "No figures" in list_figures(deps)


# ── list_tables ──────────────────────────────────────────────────────────────

def test_list_tables(deps: PaperDeps):
    result = list_tables(deps)
    assert "Table 1" in result
    assert "experimental conditions" in result.lower()


def test_list_tables_shows_all_captions():
    """list_tables always uses doc.captions, not the filtered doc.tables."""
    paper = PaperDocument(
        pdf_path="/f.pdf",
        pages=[],
        captions=[
            FigureCaption(
                figure_id="Table 1",
                label_type="Table",
                page_num=3,
                caption="Experimental conditions.",
            ),
            FigureCaption(
                figure_id="Table 2",
                label_type="Table",
                page_num=5,
                caption="Rate constants used.",
            ),
        ],
        tables=[],  # keyword filter excluded everything
    )
    deps = PaperDeps(paper=paper)
    result = list_tables(deps)
    assert "Table 1" in result
    assert "Table 2" in result


# ── list_sections ────────────────────────────────────────────────────────────

def test_list_sections(deps: PaperDeps):
    result = list_sections(deps)
    # The section detector should find at least some of the sections.
    assert len(result) > 0
    assert "pp." in result  # should contain page ranges


def test_list_sections_empty():
    paper = PaperDocument(pdf_path="/f.pdf", pages=[], captions=[])
    deps = PaperDeps(paper=paper)
    assert "No sections" in list_sections(deps)


# ── PaperDeps ────────────────────────────────────────────────────────────────

def test_paper_deps_model_species(deps: PaperDeps):
    assert deps.model_species == ["H2", "O2", "H2O", "OH"]


def test_paper_deps_sections_cached(deps: PaperDeps):
    s1 = deps.sections
    s2 = deps.sections
    assert s1 is s2


# ── Truncation tests ────────────────────────────────────────────────────────


def test_get_abstract_truncates_long_abstract():
    paper = PaperDocument(
        pdf_path="/f.pdf",
        abstract="A" * 2000,
        pages=[],
        captions=[],
    )
    deps = PaperDeps(paper=paper)
    result = get_abstract(deps)
    assert len(result) < 900  # 800 + truncation notice
    assert "truncated" in result


def test_get_section_truncates_long_section(deps: PaperDeps):
    # The fixture already has "2. Experimental Methods" on page 2.
    # Replace page 2 text with a long version.
    deps.paper.pages[1] = PageText(
        page_num=2, text="2. Experimental Methods\n" + "X" * 5000,
    )
    deps._sections = None  # reset cached sections
    result = get_section(deps, "methods")
    assert len(result) < 2100
    assert "truncated" in result


def test_search_text_truncates_long_passages():
    long_line = "keyword " + "Y" * 500
    pages = [PageText(page_num=1, text=long_line)]
    paper = PaperDocument(pdf_path="/f.pdf", pages=pages, captions=[])
    deps = PaperDeps(paper=paper)
    result = search_text(deps, "keyword")
    # Each passage capped at 200 chars
    passage = result.split("Page 1: ")[1]
    assert len(passage) <= 200


def test_get_table_truncates_large_table():
    rows = [
        TableRow(row_index=i, cells={
            "col": TableConditionValue(mode="single", raw_text="V" * 100, values=[float(i)]),
        })
        for i in range(50)
    ]
    paper = PaperDocument(
        pdf_path="/f.pdf", pages=[], captions=[],
        tables=[ParsedTable(
            table_id="Table 1", page_num=1, caption="Big table",
            columns=[TableColumn(header="col", role="value", unit="")],
            rows=rows,
        )],
    )
    deps = PaperDeps(paper=paper)
    result = get_table(deps, "1")
    assert len(result) < 1600
    assert "truncated" in result


def test_get_figure_caption_truncates_long_caption():
    paper = PaperDocument(
        pdf_path="/f.pdf", pages=[], tables=[],
        captions=[FigureCaption(
            figure_id="Figure 1", label_type="Figure", page_num=1,
            caption="C" * 500,
        )],
    )
    deps = PaperDeps(paper=paper)
    result = get_figure_caption(deps, "1")
    assert len(result) < 500
    assert "truncated" in result
