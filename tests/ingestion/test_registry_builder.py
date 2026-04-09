"""Tests for RegistryBuilder — all HTTP clients mocked."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.schemas.ingestion import PaperRecord, RegistryResult, SearchQuery


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_record(**overrides) -> PaperRecord:
    defaults = dict(
        id="test-id-1",
        title="Shock tube study of H2/O2",
        doi="10.1234/test.2024",
        abstract="We measured ignition delay times.",
        provenance="openalex",
        oa_status="oa",
    )
    defaults.update(overrides)
    return PaperRecord(**defaults)


def _mock_openalex_item() -> dict:
    """Minimal raw OpenAlex work dict."""
    return {
        "id": "https://openalex.org/W12345",
        "title": "Shock tube study of H2/O2",
        "authorships": [{"author": {"display_name": "A. Test"}}],
        "publication_year": 2024,
        "doi": "https://doi.org/10.1234/test.2024",
        "primary_location": {"source": {"display_name": "Combustion J"}},
        "open_access": {"is_oa": True, "oa_url": "https://example.com/paper.pdf"},
        "abstract_inverted_index": {},
        "referenced_works": [],
        "cited_by_count": 5,
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_clients():
    """Patch all HTTP clients so no real network calls happen."""
    record = _make_record()

    with (
        patch("src.ingestion.registry_builder.OpenAlexClient") as MockOA,
        patch("src.ingestion.registry_builder.CrossrefClient"),
        patch("src.ingestion.registry_builder.EuropePMCClient") as MockEPMC,
        patch("src.ingestion.registry_builder.SemanticScholarClient"),
    ):
        oa_instance = MockOA.return_value
        oa_instance.search.return_value = [_mock_openalex_item()]
        oa_instance.normalize.return_value = record
        oa_instance.get_citations.return_value = []
        oa_instance.get_works.return_value = []

        epmc_instance = MockEPMC.return_value
        epmc_instance.get_by_doi.return_value = None
        epmc_instance.get_by_title.return_value = None

        yield {
            "openalex": oa_instance,
            "epmc": epmc_instance,
            "record": record,
        }


# ── Core tests ───────────────────────────────────────────────────────────────

def test_build_registry_returns_registry_result(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder()
    result = builder.build_registry("H2/O2 ignition delay")

    assert isinstance(result, RegistryResult)
    assert len(result.papers) >= 1
    assert result.papers[0].title == "Shock tube study of H2/O2"


def test_build_registry_with_search_query(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder()
    sq = SearchQuery(topic="H2 kinetics", max_results=5)
    result = builder.build_registry(sq)

    assert isinstance(result, RegistryResult)
    mock_clients["openalex"].search.assert_called_once()


def test_build_registry_with_list_of_queries(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder()
    result = builder.build_registry(["H2 ignition", "O2 kinetics"])

    assert isinstance(result, RegistryResult)
    # Should search for each topic.
    assert mock_clients["openalex"].search.call_count == 2


# ── Snowball ─────────────────────────────────────────────────────────────────

def test_snowball_triggers_citation_search(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    # Make the record high-priority with a provenance containing 'openalex'.
    record = mock_clients["record"]
    record.match_reason = "relevant"
    record.provenance = "openalex"
    record.id = "https://openalex.org/W12345"
    record.priority = "high"
    record.references = ["W99999"]

    builder = RegistryBuilder()
    builder.build_registry("H2 kinetics", snowball=True)

    mock_clients["openalex"].get_citations.assert_called_once()
    mock_clients["openalex"].get_works.assert_called_once()


def test_snowball_false_skips_citation_search(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder()
    builder.build_registry("H2 kinetics", snowball=False)

    mock_clients["openalex"].get_citations.assert_not_called()
    mock_clients["openalex"].get_works.assert_not_called()


# ── Download SI ──────────────────────────────────────────────────────────────

def test_download_si_triggers_download(mock_clients, tmp_path):
    from src.ingestion.registry_builder import RegistryBuilder

    record = mock_clients["record"]
    record.si_link_found = True
    record.si_links = ["https://example.com/si.pdf"]

    with patch("src.ingestion.registry_builder.download_file") as mock_dl:
        mock_dl.return_value = tmp_path / "test_si.pdf"
        builder = RegistryBuilder()
        builder.build_registry("H2 kinetics", download_si=True, outdir=str(tmp_path))

    mock_dl.assert_called_once()


def test_download_si_false_skips_download(mock_clients, tmp_path):
    from src.ingestion.registry_builder import RegistryBuilder

    record = mock_clients["record"]
    record.si_link_found = True
    record.si_links = ["https://example.com/si.pdf"]

    with patch("src.ingestion.registry_builder.download_file") as mock_dl:
        builder = RegistryBuilder()
        builder.build_registry("H2 kinetics", download_si=False, outdir=str(tmp_path))

    mock_dl.assert_not_called()


# ── CONTACT_EMAIL ────────────────────────────────────────────────────────────

def test_contact_email_from_env(mock_clients):
    with patch.dict(os.environ, {"CONTACT_EMAIL": "test@example.org"}):
        # Re-import to pick up the env var.
        import importlib
        import src.ingestion.registry_builder as rb_mod
        importlib.reload(rb_mod)

        assert rb_mod._DEFAULT_EMAIL == "test@example.org"

    # Reload again to restore default.
    importlib.reload(rb_mod)


def test_contact_email_default_fallback(mock_clients):
    with patch.dict(os.environ, {}, clear=False):
        # Remove CONTACT_EMAIL if set.
        os.environ.pop("CONTACT_EMAIL", None)
        import importlib
        import src.ingestion.registry_builder as rb_mod
        importlib.reload(rb_mod)

        assert rb_mod._DEFAULT_EMAIL == "agent@example.com"

    importlib.reload(rb_mod)


# ── LLM-free mode ───────────────────────────────────────────────────────────

def test_no_llm_client_skips_filtering(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder(use_llm=False)
    result = builder.build_registry("H2 kinetics")

    # All papers should pass through (no LLM filter).
    assert len(result.papers) >= 1


def test_intake_sentence_no_llm(mock_clients):
    from src.ingestion.registry_builder import RegistryBuilder

    builder = RegistryBuilder(use_llm=False)
    brief = builder.intake_sentence("H2/O2 ignition delay is too fast")

    assert brief.raw_sentence == "H2/O2 ignition delay is too fast"
    assert len(brief.keywords) > 0
    assert brief.case_id.startswith("case_")


# ── ZIP mechanism extraction ─────────────────────────────────────────────────

import zipfile
import io


def _create_zip(files: dict[str, bytes]) -> bytes:
    """Create an in-memory ZIP with the given filename→content mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_extract_mechanism_from_zip_direct_inp(mock_clients, tmp_path):
    """ZIP with a direct .inp file extracts it."""
    from src.ingestion.registry_builder import RegistryBuilder

    zip_bytes = _create_zip({"mechanism.inp": b"ELEMENTS\nH O\nEND"})
    zip_path = tmp_path / "si.zip"
    zip_path.write_bytes(zip_bytes)

    builder = RegistryBuilder()
    found = builder._extract_mechanism_from_zip(zip_path, tmp_path / "out")

    assert len(found) == 1
    assert found[0].suffix == ".inp"
    assert found[0].exists()


def test_extract_mechanism_from_zip_nested_zip(mock_clients, tmp_path):
    """ZIP containing a nested ZIP with .inp extracts recursively."""
    from src.ingestion.registry_builder import RegistryBuilder

    inner_zip = _create_zip({"mech/chem.dat": b"SPECIES\nH2 O2\nEND"})
    outer_zip = _create_zip({"supplement.zip": inner_zip})
    zip_path = tmp_path / "si.zip"
    zip_path.write_bytes(outer_zip)

    builder = RegistryBuilder()
    found = builder._extract_mechanism_from_zip(zip_path, tmp_path / "out")

    assert len(found) == 1
    assert found[0].suffix == ".dat"


def test_extract_mechanism_from_zip_images_only(mock_clients, tmp_path):
    """ZIP with only images returns empty list."""
    from src.ingestion.registry_builder import RegistryBuilder

    zip_bytes = _create_zip({
        "figure1.png": b"\x89PNG",
        "figure2.jpg": b"\xff\xd8",
        "readme.pdf": b"%PDF",
    })
    zip_path = tmp_path / "si.zip"
    zip_path.write_bytes(zip_bytes)

    builder = RegistryBuilder()
    found = builder._extract_mechanism_from_zip(zip_path, tmp_path / "out")

    assert found == []


def test_extract_mechanism_from_zip_txt_with_chemkin_keywords(mock_clients, tmp_path):
    """TXT file containing ELEMENTS keyword is extracted as Chemkin."""
    from src.ingestion.registry_builder import RegistryBuilder

    zip_bytes = _create_zip({
        "mechanism.txt": b"ELEMENTS\nH O N\nEND\nSPECIES\nH2 O2 NH3\nEND",
    })
    zip_path = tmp_path / "si.zip"
    zip_path.write_bytes(zip_bytes)

    builder = RegistryBuilder()
    found = builder._extract_mechanism_from_zip(zip_path, tmp_path / "out")

    assert len(found) == 1
    assert found[0].name == "mechanism.txt"


def test_extract_mechanism_from_zip_depth_2(mock_clients, tmp_path):
    """ZIP → ZIP → ZIP → .inp extracts through 2 levels of nesting."""
    from src.ingestion.registry_builder import RegistryBuilder

    innermost = _create_zip({"deep/mech.inp": b"REACTIONS\nH+O2=OH+O\nEND"})
    middle = _create_zip({"level2.zip": innermost})
    outer = _create_zip({"level1.zip": middle})
    zip_path = tmp_path / "si.zip"
    zip_path.write_bytes(outer)

    builder = RegistryBuilder()
    found = builder._extract_mechanism_from_zip(zip_path, tmp_path / "out")

    assert len(found) == 1
    assert found[0].suffix == ".inp"


def test_select_best_mechanism_prefers_cantera(mock_clients):
    """Cantera YAML is preferred over Chemkin .inp."""
    from src.ingestion.registry_builder import RegistryBuilder

    record = _make_record()
    record.mechanism_files = [
        "/tmp/mech.inp",
        "/tmp/chem.dat",
        "/tmp/model.yaml",
    ]
    RegistryBuilder._select_best_mechanism(record)

    assert record.best_mechanism_path == "/tmp/model.yaml"
    assert record.mechanism_format == "cantera"


def test_select_best_mechanism_falls_back_to_chemkin(mock_clients):
    """When no Cantera file, select Chemkin."""
    from src.ingestion.registry_builder import RegistryBuilder

    record = _make_record()
    record.mechanism_files = ["/tmp/mech.inp", "/tmp/therm.dat"]
    RegistryBuilder._select_best_mechanism(record)

    assert record.best_mechanism_path == "/tmp/mech.inp"
    assert record.mechanism_format == "chemkin"


def test_select_best_mechanism_skips_cti(mock_clients):
    """Deprecated .cti files are not selected as best."""
    from src.ingestion.registry_builder import RegistryBuilder

    record = _make_record()
    record.mechanism_files = ["/tmp/old_model.cti"]
    RegistryBuilder._select_best_mechanism(record)

    assert record.best_mechanism_path is None
    assert record.mechanism_format is None


def test_download_si_wires_zip_extraction(mock_clients, tmp_path):
    """_download_si_files calls _extract_mechanism_from_zip for ZIPs."""
    from src.ingestion.registry_builder import RegistryBuilder

    zip_bytes = _create_zip({"mech.yaml": b"phases:\n- name: gas"})
    zip_path = tmp_path / "test_si.zip"
    zip_path.write_bytes(zip_bytes)

    record = _make_record()
    record.si_links = ["https://example.com/si.zip"]

    with patch("src.ingestion.registry_builder.download_file", return_value=zip_path):
        builder = RegistryBuilder()
        builder._download_si_files(record, tmp_path)

    assert len(record.mechanism_files) == 1
    assert record.best_mechanism_path is not None
    assert record.mechanism_format == "cantera"
