"""Tests for src/orchestrator.py — all I/O mocked."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.agents.llm_client import LLMClient, LLMConfig
from src.orchestrator import Orchestrator
from src.schemas.experimental import PaperSource, RunConfig
from src.schemas.ingestion import PaperRecord, SearchQuery


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_dataset() -> dict:
    """Minimal ExperimentalDataset that passes Pydantic validation."""
    return {
        "name": "test_dataset",
        "conditions": [
            {
                "reactor_type": "shock_tube",
                "temperature_K": 1200.0,
                "pressure_atm": 1.0,
                "mixture": {"H2": 0.5, "O2": 0.5},
                "observable_type": "idt",
                "measured_value": 0.001,
                "error_threshold": 0.1,
            }
        ],
    }


@pytest.fixture()
def workspace(tmp_path: Path, minimal_dataset: dict) -> dict:
    """Create temporary model, experiment, and config files."""
    model_file = tmp_path / "model.yaml"
    model_file.write_text("# placeholder mechanism\n")

    exp_file = tmp_path / "experiment.yaml"
    exp_file.write_text(yaml.dump(minimal_dataset))

    llm_cfg = tmp_path / "llm_config.yaml"
    llm_cfg.write_text(yaml.dump({"model": "gpt-4", "provider": "openai"}))

    output_dir = tmp_path / "reports"

    return {
        "model": model_file,
        "experiment": exp_file,
        "llm_config": llm_cfg,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture()
def make_config(workspace: dict):
    """Factory that builds a RunConfig, overridable per-test."""
    def _make(**overrides) -> RunConfig:
        defaults = {
            "original_model": workspace["model"],
            "experimental_data": workspace["experiment"],
            "paper_source": PaperSource(mode="doi", value="10.1234/test"),
            "output_dir": workspace["output_dir"],
            "llm_config": workspace["llm_config"],
        }
        defaults.update(overrides)
        return RunConfig(**defaults)
    return _make


def _fake_paper(title: str = "Test Paper", doi: str = "10.1234/test") -> PaperRecord:
    return PaperRecord(id="p1", title=title, doi=doi, provenance="test")


# ── Init validation ─────────────────────────────────────────────────────────


def test_init_raises_on_missing_model(workspace: dict):
    config = RunConfig(
        original_model=Path("/nonexistent/model.yaml"),
        experimental_data=workspace["experiment"],
        paper_source=PaperSource(mode="doi", value="10.1234/test"),
        llm_config=workspace["llm_config"],
    )
    with pytest.raises(ValueError, match="Original model not found"):
        Orchestrator(config)


def test_init_raises_on_missing_experiment(workspace: dict):
    config = RunConfig(
        original_model=workspace["model"],
        experimental_data=Path("/nonexistent/experiment.yaml"),
        paper_source=PaperSource(mode="doi", value="10.1234/test"),
        llm_config=workspace["llm_config"],
    )
    with pytest.raises(ValueError, match="Experimental data not found"):
        Orchestrator(config)


def test_init_success(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    config = make_config()
    orch = Orchestrator(config)
    assert orch.dataset is not None
    assert orch.config is config


# ── Paper ingestion: DOI mode ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_doi(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    paper = _fake_paper()
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.search",
        lambda self, query, limit=None: [{"id": "w1", "title": "Test Paper"}],
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.normalize",
        lambda self, item: paper,
    )

    orch = Orchestrator(make_config())
    result = await orch.ingest_paper(PaperSource(mode="doi", value="10.1234/test"))
    assert result.title == "Test Paper"


@pytest.mark.asyncio
async def test_ingest_doi_not_found(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.search",
        lambda self, query, limit=None: [],
    )

    orch = Orchestrator(make_config())
    with pytest.raises(ValueError, match="No paper found for DOI"):
        await orch.ingest_paper(PaperSource(mode="doi", value="10.9999/nope"))


# ── Paper ingestion: upload mode ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_upload(make_config, monkeypatch, workspace):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    # Create a fake PDF file (parse_pdf is mocked so contents don't matter)
    pdf_path = workspace["tmp_path"] / "paper.pdf"
    pdf_path.write_text("fake pdf content")

    from src.schemas.experimental import PaperDocument
    mock_doc = PaperDocument(
        pdf_path=str(pdf_path), title="Uploaded Paper", pages=[], captions=[],
    )
    monkeypatch.setattr("src.orchestrator.parse_pdf", lambda path: mock_doc)

    orch = Orchestrator(make_config())
    result = await orch.ingest_paper(
        PaperSource(mode="upload", value=str(pdf_path))
    )
    assert result.title == "Uploaded Paper"
    assert result.provenance == "local_upload"


@pytest.mark.asyncio
async def test_ingest_upload_missing_file(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    orch = Orchestrator(make_config())
    with pytest.raises(ValueError, match="PDF file not found"):
        await orch.ingest_paper(
            PaperSource(mode="upload", value="/nonexistent/paper.pdf")
        )


# ── Paper ingestion: search mode ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_search(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.search",
        lambda self, query, limit=None: [
            {"id": f"w{i}", "title": f"Paper {i}"} for i in range(1, 4)
        ],
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.normalize",
        lambda self, item: _fake_paper(item["title"], f"10.1234/{item['id']}"),
    )

    orch = Orchestrator(make_config(), confirm_fn=lambda records: 1)
    result = await orch.ingest_paper(
        PaperSource(mode="search", value="hydrogen combustion")
    )
    assert result.title == "Paper 2"


@pytest.mark.asyncio
async def test_ingest_search_invalid_choice(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.search",
        lambda self, query, limit=None: [
            {"id": "w1", "title": "Paper 1"}
        ],
    )
    monkeypatch.setattr(
        "src.orchestrator.OpenAlexClient.normalize",
        lambda self, item: _fake_paper(),
    )

    def _bad_confirm(records):
        raise ValueError("Invalid selection: banana")

    orch = Orchestrator(make_config(), confirm_fn=_bad_confirm)
    with pytest.raises(ValueError, match="Invalid selection"):
        await orch.ingest_paper(
            PaperSource(mode="search", value="hydrogen combustion")
        )


# ── run() with path flags ──────────────────────────────────────────────────


async def _async_fake_ingest(self, source):
    return _fake_paper()


async def _async_noop(self, paper):
    return None


@pytest.mark.asyncio
async def test_run_both_paths(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    calls = []

    async def _track_path1(self, paper):
        calls.append("path1")
        return None

    async def _track_path2(self, paper, path1_results=None):
        calls.append("path2")
        return None

    monkeypatch.setattr(Orchestrator, "_run_path1", _track_path1)
    monkeypatch.setattr(Orchestrator, "_run_path2", _track_path2)
    monkeypatch.setattr(Orchestrator, "ingest_paper", _async_fake_ingest)

    config = make_config(run_path1=True, run_path2=True)
    orch = Orchestrator(config)
    report_path = await orch.run()

    assert "path1" in calls
    assert "path2" in calls
    assert report_path.exists()


@pytest.mark.asyncio
async def test_run_path1_only(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    calls = []

    async def _track_path1(self, paper):
        calls.append("path1")
        return None

    async def _track_path2(self, paper, path1_results=None):
        calls.append("path2")
        return None

    monkeypatch.setattr(Orchestrator, "_run_path1", _track_path1)
    monkeypatch.setattr(Orchestrator, "_run_path2", _track_path2)
    monkeypatch.setattr(Orchestrator, "ingest_paper", _async_fake_ingest)

    config = make_config(run_path1=True, run_path2=False)
    orch = Orchestrator(config)
    await orch.run()

    assert "path1" in calls
    assert "path2" not in calls


@pytest.mark.asyncio
async def test_run_path2_only(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    calls = []

    async def _track_path1(self, paper):
        calls.append("path1")
        return None

    async def _track_path2(self, paper, path1_results=None):
        calls.append("path2")
        return None

    monkeypatch.setattr(Orchestrator, "_run_path1", _track_path1)
    monkeypatch.setattr(Orchestrator, "_run_path2", _track_path2)
    monkeypatch.setattr(Orchestrator, "ingest_paper", _async_fake_ingest)

    config = make_config(run_path1=False, run_path2=True)
    orch = Orchestrator(config)
    await orch.run()

    assert "path1" not in calls
    assert "path2" in calls


# ── Report placeholder ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_report_written_to_output_dir(make_config, monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.LLMClient.__init__",
        lambda self, **kwargs: setattr(self, "config", LLMConfig()),
    )
    monkeypatch.setattr(Orchestrator, "ingest_paper", _async_fake_ingest)

    config = make_config(run_path1=False, run_path2=False)
    orch = Orchestrator(config)
    report_path = await orch.run()

    assert report_path.exists()
    assert report_path.parent == config.output_dir

    content = yaml.safe_load(report_path.read_text())
    assert "path1" not in content
    assert "path2" not in content
    assert "original_model" in content
    assert "run_id" in content
