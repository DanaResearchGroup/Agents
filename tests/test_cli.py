"""Tests for the CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli import build_parser, cmd_convert, cmd_mae, cmd_run, cmd_validate_model, main, parse_paper_source
from src.schemas.experimental import ConversionResult, PaperSource


# ── parse_paper_source ────────────────────────────────


class TestParsePaperSource:
    def test_doi(self):
        ps = parse_paper_source("doi:10.1016/j.combustflame.2023.01.002")
        assert ps == PaperSource(mode="doi", value="10.1016/j.combustflame.2023.01.002")

    def test_file(self):
        ps = parse_paper_source("file:/tmp/paper.pdf")
        assert ps == PaperSource(mode="upload", value="/tmp/paper.pdf")

    def test_search(self):
        ps = parse_paper_source("search:methane ignition delay")
        assert ps == PaperSource(mode="search", value="methane ignition delay")

    def test_invalid_prefix_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid paper source"):
            parse_paper_source("http://example.com/paper.pdf")

    def test_empty_value_after_prefix(self):
        ps = parse_paper_source("doi:")
        assert ps.mode == "doi"
        assert ps.value == ""


# ── build_parser ──────────────────────────────────────


class TestBuildParser:
    def test_run_parses_all_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "data/models/m.yaml",
            "--experiment", "data/experimental/e.yaml",
            "--paper", "doi:10.1016/xxx",
            "--path1",
            "--output", "out/",
            "--llm-config", "config/llm.yaml",
        ])
        assert args.command == "run"
        assert args.model == Path("data/models/m.yaml")
        assert args.experiment == Path("data/experimental/e.yaml")
        assert args.paper == "doi:10.1016/xxx"
        assert args.path1 is True
        assert args.path2 is False
        assert args.output == Path("out/")
        assert args.llm_config == Path("config/llm.yaml")

    def test_run_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "m.yaml",
            "--experiment", "e.yaml",
            "--paper", "search:query",
        ])
        assert args.output == Path("data/reports")
        assert args.llm_config == Path("config/llm_config.yaml")
        assert args.path1 is False
        assert args.path2 is False

    def test_validate_model_parses(self):
        parser = build_parser()
        args = parser.parse_args(["validate-model", "--model", "m.yaml"])
        assert args.command == "validate-model"
        assert args.model == Path("m.yaml")

    def test_convert_parses(self):
        parser = build_parser()
        args = parser.parse_args([
            "convert",
            "--input", "mech.inp",
            "--output", "out/",
            "--llm-config", "cfg.yaml",
        ])
        assert args.command == "convert"
        assert getattr(args, "input") == Path("mech.inp")
        assert args.output == Path("out/")

    def test_missing_required_run_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--model", "m.yaml"])

    def test_no_command_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0


# ── cmd_run ───────────────────────────────────────────


class TestCmdRun:
    @patch("src.orchestrator.Orchestrator")
    def test_builds_correct_run_config(self, mock_orch_cls):
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=Path("data/reports/report.yaml"))
        mock_orch_cls.return_value = mock_instance

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "m.yaml",
            "--experiment", "e.yaml",
            "--paper", "doi:10.1016/xxx",
            "--path1",
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_run(args)

        config = mock_orch_cls.call_args[0][0]
        assert config.paper_source.mode == "doi"
        assert config.paper_source.value == "10.1016/xxx"
        assert config.run_path1 is True
        assert config.run_path2 is False

    @patch("src.orchestrator.Orchestrator")
    def test_both_paths_when_neither_flag(self, mock_orch_cls):
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=Path("report.yaml"))
        mock_orch_cls.return_value = mock_instance

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "m.yaml",
            "--experiment", "e.yaml",
            "--paper", "search:methane",
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_run(args)

        config = mock_orch_cls.call_args[0][0]
        assert config.run_path1 is True
        assert config.run_path2 is True


# ── cmd_validate_model ────────────────────────────────


class TestCmdValidateModel:
    @patch("src.agents.validators.ModelIsolationValidator")
    def test_prints_reaction_count(self, mock_val_cls, capsys):
        mock_val = MagicMock()
        mock_val.reaction_ids.return_value = {"A=B", "C=D", "E=F"}
        mock_val_cls.return_value = mock_val

        parser = build_parser()
        args = parser.parse_args(["validate-model", "--model", "/tmp/m.yaml"])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_validate_model(args)

        captured = capsys.readouterr()
        assert "Reactions: 3" in captured.out
        mock_val.reaction_ids.assert_called_once_with(Path("/tmp/m.yaml"))


# ── cmd_convert ───────────────────────────────────────


class TestCmdConvert:
    @patch("src.agents.conversion.ChemkinConverter")
    @patch("src.agents.llm_client.LLMClient")
    def test_success(self, mock_llm_cls, mock_conv_cls, capsys):
        mock_conv = MagicMock()
        mock_conv.convert = AsyncMock(
            return_value=ConversionResult(
                success=True, output_path=Path("/tmp/out/mech.yaml"), attempts=1,
            )
        )
        mock_conv_cls.return_value = mock_conv

        parser = build_parser()
        args = parser.parse_args([
            "convert", "--input", "/tmp/mech.inp", "--output", "/tmp/out",
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_convert(args)

        captured = capsys.readouterr()
        assert "Converted:" in captured.out

    @patch("src.agents.conversion.ChemkinConverter")
    @patch("src.agents.llm_client.LLMClient")
    def test_failure_exits_1(self, mock_llm_cls, mock_conv_cls):
        mock_conv = MagicMock()
        mock_conv.convert = AsyncMock(
            return_value=ConversionResult(
                success=False, errors=["syntax error line 42"], attempts=2,
            )
        )
        mock_conv_cls.return_value = mock_conv

        parser = build_parser()
        args = parser.parse_args([
            "convert", "--input", "/tmp/mech.inp", "--output", "/tmp/out",
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        with pytest.raises(SystemExit) as exc:
            cmd_convert(args)
        assert exc.value.code == 1


# ── mae subparser ────────────────────────────────


class TestMaeParser:
    def test_mae_parses_all_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "mae",
            "--model", "data/models/m.yaml",
            "--experiment", "data/experimental/e.yaml",
            "--end-time", "0.5",
        ])
        assert args.command == "mae"
        assert args.model == Path("data/models/m.yaml")
        assert args.experiment == Path("data/experimental/e.yaml")
        assert args.end_time == 0.5

    def test_mae_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "mae",
            "--model", "m.yaml",
            "--experiment", "e.yaml",
        ])
        assert args.end_time == 1.0

    def test_mae_missing_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["mae", "--model", "m.yaml"])


# ── cmd_mae ──────────────────────────────────────


class TestCmdMae:
    @patch("src.simulation.core.runner.run_simulation")
    def test_species_profile_pass(self, mock_run, capsys, tmp_path):
        """Single condition, simulated == measured -> MAE=0 -> PASS."""
        from src.simulation.core.runner import SimulationResult

        mock_run.return_value = SimulationResult(
            experiment_id="test_0",
            mechanism="m.yaml",
            species_histories={"CO2(12)": [0.0, 0.0005, 0.001522]},
            times=[0.0, 0.5, 1.0],
        )

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            "name: test\n"
            "conditions:\n"
            "  - reactor_type: shock_tube\n"
            "    temperature_K: 1400.0\n"
            "    pressure_atm: 1.5\n"
            "    mixture:\n"
            "      Ar: 0.99\n"
            "    observable_type: species_profile\n"
            "    observable_label: 'CO2(12)'\n"
            "    measured_value: 0.001522\n"
            "    error_threshold: 0.0005\n"
        )

        parser = build_parser()
        args = parser.parse_args([
            "mae", "--model", "m.yaml", "--experiment", str(exp_yaml),
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_mae(args)

        out = capsys.readouterr().out
        assert "T=1400.0K" in out
        assert "PASS" in out
        assert "MAE=0.000000" in out
        assert "1/1 conditions pass threshold" in out

    @patch("src.simulation.core.runner.run_simulation")
    def test_species_profile_fail(self, mock_run, capsys, tmp_path):
        """Simulated far from measured -> FAIL."""
        from src.simulation.core.runner import SimulationResult

        mock_run.return_value = SimulationResult(
            experiment_id="test_0",
            mechanism="m.yaml",
            species_histories={"CO2(12)": [0.0, 0.01, 0.05]},
            times=[0.0, 0.5, 1.0],
        )

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            "name: test\n"
            "conditions:\n"
            "  - reactor_type: shock_tube\n"
            "    temperature_K: 1400.0\n"
            "    pressure_atm: 1.5\n"
            "    mixture:\n"
            "      Ar: 0.99\n"
            "    observable_type: species_profile\n"
            "    observable_label: 'CO2(12)'\n"
            "    measured_value: 0.001522\n"
            "    error_threshold: 0.0005\n"
        )

        parser = build_parser()
        args = parser.parse_args([
            "mae", "--model", "m.yaml", "--experiment", str(exp_yaml),
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_mae(args)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "0/1 conditions pass threshold" in out

    @patch("src.simulation.core.runner.run_simulation")
    def test_simulation_error_printed(self, mock_run, capsys, tmp_path):
        """When simulation fails, print ERROR line."""
        from src.simulation.core.runner import SimulationResult

        mock_run.return_value = SimulationResult(
            experiment_id="test_0",
            mechanism="m.yaml",
            success=False,
            error="Species not found",
        )

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            "name: test\n"
            "conditions:\n"
            "  - reactor_type: shock_tube\n"
            "    temperature_K: 1400.0\n"
            "    pressure_atm: 1.5\n"
            "    mixture:\n"
            "      Ar: 0.99\n"
            "    observable_type: species_profile\n"
            "    observable_label: 'CO2(12)'\n"
            "    measured_value: 0.001522\n"
            "    error_threshold: 0.0005\n"
        )

        parser = build_parser()
        args = parser.parse_args([
            "mae", "--model", "m.yaml", "--experiment", str(exp_yaml),
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_mae(args)

        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "Species not found" in out
        assert "0/1 conditions pass threshold" in out

    @patch("src.simulation.core.runner.run_simulation")
    def test_multiple_conditions(self, mock_run, capsys, tmp_path):
        """Two conditions: one PASS, one FAIL."""
        from src.simulation.core.runner import SimulationResult

        mock_run.side_effect = [
            SimulationResult(
                experiment_id="test_0", mechanism="m.yaml",
                species_histories={"CO2(12)": [0.001522]}, times=[1.0],
            ),
            SimulationResult(
                experiment_id="test_1", mechanism="m.yaml",
                species_histories={"CO2(12)": [0.9]}, times=[1.0],
            ),
        ]

        exp_yaml = tmp_path / "exp.yaml"
        exp_yaml.write_text(
            "name: test\n"
            "conditions:\n"
            "  - reactor_type: shock_tube\n"
            "    temperature_K: 1400.0\n"
            "    pressure_atm: 1.5\n"
            "    mixture:\n"
            "      Ar: 0.99\n"
            "    observable_type: species_profile\n"
            "    observable_label: 'CO2(12)'\n"
            "    measured_value: 0.001522\n"
            "    error_threshold: 0.0005\n"
            "  - reactor_type: shock_tube\n"
            "    temperature_K: 1600.0\n"
            "    pressure_atm: 1.5\n"
            "    mixture:\n"
            "      Ar: 0.99\n"
            "    observable_type: species_profile\n"
            "    observable_label: 'CO2(12)'\n"
            "    measured_value: 0.004004\n"
            "    error_threshold: 0.001\n"
        )

        parser = build_parser()
        args = parser.parse_args([
            "mae", "--model", "m.yaml", "--experiment", str(exp_yaml),
        ])
        from src.cli import _resolve_paths
        _resolve_paths(args)

        cmd_mae(args)

        out = capsys.readouterr().out
        assert "PASS" in out
        assert "FAIL" in out
        assert "1/2 conditions pass threshold" in out
