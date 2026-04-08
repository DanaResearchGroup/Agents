"""Cantera integration tests: verify physical correctness of reactor implementations.

These tests require Cantera to be installed. They validate that:
- each reactor type produces physically meaningful results
- observables are extracted correctly
- the end-to-end pipeline (spec → run → extract → MAE) works

Skipped automatically if Cantera is not available.
"""

import pytest

try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False

from core.simulation_spec import SimulationSpec
from core.runner import run_simulation, SimulationResult
from core.observable_extractor import extract_observable
from core.mae import compute_mae

pytestmark = pytest.mark.skipif(not HAS_CANTERA, reason="Cantera not installed")


# ── Shock tube ──────────────────────────────────────────────────────────

class TestShockTube:
    """Validate constant-volume adiabatic reactor (shock tube)."""

    def test_h2_ignition_occurs(self):
        """H2/O2 at 1200 K, 5 atm should ignite (T rises significantly)."""
        spec = SimulationSpec(
            experiment_id="st_ign",
            reactor_type="shock_tube",
            observable="ignition_delay",
            temperature=1200.0,
            pressure=5 * 101325.0,
            composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
            end_time=0.01,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert result.success
        T_rise = max(result.temperature_history) - result.temperature_history[0]
        assert T_rise > 100  # dilute mixture, but should still ignite

    def test_idt_decreases_with_temperature(self):
        """Arrhenius behavior: IDT should decrease as T increases."""
        idts = []
        for T in [1000.0, 1200.0, 1400.0]:
            spec = SimulationSpec(
                experiment_id=f"st_{T:.0f}",
                reactor_type="shock_tube",
                observable="ignition_delay",
                temperature=T,
                pressure=5 * 101325.0,
                composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
                end_time=0.1,
            )
            result = run_simulation(spec, "h2o2.yaml")
            idt = extract_observable(spec, result)
            assert len(idt) == 1
            idts.append(idt[0])

        # Each IDT should be shorter than the previous (lower T = longer IDT)
        assert idts[0] > idts[1] > idts[2]

    def test_constant_volume(self):
        """Shock tube is constant-V: pressure must increase if T increases."""
        spec = SimulationSpec(
            experiment_id="st_constV",
            reactor_type="shock_tube",
            observable="ignition_delay",
            temperature=1200.0,
            pressure=5 * 101325.0,
            composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
            end_time=0.01,
        )
        result = run_simulation(spec, "h2o2.yaml")

        P_max = max(result.pressure_history)
        P0 = result.pressure_history[0]
        assert P_max > P0 * 1.05  # pressure should rise with temperature

    def test_initial_state_recorded(self):
        """t=0 should be the first recorded point."""
        spec = SimulationSpec(
            experiment_id="st_t0",
            reactor_type="shock_tube",
            observable="ignition_delay",
            temperature=1200.0,
            pressure=101325.0,
            composition={"H2": 0.01, "O2": 0.005, "AR": 0.985},
            end_time=0.001,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert result.times[0] == 0.0
        assert result.temperature_history[0] == pytest.approx(1200.0, rel=1e-3)

    def test_observable_species_tracked(self):
        """Species not in composition but set as observable_species should be tracked."""
        spec = SimulationSpec(
            experiment_id="st_oh",
            reactor_type="shock_tube",
            observable="species_profile",
            observable_species="OH",
            temperature=1200.0,
            pressure=5 * 101325.0,
            composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
            end_time=0.001,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert "OH" in result.species_histories
        oh = result.species_histories["OH"]
        assert oh[0] == pytest.approx(0.0, abs=1e-10)  # no OH initially
        assert max(oh) > 1e-6  # OH produced during combustion


# ── JSR ─────────────────────────────────────────────────────────────────

class TestJSR:
    """Validate jet-stirred reactor (isothermal CSTR)."""

    def test_isothermal(self):
        """JSR should maintain constant temperature."""
        spec = SimulationSpec(
            experiment_id="jsr_iso",
            reactor_type="jsr",
            observable="species_profile",
            observable_species="CH4",
            temperature=1400.0,
            pressure=101325.0,
            composition={"CH4": 0.01, "O2": 0.02, "N2": 0.97},
            residence_time=0.5,
        )
        result = run_simulation(spec, "gri30.yaml")

        assert result.success
        assert result.temperature_history[0] == pytest.approx(1400.0, rel=1e-3)

    def test_high_conversion_at_high_T(self):
        """CH4 at 1400 K with 500 ms residence time should nearly fully convert."""
        spec = SimulationSpec(
            experiment_id="jsr_conv",
            reactor_type="jsr",
            observable="species_profile",
            observable_species="CH4",
            temperature=1400.0,
            pressure=101325.0,
            composition={"CH4": 0.01, "O2": 0.02, "N2": 0.97},
            residence_time=0.5,
        )
        result = run_simulation(spec, "gri30.yaml")

        ch4_ss = result.species_histories["CH4"][0]
        conversion = (0.01 - ch4_ss) / 0.01
        assert conversion > 0.95  # expect near-complete conversion at 1400 K

    def test_low_conversion_at_low_T(self):
        """CH4 at 800 K should barely react."""
        spec = SimulationSpec(
            experiment_id="jsr_low",
            reactor_type="jsr",
            observable="species_profile",
            observable_species="CH4",
            temperature=800.0,
            pressure=101325.0,
            composition={"CH4": 0.01, "O2": 0.02, "N2": 0.97},
            residence_time=0.5,
        )
        result = run_simulation(spec, "gri30.yaml")

        ch4_ss = result.species_histories["CH4"][0]
        conversion = (0.01 - ch4_ss) / 0.01
        assert conversion < 0.1  # barely any reaction at 800 K


# ── Flow reactor ────────────────────────────────────────────────────────

class TestFlowReactor:
    """Validate plug flow reactor (constant-P, isothermal)."""

    def test_constant_pressure(self):
        """Flow reactor should maintain constant pressure."""
        spec = SimulationSpec(
            experiment_id="fr_constP",
            reactor_type="flow_reactor",
            observable="conversion",
            observable_species="H2",
            temperature=1100.0,
            pressure=101325.0,
            composition={"H2": 0.01, "O2": 0.005, "AR": 0.985},
            residence_time=0.1,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert result.success
        P0 = result.pressure_history[0]
        for P in result.pressure_history:
            assert P == pytest.approx(P0, rel=1e-6)

    def test_constant_temperature(self):
        """Isothermal flow reactor: temperature should not change."""
        spec = SimulationSpec(
            experiment_id="fr_constT",
            reactor_type="flow_reactor",
            observable="conversion",
            observable_species="H2",
            temperature=1100.0,
            pressure=101325.0,
            composition={"H2": 0.01, "O2": 0.005, "AR": 0.985},
            residence_time=0.1,
        )
        result = run_simulation(spec, "h2o2.yaml")

        for T in result.temperature_history:
            assert T == pytest.approx(1100.0, rel=1e-6)

    def test_species_monotonic_decay(self):
        """Fuel species should monotonically decrease in a PFR."""
        spec = SimulationSpec(
            experiment_id="fr_decay",
            reactor_type="flow_reactor",
            observable="species_profile",
            observable_species="H2",
            temperature=1100.0,
            pressure=101325.0,
            composition={"H2": 0.01, "O2": 0.005, "AR": 0.985},
            residence_time=0.1,
        )
        result = run_simulation(spec, "h2o2.yaml")

        h2 = result.species_histories["H2"]
        for i in range(1, len(h2)):
            assert h2[i] <= h2[i - 1] + 1e-12  # allow tiny float noise


# ── RCM ─────────────────────────────────────────────────────────────────

class TestRCM:
    """Validate rapid compression machine (constant-V, adiabatic)."""

    def test_constant_volume_pressure_rises(self):
        """RCM is const-V: if T rises, P must rise proportionally."""
        spec = SimulationSpec(
            experiment_id="rcm_constV",
            reactor_type="rcm",
            observable="ignition_delay",
            temperature=1000.0,
            pressure=30 * 101325.0,
            composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
            end_time=0.1,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert result.success
        P_max = max(result.pressure_history)
        P0 = result.pressure_history[0]
        T_max = max(result.temperature_history)
        T0 = result.temperature_history[0]

        # For const-V ideal gas: P/T = const (approximately, ignoring mole changes)
        if T_max > T0 * 1.01:
            assert P_max > P0 * 1.01


# ── Flame ───────────────────────────────────────────────────────────────

class TestFlame:
    """Validate freely propagating premixed flame."""

    def test_flame_speed_positive(self):
        """Flame speed should be a positive, finite number."""
        import cantera as ct
        gas = ct.Solution("h2o2.yaml")
        gas.set_equivalence_ratio(1.0, "H2", "O2:0.21, AR:0.79")
        comp = {sp: float(gas.X[i]) for i, sp in enumerate(gas.species_names) if gas.X[i] > 1e-10}

        spec = SimulationSpec(
            experiment_id="flame_h2",
            reactor_type="flame",
            observable="flame_speed",
            temperature=300.0,
            pressure=101325.0,
            composition=comp,
        )
        result = run_simulation(spec, "h2o2.yaml")

        assert result.success
        fs = extract_observable(spec, result)
        assert len(fs) == 1
        assert 0.5 < fs[0] < 10.0  # reasonable range for H2/air-like


# ── End-to-end MAE pipeline ────────────────────────────────────────────

class TestEndToEnd:
    """Test the full evaluation pipeline: spec → run → extract → MAE."""

    def test_identical_models_zero_mae(self):
        """Same model compared to itself should give MAE ≈ 0."""
        temps = [1200.0, 1400.0]
        predicted = []
        observed = []

        for T in temps:
            spec = SimulationSpec(
                experiment_id=f"e2e_{T:.0f}",
                reactor_type="shock_tube",
                observable="ignition_delay",
                temperature=T,
                pressure=5 * 101325.0,
                composition={"H2": 0.02, "O2": 0.01, "AR": 0.97},
                end_time=0.01,
            )
            r1 = run_simulation(spec, "h2o2.yaml")
            r2 = run_simulation(spec, "h2o2.yaml")

            v1 = extract_observable(spec, r1)
            v2 = extract_observable(spec, r2)
            predicted.append(v1[0])
            observed.append(v2[0])

        mae = compute_mae(predicted, observed)
        assert mae == pytest.approx(0.0, abs=1e-12)
