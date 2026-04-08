# Simulator Agent Run Summary

**Paper:** An experimental and modeling study of ammonia pyrolysis
**Pages:** 17
**Captions:** 19

## Experiment Candidates

- **shock_tube** (primary, high priority)
- **flow_reactor** (primary, high priority)
- **jsr** (secondary, low priority)

## Scenarios

| Status | Count |
|--------|-------|
| Accepted | 15 |
| Needs Review | 11 |
| Rejected | 11 |
| **Total** | **37** |

### Accepted Scenarios

- **Figure 1**: 0.45% NH3 in Ar @ 2217K, 1.22 atm
- **Figure 2**: 0.439% NH3 /Ar @ 2373K, 1.16 atm
- **Figure 5 panel a**: ∼0.5% NH3 in Ar @ 2096K, 1.25 atm
- **Figure 5 panel b**: ∼0.5% NH3 in Ar @ 2174K, 1.25 atm
- **Figure 5 panel c**: ∼0.5% NH3 in Ar @ 2487 K, 1 atm
- **Figure 5 panel d**: ∼0.5% NH3 in Ar @ 2700 K, 1 atm
- **Figure 7**: ∼0.5% NH3 in Ar @ 2400K, 1 atm
- **Figure 7**: ∼0.42% NH3 /2% H2 in Ar @ 2400K, 1 atm
- **Figure 8**: 0.5% NH3 @ 2400K, 1 atm
- **Figure 8**: 0.5% NH3 /2% H2 @ 2400K, 1 atm
- **Table 2**: ∼0.5% NH3 in Ar @ 2100 −3100K, 7 −9 atm
- **Table 2**: ∼0.42% NH3 /2% H2 in Ar @ 2100 −3100K, 7 −9 atm
- **Table 2**: 1% NH3 in Ar @ 2100 −3100K, 7 −9 atm
- **Table 2**: 0.5% N2H4 in Ar @ 2100 −3100K, 7 −9 atm
- **methods**: 0.439% NH3 @ 2373K, 1.16 atm

### Needs Review

- **Figure 5**: pressure assumed from default
- **Figure 6**: temperature from fallback (not caption), pressure assumed from default
- **Figure 6 panel ?**: missing required fields for simulation
- **Figure 6 panel ?**: missing required fields for simulation
- **Figure 7 panel ?**: missing required fields for simulation
- **Figure 7 panel ?**: missing required fields for simulation
- **Figure 7 panel ?**: missing required fields for simulation
- **Figure 7 panel ?**: missing required fields for simulation
- **Figure 9**: temperature from fallback (not caption), pressure assumed from default
- **Figure 10**: temperature from fallback (not caption), pressure assumed from default
- **Figure 10**: temperature from fallback (not caption), pressure assumed from default

## Simulation Plans

| Status | Count |
|--------|-------|
| Executable | 15 |
| Draft | 11 |
| **Total** | **26** |

### Executable Plans

- **scenario_1**: idt_const_uv | T=2217.0-2217.0 K
- **scenario_2**: idt_const_uv | T=2373.0-2373.0 K
- **scenario_5_panel_a**: idt_const_uv | T=2096.0-2096.0 K
- **scenario_5_panel_b**: idt_const_uv | T=2174.0-2174.0 K
- **scenario_5_panel_c**: idt_const_uv | T=2487.0-2487.0 K
- **scenario_5_panel_d**: idt_const_uv | T=2700.0-2700.0 K
- **scenario_7**: idt_const_uv | T=2400.0-2400.0 K
- **scenario_8**: idt_const_uv | T=2400.0-2400.0 K
- **scenario_9**: idt_const_uv | T=2400.0-2400.0 K
- **scenario_10**: idt_const_uv | T=2400.0-2400.0 K
- **scenario_12**: idt_const_uv | T=2100.0-3100.0 K
- **scenario_13**: idt_const_uv | T=2100.0-3100.0 K
- **scenario_14**: idt_const_uv | T=2100.0-3100.0 K
- **scenario_15**: idt_const_uv | T=2100.0-3100.0 K
- **scenario_27**: idt_const_uv | T=2373.0-2373.0 K

### Draft Plans (blocked)

- **scenario_5**: pressure: pressure assumed from default
- **scenario_6**: temperature: temperature from fallback (not caption), pressure: pressure assumed from default
- **scenario_6_panel_?**: unspecified
- **scenario_6_panel_?**: unspecified
- **scenario_7_panel_?**: unspecified
- **scenario_7_panel_?**: unspecified
- **scenario_8_panel_?**: unspecified
- **scenario_8_panel_?**: unspecified
- **scenario_11**: temperature: temperature from fallback (not caption), pressure: pressure assumed from default
- **scenario_16**: temperature: temperature from fallback (not caption), pressure: pressure assumed from default
- **scenario_17**: temperature: temperature from fallback (not caption), pressure: pressure assumed from default
