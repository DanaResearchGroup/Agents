# Handoff Document — Current State

## System status
Path 1 pipeline runs end-to-end. Report is generated.
ConditionReasoningAgent working (returns JSON via think-stripping).
Species mapper implemented (replaces aliases files).
PaperReaderAgent inconsistent — see open issues below.
Session C (SimulationPlannerAgent) not started.

## Open issues (fix before Session C)

### 1. PaperReaderAgent inconsistency
Sometimes returns empty summary. Need to diagnose root cause.
Run this debug command to see raw output:
  python -c "... (debug script from chat)"
Suspected causes: think stripping removing real content,
or tool calls failing silently.

### 2. Session C not started
Build src/agents/simulation_planner.py
- Takes SimConditions + model path
- Reasons about correct reactor setup
- Handles species name mapping (SpeciesMapper already exists)
- Replaces hardcoded runner.py logic
- Skill file: src/agents/skills/simulation_planner_skill.md

## Architecture decisions made (see DECISIONS.md for full rationale)
- PydanticAI for agents (not LangChain)
- OllamaModel workaround for null content bug
- _strip_thinking() for qwen3 think tag leakage
- _extract_json_array() fallback for malformed output
- SpeciesMapper for automatic cross-model species name mapping
- Plan-first architecture: deterministic pipeline → agent validates
- result_type=str for condition agent (not ConditionExtractionResult)

## Current working command
chem-agent run \
  --model data/models/chem_annotated_nh3.yaml \
  --experiment data/experimental/nh3_stagni2021_smoketest.yaml \
  --paper file:data/papers/stagni2021_nh3.pdf \
  --literature-model data/models/nh3_literature_minimal.yaml \
  --path1 \
  --output data/reports/ \
  --llm-config config/llm_config.local.yaml

## Model in use
qwen3.5:4b via Ollama (AMD RX 6600, 8GB VRAM)
Known issue: tool calling sometimes leaks think content
Real fix: use Claude/DeepSeek API (per-agent override supported)

## Next session priorities
1. Diagnose PaperReaderAgent root cause (debug script above)
2. Fix it
3. Session C — SimulationPlannerAgent
4. Path 2 smoke test
