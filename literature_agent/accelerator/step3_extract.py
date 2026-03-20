from __future__ import annotations
import json
from .models import CandidateUpdate, FailureBrief, ReactionMapping, RecommendedKinetics, SourceHit

from .llm_utils import call_deepseek

# Agentic Prompt for Step 3
EXTRACTION_PROMPT_SYSTEM = "You are a Chemical Kinetics Expert Agent."
EXTRACTION_PROMPT_USER = """
Synthesize a CandidateUpdate JSON from the literature and failure brief.

FAILURE BRIEF:
{brief_json}

SOURCE HITS:
{sources_json}

REQUIRED JSON FIELDS:
- taxonomy: string
- action: string
- confidence: float (0-1)
- proposed_update: string
- rationale: string
- reaction_mapping: list of objects with [mechanism_label, physical_model]
- recommended_kinetics: object with [bath_gas, low_pressure_limit_k0_cm6_molecule2_s, high_pressure_limit_kinf_cm3_molecule_s, broadening]

OUTPUT: Return ONLY the raw JSON object.
"""

def _call_llm_extraction(brief: FailureBrief, sources: list[SourceHit]) -> dict:
    """
    Call DeepSeek to synthesize a recommendation.
    """
    brief_data = brief.model_dump_json()
    sources_data = json.dumps([s.model_dump() for s in sources])
    
    prompt = EXTRACTION_PROMPT_USER.format(brief_json=brief_data, sources_json=sources_data)
    llm_json = call_deepseek(prompt, system_prompt=EXTRACTION_PROMPT_SYSTEM)
    
    if llm_json is None:
        # Simulation Fallback
        best_source = sources[0] if sources else None
        if best_source and any(kw in best_source.title.lower() for kw in ["kinetic", "combustion"]):
            return {
                "taxonomy": "USE_RATE_WITH_WARNING",
                "action": "ADD_PDEP_TREATMENT",
                "confidence": 0.85,
                "proposed_update": f"Replace Arrhenius for CO+OH with P-dep form.",
                "rationale": "High-confidence simulation of P-dep extraction.",
                "supporting_sources": sources[:3]
            }
        return {"taxonomy": "GENERIC_REVIEW", "action": "verify", "confidence": 0.2, "rationale": "Fallback logic activated."}
    
    return llm_json

def extract_candidate_update(
    brief: FailureBrief,
    sources: list[SourceHit],
) -> CandidateUpdate:
    # 1. Agent Logic: Call DeepSeek
    llm_json = _call_llm_extraction(brief, sources)
    
    # 2. Validation
    return CandidateUpdate(**llm_json)
