from __future__ import annotations
import hashlib
import json
from .models import FailureBrief, Conditions, Target, FocusSet

from .llm_utils import call_deepseek

# Agentic Prompt for Step 1
INTAKE_PROMPT_SYSTEM = "You are a Chemical Kinetics Expert Agent."
INTAKE_PROMPT_USER = """
Convert this human discrepancy report into a structured JSON 'Failure Brief'.

SENTENCE: "{sentence}"

REQUIRED JSON STRUCTURE:
{{
  "raw_sentence": "{sentence}",
  "targets": [
    {{
      "observable": "string (e.g., CO)",
      "direction": "one of: model_high, model_low, mismatch",
      "severity": float (0-1),
      "notes": "string"
    }}
  ],
  "conditions": {{
    "reactor": "string",
    "T_K": float,
    "P_bar": float,
    "fuel": "string"
  }},
  "focus_set": {{
    "species": ["string"],
    "tags": ["string"]
  }},
  "evidence_needs": ["string"],
  "keywords": ["string"]
}}

RULES:
- Handle temperature and pressure units correctly.
- Be precise with species names.
- Output ONLY the raw JSON.
"""

def parse_mismatch_sentence(sentence: str) -> FailureBrief:
    # 1. Agent Logic: Call DeepSeek
    prompt = INTAKE_PROMPT_USER.format(sentence=sentence)
    llm_json = call_deepseek(prompt, system_prompt=INTAKE_PROMPT_SYSTEM)
    
    # 2. Fallback to simulation if No API Key
    if llm_json is None:
        case_hash = hashlib.md5(sentence.encode()).hexdigest()[:8]
        llm_json = {
            "case_id": f"hocho_jsr_{case_hash}",
            "raw_sentence": sentence,
            "targets": [
                {"observable": "CO", "direction": "model_high", "severity": 0.7, "notes": "overpredicted"},
                {"observable": "CO2", "direction": "model_low", "severity": 0.3, "notes": "underpredicted"}
            ],
            "conditions": {
                "reactor": "JSR",
                "T_K": 900.0,
                "P_bar": 1.0,
                "fuel": "HOCHO",
                "oxidizer": "oxidation"
            },
            "focus_set": {
                "species": ["HOCHO", "CO", "CO2", "OH", "HOCO"],
                "tags": ["formic acid", "pressure dependence", "JSR"]
            },
            "evidence_needs": [
                "P-dependent branching for HOCHO",
                "HOCO thermochemistry"
            ],
            "keywords": ["HOCHO", "oxidation", "CO", "CO2", "JSR"]
        }
    
    # 3. Validation: Ensure it fits our Pydantic model
    return FailureBrief(**llm_json)
