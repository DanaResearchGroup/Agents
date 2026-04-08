"""Flame experiment family rules."""

import re
from .base import FamilyRules

flame = FamilyRules(
    name="flame",

    anchor_patterns=[
        re.compile(
            r'\b(?:(?:laminar\s+)?(?:flame|burner)\s+(?:setup|apparatus|experiment)|'
            r'bunsen\s+burner|flat[\-\s]?flame\s+burner|'
            r'counter[\-\s]?flow\s+(?:flame|burner)|heat[\-\s]?flux\s+burner)\b',
            re.IGNORECASE,
        ),
    ],

    observable_patterns={
        "flame_speed": re.compile(
            r'(?:laminar\s+)?(?:flame|burning)\s+(?:speed|velocity)',
            re.IGNORECASE,
        ),
        "species_profile": re.compile(
            r'(?:flame\s+)?(?:species|concentration)\s+pro[fﬁ]le',
            re.IGNORECASE,
        ),
    },

    temperature_bounds=(250.0, 2500.0),
    required_conditions=["temperature", "pressure", "composition"],
    template_family="flame_speed",
    default_pressure=None,
)
