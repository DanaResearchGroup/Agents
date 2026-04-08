"""Rapid compression machine experiment family rules."""

import re
from .base import FamilyRules

rcm = FamilyRules(
    name="rcm",

    anchor_patterns=[
        re.compile(
            r'\b(?:rapid\s+compression\s+machine|RCM)\b',
            re.IGNORECASE,
        ),
    ],

    observable_patterns={
        "ignition_delay": re.compile(
            r'ignition\s+delay',
            re.IGNORECASE,
        ),
        "species_profile": re.compile(
            r'(?:speciation|species|pressure)\s+(?:pro[fﬁ]le|trace|histor)',
            re.IGNORECASE,
        ),
    },

    temperature_bounds=(600.0, 1200.0),
    required_conditions=["temperature", "pressure", "composition"],
    template_family="idt_const_hp",
    default_pressure=None,
)
