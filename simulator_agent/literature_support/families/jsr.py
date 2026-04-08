"""Jet-stirred reactor experiment family rules."""

import re
from .base import FamilyRules

jsr = FamilyRules(
    name="jsr",

    anchor_patterns=[
        re.compile(
            r'\b(?:jet[\-\s]?stirred\s+reactor|JSR|PSR|'
            r'perfectly[\-\s]?stirred\s+reactor|'
            r'well[\-\s]?stirred\s+reactor|WSR|'
            r'CSTR|continuously[\-\s]?stirred(?:\s+tank)?(?:\s+reactor)?)\b',
            re.IGNORECASE,
        ),
    ],

    observable_patterns={
        "species_profile": re.compile(
            r'(?:speciation|species|concentration|mole\s+fraction)\s+pro[fﬁ]le|'
            r'(?:outlet|exit)\s+(?:composition|mole\s+fraction)',
            re.IGNORECASE,
        ),
        "conversion": re.compile(
            r'(?:fuel\s+)?conversion|'
            r'outlet\s+mole\s+fraction|'
            r'product\s+(?:yield|distribution)',
            re.IGNORECASE,
        ),
    },

    temperature_bounds=(300.0, 1500.0),
    required_conditions=["temperature", "pressure", "composition", "residence_time"],
    template_family="jsr_const_tp",
    default_pressure=None,
)
