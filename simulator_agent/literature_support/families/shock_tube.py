"""Shock tube experiment family rules."""

import re
from .base import FamilyRules

shock_tube = FamilyRules(
    name="shock_tube",

    anchor_patterns=[
        re.compile(
            r'\b(?:shock\s+tube|reflected[\-\s]?shock|incident[\-\s]?shock)\b',
            re.IGNORECASE,
        ),
    ],

    observable_patterns={
        "species_profile": re.compile(
            r'(?:speciation|species|concentration|mole\s+fraction)\s+pro[fﬁ]le|'
            r'(?:laser\s+)?(?:transmission|absorption)\s+(?:trace|signal|pro[fﬁ]le|measurement)|'
            r'time\s+histor|'
            r'decomposition\s+pro[fﬁ]le',
            re.IGNORECASE,
        ),
        "ignition_delay": re.compile(
            r'ignition\s+delay',
            re.IGNORECASE,
        ),
        "half_life": re.compile(
            r'half[\-\s]?life',
            re.IGNORECASE,
        ),
    },

    temperature_bounds=(1500.0, 4000.0),
    required_conditions=["temperature", "pressure", "composition"],
    template_family="idt_const_uv",
    default_pressure="1 atm",
)
