"""Flow reactor experiment family rules."""

import re
from .base import FamilyRules

flow_reactor = FamilyRules(
    name="flow_reactor",

    anchor_patterns=[
        re.compile(
            r'\b(?:flow\s+reactor|plug[\-\s]?flow(?:\s+reactor)?|PFR|'
            r'laminar[\-\s]?flow\s+reactor)\b',
            re.IGNORECASE,
        ),
    ],

    observable_patterns={
        "species_profile": re.compile(
            r'(?:speciation|species|concentration)\s+pro[fﬁ]le',
            re.IGNORECASE,
        ),
        "conversion": re.compile(
            r'(?:fuel\s+)?conversion|'
            r'product\s+(?:yield|distribution)',
            re.IGNORECASE,
        ),
        "temperature_profile": re.compile(
            r'temperature\s+pro[fﬁ]le',
            re.IGNORECASE,
        ),
    },

    temperature_bounds=(300.0, 2000.0),
    required_conditions=["temperature", "pressure", "composition"],
    template_family="pfr_const_tp",
    default_pressure=None,
)
