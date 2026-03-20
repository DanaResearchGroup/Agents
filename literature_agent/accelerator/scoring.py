import re
from typing import Any
from .models import PaperRecord

# Scoring Heuristics
# 1. Species/Fuel Match (High)
# 2. Keywords like mechanism, kinetics, model (Medium)
# 3. Supplemental metadata like Chemkin, Cantera (High)

KEYWORD_SCORES = {
    "mechanism": 0.2,
    "kinetics": 0.2,
    "model": 0.1,
    "pyrolysis": 0.1,
    "oxidation": 0.1,
    "discrepancy": 0.1,
    "chemkin": 0.3,
    "cantera": 0.3,
    "supplementary": 0.2,
    "si": 0.1,
    "sensitivity": 0.1,
    "flux": 0.1,
}

def calculate_relevance(record: PaperRecord, query_keywords: list[str] or None = None) -> tuple[float, list[str]]:
    """
    Calculate a relevance score between 0 and 1.
    Returns (score, reasons)
    """
    score = 0.0
    reasons = []

    text = f"{record.title or ''} {record.abstract or ''}".lower()
    
    # Check for query keywords
    if query_keywords:
        for kw in query_keywords:
            if kw.lower() in text:
                score += 0.2
                reasons.append(f"Matched query keyword: {kw}")

    # Check for hardcoded important keywords
    for kw, point in KEYWORD_SCORES.items():
        if kw in text:
            score += point
            reasons.append(f"Matched keyword: {kw}")

    # Bonus for SI links already found
    if record.si_link_found:
        score += 0.2
        reasons.append("Supplementary information link found")

    # Bonus for mechanism links already found
    if record.mechanism_link_found:
        score += 0.3
        reasons.append("Mechanism link found")

    # Cap score at 1.0
    score = min(1.0, score)
    
    return score, reasons
