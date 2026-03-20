from .models import PaperRecord
from typing import Literal

def assign_tags_and_priority(record: PaperRecord) -> None:
    """
    Assign tags and priority based on record state and score.
    """
    tags = []
    
    # Access Tags
    if record.oa_status == "oa":
        tags.append("oa")
    elif record.oa_status == "non_oa":
        tags.append("non_oa")
    else:
        tags.append("unknown_access")

    # Content Tags
    text = f"{record.title or ''} {record.abstract or ''}".lower()
    
    if "chemkin" in text:
        tags.append("possible_chemkin")
    if "cantera" in text:
        tags.append("possible_cantera")
    if "mechanism" in text:
        tags.append("possible_mechanism")
    if "kinetic" in text:
        tags.append("kinetics_relevant")
    if "supplementary" in text or record.si_link_found:
        tags.append("possible_si")
    
    # Relevance/Priority
    if record.relevance_score >= 0.7:
        record.priority = "high"
        tags.append("high_interest")
    elif record.relevance_score >= 0.4:
        record.priority = "medium"
        tags.append("medium_interest")
    else:
        record.priority = "low"
        tags.append("low_interest")

    # Manual Review
    if record.oa_status == "non_oa" and record.priority == "high":
        record.manual_review_needed = True
        record.keep_for_manual_review = True
        tags.append("manual_review_needed")

    record.tags = list(set(tags))
