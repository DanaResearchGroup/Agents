from typing import List, Dict, Any

class CandidateFilter:
    def __init__(self, keywords: List[str] = None):
        if keywords is None:
            self.keywords = [
                "flux", "reaction flux", "reaction path", "pathway",
                "dominant channel", "consumption pathway", "formation pathway",
                "path analysis", "reaction path analysis", "net flux",
                "integrated flux", "rate of production"
            ]
        else:
            self.keywords = keywords

    def score_figure(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a figure based on keyword matches in caption and context."""
        score = 0
        reasons = []
        
        text_to_search = (figure.get("caption", "") + " " + figure.get("context_text", "")).lower()
        
        for kw in self.keywords:
            if kw.lower() in text_to_search:
                # Direct match in caption is worth more
                if kw.lower() in figure.get("caption", "").lower():
                    score += 0.5
                else:
                    score += 0.2
                reasons.append(f"found keyword: '{kw}'")
        
        # Normalize score to be between 0 and 1 (roughly)
        # But for MVP, simple weighting is fine
        final_score = min(score, 1.0)
        
        figure["candidate_score"] = round(final_score, 2)
        figure["candidate_reasons"] = reasons
        return figure

    def filter_candidates(self, figures: List[Dict[str, Any]], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Filters and scores all figures."""
        scored_figures = [self.score_figure(f) for f in figures]
        # Sort by score descending
        candidates = [f for f in scored_figures if f["candidate_score"] >= threshold]
        return sorted(candidates, key=lambda x: x["candidate_score"], reverse=True)

def filter_figures(figures: List[Dict[str, Any]], threshold: float = 0.1) -> List[Dict[str, Any]]:
    filter_obj = CandidateFilter()
    return filter_obj.filter_candidates(figures, threshold)
