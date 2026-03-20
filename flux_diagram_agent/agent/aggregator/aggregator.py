from typing import List, Dict, Any

class Aggregator:
    def aggregate(self, metadata: Dict[str, Any], figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes results from all figures to create a paper-level summary."""
        flux_figures = [f for f in figures if f.get("classification", {}).get("label") == "flux diagram"]
        
        summary = {
            "paper_has_flux_diagram": len(flux_figures) > 0,
            "num_flux_figures": len(flux_figures),
            "best_figures": [f["figure_id"] for f in sorted(flux_figures, key=lambda x: x.get("classification", {}).get("confidence", 0), reverse=True)][:3],
            "overall_usefulness": self._calculate_overall_usefulness(flux_figures),
            "recommended_actions": self._generate_recommendations(flux_figures)
        }
        
        return {
            "paper": metadata,
            "figures": figures,
            "paper_summary": summary
        }

    def _calculate_overall_usefulness(self, flux_figures: List[Dict[str, Any]]) -> str:
        if not flux_figures:
            return "none"
        
        useful_count = sum(1 for f in flux_figures if f.get("flux_analysis", {}).get("usefulness") == "high")
        if useful_count > 0:
            return "high"
        return "medium" if len(flux_figures) > 0 else "low"

    def _generate_recommendations(self, flux_figures: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        if not flux_figures:
            recommendations.append("No flux diagrams were detected. Consider manual review if you suspect they were missed.")
        else:
            for f in flux_figures[:2]:
                recommendations.append(f"Review {f['figure_id']} for detailed pathway information.")
            recommendations.append("Compare extracted pathways with theoretical mechanism models.")
            
        return recommendations

def aggregate_results(metadata: Dict[str, Any], figures: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = Aggregator()
    return agg.aggregate(metadata, figures)
