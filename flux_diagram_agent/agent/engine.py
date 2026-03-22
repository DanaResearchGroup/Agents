import fitz
import re
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# --- PDF Parser ---

@dataclass
class FigureInfo:
    figure_id: str
    caption: str
    page_number: int
    context_text: str
    caption_rect: List[float]
    page_image_path: Optional[str] = None

class PDFParser:
    def __init__(self, pdf_path: str, outdir: str):
        self.pdf_path = pdf_path
        self.outdir = outdir
        self.doc = fitz.open(pdf_path)
        self.images_dir = os.path.join(outdir, "figures")
        os.makedirs(self.images_dir, exist_ok=True)

    def extract_metadata(self) -> Dict[str, str]:
        meta = self.doc.metadata
        return {"title": meta.get("title", "Unknown"), "author": meta.get("author", "Unknown"), "subject": meta.get("subject", "Unknown"), "pdf_path": self.pdf_path}

    def find_figure_captions(self) -> List[FigureInfo]:
        found_figures = {}
        fig_pattern = re.compile(r'^(Figure|Fig\.|FIGURE)\s+([A-Z]?\d+[:\.]?)\s*(.*)', re.IGNORECASE)
        for page_num in range(len(self.doc)):
            blocks = self.doc[page_num].get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                match = fig_pattern.search(text)
                if match:
                    fig_label = f"{match.group(1)} {match.group(2)}".strip(':').strip('.')
                    norm_label = re.sub(r'^(Fig\.|FIGURE)', 'Figure', fig_label, flags=re.IGNORECASE)
                    if norm_label not in found_figures or len(text) > len(found_figures[norm_label].caption):
                        found_figures[norm_label] = FigureInfo(
                            figure_id=norm_label, 
                            caption=text, 
                            page_number=page_num + 1, 
                            context_text=text,
                            caption_rect=list(b[:4])
                        )
        return list(found_figures.values())

    def render_figure(self, figure: FigureInfo, top_y: float) -> str:
        """Renders a horizontal strip of the page as an individual figure image."""
        page = self.doc[figure.page_number - 1]
        width = page.rect.width
        # Create a crop rect from top_y to the bottom of the current caption
        # Adding a small margin
        bottom_y = min(figure.caption_rect[3] + 10, page.rect.height)
        crop_rect = fitz.Rect(0, top_y, width, bottom_y)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect)
        safe_id = figure.figure_id.replace(" ", "_").replace(".", "")
        img_path = os.path.join(self.images_dir, f"{safe_id}.png")
        pix.save(img_path)
        return img_path

    def render_page(self, page_number: int) -> str:
        page = self.doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = os.path.join(self.images_dir, f"page_{page_number}.png")
        pix.save(img_path)
        return img_path

    def close(self): self.doc.close()

def parse_pdf(pdf_path: str, outdir: str) -> Dict[str, Any]:
    parser = PDFParser(pdf_path, outdir)
    try:
        metadata = parser.extract_metadata()
        figures = parser.find_figure_captions()
        
        # Sort figures by page and then by Y-coordinate for smart cropping
        figures.sort(key=lambda f: (f.page_number, f.caption_rect[1]))
        
        last_page = -1
        prev_y = 0
        for fig in figures:
            if fig.page_number != last_page:
                last_page = fig.page_number
                prev_y = 0 # Reset Y for new page
            
            # Render individual figure strip
            fig.page_image_path = parser.render_figure(fig, prev_y)
            # Next figure's top is the bottom of this one's caption
            prev_y = min(fig.caption_rect[3] + 5, parser.doc[fig.page_number-1].rect.height)
            
        return {"metadata": metadata, "figures": [asdict(f) for f in figures]}
    finally: parser.close()

# --- Candidate Filter ---

class CandidateFilter:
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or [
            "flux", "reaction flux", "reaction path", "pathway",
            "dominant channel", "consumption pathway", "formation pathway",
            "path analysis", "reaction path analysis", "net flux",
            "integrated flux", "rate of production",
            "sensitivity", "S-index", "sensitive reactions", "O'Conaire",
            "Brute-force sensitivity"
        ]

    def score_figure(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        score, reasons = 0, []
        text_to_search = (figure.get("caption", "") + " " + figure.get("context_text", "")).lower()
        for kw in self.keywords:
            if kw.lower() in text_to_search:
                score += 0.5 if kw.lower() in figure.get("caption", "").lower() else 0.2
                reasons.append(f"found keyword: '{kw}'")
        figure["candidate_score"] = round(min(score, 1.0), 2)
        figure["candidate_reasons"] = reasons
        return figure

    def filter_candidates(self, figures: List[Dict[str, Any]], threshold: float = 0.1) -> List[Dict[str, Any]]:
        scored_figures = [self.score_figure(f) for f in figures]
        candidates = [f for f in scored_figures if f["candidate_score"] >= threshold]
        return sorted(candidates, key=lambda x: x["candidate_score"], reverse=True)

def filter_figures(figures: List[Dict[str, Any]], threshold: float = 0.1) -> List[Dict[str, Any]]:
    return CandidateFilter().filter_candidates(figures, threshold)

# --- Aggregator ---

class Aggregator:
    def aggregate(self, metadata: Dict[str, Any], figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        flux_figures = [f for f in figures if f.get("classification", {}).get("label") == "flux diagram"]
        sens_figures = [f for f in figures if f.get("classification", {}).get("label") == "sensitivity analysis"]
        
        summary = {
            "paper_has_flux_diagram": len(flux_figures) > 0,
            "num_flux_figures": len(flux_figures),
            "paper_has_sensitivity_analysis": len(sens_figures) > 0,
            "num_sensitivity_figures": len(sens_figures),
            "best_figures": [f["figure_id"] for f in sorted(flux_figures + sens_figures, key=lambda x: x.get("classification", {}).get("confidence", 0), reverse=True)][:3],
            "overall_usefulness": self._calculate_overall_usefulness(flux_figures + sens_figures),
            "recommended_actions": self._generate_recommendations(flux_figures, sens_figures),
            "global_reaction_priority": self._synthesize_priority_list(flux_figures, sens_figures)
        }
        return {"paper": metadata, "figures": figures, "paper_summary": summary}

    def _synthesize_priority_list(self, flux_figures: List[Dict[str, Any]], sens_figures: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped_priority = {}
        
        # 1. Process Flux Pathways (Grouped as "Overall Mechanism Flow")
        flux_key = "Overall Mechanism Flow"
        if flux_figures:
            if flux_key not in grouped_priority: grouped_priority[flux_key] = {}
            for fig in flux_figures:
                for pw in fig.get("flux_analysis", {}).get("dominant_pathways", []):
                    rxn_str = f"{pw.get('from')} -> {pw.get('to')}"
                    if rxn_str not in grouped_priority[flux_key]:
                        grouped_priority[flux_key][rxn_str] = {"reaction": rxn_str, "importance_score": 0, "sources": []}
                    
                    weight = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(pw.get("importance", "low"), 0.2)
                    grouped_priority[flux_key][rxn_str]["importance_score"] += weight
                    grouped_priority[flux_key][rxn_str]["sources"].append(f"{fig['figure_id']}")

        # 2. Process Sensitivity Reactions (Grouped by target property)
        for fig in sens_figures:
            analysis = fig.get("sensitivity_analysis", {})
            target = analysis.get("target_property", "General Sensitivity")
            if target not in grouped_priority: grouped_priority[target] = {}
            
            for rxn in analysis.get("top_reactions", []):
                rxn_str = rxn.get("reaction")
                if not rxn_str or rxn_str.lower() == "unknown": continue
                
                if rxn_str not in grouped_priority[target]:
                    grouped_priority[target][rxn_str] = {"reaction": rxn_str, "importance_score": 0, "sources": []}
                
                # Assume top reactions in priority/sensitivity lists are important
                grouped_priority[target][rxn_str]["importance_score"] += 1.0 
                grouped_priority[target][rxn_str]["sources"].append(f"{fig['figure_id']}")

        # Sort each group by importance score
        final_ranking = {}
        for target, reactions in grouped_priority.items():
            final_ranking[target] = sorted(reactions.values(), key=lambda x: x["importance_score"], reverse=True)
            
        return final_ranking

    def _calculate_overall_usefulness(self, flux_figures: List[Dict[str, Any]]) -> str:
        if not flux_figures: return "none"
        if any(f.get("flux_analysis", {}).get("usefulness") == "high" for f in flux_figures): return "high"
        return "medium" if len(flux_figures) > 0 else "low"

    def _generate_recommendations(self, flux_figures: List[Dict[str, Any]], sens_figures: List[Dict[str, Any]]) -> List[str]:
        if not flux_figures and not sens_figures:
            return ["No flux diagrams or sensitivity analyses were detected. Consider manual review if you suspect they were missed."]
        
        recs = []
        if flux_figures:
            recs.append(f"Review {flux_figures[0]['figure_id']} for detailed pathway information.")
        if sens_figures:
            recs.append(f"Review {sens_figures[0]['figure_id']} for key sensitive reactions.")
        
        recs.append("Compare extracted pathways with theoretical mechanism models.")
        return recs

def aggregate_results(metadata: Dict[str, Any], figures: List[Dict[str, Any]]) -> Dict[str, Any]:
    return Aggregator().aggregate(metadata, figures)

# --- Report Generator ---

class ReportGenerator:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def generate_json(self, data: Dict[str, Any], filename: str = "results.json") -> str:
        path = os.path.join(self.outdir, filename)
        with open(path, "w") as f: json.dump(data, f, indent=2)
        return path

    def generate_markdown(self, data: Dict[str, Any], filename: str = "summary.md") -> str:
        path = os.path.join(self.outdir, filename)
        paper, summary, figures = data.get("paper", {}), data.get("paper_summary", {}), data.get("figures", [])
        md = [
            f"# Analysis Report: {paper.get('title', 'Unknown')}", "", 
            "## Paper Summary", 
            f"- **Found Flux Diagrams:** {'Yes' if summary.get('paper_has_flux_diagram') else 'No'} ({summary.get('num_flux_figures', 0)})",
            f"- **Found Sensitivity Analysis:** {'Yes' if summary.get('paper_has_sensitivity_analysis') else 'No'} ({summary.get('num_sensitivity_figures', 0)})",
            f"- **Overall Usefulness:** {summary.get('overall_usefulness', 'N/A')}", 
            "", "### Recommended Actions"
        ]
        for action in summary.get("recommended_actions", []): md.append(f"- {action}")
        
        grouped_priority = summary.get("global_reaction_priority", {})
        if grouped_priority:
            md.append("\n## Reaction Priority Ranking (Grouped by Target)")
            md.append("Key reactions extracted and ranked by their influence on specific targets/observables:")
            
            for target, reactions in grouped_priority.items():
                md.append(f"\n### Target: {target}")
                for i, item in enumerate(reactions[:10]):
                    md.append(f"{i+1}. **{item['reaction']}** (Score: {item['importance_score']:.1f})")
                    md.append(f"   - Identified in: {', '.join(set(item['sources']))}")

        md.append("\n## Detected Figures\n")
        for fig in figures:
            classification = fig.get("classification", {})
            md.append(f"### {fig.get('figure_id')} (Page {fig.get('page_number')})")
            img_path = fig.get("page_image_path")
            if img_path: md.append(f"![{fig.get('figure_id')}]({os.path.relpath(img_path, self.outdir)})")
            md.append(f"\n**Caption:** {fig.get('caption')}\n**Classification:** {classification.get('label')} (Confidence: {classification.get('confidence')})\n**Reasoning:** {classification.get('reasoning')}")
            
            if classification.get("label") == "flux diagram":
                analysis = fig.get("flux_analysis", {})
                md.append("\n#### Flux Analysis")
                md.append(f"- **System:** {analysis.get('system')}\n- **Conditions:** {analysis.get('conditions')}\n- **Major Species:** {', '.join(analysis.get('major_species', []))}\n\n**Dominant Pathways:**")
                for pw in analysis.get("dominant_pathways", []): md.append(f"- {pw.get('from')} → {pw.get('to')} ({pw.get('importance')} importance)")
            
            elif classification.get("label") == "sensitivity analysis":
                analysis = fig.get("sensitivity_analysis", {})
                md.append("\n#### Sensitivity Analysis")
                md.append(f"- **Target Property:** {analysis.get('target_property')}")
                md.append(f"- **Conditions:** {analysis.get('conditions')}")
                md.append("\n**Top Sensitive Reactions:**")
                for rxn in analysis.get("top_reactions", []):
                    md.append(f"- {rxn.get('reaction')}: {rxn.get('coefficient')} ({rxn.get('impact')})")
            
            md.append("\n---")
        with open(path, "w") as f: f.write("\n".join(md))
        return path

def generate_reports(data: Dict[str, Any], outdir: str):
    gen = ReportGenerator(outdir)
    return gen.generate_json(data), gen.generate_markdown(data)
