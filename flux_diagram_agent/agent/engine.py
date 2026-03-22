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
        fig_pattern = re.compile(r'^(Figure|Fig\.|FIGURE)\s+(\d+[:\.]?)\s*(.*)', re.IGNORECASE)
        for page_num in range(len(self.doc)):
            blocks = self.doc[page_num].get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                match = fig_pattern.search(text)
                if match:
                    fig_label = f"{match.group(1)} {match.group(2)}".strip(':').strip('.')
                    norm_label = re.sub(r'^(Fig\.|FIGURE)', 'Figure', fig_label, flags=re.IGNORECASE)
                    if norm_label not in found_figures or len(text) > len(found_figures[norm_label].caption):
                        found_figures[norm_label] = FigureInfo(figure_id=norm_label, caption=text, page_number=page_num + 1, context_text=text)
        return list(found_figures.values())

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
        rendered_pages = {}
        for fig in figures:
            if fig.page_number not in rendered_pages: rendered_pages[fig.page_number] = parser.render_page(fig.page_number)
            fig.page_image_path = rendered_pages[fig.page_number]
        return {"metadata": metadata, "figures": [asdict(f) for f in figures]}
    finally: parser.close()

# --- Candidate Filter ---

class CandidateFilter:
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or ["flux", "reaction flux", "reaction path", "pathway", "dominant channel", "consumption pathway", "formation pathway", "path analysis", "reaction path analysis", "net flux", "integrated flux", "rate of production"]

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
        summary = {
            "paper_has_flux_diagram": len(flux_figures) > 0,
            "num_flux_figures": len(flux_figures),
            "best_figures": [f["figure_id"] for f in sorted(flux_figures, key=lambda x: x.get("classification", {}).get("confidence", 0), reverse=True)][:3],
            "overall_usefulness": self._calculate_overall_usefulness(flux_figures),
            "recommended_actions": self._generate_recommendations(flux_figures)
        }
        return {"paper": metadata, "figures": figures, "paper_summary": summary}

    def _calculate_overall_usefulness(self, flux_figures: List[Dict[str, Any]]) -> str:
        if not flux_figures: return "none"
        if any(f.get("flux_analysis", {}).get("usefulness") == "high" for f in flux_figures): return "high"
        return "medium" if len(flux_figures) > 0 else "low"

    def _generate_recommendations(self, flux_figures: List[Dict[str, Any]]) -> List[str]:
        if not flux_figures: return ["No flux diagrams were detected. Consider manual review if you suspect they were missed."]
        recs = [f"Review {f['figure_id']} for detailed pathway information." for f in flux_figures[:2]]
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
        md = [f"# Flux Diagram Analysis: {paper.get('title', 'Unknown')}", "", "## Paper Summary", f"- **Found Flux Diagrams:** {'Yes' if summary.get('paper_has_flux_diagram') else 'No'}", f"- **Number of Flux Diagrams:** {summary.get('num_flux_figures', 0)}", f"- **Overall Usefulness:** {summary.get('overall_usefulness', 'N/A')}", "", "### Recommended Actions"]
        for action in summary.get("recommended_actions", []): md.append(f"- {action}")
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
            md.append("\n---")
        with open(path, "w") as f: f.write("\n".join(md))
        return path

def generate_reports(data: Dict[str, Any], outdir: str):
    gen = ReportGenerator(outdir)
    return gen.generate_json(data), gen.generate_markdown(data)
