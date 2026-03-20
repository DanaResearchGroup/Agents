import json
import os
from typing import Dict, Any

class ReportGenerator:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def generate_json(self, data: Dict[str, Any], filename: str = "results.json"):
        path = os.path.join(self.outdir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def generate_markdown(self, data: Dict[str, Any], filename: str = "summary.md"):
        path = os.path.join(self.outdir, filename)
        
        paper = data.get("paper", {})
        summary = data.get("paper_summary", {})
        figures = data.get("figures", [])
        
        md = [
            f"# Flux Diagram Analysis: {paper.get('title', 'Unknown')}",
            "",
            "## Paper Summary",
            f"- **Found Flux Diagrams:** {'Yes' if summary.get('paper_has_flux_diagram') else 'No'}",
            f"- **Number of Flux Diagrams:** {summary.get('num_flux_figures', 0)}",
            f"- **Overall Usefulness:** {summary.get('overall_usefulness', 'N/A')}",
            "",
            "### Recommended Actions",
        ]
        
        for action in summary.get("recommended_actions", []):
            md.append(f"- {action}")
            
        md.append("\n## Detected Figures\n")
        
        for fig in figures:
            classification = fig.get("classification", {})
            md.append(f"### {fig.get('figure_id')} (Page {fig.get('page_number')})")
            
            # Embed image if available
            img_path = fig.get("page_image_path")
            if img_path:
                # Use relative path for the report
                rel_img_path = os.path.relpath(img_path, self.outdir)
                md.append(f"![{fig.get('figure_id')}]({rel_img_path})")
            
            md.append(f"\n**Caption:** {fig.get('caption')}")
            md.append(f"**Classification:** {classification.get('label')} (Confidence: {classification.get('confidence')})")
            md.append(f"**Reasoning:** {classification.get('reasoning')}")
            
            if classification.get("label") == "flux diagram":
                analysis = fig.get("flux_analysis", {})
                md.append("\n#### Flux Analysis")
                md.append(f"- **System:** {analysis.get('system')}")
                md.append(f"- **Conditions:** {analysis.get('conditions')}")
                md.append(f"- **Major Species:** {', '.join(analysis.get('major_species', []))}")
                md.append("\n**Dominant Pathways:**")
                for pw in analysis.get("dominant_pathways", []):
                    md.append(f"- {pw.get('from')} → {pw.get('to')} ({pw.get('importance')} importance)")
                    
            md.append("\n---")
            
        with open(path, "w") as f:
            f.write("\n".join(md))
        return path

def generate_reports(data: Dict[str, Any], outdir: str):
    gen = ReportGenerator(outdir)
    json_path = gen.generate_json(data)
    md_path = gen.generate_markdown(data)
    return json_path, md_path
