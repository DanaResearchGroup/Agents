import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

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
        return {
            "title": meta.get("title", "Unknown"),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", "Unknown"),
            "pdf_path": self.pdf_path
        }

    def find_figure_captions(self) -> List[FigureInfo]:
        found_figures = {}
        # Regex for common figure caption formats
        fig_pattern = re.compile(r'^(Figure|Fig\.|FIGURE)\s+(\d+[:\.]?)\s*(.*)', re.IGNORECASE)

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("blocks")
            
            for b in blocks:
                text = b[4].strip()
                match = fig_pattern.search(text)
                if match:
                    fig_label = f"{match.group(1)} {match.group(2)}".strip(':').strip('.')
                    # Normalize label to "Figure X"
                    norm_label = re.sub(r'^(Fig\.|FIGURE)', 'Figure', fig_label, flags=re.IGNORECASE)
                    
                    caption_text = text # The whole block is likely the caption
                    
                    # Extract context (nearby blocks)
                    context = text
                    
                    # If we haven't found this figure yet, or if this caption is longer (better), update it
                    if norm_label not in found_figures or len(caption_text) > len(found_figures[norm_label].caption):
                        found_figures[norm_label] = FigureInfo(
                            figure_id=norm_label,
                            caption=caption_text,
                            page_number=page_num + 1,
                            context_text=context
                        )
        
        return list(found_figures.values())

    def render_page(self, page_number: int) -> str:
        """Renders a page to an image and returns the path."""
        page = self.doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom for better quality
        img_path = os.path.join(self.images_dir, f"page_{page_number}.png")
        pix.save(img_path)
        return img_path

    def close(self):
        self.doc.close()

def parse_pdf(pdf_path: str, outdir: str) -> Dict[str, Any]:
    parser = PDFParser(pdf_path, outdir)
    try:
        metadata = parser.extract_metadata()
        figures = parser.find_figure_captions()
        
        # Render pages for all found figures
        # In MVP, we just render the whole page where the figure was found
        rendered_pages = {}
        for fig in figures:
            if fig.page_number not in rendered_pages:
                rendered_pages[fig.page_number] = parser.render_page(fig.page_number)
            fig.page_image_path = rendered_pages[fig.page_number]
            
        return {
            "metadata": metadata,
            "figures": [asdict(f) for f in figures]
        }
    finally:
        parser.close()
