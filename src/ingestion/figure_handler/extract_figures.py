"""Extract figure images from PDF pages using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_figure_image(
    pdf_path: str,
    page_num: int,
    output_path: str | None = None,
    dpi: int = 200,
) -> bytes | None:
    """Extract a full page render as PNG image bytes.

    For targeted figure extraction, we render the full page at high DPI
    and return as bytes. The caller can crop to a specific figure region
    if needed.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-indexed page number.
        output_path: If provided, also save to this file path.
        dpi: Resolution for rendering (default 200).

    Returns:
        PNG image bytes, or None on failure.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > len(doc):
            return None

        page = doc[page_num - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(img_bytes)

        return img_bytes
    finally:
        doc.close()


def extract_all_images_from_page(
    pdf_path: str,
    page_num: int,
) -> list[dict]:
    """Extract embedded images from a specific PDF page.

    Returns list of dicts with 'xref', 'width', 'height', 'image_bytes'.
    These are the actual embedded images (figures), not a page render.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > len(doc):
            return []

        page = doc[page_num - 1]
        images = []

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if base_image:
                images.append({
                    "xref": xref,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "ext": base_image["ext"],
                    "image_bytes": base_image["image"],
                })

        return images
    finally:
        doc.close()
