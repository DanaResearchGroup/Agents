"""PDF text extraction using PyMuPDF (fitz)."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from models import PageText, PaperDocument, FigureCaption
from .caption_extractor import extract_captions
from .table_parser import parse_tables


def parse_pdf(pdf_path: str) -> PaperDocument:
    """Extract structured text from a PDF. No OCR, no images."""
    doc = fitz.open(pdf_path)
    try:
        pages = _extract_pages(doc)
        title = _extract_title(doc, pages)
        abstract = _extract_abstract(pages)
        captions = extract_captions(pages)
        tables = parse_tables(pages, captions)
        return PaperDocument(
            pdf_path=str(Path(pdf_path).resolve()),
            title=title,
            abstract=abstract,
            pages=pages,
            captions=captions,
            tables=tables,
        )
    finally:
        doc.close()


def _extract_pages(doc: fitz.Document) -> list[PageText]:
    """Extract text from each page, 1-indexed."""
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text").rstrip()
        text = _normalize_text(text)
        pages.append(PageText(page_num=i + 1, text=text))
    return pages


def _normalize_text(text: str) -> str:
    """Fix common fitz extraction artifacts."""
    # Collapse spaces within digit sequences: "210 0" -> "2100", "30 0 0" -> "3000"
    # Matches a digit, then one or more occurrences of (single space + 1-3 digits)
    # Only collapses when the fragments look like a split number, not real spacing
    text = re.sub(r'(\d) (?=\d{1,3}(?:\s|\b|[^0-9]))', r'\1', text)
    return text


def _extract_title(doc: fitz.Document, pages: list[PageText]) -> str | None:
    """Try to extract the paper title from metadata or first page."""
    # Strategy A: PDF metadata
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title and len(meta_title) > 5 and not _is_generic_title(meta_title):
        return meta_title

    # Strategy B: score candidate lines from first page
    if not pages:
        return None

    candidates: list[tuple[int, str]] = []
    for line in pages[0].text.splitlines()[:25]:
        line = line.strip()
        if len(line) < 10 or len(line) > 250:
            continue
        if _is_header_line(line):
            continue

        score = 0
        if line[0].isupper():
            score += 1
        if not re.search(r'\b(received|accepted|doi|journal|abstract)\b', line, re.I):
            score += 1
        if len(line.split()) >= 4:
            score += 1
        if not re.search(r'@\w+|\bUniversity\b|\bDepartment\b', line, re.I):
            score += 1

        candidates.append((score, line))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_abstract(pages: list[PageText]) -> str | None:
    """Extract abstract text from the first 2 pages."""
    search_text = "\n".join(p.text for p in pages[:2])

    # Try explicit section-bounded match first
    explicit = re.search(
        r'(?is)\babstract\b[:\s.\-]*\n?'
        r'(.*?)'
        r'(?=\n\s*(?:keywords?|introduction|1[\.\s]+[A-Z]|experimental|methods)\b)',
        search_text,
    )
    if explicit:
        abstract = re.sub(r'\s+', ' ', explicit.group(1)).strip()
        if len(abstract) > 20:
            return abstract

    # Fallback: take bounded chunk after "Abstract" and split at likely headers
    fallback = re.search(r'(?is)\babstract\b[:\s.\-]*\n?(.*)', search_text)
    if fallback:
        chunk = fallback.group(1)[:2500]
        chunk = re.split(
            r'\n\s*(?:keywords?|introduction|1[\.\s]+[A-Z]|experimental|methods)\b',
            chunk, maxsplit=1, flags=re.I,
        )[0]
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(chunk) > 20:
            return chunk

    return None


def _is_generic_title(title: str) -> bool:
    """Check if a metadata title is generic/useless."""
    lower = title.lower()
    return any(term in lower for term in (
        "untitled", "microsoft word", ".pdf", ".doc",
        "elsevier", "manuscript",
    ))


def _is_header_line(line: str) -> bool:
    """Check if a line looks like a journal header rather than a title."""
    lower = line.lower()
    return any(term in lower for term in (
        "volume", "vol.", "issue", "doi:", "http", "journal",
        "received", "accepted", "published", "available online",
        "copyright", "elsevier", "springer", "wiley",
        "issn", "pp.", "pages",
    ))
