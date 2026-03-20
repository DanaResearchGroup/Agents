import pytest
from unittest.mock import MagicMock, patch
from agent.parser.pdf_parser import PDFParser

@patch("fitz.open")
def test_find_figure_captions(mock_open):
    # Mocking fitz.Document and fitz.Page
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_open.return_value = mock_doc
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    
    # Sample page text with a figure caption
    mock_page.get_text.return_value = "This is some text.\nFigure 1: Reaction flux diagram of H2 oxidation.\nMore text here."
    
    parser = PDFParser("dummy.pdf", "outputs")
    figures = parser.find_figure_captions()
    
    assert len(figures) == 1
    assert figures[0].figure_id == "Figure 1"
    assert "Reaction flux diagram" in figures[0].caption
    assert "H2 oxidation" in figures[0].caption

def test_keyword_filtering():
    from agent.figure_filter.candidate_filter import CandidateFilter
    
    figures = [
        {"figure_id": "Fig 1", "caption": "Flux diagram of ethanol", "context_text": "text"},
        {"figure_id": "Fig 2", "caption": "Pressure profile", "context_text": "text"},
    ]
    
    filter_obj = CandidateFilter()
    candidates = filter_obj.filter_candidates(figures)
    
    assert len(candidates) == 1
    assert candidates[0]["figure_id"] == "Fig 1"
    assert candidates[0]["candidate_score"] > 0
