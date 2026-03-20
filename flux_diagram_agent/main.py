import argparse
import logging
import os
import sys
from typing import Dict, Any

from agent.parser.pdf_parser import parse_pdf
from agent.figure_filter.candidate_filter import filter_figures
from agent.utils.llm_provider import get_llm_provider
from agent.classifier.figure_classifier import classify_figure
from agent.flux_interpreter.flux_interpreter import interpret_flux_diagram
from agent.aggregator.aggregator import aggregate_results
from agent.report.report_generator import generate_reports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("flux_agent")

def run_pipeline(pdf_path: str, outdir: str, use_llm: bool = True):
    logger.info(f"Starting pipeline for PDF: {pdf_path}")
    
    # 1. Parsing
    logger.info("Parsing PDF and extracting figures...")
    raw_data = parse_pdf(pdf_path, outdir)
    metadata = raw_data["metadata"]
    figures = raw_data["figures"]
    logger.info(f"Found {len(figures)} figures/captions.")

    # 2. Filtering
    logger.info("Filtering candidate figures...")
    candidates = filter_figures(figures)
    logger.info(f"Identified {len(candidates)} potential flux diagrams.")

    # 3. LLM Processing (Classification & Interpretation)
    processed_figures = []
    if use_llm:
        try:
            llm = get_llm_provider("deepseek")
            for fig in candidates:
                logger.info(f"Classifying {fig['figure_id']}...")
                classification = classify_figure(fig, llm)
                fig["classification"] = classification
                
                if classification.get("label") == "flux diagram" and classification.get("confidence", 0) > 0.5:
                    logger.info(f"Interpreting flux diagram for {fig['figure_id']}...")
                    analysis = interpret_flux_diagram(fig, llm)
                    fig["flux_analysis"] = analysis
                
                processed_figures.append(fig)
        except Exception as e:
            logger.error(f"LLM processing failed: {e}. Falling back to keyword-only results.")
            processed_figures = candidates
    else:
        logger.info("LLM processing disabled. Using keyword-only filter results.")
        processed_figures = candidates

    # 4. Aggregation
    logger.info("Aggregating results...")
    final_data = aggregate_results(metadata, processed_figures)

    # 5. Report Generation
    logger.info(f"Generating reports in {outdir}...")
    json_path, md_path = generate_reports(final_data, outdir)
    
    logger.info("Pipeline completed successfully.")
    logger.info(f"JSON report: {json_path}")
    logger.info(f"Markdown report: {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Flux Diagram Extraction Agent MVP")
    parser.add_argument("--pdf", required=True, help="Path to the PDF paper")
    parser.add_argument("--outdir", default="outputs", help="Output directory for reports and images")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM processing (fallback to keyword filtering only)")
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        sys.exit(1)

    try:
        run_pipeline(args.pdf, args.outdir, use_llm=not args.no_llm)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
