import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from agent.engine import parse_pdf, filter_figures, aggregate_results, generate_reports
from agent.engine import parse_pdf, filter_figures, aggregate_results, generate_reports
from agent.brain import get_llm_provider, classify_figure, interpret_flux_diagram, interpret_sensitivity_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("flux_agent")

def run_pipeline(pdf_path: str, outdir: str, use_llm: bool = True, provider: str = "deepseek", model: Optional[str] = None):
    logger.info(f"Starting pipeline for PDF: {pdf_path} using {provider} ({model or 'default'})")
    
    # 1. Parsing
    logger.info("Parsing PDF and extracting figures...")
    raw_data = parse_pdf(pdf_path, outdir)
    metadata = raw_data["metadata"]
    figures = raw_data["figures"]
    logger.info(f"Found {len(figures)} figures/captions.")

    # 2. Filtering
    logger.info("Filtering candidate figures...")
    candidates = filter_figures(figures)
    logger.info(f"Identified {len(candidates)} potential candidates (flux or sensitivity).")

    # 3. LLM Processing (Classification & Interpretation)
    processed_figures = []
    if use_llm:
        try:
            llm = get_llm_provider(provider, model)
            for fig in candidates:
                logger.info(f"Classifying {fig['figure_id']}...")
                classification = classify_figure(fig, llm)
                fig["classification"] = classification
                
                if classification.get("label") == "flux diagram" and classification.get("confidence", 0) > 0.5:
                    logger.info(f"Interpreting flux diagram for {fig['figure_id']}...")
                    analysis = interpret_flux_diagram(fig, llm)
                    fig["flux_analysis"] = analysis
                
                elif classification.get("label") == "sensitivity analysis" and classification.get("confidence", 0) > 0.5:
                    logger.info(f"Interpreting sensitivity analysis for {fig['figure_id']}...")
                    analysis = interpret_sensitivity_analysis(fig, llm)
                    fig["sensitivity_analysis"] = analysis
                
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
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "glm"], help="LLM provider to use")
    parser.add_argument("--model", help="Specific model to use (e.g. glm-4v-plus, deepseek-reasoner)")
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        sys.exit(1)

    # Auto-generate outdir based on PDF filename if still at default
    outdir = args.outdir
    if outdir == "outputs":
        base_name = os.path.splitext(os.path.basename(args.pdf))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("outputs", f"{base_name}_{timestamp}")
    
    try:
        run_pipeline(args.pdf, outdir, use_llm=not args.no_llm, provider=args.provider, model=args.model)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
