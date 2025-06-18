import argparse
import logging
from .agents.lead_researcher import LeadResearcher
from .evaluations.evaluators.artifact_evaluator import ArtifactEvaluator
import json
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a research report')
    
    # Required arguments
    parser.add_argument('query', type=str, help='Research query')
    
    # Optional arguments
    parser.add_argument('--mode', type=str, choices=['full', 'partial'], default='full',
                      help='Report generation mode: full or partial (default: full)')
    parser.add_argument('--sections', type=str, nargs='+',
                      help='Specific sections to include (for partial mode)')
    parser.add_argument('--depth', type=int, default=3,
                      help='Maximum search depth (default: 3)')
    parser.add_argument('--results', type=int, default=5,
                      help='Maximum results per search (default: 5)')
    parser.add_argument('--output', type=str,
                      help='Output file path (default: reports/report_<timestamp>.md)')
    parser.add_argument('--citation-style', type=str, choices=['chicago', 'apa'], default='chicago',
                      help='Citation style (default: chicago)')
    parser.add_argument('--citation-threshold', type=float, default=0.6,
                      help='Similarity threshold for citation matching (default: 0.6)')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts',
                      help='Directory to save research artifacts (default: artifacts)')
    
    return parser.parse_args()

def generate_report(args):
    """Generate a research report based on command line arguments."""
    # Create output directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Set up strategy based on mode
    strategy = {
        "mode": args.mode,
        "sections": args.sections or [
            "Introduction",
            "Background",
            "Methodology",
            "Findings",
            "Discussion",
            "Conclusion",
            "References"
        ],
        "max_depth": args.depth,
        "max_results": args.results,
        "citation_style": args.citation_style,
        "citation_threshold": args.citation_threshold
    }
    
    # Initialize researcher and artifact evaluator
    researcher = LeadResearcher(config={
        "citation_style": args.citation_style,
        "citation_threshold": args.citation_threshold
    })
    artifact_evaluator = ArtifactEvaluator(artifacts_dir=args.artifacts_dir)
    
    # Generate report
    logger.info(f"Generating report for query: {args.query}")
    report = researcher.forward(args.query, strategy)
    
    # Save report in markdown format
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/report_{timestamp}.md"
    
    with open(output_path, 'w') as f:
        f.write(report.to_markdown())
    
    # Save artifacts
    artifacts_path = artifact_evaluator.save_artifacts(
        query=args.query,
        search_results=report.metadata.get("search_results", []),
        analyses=report.metadata.get("analyses", {}),
        plan=report.metadata.get("plan", {}),
        report=report.dict()
    )
    
    # Print metrics
    citation_metrics = report.metadata.get("citation_metrics", {})
    logger.info("\nCitation Metrics:")
    logger.info(f"Total Citations: {citation_metrics.get('total_citations', 0)}")
    logger.info(f"Unique Citations: {citation_metrics.get('unique_citations', 0)}")
    logger.info(f"Sections with Citations: {citation_metrics.get('sections_with_citations', 0)}")
    
    logger.info(f"\nReport saved to: {output_path}")
    logger.info(f"Artifacts saved to: {artifacts_path}")
    
    return output_path, artifacts_path

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Create output directories
    output_dir = os.path.dirname(args.output) if args.output else 'reports'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(output_dir, f'report_{timestamp}.md')
    
    # Create research strategy
    strategy = {
        "mode": args.mode,
        "sections": args.sections or [
            "Introduction",
            "Background",
            "Methodology",
            "Findings",
            "Discussion",
            "Conclusion",
            "References"
        ],
        "max_depth": args.depth,
        "max_results": args.results,
        "citation_style": args.citation_style,
        "citation_threshold": args.citation_threshold
    }
    
    # Initialize researcher and artifact evaluator
    researcher = LeadResearcher(config={
        "citation_style": args.citation_style,
        "citation_threshold": args.citation_threshold
    })
    artifact_evaluator = ArtifactEvaluator(artifacts_dir=args.artifacts_dir)
    
    # Generate report
    logger.info(f"Generating report for query: {args.query}")
    report = researcher.forward(args.query, strategy)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report.to_markdown())
    
    # Save artifacts
    artifacts_path = artifact_evaluator.save_artifacts(
        query=args.query,
        search_results=report.metadata.get("search_results", []),
        analyses=report.metadata.get("analyses", {}),
        plan=report.metadata.get("plan", {}),
        report=report.dict()
    )
    
    # Print metrics
    citation_metrics = report.metadata.get("citation_metrics", {})
    logger.info("\nCitation Metrics:")
    logger.info(f"Total Citations: {citation_metrics.get('total_citations', 0)}")
    logger.info(f"Unique Citations: {citation_metrics.get('unique_citations', 0)}")
    logger.info(f"Sections with Citations: {citation_metrics.get('sections_with_citations', 0)}")
    
    logger.info(f"\nReport saved to: {args.output}")
    logger.info(f"Artifacts saved to: {artifacts_path}")

if __name__ == '__main__':
    main() 