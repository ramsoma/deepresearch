import os
import sys
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from deep_research_agent.evaluations.evaluators.research_evaluator import ResearchEvaluator
from deep_research_agent.evaluations.evaluators.artifact_evaluator import ArtifactEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_artifacts(artifacts_path: str, ground_truth_path: Optional[str] = None) -> None:
    """Evaluate research artifacts."""
    # Initialize evaluators
    evaluator = ResearchEvaluator()
    artifact_evaluator = ArtifactEvaluator()
    
    # Load artifacts
    artifacts = artifact_evaluator.load_artifacts(artifacts_path)
    
    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path:
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = f.read()
        except Exception as e:
            logger.warning(f"Could not load ground truth: {str(e)}")
    
    # Format report for evaluation
    report = artifacts["report"]
    formatted_report = format_report_for_evaluation(report)
    
    # Prepare context with all artifacts
    context = {
        "query": artifacts["query"],
        "strategy": report.get('metadata', {}).get('strategy', {}),
        "citation_style": report.get('metadata', {}).get('citation_style', ''),
        "search_results": artifacts["search_results"],
        "analyses": artifacts["analyses"],
        "plan": artifacts["plan"],
        "metadata": artifacts["metadata"]
    }
    
    # Evaluate
    logger.info(f"Evaluating research artifacts: {artifacts_path}")
    metrics = evaluator.evaluate(
        research_output=formatted_report,
        ground_truth=ground_truth,
        context=context
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"Report: {report['title']}")
    print(f"Query: {artifacts['query']}")
    print(f"Timestamp: {artifacts['timestamp']}")
    print("\nMetrics:")
    print(f"Accuracy: {metrics.accuracy:.2f}")
    print(f"Relevance: {metrics.relevance:.2f}")
    print(f"Coherence: {metrics.coherence:.2f}")
    print("\nAdditional Metrics:")
    for key, value in metrics.additional_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"- {item}")
        else:
            print(f"{key}: {value}")
    
    # Print artifact statistics
    print("\nArtifact Statistics:")
    print(f"Total Search Results: {artifacts['metadata']['total_search_results']}")
    print(f"Topics Analyzed: {', '.join(artifacts['metadata']['topics_analyzed'])}")
    print(f"Report Sections: {', '.join(artifacts['metadata']['report_sections'])}")
    print(f"Total Citations: {artifacts['metadata']['total_citations']}")

def format_report_for_evaluation(report: Dict[str, Any]) -> str:
    """Format the report for evaluation."""
    sections = []
    
    # Add title
    sections.append(f"# {report['title']}\n")
    
    # Add each section
    for section in report['sections']:
        sections.append(f"## {section['title']}\n")
        sections.append(section['content'])
        if section['key_points']:
            sections.append("\nKey Points:")
            for point in section['key_points']:
                sections.append(f"- {point}")
        if section['citations']:
            sections.append("\nCitations:")
            for citation in section['citations']:
                sections.append(f"- {citation['text']}: {citation['reference']}")
        sections.append("\n")
    
    return "\n".join(sections)

def main():
    """Main function to evaluate research artifacts."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate research artifacts')
    parser.add_argument('artifacts_path', help='Path to the artifacts JSON file')
    parser.add_argument('--ground-truth', help='Path to ground truth file (optional)')
    args = parser.parse_args()
    
    evaluate_artifacts(args.artifacts_path, args.ground_truth)

if __name__ == "__main__":
    main() 