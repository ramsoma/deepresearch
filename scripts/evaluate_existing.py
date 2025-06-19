import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from deep_research_agent.evaluations.evaluators.artifact_evaluator import (
    ArtifactEvaluator,
)
from deep_research_agent.evaluations.evaluators.research_evaluator import (
    ResearchEvaluator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExistingEvaluator:
    def __init__(self, batch_dir: str):
        """Initialize evaluator with batch output directory."""
        self.batch_dir = Path(batch_dir)
        self.reports_dir = self.batch_dir / "reports"
        self.artifacts_dir = self.batch_dir / "artifacts"
        self.evaluations_dir = self.batch_dir / "evaluations"

        # Create evaluations directory if it doesn't exist
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.research_evaluator = ResearchEvaluator()
        self.artifact_evaluator = ArtifactEvaluator(
            artifacts_dir=str(self.artifacts_dir)
        )

    def _format_report_for_evaluation(self, report: Dict[str, Any]) -> str:
        """Format the report for evaluation."""
        sections = []

        # Add title
        sections.append(f"# {report['title']}\n")

        # Add each section
        for section in report["sections"]:
            sections.append(f"## {section['title']}\n")
            sections.append(section["content"])
            if section["key_points"]:
                sections.append("\nKey Points:")
                for point in section["key_points"]:
                    sections.append(f"- {point}")
            if section["citations"]:
                sections.append("\nCitations:")
                for citation in section["citations"]:
                    # Handle both old and new citation formats
                    if isinstance(citation, dict):
                        if "reference" in citation:
                            sections.append(
                                f"- {citation['text']}: {citation['reference']}"
                            )
                        elif "source" in citation:
                            sections.append(
                                f"- {citation['text']}: {citation['source']}"
                            )
                        else:
                            sections.append(f"- {citation['text']}")
                    else:
                        sections.append(f"- {citation}")
            sections.append("\n")

        return "\n".join(sections)

    def _evaluate_report(
        self, report_path: str, artifacts_path: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate a single report."""
        # Load artifacts
        artifacts = self.artifact_evaluator.load_artifacts(artifacts_path)

        # Format report for evaluation
        report = artifacts["report"]
        formatted_report = self._format_report_for_evaluation(report)

        # Prepare context
        context = {
            "query": artifacts["query"],
            "strategy": report.get("metadata", {}).get("strategy", {}),
            "citation_style": report.get("metadata", {}).get("citation_style", ""),
            "search_results": artifacts["search_results"],
            "analyses": artifacts["analyses"],
            "plan": artifacts["plan"],
            "metadata": artifacts["metadata"],
        }

        try:
            # Evaluate
            metrics = self.research_evaluator.evaluate(
                research_output=formatted_report,
                ground_truth=ground_truth,
                context=context,
            )

            # Convert metrics to dict, handling non-numeric values
            metrics_dict: Dict[str, Any] = {
                "accuracy": metrics.accuracy,
                "relevance": metrics.relevance,
                "coherence": metrics.coherence,
                "additional_metrics": {},
            }

            # Process additional metrics, converting non-numeric values to strings
            for key, value in metrics.additional_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_dict["additional_metrics"][key] = value
                else:
                    metrics_dict["additional_metrics"][f"{key}_text"] = str(value)

            return metrics_dict

        except Exception as e:
            logger.error(f"Error evaluating report: {str(e)}")
            return {
                "accuracy": 0.0,
                "relevance": 0.0,
                "coherence": 0.0,
                "additional_metrics": {"error": str(e)},
            }

    def evaluate_reports(self, ground_truth_file: str) -> None:
        """Evaluate all reports in the batch directory."""
        # Load ground truth
        with open(ground_truth_file, "r") as f:
            ground_truth_data = json.load(f)

        # Get all artifact files
        artifact_files = list(self.artifacts_dir.glob("*.json"))
        if not artifact_files:
            logger.error("No artifact files found!")
            return

        # Process each report
        results = []
        for report_path in self.reports_dir.glob("*.md"):
            # Use the first artifact file if we can't match by query
            artifacts_path = artifact_files[0]
            logger.info(f"Using artifact file: {artifacts_path}")

            # Get ground truth for this query
            query = report_path.stem.split("_")[1]  # Extract query from filename
            ground_truth = None

            # Try to find a matching ground truth by checking if the query is substring
            for item in ground_truth_data["queries"]:
                if query.lower() in item["query"].lower():
                    ground_truth = item.get("ground_truth")
                    logger.info(f"Found ground truth for query: {item['query']}")
                    break

            if not ground_truth:
                logger.warning(f"No ground truth found for query: {query}")
                continue

            logger.info(f"\nEvaluating report: {report_path}")
            evaluation = self._evaluate_report(
                str(report_path), str(artifacts_path), ground_truth
            )

            results.append(
                {
                    "query": query,
                    "report_path": str(report_path),
                    "artifacts_path": str(artifacts_path),
                    "evaluation": evaluation,
                }
            )

        # Generate evaluation report
        self._generate_evaluation_report(results)

    def _generate_evaluation_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate a summary report of all evaluations."""
        if not results:
            logger.warning("No evaluations to report on!")
            return

        # Create summary DataFrame
        summary_data = []
        for result in results:
            eval_data = result["evaluation"]
            row_data = {
                "Query": result["query"],
                "Accuracy": eval_data["accuracy"],
                "Relevance": eval_data["relevance"],
                "Coherence": eval_data["coherence"],
                "Report Path": result["report_path"],
                "Artifacts Path": result["artifacts_path"],
            }

            # Add additional metrics that are numeric
            for key, value in eval_data["additional_metrics"].items():
                if isinstance(value, (int, float)):
                    row_data[key] = value

            summary_data.append(row_data)

        df = pd.DataFrame(summary_data)

        # Calculate statistics for numeric columns only
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        stats = {}
        for col in numeric_cols:
            stats[f"Mean {col}"] = df[col].mean()
            stats[f"Best {col}"] = df[col].max()

        stats["Total Reports"] = len(df)

        # Save summary report
        summary_path = self.evaluations_dir / "evaluation_summary.md"
        with open(summary_path, "w") as f:
            f.write("# Evaluation Summary\n\n")
            f.write("## Statistics\n\n")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value:.3f}\n")
                else:
                    f.write(f"- {key}: {value}\n")

            f.write("\n## Detailed Results\n\n")
            f.write(df.to_markdown(index=False))

            # Add non-numeric feedback
            f.write("\n## Additional Feedback\n\n")
            for result in results:
                eval_data = result["evaluation"]
                f.write(f"### {result['query']}\n\n")
                for key, value in eval_data["additional_metrics"].items():
                    if not isinstance(value, (int, float)):
                        f.write(f"- {key}: {value}\n")
                f.write("\n")

        # Save detailed results as CSV
        df.to_csv(self.evaluations_dir / "detailed_results.csv", index=False)

        logger.info(f"\nEvaluation summary saved to: {summary_path}")
        logger.info(
            f"Detailed results saved to: {self.evaluations_dir / 'detailed_results.csv'}"  # noqa
        )


def main():
    """Main entry point for evaluating existing reports."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate existing reports in a batch output directory"
    )
    parser.add_argument("batch_dir", help="Path to batch output directory")
    parser.add_argument("ground_truth", help="Path to ground truth JSON file")
    args = parser.parse_args()

    evaluator = ExistingEvaluator(args.batch_dir)
    evaluator.evaluate_reports(args.ground_truth)


if __name__ == "__main__":
    main()
