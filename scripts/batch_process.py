import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from deep_research_agent.cli import generate_report
from deep_research_agent.core.logging_config import get_logger, setup_logging
from deep_research_agent.evaluations.evaluators.artifact_evaluator import (
    ArtifactEvaluator,
)
from deep_research_agent.evaluations.evaluators.research_evaluator import (
    ResearchEvaluator,
)

# Set up logging
setup_logging(log_level="INFO", log_dir="logs")
logger = get_logger(__name__)


class BatchProcessor:
    def __init__(self, config_path: str):
        """Initialize batch processor with config file."""
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"batch_output_{self.timestamp}")
        self.reports_dir = self.output_dir / "reports"
        self.artifacts_dir = self.output_dir / "artifacts"
        self.evaluations_dir = self.output_dir / "evaluations"

        # Create output directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.research_evaluator = ResearchEvaluator()
        self.artifact_evaluator = ArtifactEvaluator(
            artifacts_dir=str(self.artifacts_dir)
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)

    def process_queries(self) -> List[Dict[str, Any]]:
        """Process all queries in the config and return results."""
        results = []

        for query_config in self.config["queries"]:
            query = query_config["query"]
            ground_truth = query_config.get("ground_truth")

            logger.info(f"\nProcessing query: {query}")

            # Generate report
            report_path, artifacts_path = generate_report(
                args=type(
                    "Args",
                    (),
                    {
                        "query": query,
                        "mode": query_config.get("mode", "full"),
                        "sections": query_config.get("sections"),
                        "depth": query_config.get("depth", 3),
                        "results": query_config.get("results", 5),
                        "output": str(
                            self.reports_dir
                            / f"report_{query[:30]}_{self.timestamp}.md"
                        ),
                        "citation_style": query_config.get("citation_style", "chicago"),
                        "citation_threshold": query_config.get(
                            "citation_threshold", 0.6
                        ),
                        "artifacts_dir": str(self.artifacts_dir),
                    },
                )
            )

            # Evaluate if ground truth provided
            evaluation = None
            evaluation = self._evaluate_report(
                report_path, artifacts_path, ground_truth
            )

            results.append(
                {
                    "query": query,
                    "report_path": report_path,
                    "artifacts_path": artifacts_path,
                    "evaluation": evaluation,
                }
            )

        return results

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

    def generate_evaluation_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate a summary report of all evaluations."""
        # Filter results with evaluations
        evaluated_results = [r for r in results if r["evaluation"] is not None]

        if not evaluated_results:
            logger.warning("No evaluations to report on!")
            return

        # Create summary DataFrame
        summary_data = []
        for result in evaluated_results:
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
            for result in evaluated_results:
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
            f"Detailed results saved to: {self.evaluations_dir / 'detailed_results.csv'}"  # noqa: E501
        )


def main():
    """Main entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process multiple research queries"
    )
    parser.add_argument("config", help="Path to configuration JSON file")
    args = parser.parse_args()

    processor = BatchProcessor(args.config)
    results = processor.process_queries()
    processor.generate_evaluation_report(results)


if __name__ == "__main__":
    main()
