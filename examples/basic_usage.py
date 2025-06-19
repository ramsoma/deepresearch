#!/usr/bin/env python3
"""
Basic usage example for Deep Research Agent.

This script demonstrates how to use the Deep Research Agent to generate
a research report with proper citations.
"""

import os
import sys
from pathlib import Path

from deep_research_agent.agents.lead_researcher import LeadResearcher

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Demonstrate basic usage of the Deep Research Agent."""

    # Set up environment variables (you should set these in your environment)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Please set it in your environment.")
        print("export GOOGLE_API_KEY='your-api-key'")
        return

    # Initialize the researcher
    print("Initializing Deep Research Agent...")
    researcher = LeadResearcher(
        config={"citation_style": "chicago", "citation_threshold": 0.6}
    )

    # Define research strategy
    strategy = {
        "mode": "full",
        "sections": [
            "Executive Summary",
            "Introduction",
            "Main Findings",
            "Analysis and Discussion",
            "Conclusions",
            "Recommendations",
            "References",
        ],
        "max_depth": 2,
        "max_results": 5,
        "citation_style": "chicago",
    }

    # Research query
    query = "What are the latest developments in quantum computing?"

    print(f"Generating research report for: {query}")
    print("This may take a few minutes...")

    try:
        # Generate the report
        report = researcher.forward(query, strategy)

        # Save the report
        output_file = "example_report.md"
        with open(output_file, "w") as f:
            f.write(report.to_markdown())

        print("\nReport generated successfully!")
        print(f"Report saved to: {output_file}")

        # Print some metadata
        print("\nReport Metadata:")
        print(f"Title: {report.title}")
        print(f"Number of sections: {len(report.sections)}")
        print(f"Total citations: {report.metadata.get('total_citations', 0)}")

        # Print section titles
        print("\nSections:")
        for section in report.sections:
            print(f"  - {section.title}")

        # Print citation metrics
        citation_metrics = report.metadata.get("citation_metrics", {})
        if citation_metrics:
            print("\nCitation Metrics:")
            print(f"  Total citations: {citation_metrics.get('total_citations', 0)}")
            print(f"  Unique citations: {citation_metrics.get('unique_citations', 0)}")
            print(
                f"  Sections with citations: {citation_metrics.get('sections_with_citations', 0)}"  # noqa
            )

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
