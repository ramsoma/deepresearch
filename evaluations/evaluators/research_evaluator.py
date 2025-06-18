import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    @abstractmethod
    def evaluate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the report and return results."""
        pass

class ResearchContentEvaluator(BaseEvaluator):
    """Evaluates the quality of research content."""
    
    def evaluate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate research content quality."""
        results = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "confidence_score": 0.0
        }
        
        # Check content quality
        for section in report.get("sections", []):
            content = section.get("content", "")
            key_points = section.get("key_points", [])
            
            # Evaluate content length
            if len(content.split()) < 100:
                results["weaknesses"].append(f"Section '{section['title']}' is too short")
            
            # Evaluate key points
            if not key_points:
                results["weaknesses"].append(f"Section '{section['title']}' has no key points")
            
            # Check for technical depth
            if "technical" in section["title"].lower() and len(content.split()) < 200:
                results["weaknesses"].append(f"Technical section '{section['title']}' lacks depth")
        
        # Calculate confidence score
        total_checks = len(report.get("sections", [])) * 3  # content length, key points, technical depth
        passed_checks = total_checks - len(results["weaknesses"])
        results["confidence_score"] = passed_checks / total_checks if total_checks > 0 else 0
        
        return results

class ReportStructureEvaluator(BaseEvaluator):
    """Evaluates the technical structure of the report."""
    
    REQUIRED_SECTIONS = [
        "Executive Summary",
        "Introduction",
        "Main Findings",
        "Analysis and Discussion",
        "Conclusions",
        "Recommendations",
        "References"
    ]
    
    def evaluate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate report structure."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "section_coverage": {
                "required": self.REQUIRED_SECTIONS,
                "found": [],
                "missing": [],
                "extra": []
            }
        }
        
        # Check required top-level fields
        required_fields = ["title", "sections", "metadata"]
        for field in required_fields:
            if field not in report:
                results["is_valid"] = False
                results["errors"].append(f"Missing required field: {field}")
        
        # Check sections
        if not isinstance(report.get("sections"), list):
            results["is_valid"] = False
            results["errors"].append("Sections must be a list")
        
        if not report.get("sections"):
            results["is_valid"] = False
            results["errors"].append("Report has no sections")
        
        # Check each section and track coverage
        found_sections = set()
        for i, section in enumerate(report.get("sections", [])):
            if not isinstance(section, dict):
                results["is_valid"] = False
                results["errors"].append(f"Section {i} is not a dictionary")
                continue
            
            required_section_fields = ["title", "content", "key_points", "citations"]
            for field in required_section_fields:
                if field not in section:
                    results["is_valid"] = False
                    results["errors"].append(f"Section {i} missing required field: {field}")
            
            # Track section coverage
            section_title = section.get("title", "")
            found_sections.add(section_title)
            results["section_coverage"]["found"].append(section_title)
        
        # Check for missing and extra sections
        for required_section in self.REQUIRED_SECTIONS:
            if required_section not in found_sections:
                results["section_coverage"]["missing"].append(required_section)
                results["warnings"].append(f"Missing required section: {required_section}")
        
        for found_section in found_sections:
            if found_section not in self.REQUIRED_SECTIONS:
                results["section_coverage"]["extra"].append(found_section)
                results["warnings"].append(f"Extra section found: {found_section}")
        
        # Calculate section coverage score
        total_required = len(self.REQUIRED_SECTIONS)
        found_required = total_required - len(results["section_coverage"]["missing"])
        results["section_coverage"]["score"] = found_required / total_required if total_required > 0 else 0
        
        # Update overall validity
        if results["section_coverage"]["missing"]:
            results["is_valid"] = False
        
        return results

class CitationEvaluator(BaseEvaluator):
    """Evaluates citations and references."""
    
    def evaluate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate citations and references."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "citation_count": 0,
            "reference_count": 0
        }
        
        # Collect all citations
        all_citations = {}
        for section in report.get("sections", []):
            if section["title"] == "References":
                continue
            
            for citation in section.get("citations", []):
                if not isinstance(citation, dict):
                    results["errors"].append(f"Invalid citation format in section {section['title']}")
                    continue
                
                citation_id = citation.get("id")
                if not citation_id:
                    results["errors"].append(f"Citation missing ID in section {section['title']}")
                    continue
                
                all_citations[citation_id] = citation
        
        # Check References section
        references_section = None
        for section in report.get("sections", []):
            if section["title"] == "References":
                references_section = section
                break
        
        if not references_section:
            results["errors"].append("No References section found")
        else:
            # Count references
            reference_lines = [line for line in references_section["content"].split("\n") 
                             if line.strip() and not line.startswith("This section")]
            
            results["citation_count"] = len(all_citations)
            results["reference_count"] = len(reference_lines)
            
            if len(reference_lines) != len(all_citations):
                results["errors"].append(
                    f"Number of references ({len(reference_lines)}) does not match "
                    f"number of citations ({len(all_citations)})"
                )
        
        # Check in-text citations
        for section in report.get("sections", []):
            if section["title"] == "References":
                continue
            
            content = section["content"]
            for citation in section.get("citations", []):
                citation_id = citation["id"]
                in_text = citation.get("in_text")
                if not in_text:
                    results["errors"].append(f"Citation {citation_id} missing in_text field")
                    continue
                
                expected_citation = f"[{in_text}]"
                if expected_citation not in content:
                    results["errors"].append(
                        f"Expected citation {expected_citation} not found in section {section['title']}"
                    )
        
        results["is_valid"] = len(results["errors"]) == 0
        return results

class ResearchEvaluator:
    """Main evaluator class that combines all evaluation aspects."""
    
    def __init__(self):
        self.content_evaluator = ResearchContentEvaluator()
        self.structure_evaluator = ReportStructureEvaluator()
        self.citation_evaluator = CitationEvaluator()
    
    def evaluate_report(self, report_path: str) -> Dict[str, Any]:
        """Evaluate a report file."""
        try:
            # Read the report file
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Determine file type and parse accordingly
            if report_path.endswith('.json'):
                report = json.loads(content)
            elif report_path.endswith('.md'):
                report = self._parse_markdown_report(content)
            else:
                raise ValueError(f"Unsupported file format: {report_path}")
            
            # Run all evaluations
            content_results = self.content_evaluator.evaluate(report)
            structure_results = self.structure_evaluator.evaluate(report)
            citation_results = self.citation_evaluator.evaluate(report)
            
            # Combine results
            results = {
                "content_evaluation": content_results,
                "structure_evaluation": structure_results,
                "citation_evaluation": citation_results,
                "overall_score": self._calculate_overall_score(
                    content_results,
                    structure_results,
                    citation_results
                )
            }
            
            # Print summary
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating report: {str(e)}")
            return {
                "error": str(e),
                "overall_score": 0.0
            }
    
    def _parse_markdown_report(self, md_content: str) -> Dict[str, Any]:
        """Parse a markdown report into a structured format."""
        sections = []
        current_section = None
        current_content = []
        current_key_points = []
        current_citations = []
        
        # Regular expressions for parsing
        section_pattern = re.compile(r'^#\s+(.+)$')
        key_point_pattern = re.compile(r'^\*\s+(.+)$')
        citation_pattern = re.compile(r'\[([^\]]+)\]')
        
        for line in md_content.split('\n'):
            # Check for section header
            section_match = section_pattern.match(line)
            if section_match:
                # Save previous section if exists
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": "\n".join(current_content),
                        "key_points": current_key_points,
                        "citations": current_citations
                    })
                
                # Start new section
                current_section = section_match.group(1)
                current_content = []
                current_key_points = []
                current_citations = []
                continue
            
            # Check for key points
            key_point_match = key_point_pattern.match(line)
            if key_point_match:
                current_key_points.append(key_point_match.group(1))
                continue
            
            # Collect citations
            citations = citation_pattern.findall(line)
            for citation in citations:
                current_citations.append({
                    "in_text": citation,
                    "id": f"cite{len(current_citations) + 1}"
                })
            
            # Add line to content
            current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_content),
                "key_points": current_key_points,
                "citations": current_citations
            })
        
        # Extract title from first section
        title = sections[0]["title"] if sections else "Untitled Report"
        
        return {
            "title": title,
            "sections": sections,
            "metadata": {
                "format": "markdown"
            }
        }
    
    def _calculate_overall_score(self,
                               content_results: Dict[str, Any],
                               structure_results: Dict[str, Any],
                               citation_results: Dict[str, Any]) -> float:
        """Calculate overall evaluation score."""
        # Weight different aspects
        content_weight = 0.3
        structure_weight = 0.3
        citation_weight = 0.4  # Increased weight for citations
        
        # Calculate scores
        content_score = content_results.get("confidence_score", 0.0)
        structure_score = structure_results.get("section_coverage", {}).get("score", 0.0)
        citation_score = 1.0 if citation_results.get("is_valid", False) else 0.0
        
        # Calculate weighted average
        overall_score = (
            content_score * content_weight +
            structure_score * structure_weight +
            citation_score * citation_weight
        )
        
        return overall_score
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print("\nEvaluation Summary:")
        print(f"Content Quality: {'✓' if results['content_evaluation']['confidence_score'] > 0.7 else '✗'}")
        print(f"Report Structure: {'✓' if results['structure_evaluation']['is_valid'] else '✗'}")
        print(f"Citations: {'✓' if results['citation_evaluation']['is_valid'] else '✗'}")
        print(f"Overall Score: {results['overall_score']:.2f}")
        
        # Print section coverage
        section_coverage = results['structure_evaluation']['section_coverage']
        print("\nSection Coverage:")
        print(f"Required Sections: {len(section_coverage['required'])}")
        print(f"Found Sections: {len(section_coverage['found'])}")
        print(f"Missing Sections: {len(section_coverage['missing'])}")
        if section_coverage['missing']:
            print("\nMissing Sections:")
            for section in section_coverage['missing']:
                print(f"- {section}")
        if section_coverage['extra']:
            print("\nExtra Sections:")
            for section in section_coverage['extra']:
                print(f"- {section}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a research report.')
    parser.add_argument('report_path', help='Path to the report file (JSON or Markdown)')
    args = parser.parse_args()
    
    evaluator = ResearchEvaluator()
    results = evaluator.evaluate_report(args.report_path)
    
    exit(0 if results.get("overall_score", 0) > 0.7 else 1)

if __name__ == "__main__":
    main() 