import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtifactEvaluator:
    """Evaluates generated content against intermediate artifacts."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def save_artifacts(self, 
                      query: str,
                      search_results: List[Dict[str, Any]],
                      analyses: Dict[str, Any],
                      plan: Dict[str, Any],
                      report: Dict[str, Any]) -> str:
        """Save all intermediate artifacts for a research run."""
        timestamp = report["metadata"].get("timestamp", "unknown")
        run_dir = self.artifacts_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save query
        with open(run_dir / "query.txt", "w") as f:
            f.write(query)
        
        # Save search results
        with open(run_dir / "search_results.json", "w") as f:
            json.dump(search_results, f, indent=2)
        
        # Save analyses
        with open(run_dir / "analyses.json", "w") as f:
            json.dump(analyses, f, indent=2)
        
        # Save plan
        with open(run_dir / "plan.json", "w") as f:
            json.dump(plan, f, indent=2)
        
        # Save final report
        with open(run_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return str(run_dir)
    
    def evaluate_against_artifacts(self, run_dir: str) -> Dict[str, Any]:
        """Evaluate the generated report against its artifacts."""
        run_dir = Path(run_dir)
        results = {
            "hallucinations": [],
            "source_coverage": {},
            "citation_accuracy": {},
            "reasoning_chain": [],
            "overall_score": 0.0
        }
        
        # Load artifacts
        with open(run_dir / "search_results.json") as f:
            search_results = json.load(f)
        with open(run_dir / "analyses.json") as f:
            analyses = json.load(f)
        with open(run_dir / "report.json") as f:
            report = json.load(f)
        
        # Create source content index
        source_content = {}
        for result in search_results:
            source_content[result["url"]] = result["content"]
        
        # Evaluate each section
        for section in report["sections"]:
            if section["title"] == "References":
                continue
            
            section_results = self._evaluate_section(
                section,
                source_content,
                search_results,
                analyses
            )
            
            # Add section results to overall results
            results["hallucinations"].extend(section_results["hallucinations"])
            results["source_coverage"][section["title"]] = section_results["source_coverage"]
            results["citation_accuracy"][section["title"]] = section_results["citation_accuracy"]
            results["reasoning_chain"].extend(section_results["reasoning_chain"])
        
        # Calculate overall score
        results["overall_score"] = self._calculate_score(results)
        
        return results
    
    def _evaluate_section(self,
                         section: Dict[str, Any],
                         source_content: Dict[str, str],
                         search_results: List[Dict[str, Any]],
                         analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single section against artifacts."""
        results = {
            "hallucinations": [],
            "source_coverage": {},
            "citation_accuracy": {},
            "reasoning_chain": []
        }
        
        # Split content into sentences
        sentences = self._split_into_sentences(section["content"])
        
        # Check each sentence against sources
        for sentence in sentences:
            # Skip sentences that are just citations
            if re.match(r'^\[.*\]$', sentence.strip()):
                continue
            
            # Find best matching source
            best_match = None
            best_score = 0
            best_source = None
            
            for source_url, source_text in source_content.items():
                # Split source into sentences for comparison
                source_sentences = self._split_into_sentences(source_text)
                
                for source_sentence in source_sentences:
                    score = SequenceMatcher(None, sentence.lower(), source_sentence.lower()).ratio()
                    if score > best_score and score > 0.6:  # Threshold for considering a match
                        best_score = score
                        best_match = source_sentence
                        best_source = source_url
            
            if best_match:
                # Sentence has a source
                results["source_coverage"][sentence] = {
                    "source": best_source,
                    "match_score": best_score,
                    "matched_text": best_match
                }
                results["reasoning_chain"].append({
                    "sentence": sentence,
                    "source": best_source,
                    "confidence": best_score
                })
            else:
                # Potential hallucination
                results["hallucinations"].append({
                    "sentence": sentence,
                    "section": section["title"]
                })
        
        # Check citation accuracy
        for citation in section.get("citations", []):
            citation_id = citation.get("id")
            citation_text = citation.get("text", "")
            
            # Find matching source
            matching_source = None
            for result in search_results:
                if citation_text in result["content"]:
                    matching_source = result["url"]
                    break
            
            results["citation_accuracy"][citation_id] = {
                "has_source": matching_source is not None,
                "source_url": matching_source
            }
        
        return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall evaluation score."""
        # Weights for different aspects
        hallucination_weight = 0.4
        source_coverage_weight = 0.3
        citation_accuracy_weight = 0.3
        
        # Calculate hallucination score (inverse of number of hallucinations)
        total_sentences = sum(len(coverage) for coverage in results["source_coverage"].values())
        hallucination_score = 1.0 - (len(results["hallucinations"]) / total_sentences if total_sentences > 0 else 0)
        
        # Calculate source coverage score
        coverage_scores = []
        for section_coverage in results["source_coverage"].values():
            if section_coverage:
                section_score = sum(item["match_score"] for item in section_coverage.values()) / len(section_coverage)
                coverage_scores.append(section_score)
        source_coverage_score = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        
        # Calculate citation accuracy score
        citation_scores = []
        for section_accuracy in results["citation_accuracy"].values():
            if section_accuracy:
                section_score = sum(1 for item in section_accuracy.values() if item["has_source"]) / len(section_accuracy)
                citation_scores.append(section_score)
        citation_accuracy_score = sum(citation_scores) / len(citation_scores) if citation_scores else 0
        
        # Calculate weighted average
        overall_score = (
            hallucination_score * hallucination_weight +
            source_coverage_score * source_coverage_weight +
            citation_accuracy_score * citation_accuracy_weight
        )
        
        return overall_score

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a research report against its artifacts.')
    parser.add_argument('run_dir', help='Path to the run directory containing artifacts')
    args = parser.parse_args()
    
    evaluator = ArtifactEvaluator()
    results = evaluator.evaluate_against_artifacts(args.run_dir)
    
    # Print summary
    print("\nArtifact Evaluation Summary:")
    print(f"Hallucinations: {len(results['hallucinations'])}")
    print(f"Source Coverage: {sum(len(coverage) for coverage in results['source_coverage'].values())} sentences")
    print(f"Citation Accuracy: {sum(len(accuracy) for accuracy in results['citation_accuracy'].values())} citations")
    print(f"Overall Score: {results['overall_score']:.2f}")
    
    # Print detailed results
    if results['hallucinations']:
        print("\nPotential Hallucinations:")
        for hallucination in results['hallucinations']:
            print(f"- In {hallucination['section']}: {hallucination['sentence']}")
    
    exit(0 if results['overall_score'] > 0.7 else 1)

if __name__ == "__main__":
    main() 