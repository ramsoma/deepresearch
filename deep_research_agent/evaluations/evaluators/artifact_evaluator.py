import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ArtifactEvaluator:
    """Handles saving and loading of research artifacts for evaluation."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize the artifact evaluator.
        
        Args:
            artifacts_dir: Directory to store artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_artifacts(self,
                      query: str,
                      search_results: List[Dict[str, Any]],
                      analyses: Dict[str, Any],
                      plan: Dict[str, Any],
                      report: Dict[str, Any]) -> str:
        """Save all research artifacts to a JSON file.
        
        Args:
            query: The research query
            search_results: List of search results
            analyses: Content analyses
            plan: Research plan
            report: Generated report
            
        Returns:
            str: Path to the saved artifacts file
        """
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_artifacts_{timestamp}.json"
        filepath = self.artifacts_dir / filename
        
        # Prepare artifacts data
        artifacts = {
            "timestamp": timestamp,
            "query": query,
            "search_results": search_results,
            "analyses": analyses,
            "plan": plan,
            "report": report,
            "metadata": {
                "total_search_results": len(search_results),
                "topics_analyzed": list(analyses.keys()),
                "report_sections": [s["title"] for s in report["sections"]],
                "total_citations": sum(len(s.get("citations", [])) for s in report["sections"])
            }
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(artifacts, f, indent=2)
            logger.info(f"Saved research artifacts to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise
    
    def load_artifacts(self, filepath: str) -> Dict[str, Any]:
        """Load research artifacts from a JSON file.
        
        Args:
            filepath: Path to the artifacts file
            
        Returns:
            Dict[str, Any]: Loaded artifacts
        """
        try:
            with open(filepath, 'r') as f:
                artifacts = json.load(f)
            logger.info(f"Loaded research artifacts from {filepath}")
            return artifacts
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def list_artifacts(self) -> List[str]:
        """List all available artifact files.
        
        Returns:
            List[str]: List of artifact file paths
        """
        return [str(f) for f in self.artifacts_dir.glob("research_artifacts_*.json")] 