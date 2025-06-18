from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
from datetime import datetime

class ResearchMemory(BaseModel):
    """Memory system for persisting research context and findings."""
    
    strategy: Optional[Dict[str, Any]] = None
    findings: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    
    def save_strategy(self, strategy: Dict[str, Any]) -> None:
        """Save research strategy."""
        self.strategy = strategy
        self.metadata["strategy_updated_at"] = datetime.now()
    
    def add_finding(self, finding: Dict[str, Any]) -> None:
        """Add a research finding."""
        finding["timestamp"] = datetime.now()
        self.findings.append(finding)
    
    def get_findings(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve findings based on filter criteria."""
        if not filter_criteria:
            return self.findings
        
        filtered_findings = []
        for finding in self.findings:
            if all(finding.get(k) == v for k, v in filter_criteria.items()):
                filtered_findings.append(finding)
        return filtered_findings
    
    def save_to_file(self, filepath: str) -> None:
        """Save memory state to file."""
        with open(filepath, 'w') as f:
            json.dump(self.dict(), f, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ResearchMemory':
        """Load memory state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data) 