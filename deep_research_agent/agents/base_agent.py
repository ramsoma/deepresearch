from typing import Any, Dict, Optional
import dspy
from pydantic import BaseModel

class BaseAgent(dspy.Module):
    """Base class for all research agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
    def forward(self, *args, **kwargs):
        """Forward pass to be implemented by specific agents."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def validate_output(self, output: Any) -> bool:
        """Validate the output of the agent."""
        raise NotImplementedError("Subclasses must implement validate_output method") 