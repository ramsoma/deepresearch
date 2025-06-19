from typing import Any, Dict, Optional

import dspy
from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    """Base class for evaluation metrics."""

    accuracy: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    additional_metrics: Dict[str, Any] = {}


class BaseEvaluator(dspy.Module):
    """Base class for all research evaluators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

    def evaluate(
        self,
        research_output: Any,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationMetrics:
        """Evaluate research output against ground truth and context."""
        raise NotImplementedError("Subclasses must implement evaluate method")

    def validate_metrics(self, metrics: EvaluationMetrics) -> bool:
        """Validate the evaluation metrics."""
        return all(
            0 <= value <= 1
            for value in metrics.dict().values()
            if isinstance(value, (int, float))
        )
