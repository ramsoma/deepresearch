from typing import Any, Dict, Optional, List
import dspy
import google.generativeai as genai
import os
import logging
from ..base_evaluator import BaseEvaluator, EvaluationMetrics
from ..metrics import ResearchMetrics

logger = logging.getLogger(__name__)

class ResearchEvaluation(dspy.Signature):
    """Structure for research evaluation results."""
    factual_accuracy: float = dspy.OutputField(desc="Score for factual accuracy (0.0-1.0)")
    citation_accuracy: float = dspy.OutputField(desc="Score for citation accuracy (0.0-1.0)")
    completeness: float = dspy.OutputField(desc="Score for completeness (0.0-1.0)")
    source_quality: float = dspy.OutputField(desc="Score for source quality (0.0-1.0)")
    tool_efficiency: float = dspy.OutputField(desc="Score for tool efficiency (0.0-1.0)")
    overall_score: float = dspy.OutputField(desc="Overall evaluation score (0.0-1.0)")
    passed: bool = dspy.OutputField(desc="Whether the research output passes quality threshold")
    feedback: List[str] = dspy.OutputField(desc="Specific feedback points")
    confidence: float = dspy.OutputField(desc="Confidence in the evaluation (0.0-1.0)")

class ResearchEvaluator(BaseEvaluator):
    """Evaluator for research outputs using LLM-as-judge approach."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
        self.metrics = ResearchMetrics()
        self.quality_threshold = config.get("quality_threshold", 0.7) if config else 0.7
    
    def evaluate(self,
                research_output: Any,
                ground_truth: Optional[Any] = None,
                context: Optional[Dict[str, Any]] = None) -> EvaluationMetrics:
        """Evaluate research output quality using LLM-as-judge."""
        logger.info("Starting LLM-as-judge evaluation")
        
        # Prepare evaluation prompt
        prompt = self._create_evaluation_prompt(research_output, ground_truth, context)
        
        # Get LLM evaluation
        try:
            response = self.model.generate_content(prompt)
            evaluation = self._parse_evaluation_response(response.text)
            
            # Convert to EvaluationMetrics
            metrics = EvaluationMetrics(
                accuracy=evaluation.factual_accuracy,
                relevance=evaluation.completeness,
                coherence=evaluation.overall_score,
                additional_metrics={
                    "citation_accuracy": evaluation.citation_accuracy,
                    "source_quality": evaluation.source_quality,
                    "tool_efficiency": evaluation.tool_efficiency,
                    "passed": evaluation.passed,
                    "confidence": evaluation.confidence,
                    "feedback": evaluation.feedback
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            return EvaluationMetrics(
                accuracy=0.0,
                relevance=0.0,
                coherence=0.0,
                additional_metrics={
                    "error": str(e)
                }
            )
    
    def _create_evaluation_prompt(self,
                                research_output: Any,
                                ground_truth: Optional[Any],
                                context: Optional[Dict[str, Any]]) -> str:
        """Create the evaluation prompt for the LLM judge."""
        prompt = f"""You are an expert research evaluator. Your task is to evaluate the following research output and provide detailed scores and feedback.

Research Output to Evaluate:
{research_output}

"""

        if ground_truth:
            prompt += f"""
Ground Truth/Expected Output:
{ground_truth}
"""

        if context:
            prompt += f"""
Context:
{context}
"""

        prompt += """
Evaluate the research output based on the following criteria:

1. Factual Accuracy (0.0-1.0)
   - Do claims match the sources?
   - Are statements supported by evidence?
   - Are there any factual errors?

2. Citation Accuracy (0.0-1.0)
   - Do cited sources match the claims?
   - Are citations properly formatted?
   - Are all claims properly cited?

3. Completeness (0.0-1.0)
   - Are all requested aspects covered?
   - Is the analysis thorough?
   - Are important points addressed?

4. Source Quality (0.0-1.0)
   - Are primary sources used over secondary sources?
   - Are sources reliable and authoritative?
   - Is there a good mix of recent and foundational sources?

5. Tool Efficiency (0.0-1.0)
   - Were the right tools used?
   - Was the number of tool calls reasonable?
   - Were tools used effectively?

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text or explanation.

Provide your evaluation in the following structure:
{
    "factual_accuracy": 0.0,
    "citation_accuracy": 0.0,
    "completeness": 0.0,
    "source_quality": 0.0,
    "tool_efficiency": 0.0,
    "overall_score": 0.0,
    "passed": true,
    "feedback": [
        "List specific feedback points",
        "Include both strengths and areas for improvement"
    ],
    "confidence": 0.0
}

Remember:
- Scores should be between 0.0 and 1.0
- Overall score should be a weighted average of individual scores
- Pass threshold is 0.7
- Confidence should reflect your certainty in the evaluation
- Feedback should be specific and actionable

Respond with ONLY the JSON object, no other text."""

        return prompt
    
    def _parse_evaluation_response(self, response_text: str) -> ResearchEvaluation:
        """Parse the LLM's evaluation response."""
        try:
            import json
            # Clean the response text
            response_text = response_text.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = [
                "factual_accuracy", "citation_accuracy", "completeness",
                "source_quality", "tool_efficiency", "overall_score",
                "passed", "feedback", "confidence"
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create ResearchEvaluation object
            return ResearchEvaluation(
                factual_accuracy=float(result["factual_accuracy"]),
                citation_accuracy=float(result["citation_accuracy"]),
                completeness=float(result["completeness"]),
                source_quality=float(result["source_quality"]),
                tool_efficiency=float(result["tool_efficiency"]),
                overall_score=float(result["overall_score"]),
                passed=bool(result["passed"]),
                feedback=list(result["feedback"]),
                confidence=float(result["confidence"])
            )
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {str(e)}")
            raise ValueError(f"Failed to parse evaluation response: {str(e)}") 