from typing import List, Dict, Any, Optional
import dspy
import google.generativeai as genai
import os
import logging

logger = logging.getLogger(__name__)

class ReviewFeedback(dspy.Signature):
    """Structure for review feedback."""
    strengths: List[str] = dspy.OutputField(desc="Strengths of the report")
    weaknesses: List[str] = dspy.OutputField(desc="Areas that need improvement")
    missing_elements: List[str] = dspy.OutputField(desc="Important elements that are missing")
    suggestions: List[str] = dspy.OutputField(desc="Specific suggestions for improvement")
    confidence_score: float = dspy.OutputField(desc="Confidence in the review (0-1)")
    priority_fixes: List[str] = dspy.OutputField(desc="High-priority items that should be addressed first")

class ReviewerAgent(dspy.Module):
    """Agent that reviews research reports and provides feedback for improvement."""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing ReviewerAgent")
        self.model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
        
    def forward(self, report: Dict[str, Any], query: str, strategy: Dict[str, Any]) -> ReviewFeedback:
        """Review a research report and provide feedback."""
        logger.info("Starting report review")
        
        # Combine all section contents for review
        report_content = "\n\n".join([
            f"# {section.title}\n{section.content}"
            for section in report.sections
        ])
        
        # Log the initial report content for debugging
        logger.info(f"Reviewing report with content: {report_content[:500] if report_content else 'No content'}...")
        
        prompt = f"""You are a research report reviewer. Your task is to evaluate the following research report and provide detailed feedback.

Research Query: {query}
Research Strategy: {strategy}

Report to Review:
{report_content}

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text or explanation.

Evaluate the report and provide feedback in the following structure:
{{
    "strengths": [
        "List the main strengths of the report",
        "Focus on content quality, structure, and analysis"
    ],
    "weaknesses": [
        "List areas that need improvement",
        "Be specific about what's not working well"
    ],
    "missing_elements": [
        "List important elements that are missing",
        "Consider both content and structural elements"
    ],
    "suggestions": [
        "Provide specific suggestions for improvement",
        "Include concrete examples where possible"
    ],
    "confidence_score": 0.85,
    "priority_fixes": [
        "List the most critical issues that should be addressed first",
        "Order by importance and impact"
    ]
}}

Consider the following aspects in your review:
1. Content Quality
   - Accuracy and reliability of information
   - Depth of analysis
   - Use of evidence and citations
   - Technical accuracy

2. Structure and Organization
   - Logical flow of information
   - Section organization
   - Transitions between sections
   - Balance of content
   - Check for duplicate sections (especially References)
   - Verify section ordering and hierarchy

3. Research Methodology
   - Alignment with research strategy
   - Coverage of key topics
   - Depth of investigation
   - Use of sources

4. Clarity and Communication
   - Writing style and clarity
   - Technical explanations
   - Use of examples and illustrations
   - Audience appropriateness

5. Completeness
   - Coverage of all required sections
   - Addressing of research query
   - Inclusion of all necessary elements
   - Balance of breadth and depth

6. Citation and Reference Quality
   - Proper citation formatting
   - Consistent citation style
   - No literal "(Year)" placeholders
   - Citations properly linked to references
   - No duplicate citations
   - References section properly formatted
   - All citations have corresponding references
   - Citations should only appear in the text where referenced, not as lists at the end of sections
   - All citations should be collated in a single References section at the end of the report
   - No "Citations" or "References" subsections within other sections

Remember: Respond with ONLY the JSON object, no other text."""

        logger.info("Sending review prompt to LLM")
        response = self.model.generate_content(prompt)
        logger.info(f"Received raw response from LLM: {response.text}")
        
        try:
            import json
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block markers
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            logger.info(f"Successfully parsed JSON response with {len(result.get('weaknesses', []))} weaknesses and {len(result.get('suggestions', []))} suggestions")
            
            # Log specific citation-related feedback
            citation_issues = [w for w in result.get('weaknesses', []) if any(term in w.lower() for term in ['citation', 'reference', 'cite'])]
            if citation_issues:
                logger.warning(f"Citation-related issues found: {citation_issues}")
            
            # Validate the required fields
            if not isinstance(result.get("strengths"), list):
                raise ValueError("strengths must be a list")
            if not isinstance(result.get("weaknesses"), list):
                raise ValueError("weaknesses must be a list")
            if not isinstance(result.get("missing_elements"), list):
                raise ValueError("missing_elements must be a list")
            if not isinstance(result.get("suggestions"), list):
                raise ValueError("suggestions must be a list")
            if not isinstance(result.get("confidence_score"), (int, float)):
                raise ValueError("confidence_score must be a number")
            if not isinstance(result.get("priority_fixes"), list):
                raise ValueError("priority_fixes must be a list")
            
            feedback = ReviewFeedback(
                strengths=result.get("strengths", []),
                weaknesses=result.get("weaknesses", []),
                missing_elements=result.get("missing_elements", []),
                suggestions=result.get("suggestions", []),
                confidence_score=result.get("confidence_score", 0.0),
                priority_fixes=result.get("priority_fixes", [])
            )
            
            # Log the final feedback object
            logger.info(f"Generated feedback with confidence score: {feedback.confidence_score}")
            logger.info(f"Priority fixes: {feedback.priority_fixes}")
            
            return feedback
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response was: {response.text}")
            # Return a minimal feedback object with error information
            return ReviewFeedback(
                strengths=[],
                weaknesses=["Failed to parse review feedback"],
                missing_elements=[],
                suggestions=["Please try reviewing the report again"],
                confidence_score=0.0,
                priority_fixes=["Fix the review process"]
            ) 