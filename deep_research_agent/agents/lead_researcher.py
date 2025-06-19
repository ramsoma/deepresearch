import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy
import google.generativeai as genai
from evaluations.evaluators.artifact_evaluator import ArtifactEvaluator

from deep_research_agent.agents.subagents.llm_citation_agent import LLMCitationAgent

from ..core.logging_config import get_logger
from ..core.memory import ResearchMemory
from ..core.template_manager import TemplateManager
from .subagents.reviewer_agent import ReviewerAgent, ReviewFeedback
from .subagents.web_searcher import WebSearcher

# Get logger
logger = get_logger(__name__)


class ResearchStrategy(dspy.Signature):
    """Structure for research strategy."""

    approach: str = dspy.InputField(
        desc="Research approach (breadth-first or depth-first)"
    )
    max_depth: int = dspy.InputField(desc="Maximum search depth")
    max_results: int = dspy.InputField(desc="Maximum results per search")
    focus_areas: List[str] = dspy.InputField(desc="Areas to focus the research")


class ResearchPlan(dspy.Signature):
    """Research plan structure."""

    main_topics: List[str] = dspy.OutputField(desc="Main topics to research")
    subtopics: Dict[str, List[str]] = dspy.OutputField(
        desc="Subtopics for each main topic"
    )
    key_questions: List[str] = dspy.OutputField(desc="Key questions to answer")
    required_sources: Dict[str, int] = dspy.OutputField(
        desc="Number of sources needed per topic"
    )
    search_results: List[dict] = dspy.OutputField(desc="Search results for each topic")


class ContentAnalysis(dspy.Signature):
    """Content analysis structure."""

    main_points: List[str] = dspy.OutputField(desc="Main points from the content")
    key_insights: List[str] = dspy.OutputField(desc="Key insights and implications")
    technical_details: List[str] = dspy.OutputField(
        desc="Technical details and specifications"
    )
    applications: List[str] = dspy.OutputField(desc="Applications and use cases")
    future_directions: List[str] = dspy.OutputField(
        desc="Future directions and developments"
    )
    confidence_score: float = dspy.OutputField(desc="Confidence score (0-1)")
    citations: List[Dict[str, Any]] = dspy.OutputField(
        desc="Citations and references used in the analysis"
    )


class ReportSection(dspy.Signature):
    """Structure for a report section."""

    title: str = dspy.OutputField(desc="Section title")
    content: str = dspy.OutputField(desc="Section content")
    key_points: List[str] = dspy.OutputField(desc="Key points from the section")
    citations: List[Dict[str, Any]] = dspy.OutputField(desc="Citations and references")

    def __init__(self, **data):
        # Convert Citation objects to dictionaries if needed
        if "citations" in data and data["citations"]:
            data["citations"] = [
                citation if isinstance(citation, dict) else citation.dict()
                for citation in data["citations"]
            ]
        super().__init__(**data)


class ResearchReport(dspy.Signature):
    """Structure for a complete research report."""

    title: str = dspy.OutputField(desc="Report title")
    sections: List[ReportSection] = dspy.OutputField(desc="Report sections")
    metadata: Dict[str, Any] = dspy.OutputField(desc="Report metadata")

    def to_markdown(self) -> str:
        """Convert the report to markdown format."""
        md = f"# {self.title}\n\n"

        # Add all sections
        for section in self.sections:
            md += f"## {section.title}\n\n{section.content}\n\n"
            if hasattr(section, "key_points") and section.key_points:
                md += "### Key Points\n"
                for point in section.key_points:
                    md += f"- {point}\n"
                md += "\n"

        # Add metadata at the end
        md += "## Metadata\n\n"
        for k, v in self.metadata.items():
            md += f"- {k}: {v}\n"
        return md


class PlanGenerator(dspy.Module):
    """Generates research plans."""

    def __init__(self):
        super().__init__()
        logger.info("Initializing PlanGenerator")
        self.model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
        self.template_manager = TemplateManager()

    def forward(self, query: str, strategy: Dict[str, Any]) -> ResearchPlan:
        """Generate a research plan."""
        logger.info(f"Generating research plan for query: {query}")

        # Render the plan generation template
        prompt = self.template_manager.render_template(
            "plan_generation.jinja2", query=query, strategy=strategy
        )

        logger.info("Sending prompt to LLM for plan generation")
        response = self.model.generate_content(prompt)
        logger.info(f"Received response from LLM: {response.text[:200]}...")

        try:
            import json

            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            logging.info(f"# Response text: {response_text}")
            # Remove any markdown code block markers
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            # Remove any explanatory text before or after the JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            logging.info(f"# Response text: {response_text}")
            result = json.loads(response_text)
            logger.info(f"Successfully parsed JSON response: {result}")

            # Validate the required fields
            if not isinstance(result.get("main_topics"), list):
                raise ValueError("main_topics must be a list")
            if not isinstance(result.get("subtopics"), dict):
                raise ValueError("subtopics must be a dictionary")
            if not isinstance(result.get("key_questions"), list):
                raise ValueError("key_questions must be a list")
            if not isinstance(result.get("required_sources"), dict):
                raise ValueError("required_sources must be a dictionary")

            return ResearchPlan(
                main_topics=result.get("main_topics", []),
                subtopics=result.get("subtopics", {}),
                key_questions=result.get("key_questions", []),
                required_sources=result.get("required_sources", {}),
                search_results=[],
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.info("Falling back to text parsing")
            # Fallback to text parsing if JSON parsing fails
            return ResearchPlan(
                main_topics=[],
                subtopics={},
                key_questions=[],
                required_sources={},
                search_results=[],
            )


class ContentAnalyzer(dspy.Module):
    """Analyzes content and extracts key insights."""

    def __init__(self):
        super().__init__()
        logger.info("Initializing ContentAnalyzer")
        self.model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))

    def forward(self, content: str) -> ContentAnalysis:
        """Analyze content and extract key insights."""
        logger.info("Starting content analysis")

        prompt = f"""You are a content analyzer. Your task is to analyze the following
          content and extract key insights in JSON format.

Content to analyze:
{content}

IMPORTANT: You must respond with ONLY a valid JSON object.
Do not include any other text or explanation.
The response must be a single JSON object with the following structure:
{{
    "main_points": ["point1", "point2", ...],
    "key_insights": ["insight1", "insight2", ...],
    "technical_details": ["detail1", "detail2", ...],
    "applications": ["application1", "application2", ...],
    "future_directions": ["direction1", "direction2", ...],
    "confidence_score": 0.95,
    "citations": [
        {{
            "id": "cite1",
            "text": "Author, A. (Year). Title. URL",
            "url": "https://example.com/paper1"
        }}
    ]
}}

Remember:
1. Use double quotes for all strings
2. Use square brackets for arrays
3. Use curly braces for objects
4. Include a comma between array elements
5. Do not include any trailing commas
6. Do not include any markdown formatting
7. Do not include any explanatory text outside the JSON
8. Include citations for any claims or information that needs attribution

Respond with ONLY the JSON object."""

        logger.info("Sending prompt to LLM for content analysis")
        response = self.model.generate_content(prompt)
        logger.info(f"Received response from LLM: {response.text[:200]}...")

        try:
            import json

            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block markers
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            # Remove any explanatory text before or after the JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            result = json.loads(response_text)
            logger.info(f"Successfully parsed JSON response: {result}")

            # Validate the required fields
            if not isinstance(result.get("main_points"), list):
                raise ValueError("main_points must be a list")
            if not isinstance(result.get("key_insights"), list):
                raise ValueError("key_insights must be a list")
            if not isinstance(result.get("technical_details"), list):
                raise ValueError("technical_details must be a list")
            if not isinstance(result.get("applications"), list):
                raise ValueError("applications must be a list")
            if not isinstance(result.get("future_directions"), list):
                raise ValueError("future_directions must be a list")
            if not isinstance(result.get("confidence_score"), (int, float)):
                raise ValueError("confidence_score must be a number")

            return ContentAnalysis(
                main_points=result.get("main_points", []),
                key_insights=result.get("key_insights", []),
                technical_details=result.get("technical_details", []),
                applications=result.get("applications", []),
                future_directions=result.get("future_directions", []),
                confidence_score=result.get("confidence_score", 0.0),
                citations=result.get("citations", []),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.info("Falling back to text parsing")
            # Fallback to text parsing if JSON parsing fails
            return ContentAnalysis(
                main_points=[],
                key_insights=[],
                technical_details=[],
                applications=[],
                future_directions=[],
                confidence_score=0.0,
                citations=[],
            )


class SectionGenerator(dspy.Module):
    """Module for generating report sections."""

    def __init__(self):
        super().__init__()
        self.model = genai.GenerativeModel(os.getenv("LLM_MODEL"))
        self.template_manager = TemplateManager()

    def forward(
        self, topic: str, analysis: Dict[str, Any], section_type: Optional[str] = None
    ) -> ReportSection:
        """Generate a report section."""
        # Render the section generation template
        prompt = self.template_manager.render_template(
            "section_generation.jinja2",
            topic=topic,
            analysis=analysis,
            section_type=section_type,
        )

        response = self.model.generate_content(prompt)
        try:
            import json

            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block markers
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            # Remove any explanatory text before or after the JSON
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            result = json.loads(response_text)
            return ReportSection(
                title=result.get("title", ""),
                content=result.get("content", ""),
                key_points=result.get("key_points", []),
                citations=result.get("citations", []),
            )
        except json.JSONDecodeError:
            # Fallback to parsing text if JSON parsing fails
            lines = response.text.split("\n")
            title = ""
            content = []
            key_points = []
            citations = []

            current_section = "title"
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("# "):
                    title = line[2:].strip()
                    current_section = "content"
                elif line.startswith("## Key Points"):
                    current_section = "key_points"
                elif line.startswith("## Citations"):
                    current_section = "citations"
                else:
                    if current_section == "content":
                        content.append(line)
                    elif current_section == "key_points":
                        if line.startswith("- "):
                            key_points.append(line[2:].strip())
                    elif current_section == "citations":
                        if line.startswith("- "):
                            citations.append(line[2:].strip())

            return ReportSection(
                title=title,
                content="\n".join(content),
                key_points=key_points,
                citations=citations,
            )


class LeadResearcher(dspy.Module):
    """Lead researcher agent that coordinates the research process."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the lead researcher."""
        super().__init__()
        logger.info("Initializing LeadResearcher")
        self.config = config or {}
        self.memory = ResearchMemory()
        self.web_searcher = WebSearcher()
        self.citation_agent = LLMCitationAgent(
            citation_style=self.config.get("citation_style", "chicago")
        )
        self.reviewer = ReviewerAgent()
        self.template_manager = TemplateManager()
        self.plan_generator = PlanGenerator()
        self.content_analyzer = ContentAnalyzer()
        self.section_generator = SectionGenerator()
        self.evaluator = ArtifactEvaluator()
        logger.info("LeadResearcher initialized successfully")

    def forward(self, query: str, strategy: Dict[str, Any]) -> ResearchReport:
        """Generate a complete research report."""
        logger.info(f"Starting research process for query: {query}")
        logger.info(f"Using strategy: {strategy}")

        # Update citation style from strategy if provided
        if "citation_style" in strategy:
            self.citation_style = strategy["citation_style"]
            logger.info(f"Using citation style: {self.citation_style}")

        try:
            # Generate research plan
            logger.info("Generating research plan...")
            plan = self.plan_generator.forward(query, strategy)
            logger.info(f"Generated plan: {plan}")

            # Convert plan to dict for template
            plan_dict = self._convert_to_dict(plan)

            # Collect search results
            logger.info("Collecting search results...")
            search_results = []
            for topic in plan.main_topics:
                logger.info(f"Searching for topic: {topic}")
                # Extract search parameters from strategy
                search_depth = strategy.get("max_depth", 3)
                max_results = strategy.get("max_results", 5)
                search_type = strategy.get("search_type", "basic")

                results = self.web_searcher.forward(
                    query=topic,
                    search_depth=search_depth,
                    max_results=max_results,
                    search_type=search_type,
                )
                search_results.extend(results.get("results", []))
            logger.info(f"# Search results: {len(search_results)}")
            # Analyze content
            logger.info("Analyzing content...")
            analyses = {}
            analyses_dict = {}

            for topic in plan.main_topics:
                logger.info(f"Analyzing topic: {topic}")

                # Generate more detailed content for analysis
                content_prompt = f"""Generate detailed content about the
                following topic in the context of the research query.

Research Query: {query}
Topic to analyze: {topic}
Research Strategy: {strategy}

Generate comprehensive content that includes:
1. Key concepts and definitions
2. Current state of the field
3. Recent developments and breakthroughs
4. Technical challenges and solutions
5. Future implications and directions

The content should be detailed and informative, suitable for a research report."""

                logger.info("Generating content for analysis...")
                model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
                content_response = model.generate_content(content_prompt)
                content = content_response.text
                logger.info(f"Generated content: {content[:200]}...")

                # Analyze the generated content
                analyses[topic] = self.content_analyzer.forward(content)
                analyses_dict[topic] = self._convert_to_dict(analyses[topic])
                logger.info(f"Analysis for {topic}: {analyses[topic]}")

            # Generate initial report
            logger.info("Generating initial report...")
            initial_report = self._generate_report(
                query, strategy, plan, analyses, search_results
            )

            # Review the report
            logger.info("Reviewing the report...")
            review_feedback = self.reviewer.forward(initial_report, query, strategy)
            logger.info(f"Review feedback: {review_feedback}")

            # Log specific citation-related feedback
            citation_issues = [
                w
                for w in review_feedback.weaknesses
                if any(term in w.lower() for term in ["citation", "reference", "cite"])
            ]
            if citation_issues:
                logger.warning(
                    f"Citation-related issues found in review: {citation_issues}"
                )

            # If there are significant issues, regenerate the report
            if (
                review_feedback.confidence_score < 0.7
                or len(review_feedback.priority_fixes) > 0
            ):
                logger.info(
                    f"""Regenerating report based on review feedback.
                    Confidence score: {review_feedback.confidence_score},
                    Priority fixes: {review_feedback.priority_fixes}"""
                )
                improved_report = self._generate_report(
                    query,
                    strategy,
                    plan,
                    analyses,
                    search_results,
                    review_feedback=review_feedback,
                )

                # Review the improved report
                logger.info("Reviewing the improved report...")
                final_review = self.reviewer.forward(improved_report, query, strategy)
                logger.info(f"Final review feedback: {final_review}")

                # Log if citation issues were resolved
                final_citation_issues = [
                    w
                    for w in final_review.weaknesses
                    if any(
                        term in w.lower() for term in ["citation", "reference", "cite"]
                    )
                ]
                if final_citation_issues:
                    logger.warning(
                        f"""Citation issues still present after regeneration:
                         {final_citation_issues}"""
                    )
                else:
                    logger.info(
                        "All citation issues were resolved in the improved report"
                    )

                # Add review feedback to metadata
                improved_report.metadata["initial_review"] = review_feedback
                improved_report.metadata["final_review"] = final_review

                # Save artifacts for evaluation
                self.evaluator.save_artifacts(
                    query=query,
                    search_results=search_results,
                    analyses=analyses_dict,
                    plan=plan_dict,
                    report=self._convert_to_dict(improved_report),
                )

                return improved_report

            # Add review feedback to metadata
            initial_report.metadata["review"] = review_feedback

            # Save artifacts for evaluation
            self.evaluator.save_artifacts(
                query=query,
                search_results=search_results,
                analyses=analyses_dict,
                plan=plan_dict,
                report=self._convert_to_dict(initial_report),
            )

            return initial_report

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            # Return a minimal report with error information
            return ResearchReport(
                title="Research Report (Error)",
                sections=[
                    ReportSection(
                        title="Error",
                        content=f"""An error occurred while generating the report:
                                    {str(e)}""",
                        key_points=[],
                        citations=[],
                    )
                ],
                metadata={"query": query, "strategy": strategy, "error": str(e)},
            )

    def _convert_to_dict(self, obj):
        """Convert DSPy objects to dictionaries for JSON serialization."""
        if hasattr(obj, "__dict__"):
            return {k: self._convert_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    def _generate_report(
        self,
        query: str,
        strategy: Dict[str, Any],
        plan: ResearchPlan,
        analyses: Dict[str, ContentAnalysis],
        search_results: List[dict],
        review_feedback: Optional[ReviewFeedback] = None,
    ) -> ResearchReport:
        """Generate the final research report."""
        logger.info("Starting report generation")

        # Generate title
        title = self._generate_title(query)
        logger.info(f"Generated title: {title}")

        # Collect all sources
        sources = self._collect_sources(plan, analyses, search_results)
        logger.info(f"Collected {len(sources)} sources")

        # Get sections list from strategy
        sections_list = strategy.get("sections", [])
        logger.info(f"Using sections: {sections_list}")

        # Generate sections
        sections = []
        section_texts = []  # Track all section texts

        # First pass: Generate all sections except References
        for section_type in [s for s in sections_list if s != "References"]:
            logger.info(f"Generating section: {section_type}")

            # Get analysis for this section
            analysis: Dict[str, Any] = analyses.get(section_type.lower(), {})

            # Generate section content
            section = self.section_generator.forward(
                topic=query, analysis=analysis, section_type=section_type
            )

            if section.content:
                section_texts.append(section.content)
                sections.append(section)
                logger.info(f"Completed section: {section_type}")

        logging.info(f"# Sources: {len(sources)}")
        # Add citations to all sections at once
        if section_texts and search_results:
            # Combine all text
            logging.info("Creating citations for..")
            combined_text = "\n\n".join(section_texts)

            # Get citations for the entire text
            citation_result = self.citation_agent.cite_text(
                text=combined_text,
                sources=sources,
                style=self.config.get("citation_style", "chicago"),
            )

            # Split the cited text back into sections
            cited_sections = citation_result["cited_text"].split("\n\n")

            # Update each section with its cited content
            for i, section in enumerate(sections):
                if i < len(cited_sections):
                    section.content = cited_sections[i]
                    # Find citations that belong to this section
                    section.citations = [
                        c
                        for c in citation_result["citations"]
                        if c["text"] in cited_sections[i]
                    ]

            # Add References section if it's in the sections list
            references_section = ReportSection(
                title="References",
                content=citation_result["references"],
                key_points=[],
                citations=citation_result["citations"],
            )
            sections.append(references_section)
            logger.info(
                f"""Added References section with
                        {len(citation_result['citations'])} citations"""
            )

        # Create the report
        report = ResearchReport(
            title=title,
            sections=sections,
            metadata={
                "query": query,
                "strategy": strategy,
                "generation_timestamp": datetime.now().isoformat(),
                "review_feedback": (
                    self._convert_to_dict(review_feedback) if review_feedback else None
                ),
                "total_citations": (
                    len(citation_result["citations"])
                    if section_texts and sources
                    else 0
                ),
                "citation_style": self.config.get("citation_style", "chicago"),
                "citation_metrics": {
                    "total_citations": (
                        len(citation_result["citations"])
                        if section_texts and sources
                        else 0
                    ),
                    "unique_citations": (
                        len(set(c["text"] for c in citation_result["citations"]))
                        if section_texts and sources
                        else 0
                    ),
                    "sections_with_citations": len(
                        [s for s in sections if s.citations]
                    ),
                },
            },
        )

        logger.info("Report generation completed")
        return report

    def _generate_title(self, query: str) -> str:
        """Generate a title for the research report."""
        logger.info(f"Generating title for query: {query}")
        try:
            # Initialize the model
            model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))

            prompt = f"""Generate a clear, concise title for a research
            report about: {query}

The title should:
1. Be specific and descriptive
2. Be no longer than 15 words
3. Not include quotes or special characters
4. Not end with a period
5. Be in title case

Respond with ONLY the title, no additional text or explanation."""

            response = model.generate_content(prompt)
            title = response.text.strip()

            # Clean up the title
            title = title.replace('"', "").replace('"', "").replace('"', "")
            if title.endswith("."):
                title = title[:-1]

            logger.info(f"Generated title: {title}")
            return title

        except Exception as e:
            logger.warning(f"Error generating title: {str(e)}")
            # Fallback to a simple title
            return f"Research Report: {query}"

    def _update_sections_based_on_feedback(
        self, required_sections: List[str], review_feedback: Dict
    ) -> List[str]:
        """Update required sections based on reviewer feedback."""
        try:
            # If reviewer suggests adding sections, add them
            if "suggested_sections" in review_feedback:
                for section in review_feedback["suggested_sections"]:
                    if section not in required_sections:
                        required_sections.append(section)

            # If reviewer suggests removing sections, remove them
            if "sections_to_remove" in review_feedback:
                required_sections = [
                    s
                    for s in required_sections
                    if s not in review_feedback["sections_to_remove"]
                ]

            return required_sections

        except Exception as e:
            logger.warning(f"Error updating sections based on feedback: {str(e)}")
            return required_sections

    def _compose_draft_text(self, query, strategy, plan, analyses):
        """Compose draft text for the report."""
        logger.info("Composing draft text")
        try:
            # Initialize the model
            model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
            prompt_parms = {
                "plan": self._convert_to_dict(plan),
                "analyses": {k: self._convert_to_dict(v) for k, v in analyses.items()},
            }
            content_prompt = f"""Compose a draft research report about: {query}

Use the following research plan and analyses:
{json.dumps(prompt_parms, indent=2)}

The report should:
1. Be well-structured and organized
2. Include all key findings
3. Be written in a clear, academic style
4. Include placeholders for citations
5. Follow the standard research report format

Respond with ONLY the draft text, no additional explanation."""

            content_response = model.generate_content(content_prompt)
            return content_response.text.strip()

        except Exception as e:
            logger.warning(f"Error composing draft text: {str(e)}")
            return ""

    def _collect_sources(self, plan, analyses, search_results):
        """
        Collect sources from the plan and analyses.
        Args:
            plan: The research plan
            analyses: The research analyses
            search_results: The search results
        Returns:
            List[Dict[str, Any]]: List of sources with metadata
        """
        sources = []
        # Extract sources from search results
        if search_results:
            for result in search_results:
                if isinstance(result, dict):
                    source = {
                        "title": result.get("title", "Untitled"),
                        "authors": result.get("authors", ["Unknown Author"]),
                        "year": result.get("year", datetime.now().year),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                    }
                    sources.append(source)
                else:
                    logging.warning(f"Source is not a dict: {result}")
        else:
            logging.warning(f"Plan has no search results: {plan}")
        # Extract sources from analyses
        if analyses:
            for topic, analysis in analyses.items():
                if hasattr(analysis, "sources"):
                    for source in analysis.sources:
                        if isinstance(source, dict):
                            sources.append(source)
                else:
                    logging.warning(f"Analysis has no source is not a dict: {analysis}")
        else:
            logging.warning(f"Analyses is not present: {analyses}")
        # Remove duplicates based on URL
        unique_sources = {}
        for source in sources:
            url = source.get("url", "")
            if url and url not in unique_sources:
                unique_sources[url] = source

        return list(unique_sources.values())
