import json
import unittest
from unittest.mock import MagicMock, patch

from deep_research_agent.agents.lead_researcher import LeadResearcher, ReportSection


class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Set up the test environment with mock dependencies."""
        # Create mock responses for different stages
        self.mock_responses = {
            "plan": {
                "main_topics": ["Quantum Computing Basics", "Recent Developments"],
                "subtopics": {
                    "Quantum Computing Basics": ["Qubits", "Superposition"],
                    "Recent Developments": ["Error Correction", "Quantum Supremacy"],
                },
                "key_questions": [
                    "What are the latest breakthroughs?",
                    "How close are we to practical applications?",
                ],
                "required_sources": {
                    "Quantum Computing Basics": 2,
                    "Recent Developments": 3,
                },
            },
            "analysis": {
                "main_points": [
                    "Quantum computing uses qubits",
                    "Recent breakthroughs in error correction",
                ],
                "key_insights": [
                    "Quantum supremacy achieved",
                    "Practical applications emerging",
                ],
                "technical_details": [
                    "Qubit coherence times improving",
                    "Error correction codes advancing",
                ],
                "applications": ["Cryptography", "Drug discovery"],
                "future_directions": [
                    "Fault-tolerant quantum computers",
                    "Quantum internet",
                ],
                "confidence_score": 0.9,
            },
            "review": {
                "strengths": [
                    "Clear explanation of basics",
                    "Good coverage of recent developments",
                ],
                "weaknesses": [
                    "Could use more examples",
                    "Technical details could be deeper",
                ],
                "missing_elements": ["More examples", "Deeper technical details"],
                "suggestions": [
                    "Add more real-world examples",
                    "Include more technical details",
                ],
                "confidence_score": 0.85,
                "priority_fixes": ["Add more examples", "Deepen technical details"],
            },
            "section": {
                "title": "Introduction",
                "content": "Quantum computing represents a revolutionary approach to computation, leveraging the principles of quantum mechanics to process information in fundamentally new ways. Recent developments have brought us closer to practical applications, with significant breakthroughs in error correction and quantum supremacy demonstrations.",  # noqa
                "key_points": [
                    "Quantum computing basics",
                    "Recent breakthroughs",
                    "Future potential",
                ],
                "citations": [
                    {
                        "key": "smith2023",
                        "text": "Smith et al. (2023)",
                        "reference": "Smith, J., et al. (2023). Quantum Computing Advances. Nature, 123(4), 567-589.",  # noqa
                        "url": "https://example.com/smith2023",
                    },
                    {
                        "key": "jones2021",
                        "text": "Jones (2021)",
                        "reference": "Jones, R. (2021). The Future of Quantum Computing. Science, 456(7), 890-912.",  # noqa
                        "url": "https://example.com/jones2021",
                    },
                ],
            },
            "references": {
                "title": "References",
                "content": "1. Smith, J., et al. (2023). Quantum Computing Advances. Nature, 123(4), 567-589.\n2. Jones, R. (2021). The Future of Quantum Computing. Science, 456(7), 890-912.",  # noqa
                "key_points": [],
                "citations": [
                    {
                        "key": "smith2023",
                        "text": "Smith et al. (2023)",
                        "reference": "Smith, J., et al. (2023). Quantum Computing Advances. Nature, 123(4), 567-589.",  # noqa
                        "url": "https://example.com/smith2023",
                    },
                    {
                        "key": "jones2021",
                        "text": "Jones (2021)",
                        "reference": "Jones, R. (2021). The Future of Quantum Computing. Science, 456(7), 890-912.",  # noqa
                        "url": "https://example.com/jones2021",
                    },
                ],
            },
        }

        # Create a mock model that returns different responses based on the prompt
        self.mock_model = MagicMock()

        def mock_generate_content(prompt):
            response = MagicMock()
            if "research plan" in prompt.lower():
                response.text = json.dumps(self.mock_responses["plan"])
            elif "content analysis" in prompt.lower():
                response.text = json.dumps(self.mock_responses["analysis"])
            elif "review" in prompt.lower():
                response.text = json.dumps(self.mock_responses["review"])
            elif "references" in prompt.lower():
                response.text = json.dumps(self.mock_responses["references"])
            else:
                response.text = json.dumps(
                    self.mock_responses["section"]
                )  # Default response
            return response

        self.mock_model.generate_content.side_effect = mock_generate_content

        # Create a mock web search response
        self.mock_search_response = {
            "results": [
                {
                    "title": "Quantum Computing Basics",
                    "url": "https://example.com/quantum-basics",
                    "content": "An introduction to quantum computing concepts.",
                },
                {
                    "title": "Recent Developments in Quantum Computing",
                    "url": "https://example.com/quantum-developments",
                    "content": "Latest breakthroughs in quantum computing technology.",
                },
            ]
        }

        # Create patches for all external dependencies
        self.patches = {
            "llm": patch(
                "deep_research_agent.agents.lead_researcher.genai.GenerativeModel"
            ),
            "web_searcher": patch(
                "deep_research_agent.agents.subagents.web_searcher.WebSearcher"
            ),
            "tavily": patch(
                "deep_research_agent.agents.subagents.web_searcher.TavilyClient"
            ),
            "reviewer": patch(
                "deep_research_agent.agents.subagents.reviewer_agent.ReviewerAgent"
            ),
            "section_generator": patch(
                "deep_research_agent.agents.lead_researcher.SectionGenerator"
            ),
            "citation_agent": patch(
                "deep_research_agent.agents.lead_researcher.CitationAgent"
            ),
        }

        # Start all patches
        self.mocks = {}
        for name, patcher in self.patches.items():
            self.mocks[name] = patcher.start()

        # Configure the mocks
        self.mocks["llm"].return_value = self.mock_model
        self.mocks[
            "web_searcher"
        ].return_value.search.return_value = self.mock_search_response

        # Mock the section generator
        def mock_section_generator(topic, analysis, section_type=None):
            if section_type == "References":
                return ReportSection(**self.mock_responses["references"])
            return ReportSection(**self.mock_responses["section"])

        self.mocks[
            "section_generator"
        ].return_value.forward.side_effect = mock_section_generator

        # Mock the citation agent
        def mock_citation_agent(content, search_results, citation_style=None):
            return {
                "content": content,
                "citations": self.mock_responses["section"]["citations"],
            }

        self.mocks[
            "citation_agent"
        ].return_value.forward.side_effect = mock_citation_agent

        # Initialize the lead researcher
        self.lead_researcher = LeadResearcher()

    def tearDown(self):
        """Clean up after the test."""
        for patcher in self.patches.values():
            patcher.stop()

    def test_full_pipeline(self):
        """Test the full research pipeline with a real-world query."""
        # Test query and strategy
        query = "What are the latest developments in quantum computing?"
        strategy = {
            "approach": "breadth-first",
            "max_depth": 2,
            "max_results": 3,
            "focus_areas": ["basics", "applications"],
            "test_mode": True,
        }

        # Run the pipeline
        report = self.lead_researcher.forward(query, strategy)

        # Basic structure checks
        self.assertIsNotNone(report)
        self.assertIsNotNone(report.title)
        self.assertTrue(len(report.sections) > 0)

        # Check for essential sections
        section_titles = [section.title for section in report.sections]
        self.assertIn("Introduction", section_titles)
        self.assertIn("References", section_titles)

        # Check that sections have content
        for section in report.sections:
            self.assertTrue(
                len(section.content) > 0, f"Section {section.title} has no content"
            )

        # Check for citations in the content
        has_citations = any(
            "[" in section.content or "(" in section.content
            for section in report.sections
        )
        self.assertTrue(has_citations, "No citations found in the report")

        # Check metadata
        self.assertIsNotNone(report.metadata)
        self.assertIn("query", report.metadata)
        self.assertIn("strategy", report.metadata)
        self.assertIn("generation_timestamp", report.metadata)


if __name__ == "__main__":
    unittest.main()
