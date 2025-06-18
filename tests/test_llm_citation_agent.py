import unittest
from unittest.mock import Mock, patch
import google.generativeai as genai
from deep_research_agent.agents.subagents.llm_citation_agent import LLMCitationAgent

class TestLLMCitationAgent(unittest.TestCase):
    def setUp(self):
        # Mock the LLM
        self.mock_llm = Mock(spec=genai.GenerativeModel)
        self.agent = LLMCitationAgent(self.mock_llm, citation_style="chicago")
        
        # Sample test data
        self.test_text = """
        Quantum computing has made significant progress in recent years.
        Researchers have achieved breakthroughs in qubit stability and error correction.
        New algorithms are being developed for practical applications.
        """
        
        self.test_sources = [
            {
                "authors": ["Smith", "Jones"],
                "year": 2023,
                "title": "Advances in Quantum Computing",
                "url": "https://example.com/quantum"
            },
            {
                "authors": ["Brown"],
                "year": 2024,
                "title": "Quantum Error Correction",
                "url": "https://example.com/error"
            }
        ]

    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.citation_style, "chicago")
        self.assertIsNotNone(self.agent.llm)

    def test_cite_text_basic(self):
        """Test basic citation functionality."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """
        Quantum computing has made significant progress in recent years [Smith, Jones, 2023].
        Researchers have achieved breakthroughs in qubit stability and error correction [Brown, 2024].
        New algorithms are being developed for practical applications.

        ## References
        Smith, Jones. 2023. "Advances in Quantum Computing." https://example.com/quantum
        Brown. 2024. "Quantum Error Correction." https://example.com/error
        """
        self.mock_llm.generate_content.return_value = mock_response

        # Call the method
        result = self.agent.cite_text(self.test_text, self.test_sources)

        # Verify the result
        self.assertIn("cited_text", result)
        self.assertIn("references", result)
        self.assertIn("citations", result)
        self.assertIn("## References", result["references"])
        self.assertEqual(len(result["citations"]), 2)

    def test_cite_text_no_references_section(self):
        """Test handling of LLM response without References section."""
        # Mock LLM response without References section
        mock_response = Mock()
        mock_response.text = """
        Quantum computing has made significant progress in recent years [Smith, Jones, 2023].
        Researchers have achieved breakthroughs in qubit stability and error correction [Brown, 2024].
        New algorithms are being developed for practical applications.
        """
        self.mock_llm.generate_content.return_value = mock_response

        # Call the method
        result = self.agent.cite_text(self.test_text, self.test_sources)

        # Verify that a References section was generated
        self.assertIn("## References", result["references"])
        self.assertEqual(len(result["citations"]), 2)

    def test_cite_text_llm_error(self):
        """Test handling of LLM errors."""
        # Mock LLM error
        self.mock_llm.generate_content.side_effect = Exception("LLM error")

        # Verify that the error is propagated
        with self.assertRaises(Exception):
            self.agent.cite_text(self.test_text, self.test_sources)

    def test_citation_validation(self):
        """Test citation validation functionality."""
        # Test text with citations
        text_with_citations = """
        Quantum computing has made significant progress [Smith, Jones, 2023].
        New developments in error correction [Brown, 2024].
        """
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = text_with_citations
        self.mock_llm.generate_content.return_value = mock_response

        # Call the method
        result = self.agent.cite_text(text_with_citations, self.test_sources)

        # Verify that all citations have corresponding references
        self.assertEqual(len(result["citations"]), 2)
        self.assertIn("Smith, Jones", result["references"])
        self.assertIn("Brown", result["references"])

if __name__ == '__main__':
    unittest.main() 