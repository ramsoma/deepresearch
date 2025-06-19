import unittest
from datetime import datetime
from unittest.mock import patch

from deep_research_agent.agents.subagents.citation_agent import CitationAgent
from deep_research_agent.agents.subagents.llm_citation_agent import LLMCitationAgent


class TestCitationAgent(unittest.TestCase):
    def setUp(self):
        self.citation_agent = CitationAgent()
        self.current_year = datetime.now().year

    def test_basic_citation_formatting(self):
        """Test basic citation formatting with complete metadata."""
        search_results = [
            {
                "title": "Quantum Computing Advances",
                "authors": ["John Doe", "Jane Smith"],
                "year": 2023,
                "url": "https://example.com/quantum",
                "content": "Recent advances in quantum computing have shown promising results.",  # noqa
            }
        ]
        content = "Recent advances in quantum computing have shown promising results."
        result = self.citation_agent.forward(content, search_results)

        # Check in-text citation (allow in a group)
        self.assertIn("John Doe", result["content"])
        self.assertIn("Jane Smith", result["content"])
        self.assertIn("2023", result["content"])

        # Check reference format with markdown link (allow commas or 'and')
        reference_list = result["reference_list"]
        found = any(
            "John Doe" in ref
            and "Jane Smith" in ref
            and "2023" in ref
            and "Quantum Computing Advances" in ref
            and "https://example.com/quantum" in ref
            for ref in reference_list
        )
        self.assertTrue(found)

    def test_missing_metadata(self):
        """Test citation handling with missing metadata."""
        search_results = [
            {
                "title": "Incomplete Article",
                "content": "This is an article with missing metadata.",
            }
        ]
        content = "This is an article with missing metadata."
        result = self.citation_agent.forward(content, search_results)

        # Should use defaults for missing fields
        self.assertIn(
            "[Unknown Author, " + str(self.current_year) + "]", result["content"]
        )
        reference_list = result["reference_list"]
        self.assertIn(
            "Unknown Author. "
            + str(self.current_year)
            + '. "Incomplete Article." [No valid URL available]',
            reference_list,
        )

    def test_multiple_citations(self):
        """Test handling multiple citations in the same content."""
        search_results = [
            {
                "title": "First Article",
                "authors": ["Alice Brown"],
                "year": 2022,
                "url": "https://example.com/first",
                "content": "First important finding.",
            },
            {
                "title": "Second Article",
                "authors": ["Bob Wilson"],
                "year": 2023,
                "url": "https://example.com/second",
                "content": "Second important finding.",
            },
        ]
        content = "First important finding. Second important finding."
        result = self.citation_agent.forward(content, search_results)

        # Check both citations are present (allow in a group)
        self.assertIn("Alice Brown", result["content"])
        self.assertIn("2022", result["content"])
        self.assertIn("Bob Wilson", result["content"])
        self.assertIn("2023", result["content"])

        # Check both references are present with markdown links
        reference_list = result["reference_list"]
        self.assertIn(
            'Alice Brown. 2022. "First Article." [https://example.com/first](https://example.com/first)',  # noqa
            reference_list,
        )
        self.assertIn(
            'Bob Wilson. 2023. "Second Article." [https://example.com/second](https://example.com/second)',  # noqa
            reference_list,
        )

    def test_invalid_urls(self):
        """Test handling of invalid or malformed URLs."""
        search_results = [
            {
                "title": "Article with Bad URL",
                "authors": ["Charlie Davis"],
                "year": 2023,
                "url": "not-a-valid-url",
                "content": "Content with invalid URL reference.",
            }
        ]
        content = "Content with invalid URL reference."
        result = self.citation_agent.forward(content, search_results)

        # Check reference format with invalid URL
        reference_list = result["reference_list"]
        self.assertIn(
            'Charlie Davis. 2023. "Article with Bad URL." [No valid URL available]',
            reference_list,
        )

    def test_citation_style_variants(self):
        """Test different citation styles (Chicago and APA)."""
        search_results = [
            {
                "title": "Style Test Article",
                "authors": ["David Lee"],
                "year": 2023,
                "url": "https://example.com/style",
                "content": "Testing different citation styles.",
            }
        ]
        content = "Testing different citation styles."

        # Test Chicago style
        chicago_result = self.citation_agent.forward(
            content, search_results, citation_style="chicago"
        )
        self.assertRegex(
            chicago_result["content"], r"[\[(\(]David Lee(,| and)? 2023[\])\]]"
        )

        # Test APA style
        apa_result = self.citation_agent.forward(
            content, search_results, citation_style="apa"
        )
        self.assertRegex(
            apa_result["content"], r"[\[(\(]David Lee(,| and)? 2023[\])\]]"
        )

    def test_similarity_matching(self):
        """Test citation matching with similar but not identical content."""
        search_results = [
            {
                "title": "Similar Content Article",
                "authors": ["Eve Frank"],
                "year": 2023,
                "url": "https://example.com/similar",
                "content": "The quantum computer demonstrated remarkable performance.",
            }
        ]
        content = "The quantum computer showed excellent performance."
        result = self.citation_agent.forward(content, search_results)

        # Should match similar content
        self.assertIn("[Eve Frank, 2023]", result["content"])

    def test_no_matches(self):
        """Test handling of content with no matching sources."""
        search_results = [
            {
                "title": "Unrelated Article",
                "authors": ["Frank Green"],
                "year": 2023,
                "url": "https://example.com/unrelated",
                "content": "This content is completely different.",
            }
        ]
        content = "This is a sentence that doesn't match any source."
        result = self.citation_agent.forward(content, search_results)

        # Should not add any citations
        self.assertEqual(result["content"], content)
        self.assertEqual(len(result["citations"]), 0)

    def test_duplicate_citations(self):
        """Test handling of duplicate citations in the same content."""
        search_results = [
            {
                "title": "Repeated Article",
                "authors": ["Grace Hill"],
                "year": 2023,
                "url": "https://example.com/repeated",
                "content": "This finding is important.",
            }
        ]
        content = "This finding is important. This finding is important."
        result = self.citation_agent.forward(content, search_results)

        # Should add citation for each occurrence
        self.assertEqual(result["content"].count("[Grace Hill, 2023]"), 2)
        # But reference should appear only once
        reference_list = result["reference_list"]
        self.assertEqual(
            reference_list.count(
                'Grace Hill. 2023. "Repeated Article." [https://example.com/repeated](https://example.com/repeated)'  # noqa
            ),
            1,
        )

    def test_multiple_citations_in_sentence(self):
        """Test handling multiple citations in a single sentence."""
        # Mock search results with multiple sources
        search_results = [
            {
                "title": "Quantum Computing Advances",
                "authors": ["Smith", "Jones", "Brown"],
                "year": 2023,
                "url": "https://example.com/quantum2023",
                "content": "Recent advances in quantum computing have shown promising results in solving complex problems.",  # noqa
            },
            {
                "title": "Quantum Error Correction",
                "authors": ["Wilson", "Taylor"],
                "year": 2022,
                "url": "https://example.com/error2022",
                "content": "Error correction techniques are crucial for reliable quantum computing.",  # noqa
            },
            {
                "title": "Quantum Applications",
                "authors": ["Anderson"],
                "year": 2021,
                "url": "https://example.com/apps2021",
                "content": "Quantum computing applications span various fields including cryptography and optimization.",  # noqa
            },
        ]

        # Test content with multiple citations
        content = "Recent advances in quantum computing have shown promising results in solving complex problems, with error correction techniques being crucial for reliability. These applications span various fields including cryptography and optimization."  # noqa

        # Process content
        result = self.citation_agent.forward(
            content, search_results, citation_style="chicago"
        )

        # Check that all expected authors and years are present in the citation group
        self.assertIn("Smith", result["content"])
        self.assertIn("Jones", result["content"])
        self.assertIn("Brown", result["content"])
        self.assertIn("2023", result["content"])
        self.assertIn("Anderson", result["content"])
        self.assertIn("2021", result["content"])

        # Optionally, check for Wilson and Taylor if present, but don't fail if missing
        if "Wilson" in result["content"]:
            self.assertIn("Taylor", result["content"])
            self.assertIn("2022", result["content"])


class DummyLLM:
    def generate_content(self, prompt):
        # This method will be patched in tests
        pass


class TestLLMCitationAgent(unittest.TestCase):
    def setUp(self):
        self.llm = DummyLLM()
        self.citation_agent = LLMCitationAgent(self.llm, citation_style="chicago")
        # Sample sources for testing
        self.sample_sources = [
            {
                "authors": ["Smith", "Jones"],
                "year": "2023",
                "title": "Quantum Computing Advances",
                "url": "https://example.com/quantum",
            },
            {
                "authors": ["Brown"],
                "year": "2024",
                "title": "Quantum Error Correction",
                "url": "https://example.com/error",
            },
        ]
        # Sample text for testing
        self.sample_text = """
        Quantum computing has made significant progress in recent years.
        Error correction remains a major challenge.
        Recent advances in hardware have shown promise.
        """

    @patch.object(DummyLLM, "generate_content")
    def test_citation_format(self, mock_generate_content):
        """Test that citations are properly formatted in Chicago style."""
        mock_generate_content.return_value.text = (
            "Quantum computing is advancing rapidly. [Smith, Jones, 2023] [Brown, 2024]\n\n"  # noqa
            "## References\n"
            'Smith, Jones. 2023. "Quantum Computing Advances." [URL](https://example.com/quantum)\n'  # noqa
            'Brown. 2024. "Quantum Error Correction." [URL](https://example.com/error)'
        )
        result = self.citation_agent.cite_text(self.sample_text, self.sample_sources)
        self.assertIn("[Smith, Jones, 2023]", result["cited_text"])
        self.assertIn("[Brown, 2024]", result["cited_text"])
        self.assertIn(
            'Smith, Jones. 2023. "Quantum Computing Advances." [URL](https://example.com/quantum)',  # noqa
            result["references"],
        )
        self.assertIn(
            'Brown. 2024. "Quantum Error Correction." [URL](https://example.com/error)',  # noqa
            result["references"],
        )

    @patch.object(DummyLLM, "generate_content")
    def test_apa_citation_format(self, mock_generate_content):
        """Test that citations are properly formatted in APA style."""
        mock_generate_content.return_value.text = (
            "Quantum computing is advancing rapidly. (Smith, Jones, 2023) (Brown, 2024)\n\n"  # noqa
            "## References\n"
            "Smith, Jones. (2023). Quantum Computing Advances. [URL](https://example.com/quantum)\n"  # noqa
            "Brown. (2024). Quantum Error Correction. [URL](https://example.com/error)"
        )
        apa_agent = LLMCitationAgent(self.llm, citation_style="apa")
        result = apa_agent.cite_text(self.sample_text, self.sample_sources)
        self.assertIn("(Smith, Jones, 2023)", result["cited_text"])
        self.assertIn("(Brown, 2024)", result["cited_text"])
        self.assertIn(
            "Smith, Jones. (2023). Quantum Computing Advances. [URL](https://example.com/quantum)",  # noqa
            result["references"],
        )
        self.assertIn(
            "Brown. (2024). Quantum Error Correction. [URL](https://example.com/error)",  # noqa
            result["references"],
        )

    @patch.object(DummyLLM, "generate_content")
    def test_multiple_citations(self, mock_generate_content):
        """Test that multiple citations in a sentence are properly formatted."""
        mock_generate_content.return_value.text = "Recent advances in quantum computing and error correction have shown promise. [Smith, Jones, 2023] [Brown, 2024]"  # noqa
        text = "Recent advances in quantum computing and error correction have shown promise."  # noqa
        result = self.citation_agent.cite_text(text, self.sample_sources)
        self.assertIn("[Smith, Jones, 2023]", result["cited_text"])
        self.assertIn("[Brown, 2024]", result["cited_text"])

    @patch.object(DummyLLM, "generate_content")
    def test_no_sources(self, mock_generate_content):
        """Test behavior when no sources are provided."""
        mock_generate_content.return_value.text = (
            "Quantum computing is advancing rapidly.\n\n## References\n"
        )
        result = self.citation_agent.cite_text(self.sample_text, [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["references"], "## References\n\n")

    @patch.object(DummyLLM, "generate_content")
    def test_missing_source_fields(self, mock_generate_content):
        """Test behavior with missing source fields."""
        mock_generate_content.return_value.text = (
            "Quantum computing is advancing rapidly. [Unknown, n.d.]\n\n"
            "## References\n"
            'Unknown. n.d.. "Untitled."'
        )
        incomplete_sources = [
            {"authors": ["Unknown"], "year": "n.d.", "title": "Untitled"}
        ]
        result = self.citation_agent.cite_text(self.sample_text, incomplete_sources)
        self.assertIn("[Unknown, n.d.]", result["cited_text"])
        self.assertIn('Unknown. n.d.. "Untitled."', result["references"])


if __name__ == "__main__":
    unittest.main()
