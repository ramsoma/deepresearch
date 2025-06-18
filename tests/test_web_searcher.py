import unittest
from deep_research_agent.agents.subagents.web_searcher import WebSearcher
from unittest.mock import patch, MagicMock
import os

class TestWebSearcher(unittest.TestCase):
    def setUp(self):
        # Patch TavilyClient at the correct import path before instantiating WebSearcher
        self.tavily_patcher = patch('deep_research_agent.agents.subagents.web_searcher.TavilyClient')
        self.mock_tavily = self.tavily_patcher.start()
        # Mock the TAVILY_API_KEY environment variable
        self.env_patcher = patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
        self.env_patcher.start()
        self.searcher = WebSearcher()
    
    def tearDown(self):
        self.tavily_patcher.stop()
        self.env_patcher.stop()
    
    def test_metadata_extraction_and_url_validation(self):
        # Mock Tavily API response with valid and invalid URLs
        mock_response = {
            "results": [
                {
                    "title": "Valid URL Article",
                    "url": "https://example.com/valid",
                    "content": "Valid content.",
                    "score": 0.9,
                    "metadata": {
                        "authors": ["John Doe"],
                        "date": "2023-05-15"
                    }
                },
                {
                    "title": "Invalid URL Article",
                    "url": "not-a-valid-url",
                    "content": "Invalid URL content.",
                    "score": 0.8,
                    "metadata": {
                        "author": "Jane Smith",
                        "published_date": "2022-12-01"
                    }
                },
                {
                    "title": "No URL Article",
                    "content": "No URL content.",
                    "score": 0.7,
                    "metadata": {}
                }
            ],
            "answer": None  # Add this to match the expected response structure
        }
        self.searcher.tavily_client.search.return_value = mock_response
        results = self.searcher.forward("test query", max_results=3, search_depth=1)
        processed_results = results.get("results", [])
        print("Processed result titles:", [r["title"] for r in processed_results])
        self.assertEqual(len(processed_results), 2)  # Only valid and no-url articles should be included
        # Valid URL
        valid = next(r for r in processed_results if r["title"] == "Valid URL Article")
        self.assertEqual(valid["url"], "https://example.com/valid")
        self.assertEqual(valid["authors"], ["John Doe"])
        self.assertEqual(valid["year"], 2023)
        # No URL (should be empty string)
        no_url = next(r for r in processed_results if r["title"] == "No URL Article")
        self.assertEqual(no_url["url"], "")
        self.assertEqual(no_url["authors"], ["Unknown Author"])  # Domain fallback fails, so Unknown Author
        self.assertIsInstance(no_url["year"], int)
    
    def test_content_filtering(self):
        # Mock Tavily API response with empty content
        mock_response = {
            "results": [
                {
                    "title": "Empty Content Article",
                    "url": "https://example.com/empty",
                    "content": "",
                    "score": 0.5,
                    "metadata": {"authors": ["Nobody"]}
                }
            ]
        }
        self.searcher.tavily_client.search.return_value = mock_response
        results = self.searcher.forward("test query", max_results=1)
        processed_results = results.get("results", [])
        self.assertEqual(len(processed_results), 0)  # Should filter out empty content

if __name__ == '__main__':
    unittest.main() 