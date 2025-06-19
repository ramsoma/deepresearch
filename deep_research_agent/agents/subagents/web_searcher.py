import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import dspy
from tavily import TavilyClient

from ...core.memory import ResearchMemory
from ..base_agent import BaseAgent


class SearchResult(dspy.Signature):
    """Structure for search results."""

    title: str = dspy.InputField(desc="Title of the search result")
    url: str = dspy.InputField(desc="URL of the result")
    content: str = dspy.InputField(desc="Main content of the result")
    score: float = dspy.InputField(desc="Relevance score")


class WebSearcher(BaseAgent):
    """Specialized agent for web search tasks using Tavily API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.memory = ResearchMemory()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def forward(
        self,
        query: str,
        search_depth: int = 3,
        max_results: int = 5,
        search_type: str = "basic",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute web search with iterative refinement."""
        findings = []

        # Initial search
        initial_results = self._perform_search(query, search_type)
        findings.extend(initial_results)

        # Iterative refinement
        for depth in range(search_depth - 1):
            refined_query = self._refine_query(query, findings)
            new_results = self._perform_search(refined_query, search_type)
            findings.extend(new_results)

        # Rank and return results
        ranked_results = self._rank_results(findings)
        return {
            "query": query,
            "results": ranked_results[:max_results],
            "metadata": {
                "search_depth": search_depth,
                "total_results": len(findings),
                "search_type": search_type,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _perform_search(
        self, query: str, search_type: str = "basic"
    ) -> List[Dict[str, Any]]:
        """Perform web search using Tavily API."""
        try:
            # Execute search
            search_response = self.tavily_client.search(
                query=query,
                search_depth=search_type,
                include_answer=True,
                include_raw_content=True,
                max_results=5,  # Limit to 5 most relevant results
                answer_length=500,  # Limit answer to 500 characters
                content_length=1000,  # Limit raw content to 1000 characters
            )

            # Process results
            results = []

            # Handle search results
            if isinstance(search_response, dict) and "results" in search_response:
                for result in search_response["results"]:
                    if isinstance(result, dict):
                        processed_result = self._process_search_result(result)
                        if processed_result:
                            results.append(processed_result)

            # Handle answer if available
            if isinstance(search_response, dict) and "answer" in search_response:
                answer = search_response["answer"]
                if isinstance(answer, dict):
                    results.append(self._process_answer(answer))

            return results

        except Exception as e:
            print(f"Error performing search: {str(e)}")
            return []

    def _process_search_result(
        self, result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single search result from Tavily."""
        try:
            # Extract metadata from the result
            metadata = result.get("metadata", {})

            # Extract authors - try different possible fields
            authors = []
            if "authors" in metadata:
                authors = metadata["authors"]
            elif "author" in metadata:
                authors = [metadata["author"]]
            elif "creator" in metadata:
                authors = [metadata["creator"]]

            # Extract year - try different possible fields
            year = None
            if "date" in metadata:
                try:
                    year = int(metadata["date"][:4])  # Try to get year from date string
                except (ValueError, TypeError):
                    pass
            elif "published_date" in metadata:
                try:
                    year = int(metadata["published_date"][:4])
                except (ValueError, TypeError):
                    pass

            # If no year found, use current year
            if not year:
                year = datetime.now().year

            # If no authors found, use domain as author
            if not authors:
                try:
                    domain = urlparse(result.get("url", "")).netloc
                    if domain:
                        authors = [domain.replace("www.", "")]
                    else:
                        authors = ["Unknown Author"]
                except:  # noqa
                    authors = ["Unknown Author"]

            # Handle URL validation
            url = result.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                return None  # Skip results with invalid URLs
            # Note: Empty URLs are allowed and will be kept as empty strings

            # Validate content
            content = result.get("content", "")
            if not content or len(content.strip()) == 0:
                return None  # Skip results with no content

            return {
                "title": result.get("title", ""),
                "url": url,
                "content": content,
                "score": result.get("score", 0.0),
                "authors": authors,
                "year": year,
                "metadata": metadata,
            }
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            return None

    def _process_answer(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Process Tavily's generated answer."""
        return {
            "title": "Generated Answer",
            "url": "",
            "content": answer.get("answer", ""),
            "score": 1.0,  # Highest relevance for generated answer
        }

    def _refine_query(
        self, original_query: str, current_findings: List[Dict[str, Any]]
    ) -> str:
        """Refine search query based on current findings."""
        # Simple refinement: add "latest" to get more recent results
        return f"{original_query} latest"

    def _rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank search results by relevance score."""
        # Sort by score in descending order
        return sorted(results, key=lambda x: float(x.get("score", 0)), reverse=True)
