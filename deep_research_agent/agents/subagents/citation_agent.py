import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy

from deep_research_agent.utils.text_processing import calculate_similarity

from ...core.memory import ResearchMemory
from ..base_agent import BaseAgent


class Citation(dspy.Signature):
    """Structure for citation information."""

    title: str = dspy.InputField(desc="Title of the cited work")
    authors: List[str] = dspy.InputField(desc="List of authors")
    year: int = dspy.InputField(desc="Publication year")
    url: str = dspy.InputField(desc="URL of the source")
    doi: Optional[str] = dspy.InputField(default=None, desc="Digital Object Identifier")
    content: str = dspy.InputField(desc="Content being cited")
    citation_key: str = dspy.InputField(desc="Unique key for the citation")
    reference: str = dspy.InputField(desc="Formatted reference")


class CitationAgent(BaseAgent):
    """Specialized agent for managing citations and references."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.memory = ResearchMemory()
        self.citation_db: Dict[str, Citation] = {}  # citation_key -> Citation
        self.citation_style = "chicago"  # Default to Chicago style
        self.similarity_threshold = 0.6  # Lower threshold for better matching

    def forward(
        self,
        content: str,
        search_results: List[Dict[str, Any]],
        citation_style: str = "chicago",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process content and add citations based on search results.

        Args:
            content: The text content to process
            search_results: List of search results with metadata
            citation_style: Citation style to use (chicago or apa)

        Returns:
            Dict containing processed content, citations, and reference list
        """
        self.citation_style = citation_style
        self.citation_db.clear()  # Clear previous citations

        # First pass: Process all search results and build citation database
        for result in search_results:
            self._process_search_result(result)

        # Second pass: Add citations to content
        processed_content = self._add_citations_to_content(content)

        # Only include citations that were actually referenced
        referenced_citations = [
            self.citation_db[k] for k in self.referenced_citation_keys
        ]

        # Generate reference list from referenced citations
        reference_list = sorted([c.reference for c in referenced_citations])

        return {
            "content": processed_content,
            "citations": referenced_citations,
            "reference_list": reference_list,
            "metadata": {
                "citation_style": citation_style,
                "total_citations": len(referenced_citations),
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _process_search_result(self, result: Dict[str, Any]) -> None:
        """Process a search result and add it to the citation database."""
        # Extract metadata with defaults
        title = result.get("title", "Untitled")
        authors = result.get("authors", ["Unknown Author"])
        year = result.get("year", datetime.now().year)
        url = result.get("url", "")
        doi = result.get("doi", None)  # Get DOI if available
        source_content = result.get("content", "")

        # Skip results with no content
        if not source_content or len(source_content.strip()) == 0:
            return

        # Skip results with invalid URLs
        if not url or not url.startswith(("http://", "https://")):
            url = ""

        # Generate citation key
        citation_key = self._generate_citation_key(authors, year)

        # Format reference
        reference = self._format_reference(title, authors, year, url)

        # Add to citation database
        self.citation_db[citation_key] = Citation(
            title=title,
            authors=authors,
            year=year,
            url=url,
            doi=doi,  # Pass the DOI (which may be None)
            content=source_content,
            citation_key=citation_key,
            reference=reference,
        )

    def _add_citations_to_content(self, content: str) -> str:
        """Add citations to content using the citation database and only
        include referenced citations."""
        # Track which citation keys are actually used
        self.referenced_citation_keys = set()
        sentences = re.split(r"(?<=[.!?])\s+", content)
        processed_sentences = []
        for sentence in sentences:
            matching_citations = []
            for key, citation in self.citation_db.items():
                if (
                    calculate_similarity(sentence, citation.content)
                    >= self.similarity_threshold
                ):
                    matching_citations.append(citation)
                    self.referenced_citation_keys.add(key)
            if matching_citations:
                matching_citations.sort(key=lambda x: x.year, reverse=True)
                citations_text = self._format_citation_group(matching_citations)
                processed_sentences.append(f"{sentence} {citations_text}")
            else:
                processed_sentences.append(sentence)
        return " ".join(processed_sentences)

    def _format_citation_group(self, citations: List[Citation]) -> str:
        """Format a group of citations together."""
        if not citations:
            return ""
        # Sort citations by their index in the citation database
        sorted_citations = sorted(
            citations, key=lambda x: list(self.citation_db.keys()).index(x.citation_key)
        )
        citation_numbers = [str(i + 1) for i, _ in enumerate(sorted_citations)]
        return f"[{','.join(citation_numbers)}]"

    def _generate_reference_list(self) -> List[str]:
        """Generate a sorted list of references from the citation database."""
        # Sort citations by their order in the database
        sorted_citations = sorted(
            self.citation_db.values(),
            key=lambda x: list(self.citation_db.keys()).index(x.citation_key),
        )
        # Format references with numbers
        return [
            f"[{i+1}] {citation.reference}"
            for i, citation in enumerate(sorted_citations)
        ]

    def _generate_citation_key(self, authors: List[str], year: int) -> str:
        """Generate a citation key from authors and year."""
        return f"{authors[0]}_{year}"

    def _format_reference(
        self, title: str, authors: List[str], year: int, url: str
    ) -> str:
        """Format reference based on style."""
        # Format authors based on style
        if self.citation_style == "apa":
            if len(authors) > 2:
                authors_str = f"{authors[0]}, {authors[1]}, & {authors[2]}"
            elif len(authors) == 2:
                authors_str = f"{authors[0]} & {authors[1]}"
            else:
                authors_str = authors[0]
        else:  # Chicago style
            if len(authors) > 2:
                authors_str = f"{authors[0]}, {authors[1]}, and {authors[2]}"
            elif len(authors) == 2:
                authors_str = f"{authors[0]} and {authors[1]}"
            else:
                authors_str = authors[0]

        # Validate URL
        if not url or not url.startswith(("http://", "https://")):
            url_text = "[No valid URL available]"
        else:
            # Format URL as a markdown link
            url_text = f"[{url}]({url})"

        # Get the citation number from the database
        citation_key = self._generate_citation_key(authors, year)
        if citation_key in self.citation_db:
            citation_number = list(self.citation_db.keys()).index(citation_key) + 1
        else:
            # If not in database yet, use the next available number
            citation_number = len(self.citation_db) + 1

        if self.citation_style == "apa":
            return f"[{citation_number}] {authors_str}. ({year}). {title}. {url_text}"
        return f'[{citation_number}] {authors_str}. {year}. "{title}." {url_text}'

    def add_citation(self, citation: Citation) -> str:
        """Add a new citation to the collection."""
        citation_key = citation.get("citation_key", f"cite_{len(self.citation_db)}")
        self.citation_db[citation_key] = citation
        return citation_key

    def get_citation(self, citation_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a citation by its key."""
        return self.citation_db.get(citation_key)

    def list_citations(self) -> List[Dict[str, Any]]:
        """List all citations."""
        return list(self.citation_db.values())
