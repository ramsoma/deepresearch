import json
from typing import List, Dict, Any, Optional
import logging
import re
import os
import google.generativeai as genai
from ...core.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

class LLMCitationAgent:
    """
    LLM-based agent for inserting citations and generating a references section.
    Uses Gemini Pro for improved citation handling.
    """
    def __init__(self, llm=None, citation_style: str = "chicago"):
        # Initialize with Gemini Pro if no model provided
        self.llm = llm or genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-pro"))
        self.citation_style = citation_style
        logger.info(f"Initialized LLMCitationAgent with citation style: {citation_style}")

    def cite_text(self, text: str, sources: List[Dict[str, Any]], style: Optional[str] = None) -> Dict[str, str]:
        """
        Insert citations into text using Gemini Pro and generate a references section.
        Args:
            text: The draft research text (section or full report)
            sources: List of dicts with keys: author(s), year, title, url
            style: Citation style (default: self.citation_style)
        Returns:
            Dict with 'cited_text', 'references', and 'citations'
        """
        logger.info(f"Starting citation generation for text of length {len(text)}")
        logger.debug(f"Input text: {text[:200]}...")  # Log first 200 chars of text
        
        citation_style = style or self.citation_style
        logger.info(f"Using citation style: {citation_style}")
        
        # Log sources
        logger.info(f"Processing {len(sources)} sources")
        for i, source in enumerate(sources):
            logger.debug(f"Source {i+1}: {json.dumps(source, indent=2)}")
        
        # Format sources for the prompt
        sources_md = "\n".join([
            f"{i+1}. {', '.join(s.get('authors', ['Unknown Author']))} ({s.get('year', 'n.d.')}). {s.get('title', 'Untitled')}. {s.get('url', '[No URL available]')}"
            for i, s in enumerate(sources)
        ])
        logger.debug(f"Formatted sources markdown:\n{sources_md}")
        
        prompt = f"""
You are a research assistant specializing in academic citations. Your task is to insert in-text citations into the following text using ONLY the provided sources. Follow these rules EXACTLY:

Citation Rules:
1. Use numeric citations in square brackets: [1], [2,3], etc.
2. Multiple citations should be comma-separated: [1,2,3]
3. Citations should be placed at the end of the relevant sentence
4. Every significant claim MUST be cited
5. Use ONLY the provided sources, do not invent or hallucinate
6. Do not use generic citations like [Analysis, Year]
7. Do not use placeholder citations like [cite1]
8. Do not hallucinate or modify URLs - use them exactly as provided
9. Do not add markdown formatting to URLs in the references section
10. Only cite sources that are actually used in the text

Text to cite:
{text}

Available sources:
{sources_md}

IMPORTANT: Format your response exactly like this:
[Your cited text here]

## References
[Your references here]

Remember:
- Every claim needs a citation
- Use only the provided sources
- Use numeric citations [1], [2,3], etc.
- Include a complete References section
- Do not use placeholder or generic citations
- Do not modify or hallucinate URLs
- Only include references for sources that are actually cited
"""
        logger.debug(f"Generated prompt:\n{prompt}")
        
        try:
            response = self.llm.generate_content(prompt)
            logger.info("Successfully received response from Gemini Pro")
            logger.debug(f"LLM response:\n{response.text[:500]}...")  # Log first 500 chars of response
        except Exception as e:
            logger.error(f"Error getting response from Gemini Pro: {str(e)}")
            raise
        
        cited_text = response.text.strip()
        
        # Extract the References section
        references_section = ""
        if "## References" in cited_text:
            references_section = cited_text.split("## References")[1].strip()
            cited_text = cited_text.split("## References")[0].strip()
            logger.info("Successfully extracted References section")
        else:
            logger.warning("No References section found in Gemini Pro response")
            references_section = ""
        
        # Format citations with both text and reference fields
        citations = []
        for i, s in enumerate(sources):
            authors = ', '.join(s.get('authors', ['Unknown Author']))
            year = s.get('year', 'n.d.')
            title = s.get('title', 'Untitled')
            url = s.get('url', '')
            
            # Format the citation text to match in-text citations
            text = f"[{i+1}]"
            
            # Format the reference with proper URL handling
            if url:
                if citation_style == "apa":
                    reference = f"{authors}. ({year}). {title}. {url}"
                else:  # chicago
                    reference = f"{authors}. {year}. \"{title}.\" {url}"
            else:
                if citation_style == "apa":
                    reference = f"{authors}. ({year}). {title}."
                else:  # chicago
                    reference = f"{authors}. {year}. \"{title}.\""
            
            citations.append({
                "id": str(i+1),
                "text": text,
                "reference": reference,
                "url": url
            })
            logger.debug(f"Formatted citation: {text} -> {reference}")
        
        # Validate that all citations in the text have corresponding references
        citation_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        found_citations = []
        for match in re.finditer(citation_pattern, cited_text):
            citation_text = match.group(0)
            # Extract citation numbers
            numbers = [int(n) for n in match.group(1).split(',')]
            found_citations.extend(numbers)
        
        # Filter citations to only include those that are actually used in the text
        used_citations = [c for c in citations if int(c["id"]) in found_citations]
        
        formatted_references = ""
        # Format references section with only used citations
        # Sort used citations by their numeric ID
        used_citations.sort(key=lambda x: int(x["id"]))
        for citation in used_citations:
            # Format reference with number at the start
            formatted_references += f"[{citation['id']}] {citation['reference']}\n\n"
        
        logger.info(f"Generated {len(used_citations)} citations")
        logger.debug(f"Final references section:\n{formatted_references}")
        
        # Log any missing references
        for i in range(1, len(sources) + 1):
            if i not in found_citations:
                logger.warning(f"Citation [{i}] not found in text")
        
        # Check for duplicate citations
        citation_counts = {}
        for match in re.finditer(citation_pattern, cited_text):
            citation_text = match.group(0)
            citation_counts[citation_text] = citation_counts.get(citation_text, 0) + 1
            if citation_counts[citation_text] > 1:
                logger.warning(f"Duplicate citation found: {citation_text}")
        
        return {
            "cited_text": cited_text,
            "references": formatted_references,
            "citations": used_citations
        } 