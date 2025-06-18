import pytest
from deep_research_agent.agents.lead_researcher import LeadResearcher, ResearchPlan, ContentAnalysis, ReportSection
import logging
import json
from unittest.mock import MagicMock, patch
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'LLM_MODEL': 'gemini-pro',
        'TAVILY_API_KEY': 'test-key'
    }):
        yield

@pytest.fixture
def mock_llm():
    """Create a mock LLM model that returns a realistic JSON string with sections."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "title": "Test Report",
        "sections": [
            {
                "title": "Introduction",
                "content": "This is an introduction with a citation [Smith, J. 2023].",
                "key_points": ["Point 1"],
                "citations": [
                    {
                        "id": "cite1",
                        "text": "Smith, J. (2023). Test Paper",
                        "url": "https://example.com/paper1"
                    }
                ]
            },
            {
                "title": "Background",
                "content": "Background content.",
                "key_points": ["Background Point"],
                "citations": []
            },
            {
                "title": "References",
                "content": "This section lists all sources cited in the report.\n\n1. Smith, J. (2023). Test Paper. https://example.com/paper1",
                "key_points": [],
                "citations": []
            }
        ],
        "metadata": {
            "query": "Test query",
            "strategy": {"test_mode": True},
            "plan": {
                "main_topics": ["Topic 1"],
                "subtopics": {"Topic 1": ["Subtopic 1"]},
                "key_questions": ["Question 1"],
                "required_sources": {"Topic 1": 2}
            },
            "analyses": {
                "Topic 1": {
                    "main_points": ["Point 1"],
                    "key_insights": ["Insight 1"],
                    "technical_details": ["Detail 1"],
                    "applications": ["Application 1"],
                    "future_directions": ["Direction 1"],
                    "confidence_score": 0.9
                }
            }
        }
    })
    mock_model.generate_content.return_value = mock_response
    return mock_model

@pytest.fixture
def mock_tavily():
    """Create a mock Tavily client."""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com/test",
                "content": "Test content",
                "score": 0.9,
                "metadata": {
                    "authors": ["Test Author"],
                    "date": "2024-01-01"
                }
            }
        ]
    }
    return mock_client

@pytest.fixture
def test_config():
    """Test configuration with minimal parameters."""
    return {
        "sections": ["Introduction", "Background"],  # Only 2 sections for testing
        "max_depth": 1,  # Minimal depth
        "max_results": 2,  # Few results for testing
        "citation_style": "chicago",
        "test_mode": True  # Flag to indicate test mode
    }

@pytest.fixture
def lead_researcher(mock_llm, mock_tavily):
    """Create a LeadResearcher instance for testing."""
    with patch('deep_research_agent.agents.lead_researcher.genai.GenerativeModel', return_value=mock_llm), \
         patch('deep_research_agent.agents.subagents.web_searcher.TavilyClient', return_value=mock_tavily):
        return LeadResearcher()

@pytest.fixture(autouse=True)
def patch_section_generator():
    """Patch section_generator.forward to return predictable sections for testing."""
    def fake_forward(topic, analysis, section_type, **kwargs):
        # For References, include a dummy reference in the content
        if section_type == "References":
            return ReportSection(
                title=section_type,
                content="1. Doe, 2023. Test Reference. https://example.com",
                key_points=[],
                citations=[]
            )
        return ReportSection(
            title=section_type,
            content=f"Content for {section_type}",
            key_points=[f"Key point for {section_type}"],
            citations=[{"text": f"[Doe, 2023]"}] if section_type != "References" else []
        )
    with patch('deep_research_agent.agents.lead_researcher.SectionGenerator') as MockSectionGen:
        instance = MockSectionGen.return_value
        instance.forward.side_effect = fake_forward
        yield

def test_minimal_report_generation(lead_researcher, test_config):
    """Test report generation with minimal sections and depth."""
    query = "Test quantum computing basics"
    strategy = {
        "approach": "breadth-first",
        "max_depth": test_config["max_depth"],
        "max_results": test_config["max_results"],
        "focus_areas": ["basics", "applications"],
        "test_mode": test_config["test_mode"]
    }
    # Generate report
    report = lead_researcher.forward(query, strategy)
    # Log the actual section titles for debugging
    print("Generated section titles:", [section.title for section in report.sections])
    # Basic validation
    assert report is not None
    # Check that all expected section titles are present
    expected_sections = ["Executive Summary", "Introduction", "Main Findings", "Analysis and Discussion", "Conclusions", "Recommendations", "References"]
    for expected in expected_sections:
        assert any(expected in section.title for section in report.sections), f"Section '{expected}' not found in report."
    # Check citation formatting
    for section in report.sections:
        # Log section content for debugging
        logger.info(f"Section: {section.title}")
        logger.info(f"Content: {section.content[:200]}...")
        # Check for properly formatted citations (if any)
        if "[" in section.content and "]" in section.content:
            citations = section.content[section.content.find("["):section.content.find("]")+1]
            logger.info(f"Found citation: {citations}")
            assert "," in citations, "Citation not properly formatted"

def test_citation_handling(mock_llm, mock_tavily):
    """Test that citations are properly processed and referenced."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "title": "Test Report",
        "sections": [
            {
                "title": "Introduction",
                "content": "This is an introduction with a citation [Smith, J. 2023].",
                "key_points": ["Point 1"],
                "citations": [
                    {
                        "id": "cite1",
                        "text": "Smith, J. (2023). Test Paper",
                        "url": "https://example.com/paper1"
                    }
                ]
            },
            {
                "title": "Main Findings",
                "content": "Main findings with another citation [Johnson, A. 2024].",
                "key_points": ["Finding 1"],
                "citations": [
                    {
                        "id": "cite2",
                        "text": "Johnson, A. (2024). Another Paper",
                        "url": "https://example.com/paper2"
                    }
                ]
            }
        ],
        "metadata": {
            "query": "Test query",
            "strategy": {"test_mode": True},
            "plan": {
                "main_topics": ["Topic 1"],
                "subtopics": {"Topic 1": ["Subtopic 1"]},
                "key_questions": ["Question 1"],
                "required_sources": {"Topic 1": 2}
            },
            "analyses": {
                "Topic 1": {
                    "main_points": ["Point 1"],
                    "key_insights": ["Insight 1"],
                    "technical_details": ["Detail 1"],
                    "applications": ["Application 1"],
                    "future_directions": ["Direction 1"],
                    "confidence_score": 0.9
                }
            }
        }
    })
    # Configure mock model
    mock_llm.generate_content.return_value = mock_response
    # Create the researcher with the mock model
    with patch('deep_research_agent.agents.lead_researcher.genai.GenerativeModel', return_value=mock_llm), \
         patch('deep_research_agent.agents.subagents.web_searcher.TavilyClient', return_value=mock_tavily):
        researcher = LeadResearcher()
        # Create a test query and strategy
        query = "Test research query"
        strategy = {
            "test_mode": True,
            "sections": ["Introduction", "Main Findings"]
        }
        # Generate the report
        report = researcher.forward(query, strategy)
        # Verify the report structure
        assert report.title is not None
        assert len(report.sections) > 0
        # Find the References section
        references_section = None
        for section in report.sections:
            if "References" in section.title:
                references_section = section
                break
        print(references_section)
        assert references_section is not None, "References section not found"
        # Check if the References section has content
        assert len(references_section.content.strip()) > 0, "References section is empty"
        # Check that at least one reference is present
        assert any("http" in line or "[" in line for line in references_section.content.splitlines()), "No references found in References section."

def test_report_to_markdown(lead_researcher, test_config):
    """Test that ResearchReport.to_markdown produces expected markdown output."""
    query = "Test quantum computing basics"
    strategy = {
        "approach": "breadth-first",
        "max_depth": test_config["max_depth"],
        "max_results": test_config["max_results"],
        "focus_areas": ["basics", "applications"],
        "test_mode": test_config["test_mode"]
    }
    report = lead_researcher.forward(query, strategy)
    md = report.to_markdown()
    # Basic checks
    assert md.startswith(f"# {report.title}")
    for section in report.sections:
        assert f"## {section.title}" in md
        # Check that the section has some content (not necessarily the exact string)
        assert len(section.content) > 0, f"Section {section.title} has no content"
        # For References section, just check that it has content
        if section.title == "References":
            assert "## References" in md
            assert len(section.content) > 0, "References section has no content"
        else:
            assert section.content in md, f"Section {section.title} content not found in markdown"

if __name__ == "__main__":
    # Run tests directly
    test_config = {
        "sections": ["Introduction", "Background"],
        "max_depth": 1,
        "max_results": 2,
        "citation_style": "chicago",
        "test_mode": True
    }
    
    researcher = LeadResearcher()
    test_minimal_report_generation(researcher, test_config)
    test_citation_handling()
    test_report_to_markdown(researcher, test_config) 