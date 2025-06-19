import logging
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages Jinja templates for the research system."""

    def __init__(self):
        """Initialize the template manager."""
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the package root
        package_root = os.path.dirname(current_dir)
        # Set up the template directory
        template_dir = os.path.join(package_root, "templates")

        # Create templates directory if it doesn't exist
        os.makedirs(template_dir, exist_ok=True)

        # Set up Jinja environment
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=select_autoescape(["html", "xml"]),
        )

        logger.info(
            f"Initialized template manager with template directory: {template_dir}"
        )

    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with the given context."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}")
            raise

    def get_template_names(self) -> list:
        """Get a list of available template names."""
        return self.env.list_templates()

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists."""
        return self.env.get_template(template_name) is not None

    def get_required_sections(self) -> list:
        """Return the standard required sections for a research report."""
        return [
            "Executive Summary",
            "Introduction",
            "Main Findings",
            "Analysis and Discussion",
            "Conclusions",
            "Recommendations",
            "References",
        ]
