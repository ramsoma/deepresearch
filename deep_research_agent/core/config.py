import os
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for individual agents."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    additional_params: Dict[str, Any] = {}


class SystemConfig(BaseModel):
    """System-wide configuration."""

    openai_api_key: str
    default_model: str = "gpt-4"
    agents: Dict[str, AgentConfig] = {}


def load_config() -> SystemConfig:
    """Load configuration from environment variables."""
    load_dotenv()

    return SystemConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4"),
        agents={},
    )
