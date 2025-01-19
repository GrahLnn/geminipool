from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    gemini_api_keys: Optional[List[str]] = Field(default=None, validate_default=True)
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com")
    rpm: int = Field(default=10)
    allow_concurrent: bool = Field(default=False)
    model: str = Field(default="gemini-2.0-flash-exp")