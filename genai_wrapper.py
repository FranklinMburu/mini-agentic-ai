"""
Wrapper for Google Generative AI Client to match expected API.
Maps the old google.generativeai API to a Client-based interface.
"""

import google.generativeai as _genai
from typing import Optional


class GenerateContentResponse:
    """Wrapper for generate_content response."""
    def __init__(self, text: str):
        self.text = text


class ModelsAPI:
    """Wrapper for models API."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            _genai.configure(api_key=api_key)
    
    def generate_content(self, model: str, contents: str) -> GenerateContentResponse:
        """Generate content using the provided model."""
        try:
            response = _genai.generate_text(
                prompt=contents,
                model=model or "models/text-bison-001"
            )
            text = response.result or ""
            return GenerateContentResponse(text)
        except Exception as e:
            # Try alternative API if available
            raise Exception(f"Failed to generate content: {str(e)}")


class Client:
    """Google Generative AI Client wrapper."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.models = ModelsAPI(api_key)
        if api_key:
            _genai.configure(api_key=api_key)


# Export a default module-level interface that matches "from google import genai"
class _GenaiModule:
    """Module wrapper to mimic 'from google import genai'."""
    Client = Client
    
    def __getattr__(self, name):
        """Delegate to the actual google.generativeai module."""
        return getattr(_genai, name)


# Create module instance
_instance = _GenaiModule()

# Override module in sys.modules to allow imports like "from google import genai"
import sys
sys.modules['google.genai'] = _instance

# Make Client available
Client = Client
