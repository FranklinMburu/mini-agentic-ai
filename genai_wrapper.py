"""
Thin compatibility layer over Google's current Gen AI SDK (google-genai).

The project originally wrapped the deprecated google-generativeai package and
called generate_text(), which no longer exists. This module now re-exports the
official google-genai Client, so callers keep using:

    client = Client(api_key=...)
    client.models.generate_content(model="gemini-2.5-flash", contents="...")
"""

from google.genai import Client  # official SDK: pip install google-genai

__all__ = ["Client"]
