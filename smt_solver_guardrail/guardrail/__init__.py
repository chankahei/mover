"""SMT Solver Guardrail package.

Loads environment variables from a local ``.env`` (if present) so the LLM
client picks up ``OPENAI_API_KEY`` / ``BASE_URL`` / model settings.
"""

from __future__ import annotations

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # python-dotenv optional at import time
    pass

__all__ = ["__version__"]
__version__ = "0.1.0"
