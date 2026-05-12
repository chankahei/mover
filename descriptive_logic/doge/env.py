from __future__ import annotations


def load_environment() -> None:
    """Load .env from the current working directory or its parents."""

    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
