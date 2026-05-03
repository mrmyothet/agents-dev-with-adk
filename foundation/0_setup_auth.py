"""Part 0: Setup and authentication for ADK examples.

Based on DevFest_BwAI/ADK_Learning_tools.ipynb.
"""

import os
from getpass import getpass

import google.generativeai as genai


def configure_google_api_key() -> str:
    """Prompt for a Google API key and configure SDKs."""
    api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API Key: ")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


def main() -> None:
    configure_google_api_key()
    print("Setup complete. GOOGLE_API_KEY is configured.")


if __name__ == "__main__":
    main()
