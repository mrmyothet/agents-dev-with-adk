"""Part 0: Setup, authentication, and shared helper.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"


def configure_google_api_key() -> None:
    """Configure GOOGLE_API_KEY for Gemini and ADK."""
    api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API Key: ")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key


async def run_agent_query(
    agent: Agent,
    query: str,
    session: Session,
    user_id: str,
    *,
    is_router: bool = False,
) -> str:
    """Run one query against an ADK agent and return final text response."""
    print(f"\nRunning query for agent '{agent.name}' in session '{session.id}'...")

    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)
    final_response = ""

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user"),
        ):
            if not is_router:
                print(f"EVENT: {event}")
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
    except Exception as exc:  # pragma: no cover
        final_response = f"An error occurred: {exc}"

    if not is_router:
        print("\n" + "-" * 50)
        print("Final Response:")
        display(Markdown(final_response))
        print("-" * 50 + "\n")

    return final_response


def main() -> None:
    configure_google_api_key()
    print("Setup complete.")


if __name__ == "__main__":
    main()
