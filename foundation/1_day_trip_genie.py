"""Part 1: Day Trip Genie agent.

Based on DevFest_BwAI/ADK_Learning_tools.ipynb.
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search
from google.genai.types import Content, Part

session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"


def configure_google_api_key() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API Key: ")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key


def create_day_trip_agent() -> Agent:
    return Agent(
        name="day_trip_agent",
        model="gemini-2.5-flash",
        description=(
            "Agent specialized in generating spontaneous full-day itineraries "
            "based on mood, interests, and budget."
        ),
        instruction="""
You are the Spontaneous Day Trip Generator, a specialized AI assistant
that creates engaging full-day itineraries.

Guidelines:
1. Budget aware: respect hints like cheap, affordable, or splurge.
2. Full-day structure: include morning, afternoon, and evening.
3. Real-time focus: search current operating hours and events.
4. Mood matching: align with the requested style.

Return itinerary in markdown with clear time blocks and venue names.
""",
        tools=[google_search],
    )


async def run_agent_query(
    agent: Agent, query: str, session: Session, user_id: str
) -> str:
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=Content(parts=[Part(text=query)], role="user"),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    return final_response


async def demo() -> None:
    configure_google_api_key()
    agent = create_day_trip_agent()

    session = await session_service.create_session(
        app_name=agent.name, user_id=my_user_id
    )
    query = "Plan a relaxing and artsy day trip near Sunnyvale, CA. Keep it affordable."
    response = await run_agent_query(agent, query, session, my_user_id)

    print("User query:")
    print(query)
    print("\nAgent response:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(demo())
