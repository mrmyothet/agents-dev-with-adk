"""Part 3: Memory with adaptive multi-day planner.

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


def create_multi_day_trip_agent() -> Agent:
    return Agent(
        name="multi_day_trip_agent",
        model="gemini-2.5-flash",
        description=(
            "Agent that progressively plans a multi-day trip, remembers previous "
            "days, and adapts to feedback."
        ),
        instruction="""
You are an Adaptive Trip Planner.

Mission:
1. Start by asking destination, trip duration, and interests.
2. Plan one day at a time, then ask for confirmation.
3. Handle feedback by replacing disliked suggestions with suitable alternatives.
4. Maintain context and avoid repeated activities across days.
5. Return each day in markdown format.
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


async def scenario_with_memory(agent: Agent) -> None:
    print("Scenario A: same session (memory preserved)")
    trip_session = await session_service.create_session(
        app_name=agent.name, user_id=my_user_id
    )

    query1 = "I want a 2-day trip to Lisbon focused on history and local food."
    query2 = (
        "I do not like castles. Replace Day 1 morning with another historical option."
    )
    query3 = "Great. Now plan Day 2 and keep the food theme."

    print("\nTurn 1:\n", await run_agent_query(agent, query1, trip_session, my_user_id))
    print("\nTurn 2:\n", await run_agent_query(agent, query2, trip_session, my_user_id))
    print("\nTurn 3:\n", await run_agent_query(agent, query3, trip_session, my_user_id))


async def scenario_without_memory(agent: Agent) -> None:
    print("\nScenario B: separate sessions (memory lost)")

    query1 = "I want a 2-day trip to Lisbon focused on history and local food."
    session_one = await session_service.create_session(
        app_name=agent.name, user_id=my_user_id
    )
    print("\nTurn 1:\n", await run_agent_query(agent, query1, session_one, my_user_id))

    query2 = "Looks good. Please plan Day 2."
    session_two = await session_service.create_session(
        app_name=agent.name, user_id=my_user_id
    )
    print("\nTurn 2:\n", await run_agent_query(agent, query2, session_two, my_user_id))


async def main() -> None:
    configure_google_api_key()
    agent = create_multi_day_trip_agent()
    await scenario_with_memory(agent)
    await scenario_without_memory(agent)


if __name__ == "__main__":
    asyncio.run(main())
