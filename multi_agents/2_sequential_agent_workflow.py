"""Part 2: SequentialAgent workflow.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
from google.adk.agents import Agent, SequentialAgent
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


async def run_agent_query(
    agent: Agent, query: str, session: Session, user_id: str, is_router: bool = False
) -> str:
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)
    final_response = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=Content(parts=[Part(text=query)], role="user"),
    ):
        if not is_router:
            print(f"EVENT: {event}")
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    if not is_router:
        print("\n" + "-" * 50)
        print("Final Response:")
        display(Markdown(final_response))
        print("-" * 50 + "\n")

    return final_response


def build_agents():
    day_trip_agent = Agent(
        name="day_trip_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="You are a day trip planner. Return a markdown day plan.",
    )

    foodie_agent = Agent(
        name="foodie_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Output only the best place name for the user's food request.",
        output_key="destination",
    )

    transportation_agent = Agent(
        name="transportation_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=(
            "You are a navigation assistant. User destination is {destination}. "
            "Infer start point from user query and provide clear directions."
        ),
    )

    find_and_navigate_agent = SequentialAgent(
        name="find_and_navigate_agent",
        sub_agents=[foodie_agent, transportation_agent],
        description="Find a place first, then provide directions.",
    )

    weekend_guide_agent = Agent(
        name="weekend_guide_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find events, concerts, and activities for the requested weekend.",
    )

    router_agent = Agent(
        name="router_agent",
        model="gemini-2.5-flash",
        instruction="""
You are a request router. Return exactly one option name:
- foodie_agent
- weekend_guide_agent
- day_trip_agent
- find_and_navigate_agent
""",
    )

    worker_agents = {
        "day_trip_agent": day_trip_agent,
        "foodie_agent": foodie_agent,
        "weekend_guide_agent": weekend_guide_agent,
        "find_and_navigate_agent": find_and_navigate_agent,
    }
    return router_agent, worker_agents


async def run_sequential_agent_app() -> None:
    router_agent, worker_agents = build_agents()
    queries = [
        "I want to eat the best sushi in Palo Alto.",
        "Are there any cool outdoor concerts this weekend?",
        "Find me the best sushi in Palo Alto and then tell me how to get there from the Caltrain station.",
    ]

    for query in queries:
        print(f"\n{'=' * 60}\nProcessing query: {query}\n{'=' * 60}")
        router_session = await session_service.create_session(
            app_name=router_agent.name, user_id=my_user_id
        )
        chosen_route = await run_agent_query(
            router_agent, query, router_session, my_user_id, is_router=True
        )
        chosen_route = chosen_route.strip().replace("'", "")
        print(f"Router selected: {chosen_route}")

        if chosen_route in worker_agents:
            worker_agent = worker_agents[chosen_route]
            worker_session = await session_service.create_session(
                app_name=worker_agent.name, user_id=my_user_id
            )
            await run_agent_query(worker_agent, query, worker_session, my_user_id)
        else:
            print(f"Unknown route selected: {chosen_route}")


async def main() -> None:
    configure_google_api_key()
    await run_sequential_agent_app()


if __name__ == "__main__":
    asyncio.run(main())
