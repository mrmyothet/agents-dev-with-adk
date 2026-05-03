"""Part 1: Manual multi-agent sequential workflow.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
import re
import asyncio
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
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
        description="Generates full-day itineraries based on mood, interests, and budget.",
        instruction=(
            "You are a day trip generator. Create morning, afternoon, and evening plans "
            "that match the user mood and budget. Return markdown with venue names."
        ),
        tools=[google_search],
    )

    foodie_agent = Agent(
        name="foodie_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=(
            "You are a food critic. Recommend one best place and include the place name "
            "in bold markdown, for example: The best sushi is at **Jin Sho**."
        ),
    )

    weekend_guide_agent = Agent(
        name="weekend_guide_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="You are an events guide. Find activities and events for the requested weekend.",
    )

    transportation_agent = Agent(
        name="transportation_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="You are a navigation assistant. Provide directions from start to destination.",
    )

    router_agent = Agent(
        name="router_agent",
        model="gemini-2.5-flash",
        instruction="""
You are a request router. Return exactly one option name:
- foodie_agent
- weekend_guide_agent
- day_trip_agent
- find_and_navigate_combo

Choose find_and_navigate_combo only when user asks to find a place first and then get directions.
""",
    )

    worker_agents = {
        "day_trip_agent": day_trip_agent,
        "foodie_agent": foodie_agent,
        "weekend_guide_agent": weekend_guide_agent,
        "transportation_agent": transportation_agent,
    }

    return router_agent, worker_agents, foodie_agent, transportation_agent


async def run_sequential_app() -> None:
    router_agent, worker_agents, foodie_agent, transportation_agent = build_agents()

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

        if chosen_route == "find_and_navigate_combo":
            foodie_session = await session_service.create_session(
                app_name=foodie_agent.name, user_id=my_user_id
            )
            foodie_response = await run_agent_query(
                foodie_agent, query, foodie_session, my_user_id
            )

            match = re.search(r"\*\*(.*?)\*\*", foodie_response)
            if not match:
                print("Could not extract restaurant name from foodie response.")
                continue

            destination = match.group(1)
            directions_query = f"Give me directions to {destination} from the Palo Alto Caltrain station."
            transport_session = await session_service.create_session(
                app_name=transportation_agent.name,
                user_id=my_user_id,
            )
            await run_agent_query(
                transportation_agent, directions_query, transport_session, my_user_id
            )

        elif chosen_route in worker_agents:
            worker_agent = worker_agents[chosen_route]
            worker_session = await session_service.create_session(
                app_name=worker_agent.name, user_id=my_user_id
            )
            await run_agent_query(worker_agent, query, worker_session, my_user_id)
        else:
            print(f"Unknown route selected: {chosen_route}")


async def main() -> None:
    configure_google_api_key()
    await run_sequential_app()


if __name__ == "__main__":
    asyncio.run(main())
