"""Part 2: Custom tools and Agent-as-a-Tool pattern.

Based on DevFest_BwAI/ADK_Learning_tools.ipynb.
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
import requests
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Content, Part

session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"


def configure_google_api_key() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API Key: ")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key


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


# ---------------------------------------------------------------------------
# 2.1 FunctionTool example: live weather API
# ---------------------------------------------------------------------------

LOCATION_COORDINATES = {
    "sunnyvale": "37.3688,-122.0363",
    "san francisco": "37.7749,-122.4194",
    "lake tahoe": "39.0968,-120.0324",
}


def get_live_weather_forecast(location: str) -> dict:
    """Get a real-time weather forecast for a supported US location.

    Args:
        location: A city or place name.

    Returns:
        A dictionary with status and either weather data or an error message.
    """
    normalized_location = location.lower()
    coords = None
    for key, value in LOCATION_COORDINATES.items():
        if key in normalized_location:
            coords = value
            break

    if not coords:
        return {
            "status": "error",
            "message": f"No coordinates configured for {location}.",
        }

    try:
        headers = {"User-Agent": "ADK Foundation Example"}
        points_url = f"https://api.weather.gov/points/{coords}"
        points_response = requests.get(points_url, headers=headers, timeout=20)
        points_response.raise_for_status()

        forecast_url = points_response.json()["properties"]["forecast"]
        forecast_response = requests.get(forecast_url, headers=headers, timeout=20)
        forecast_response.raise_for_status()

        current_period = forecast_response.json()["properties"]["periods"][0]
        return {
            "status": "success",
            "temperature": f"{current_period['temperature']} {current_period['temperatureUnit']}",
            "forecast": current_period["detailedForecast"],
        }
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Weather API request failed: {exc}"}


weather_agent = Agent(
    name="weather_aware_planner",
    model="gemini-2.5-flash",
    description="Trip planner that checks real-time weather before suggestions.",
    instruction=(
        "You are a cautious trip planner. Before suggesting outdoor activities, "
        "use get_live_weather_forecast and include weather details in your recommendation."
    ),
    tools=[get_live_weather_forecast],
)


# ---------------------------------------------------------------------------
# 2.2 Agent-as-a-Tool example
# ---------------------------------------------------------------------------

db_agent = Agent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a database agent. Return this mock JSON when asked for hotel data: "
        "{'status':'success','data':[{'name':'The Grand Hotel','rating':5,'reviews':450},"
        "{'name':'Seaside Inn','rating':4,'reviews':620}]}"
    ),
)

food_critic_agent = Agent(
    name="food_critic_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a snobby but brilliant food critic. Reply with one witty restaurant "
        "suggestion near the given location."
    ),
)

concierge_agent = Agent(
    name="concierge_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a five-star hotel concierge. If asked for a restaurant recommendation, "
        "use the food_critic_agent tool and present the result politely."
    ),
    tools=[AgentTool(agent=food_critic_agent)],
)


async def call_db_agent(question: str, tool_context: ToolContext):
    """Use first: fetch hotel/place data from db_agent."""
    db_tool = AgentTool(agent=db_agent)
    db_output = await db_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["retrieved_data"] = db_output
    return db_output


async def call_concierge_agent(question: str, tool_context: ToolContext):
    """Use after call_db_agent: get recommendations from concierge_agent."""
    input_data = tool_context.state.get("retrieved_data", "No data found")
    prompt = f"Context data: {input_data}\n\nUser request: {question}"

    concierge_tool = AgentTool(agent=concierge_agent)
    return await concierge_tool.run_async(
        args={"request": prompt}, tool_context=tool_context
    )


trip_data_concierge_agent = Agent(
    name="trip_data_concierge",
    model="gemini-2.5-flash",
    description="Top-level agent that queries data then gets concierge recommendations.",
    tools=[call_db_agent, call_concierge_agent],
    instruction="""
You are a master travel planner.
1. Always call call_db_agent first to fetch data.
2. Then call call_concierge_agent for recommendations based on that data.
""",
)


async def demo_weather() -> None:
    session = await session_service.create_session(
        app_name=weather_agent.name, user_id=my_user_id
    )
    query = "I want to go hiking near Lake Tahoe. What is the weather like?"
    response = await run_agent_query(weather_agent, query, session, my_user_id)
    print("Weather demo response:\n")
    print(response)


async def demo_agent_as_tool() -> None:
    session = await session_service.create_session(
        app_name=trip_data_concierge_agent.name,
        user_id=my_user_id,
    )
    query = "Find top-rated hotels from the database, then suggest dinner nearby."
    response = await run_agent_query(
        trip_data_concierge_agent, query, session, my_user_id
    )
    print("\nAgent-as-a-tool demo response:\n")
    print(response)


async def main() -> None:
    configure_google_api_key()
    await demo_weather()
    await demo_agent_as_tool()


if __name__ == "__main__":
    asyncio.run(main())
