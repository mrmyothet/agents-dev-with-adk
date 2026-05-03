"""Part 4: Parallel research workflow with ParallelAgent.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
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
    agent: Agent, query: str, session: Session, user_id: str
) -> str:
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)
    final_response = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=Content(parts=[Part(text=query)], role="user"),
    ):
        print(f"EVENT: {event}")
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    print("\n" + "-" * 50)
    print("Final Response:")
    display(Markdown(final_response))
    print("-" * 50 + "\n")
    return final_response


def build_parallel_planner() -> Agent:
    museum_finder_agent = Agent(
        name="museum_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one best museum for user query. Output only museum name.",
        output_key="museum_result",
    )

    concert_finder_agent = Agent(
        name="concert_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one concert for user query. Output concert and artist only.",
        output_key="concert_result",
    )

    restaurant_finder_agent = Agent(
        name="restaurant_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one best restaurant for user query. Output only restaurant name.",
        output_key="restaurant_result",
    )

    parallel_research_agent = ParallelAgent(
        name="parallel_research_agent",
        sub_agents=[museum_finder_agent, concert_finder_agent, restaurant_finder_agent],
    )

    synthesis_agent = Agent(
        name="synthesis_agent",
        model="gemini-2.5-flash",
        instruction="""
Combine research results into a short bulleted list:
- Museum: {museum_result}
- Concert: {concert_result}
- Restaurant: {restaurant_result}
""",
    )

    return SequentialAgent(
        name="parallel_planner_agent",
        sub_agents=[parallel_research_agent, synthesis_agent],
        description="Find multiple items in parallel then summarize results.",
    )


async def main() -> None:
    configure_google_api_key()
    parallel_planner_agent = build_parallel_planner()

    query = "Help me plan a trip to SF. I need one museum, one concert, and one great restaurant."
    session = await session_service.create_session(
        app_name=parallel_planner_agent.name, user_id=my_user_id
    )
    await run_agent_query(parallel_planner_agent, query, session, my_user_id)


if __name__ == "__main__":
    asyncio.run(main())
