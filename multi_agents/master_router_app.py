"""Part 5: Fully loaded router app combining sequential, loop, and parallel workflows.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import ToolContext, google_search
from google.genai.types import Content, Part

session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"
COMPLETION_PHRASE = "The plan is feasible and meets all constraints."


def configure_google_api_key() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API Key: ")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key


def exit_loop(tool_context: ToolContext):
    tool_context.actions.escalate = True
    return {}


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
        instruction="You are a day trip planner. Return a concise markdown itinerary.",
    )

    foodie_agent = Agent(
        name="foodie_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Output only the best place name for the food request.",
        output_key="destination",
    )

    transportation_agent = Agent(
        name="transportation_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=(
            "Destination is {destination}. Infer start point from user query and "
            "provide clear directions."
        ),
    )

    find_and_navigate_agent = SequentialAgent(
        name="find_and_navigate_agent",
        sub_agents=[foodie_agent, transportation_agent],
        description="Find destination then navigate.",
    )

    planner_agent = Agent(
        name="planner_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=(
            "Propose one activity and one restaurant. Output only names in format "
            "'Activity: X, Restaurant: Y'."
        ),
        output_key="current_plan",
    )

    critic_agent = Agent(
        name="critic_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=f"""
Evaluate travel time in current plan: {{current_plan}}
If over 45 minutes, provide critique.
Else respond exactly with: {COMPLETION_PHRASE}
""",
        output_key="criticism",
    )

    refiner_agent = Agent(
        name="refiner_agent",
        model="gemini-2.5-flash",
        tools=[google_search, exit_loop],
        instruction=f"""
Original request: {{session.query}}
Critique: {{criticism}}
If critique equals '{COMPLETION_PHRASE}', call exit_loop.
Else output improved 'Activity: X, Restaurant: Y'.
""",
        output_key="current_plan",
    )

    refinement_loop = LoopAgent(
        name="refinement_loop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=3,
    )

    iterative_planner_agent = SequentialAgent(
        name="iterative_planner_agent",
        sub_agents=[planner_agent, refinement_loop],
    )

    museum_finder_agent = Agent(
        name="museum_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one museum for query. Output only museum name.",
        output_key="museum_result",
    )

    concert_finder_agent = Agent(
        name="concert_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one concert for query. Output concert and artist.",
        output_key="concert_result",
    )

    restaurant_finder_agent = Agent(
        name="restaurant_finder_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction="Find one restaurant for query. Output restaurant name only.",
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
Combine into bullets:
- Museum: {museum_result}
- Concert: {concert_result}
- Restaurant: {restaurant_result}
""",
    )

    parallel_planner_agent = SequentialAgent(
        name="parallel_planner_agent",
        sub_agents=[parallel_research_agent, synthesis_agent],
    )

    router_agent = Agent(
        name="router_agent",
        model="gemini-2.5-flash",
        instruction="""
Return exactly one option name based on user query:
- foodie_agent
- find_and_navigate_agent
- iterative_planner_agent
- parallel_planner_agent
- day_trip_agent
""",
    )

    worker_agents = {
        "day_trip_agent": day_trip_agent,
        "foodie_agent": foodie_agent,
        "find_and_navigate_agent": find_and_navigate_agent,
        "iterative_planner_agent": iterative_planner_agent,
        "parallel_planner_agent": parallel_planner_agent,
    }

    return router_agent, worker_agents


async def run_fully_loaded_app() -> None:
    router_agent, worker_agents = build_agents()

    queries = [
        "Find me the best sushi in Palo Alto and then tell me how to get there from the Caltrain station.",
        "Plan me a day in San Francisco with a museum and a nice dinner, but make sure the travel time between them is very short.",
        "Help me plan a trip to SF. I need one museum, one concert, and one great restaurant.",
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
    await run_fully_loaded_app()


if __name__ == "__main__":
    asyncio.run(main())
