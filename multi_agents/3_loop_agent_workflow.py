"""Part 3: Iterative workflow with LoopAgent.

Source: DevFest_BwAI/ADK_Learning_tool_multi_agents.ipynb
"""

import os
import asyncio
from getpass import getpass

import google.generativeai as genai
from IPython.display import Markdown, display
from google.adk.agents import Agent, LoopAgent, SequentialAgent
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
    """Call only when plan is approved and loop should end."""
    print(f"[Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}


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


def build_iterative_agent() -> Agent:
    planner_agent = Agent(
        name="planner_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=(
            "Propose one activity and one restaurant based on user request. "
            "Output only names, like 'Activity: X, Restaurant: Y'."
        ),
        output_key="current_plan",
    )

    critic_agent = Agent(
        name="critic_agent",
        model="gemini-2.5-flash",
        tools=[google_search],
        instruction=f"""
You are a logistics critic with a strict travel-time constraint.
Current plan: {{current_plan}}
Check travel time between the two places.
If over 45 minutes, critique the plan and suggest improvement.
Else respond with exactly: {COMPLETION_PHRASE}
""",
        output_key="criticism",
    )

    refiner_agent = Agent(
        name="refiner_agent",
        model="gemini-2.5-flash",
        tools=[google_search, exit_loop],
        instruction=f"""
You refine plans based on criticism.
Original request: {{session.query}}
Critique: {{criticism}}
If critique equals '{COMPLETION_PHRASE}', call exit_loop.
Else produce a new plan in format: 'Activity: X, Restaurant: Y'.
""",
        output_key="current_plan",
    )

    refinement_loop = LoopAgent(
        name="refinement_loop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=3,
    )

    return SequentialAgent(
        name="iterative_planner_agent",
        sub_agents=[planner_agent, refinement_loop],
        description="Iteratively plans and refines a trip until constraints are met.",
    )


async def main() -> None:
    configure_google_api_key()
    iterative_planner_agent = build_iterative_agent()

    query = (
        "Plan me a day in San Francisco with a museum and a nice dinner, "
        "but make sure the travel time between them is very short."
    )
    session = await session_service.create_session(
        app_name=iterative_planner_agent.name, user_id=my_user_id
    )
    await run_agent_query(iterative_planner_agent, query, session, my_user_id)


if __name__ == "__main__":
    asyncio.run(main())
