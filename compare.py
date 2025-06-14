# compare.py
from crewai import Task
from agents import get_all_agents

# Unpack agents — we only need the comparison agent
_, _, _, comparison_agent = get_all_agents()


def create_comparison_task(agent_responses: dict, user_prompt: str):
    """Generate a task for comparing multiple LLM responses."""

    formatted_responses = "\n\n".join(
        f"{agent}:\n{response}" for agent, response in agent_responses.items()
    )

    return Task(
        description=(
            f"The user asked: '{user_prompt}'\n\n"
            f"The following are responses from different language models:\n\n"
            f"{formatted_responses}\n\n"
            "Please evaluate and compare the answers for clarity, accuracy, completeness, and helpfulness. "
            "State which answer is the best and explain why."
        ),
        expected_output="A concise summary comparing the responses and declaring a winner with rationale.",
        agent=comparison_agent
    )
