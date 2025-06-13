# compare.py
"""Utilities for comparing LLM responses."""

from crewai import Task
from agents import get_all_agents

_, _, _, comparison_agent = get_all_agents()


def create_comparison_task(agent_responses: dict, user_prompt: str) -> Task:
    """Create a task instructing the comparison agent to evaluate responses.

    Parameters
    ----------
    agent_responses : dict
        Mapping of agent names to their responses.
    user_prompt : str
        The original question provided by the user.
    """

    formatted_responses = "\n".join(
        f"{name}: {text}" for name, text in agent_responses.items()
    )
    description = (
        "Compare the following answers provided by different LLMs "
        f"to the prompt '{user_prompt}'.\n\n{formatted_responses}\n\n"
        "Summarize the relative strengths of each answer and recommend "
        "which is best and why."
    )

    return Task(description=description, agent=comparison_agent)
