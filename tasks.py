from crewai import Task
from agents import get_all_agents

# Unpack all four agents
chatgpt_agent, claude_agent, gemini_agent, _ = get_all_agents()


def create_response_tasks(user_prompt):
    """Generate tasks asking each LLM agent to answer the prompt."""
    return [
        Task(
            description=f"Answer the following question: {user_prompt}",
            expected_output="A complete and helpful answer to the user's question.",
            agent=chatgpt_agent
        ),
        Task(
            description=f"Answer the following question: {user_prompt}",
            expected_output="A complete and helpful answer to the user's question.",
            agent=claude_agent
        ),
        Task(
            description=f"Answer the following question: {user_prompt}",
            expected_output="A complete and helpful answer to the user's question.",
            agent=gemini_agent
        )
    ]
