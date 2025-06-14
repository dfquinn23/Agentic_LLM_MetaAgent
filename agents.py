# agents.py

import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import LiteLLM

# Load environment variables
load_dotenv()

# Handle Claude fallback
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("CLAUDE_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("CLAUDE_API_KEY")

# Warn if keys are missing
for key in ["OPENAI_API_KEY", "CLAUDE_API_KEY", "GEMINI_API_KEY"]:
    if not os.getenv(key):
        print(f"⚠️ Warning: {key} is not set in the .env file")

# --- ChatGPT Agent ---
chatgpt_agent = Agent(
    role="ChatGPT",
    goal="Answer user prompts using OpenAI's GPT-4",
    backstory="You are GPT-4, providing high-quality answers.",
    llm=ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
)

# --- Claude Agent ---
claude_agent = Agent(
    role="Claude",
    goal="Respond using Claude with clarity and ethics",
    backstory="You are Claude from Anthropic, known for nuanced reasoning.",
    llm=ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229"),
        temperature=0.7,
        anthropic_api_key=os.getenv("CLAUDE_API_KEY")
    )
)

# --- Gemini Agent using LiteLLM ---
gemini_agent = Agent(
    role="Gemini",
    goal="Respond using Google's Gemini model",
    backstory="You are Gemini, optimized for helpful, direct answers.",
    llm=LiteLLM(
        model="gemini-pro",  # NOTE: LiteLLM must receive this exact model name
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )
)

# --- Comparison Agent ---
comparison_agent = Agent(
    role="Comparison Agent",
    goal="Compare LLM responses and summarize quality and differences",
    backstory="You review and evaluate answers from different LLMs for fairness, clarity, and accuracy.",
    llm=ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
)

# --- Export all agents ---


def get_all_agents():
    return chatgpt_agent, claude_agent, gemini_agent, comparison_agent
