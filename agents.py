# agents.py
from langchain.llms.base import LLM
from google.generativeai import GenerativeModel
import os
from dotenv import load_dotenv

from crewai import Agent
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Warn if critical env vars are missing
required_keys = ["OPENAI_API_KEY", "CLAUDE_API_KEY", "GEMINI_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        print(f"\u26a0\ufe0f Warning: {key} is not set in the .env file")

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
claude_llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229"),
    temperature=0.7,
    anthropic_api_key=os.getenv("CLAUDE_API_KEY")
)

claude_agent = Agent(
    role="Claude",
    goal="Respond using Claude with clarity and ethics",
    backstory="You are Claude from Anthropic, known for nuanced reasoning.",
    llm=claude_llm
)

# --- Gemini Agent (Direct SDK) ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = GenerativeModel(model_name=os.getenv(
    "GEMINI_MODEL_NAME", "gemini-1.5-pro-latest"))


def gemini_generate(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text


class GeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        return gemini_generate(prompt)

    @property
    def _llm_type(self):
        return "gemini"


gemini_agent = Agent(
    role="Gemini",
    goal="Respond using Google's Gemini model",
    backstory="You are Gemini, optimized for helpful, direct answers.",
    llm=GeminiLLM()
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

# Export agents for external modules


def get_all_agents():
    return chatgpt_agent, claude_agent, gemini_agent, comparison_agent
