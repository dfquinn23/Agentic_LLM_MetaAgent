from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

resp = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hi"}]
)

print(resp.content[0].text)


load_dotenv()

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    anthropic_api_key=os.getenv("CLAUDE_API_KEY")
)

response = llm.invoke("Say hi as Claude.")
print(response)
print("Gemini config:", {
    "model": os.getenv("GEMINI_MODEL_NAME"),
    "api_key": os.getenv("GOOGLE_API_KEY"),
    "use_google_ai_studio": os.getenv("LITELLM_USE_GOOGLE_AI_STUDIO")
})

print("🧪 Gemini via LiteLLM:", os.getenv("GOOGLE_API_KEY"),
      os.getenv("LITELLM_USE_GOOGLE_AI_STUDIO"))
