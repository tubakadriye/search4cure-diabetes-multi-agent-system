from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import agent_tool
from pydantic import BaseModel
from google.adk.events import Event
from google.genai import types

class ImageGeneratorAgent(BaseAgent):
    name: str = "ImageGen"
    description: str = "Generates an image based on a prompt."

    async def _run_async_impl(self, ctx):
        prompt = ctx.session.state.get("image_prompt", "default prompt")
        image_bytes = b"..."  # replace with actual logic
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part.from_bytes(image_bytes, "image/png")])
        )

# Wrap the custom agent as a tool
image_tool = agent_tool.AgentTool(agent=ImageGeneratorAgent())

# Use the tool in an LlmAgent
artist_agent = LlmAgent(
    name="Artist",
    model="gemini-2.0-flash",
    instruction="Create a prompt and use the ImageGen tool to generate the image.",
    tools=[image_tool]
)
