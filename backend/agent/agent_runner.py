# agent_runner.py
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from backend.agent.agent import root_agent

APP_NAME = "search4cure_ai"
USER_ID = "user1"
SESSION_ID = "sess1"

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

def call_agent(query: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event.is_final_response():
            return event.content.parts[0].text
    return "No final response from agent."
