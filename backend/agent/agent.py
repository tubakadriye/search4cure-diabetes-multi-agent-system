# agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.agents import Agent

# Import your agents/tools
from sub_agents.csv_agents import csv_files_vector_search_agent
from sub_agents.article_agents import article_page_vector_search_agent
from sub_agents.image_generator_agent import artist_agent
from sub_agents.csv_agents import create_record_agent
from sub_agents.article_agents import vector_search_image_agent
from prompts.agent_prompt import get_agent_prompt

agent_purpose = get_agent_prompt()


# Coordinator (“mid-level”) agent
research_assistant = LlmAgent(
    name="research_assistant_for_the_cure_Diabetes",
    model="gemini-2.0-flash",
    instruction = agent_purpose,
    description="Decides on which agent to work with based on the queries about articles, csvs or images.",
    sub_agents=[csv_files_vector_search_agent, create_record_agent, article_page_vector_search_agent, vector_search_image_agent,
                artist_agent],
    #tools = [coordinator_tool]
    
)

# Root ("high-level") agent – this must be named `root_agent`
root_agent = LlmAgent(
    name="ReportWriter",
    model="gemini-2.0-flash",
    instruction="Write a report on the given question. Use the ResearchAssistant to gather information.",
    sub_agents=[research_assistant],
)
