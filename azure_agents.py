import os
from dotenv import load_dotenv
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.agents import AgentServiceFactory
from prompts import RECIPIENT_INSTRUCTIONS, INQUIRER_INSTRUCTIONS

load_dotenv()  # Load environment variables from .env file

# Initialize the Factory to connect to Azure AI Foundry
factory = AgentServiceFactory(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

MODEL_NAME = os.environ.get("MODEL_DEPLOYMENT_NAME")

# ------------------------------------------------------------------
# Initialize the Foundry Chat Client
# ------------------------------------------------------------------
chat_client = FoundryChatClient(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    model=os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    credential=DefaultAzureCredential(),
)

# ------------------------------------------------------------------
# 1. Recipient Agent (HPI Structuring)
# ------------------------------------------------------------------

recipient_agent = chat_client.as_agent(
    name="recipient-agent",
    instructions=RECIPIENT_INSTRUCTIONS,
)

# ------------------------------------------------------------------
# 2. Inquirer Agent (Intake Interviewer)
# ------------------------------------------------------------------

inquirer_agent = chat_client.as_agent(
    name="inquirer-agent",
    instructions=INQUIRER_INSTRUCTIONS,
)

# 4. Triage Agent does not return a JSON, so it can be simply imported and ran with .ainvoke
triage_agent = factory.get_agent_node(name="triage-agent")
