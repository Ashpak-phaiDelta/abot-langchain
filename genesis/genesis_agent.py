

from langchain.llms.base import BaseLLM
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import BaseTool, Tool
from langchain.agents import AgentExecutor, initialize_agent

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory

from langchain.requests import Requests

# All tools
from .genesis_api import *

from typing import List

from .config import fetch_genesis_spec, get_agent_is_verbose, get_tool_is_verbose, get_auth_token

# Prompts
from .prompts import is_chat_model, get_agent_prompt, GENESIS_AGENT_PROMPT_PREFIX


# Trying out the ready-made way. It isn't as powerful, prompt is fixed

# from langchain.requests import RequestsWrapper
# from langchain.agents.agent_toolkits.openapi.planner import create_openapi_agent
# from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
# from langchain.agents.agent_toolkits import NLAToolkit


def get_genesis_api_agent(llm: BaseLLM, *additional_tools: BaseTool, llm_for_tool: BaseLLM = None) -> AgentExecutor:
    """Create an Agent that executes queries for Genesis server"""
    # Requests with auth token
    requests = Requests(headers={"Authorization": "Bearer %s" % get_auth_token()})

    # Genesis API specifications (OpenAPI)
    spec = fetch_genesis_spec()

    agent_verbose = get_agent_is_verbose()
    tool_verbose = get_tool_is_verbose()

    if llm_for_tool is None:
        llm_for_tool = llm

    # Agent's memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Create Agents for various levels
    unit_level_agent = get_unit_level_query_agent(llm, llm_for_tool=llm_for_tool, spec=spec, requests=requests, memory=memory, verbose=tool_verbose)

    # List of tools available to the agent
    genesis_tools: List[BaseTool] = [
        get_tool_genesis_location_list(llm_for_tool, spec, requests, verbose=tool_verbose),
        get_tool_genesis_location_summary(llm_for_tool, spec, requests, verbose=tool_verbose),
        # get_tool_genesis_warehouse_summary(llm_for_tool, spec, requests, verbose=tool_verbose),
        # get_tool_genesis_warehouse_unit_summary(llm_for_tool, spec, requests, verbose=tool_verbose),
        ## get_tool_genesis_unit_sensor_list(llm_for_tool, spec, requests, verbose=tool_verbose),
        # get_tool_genesis_sensor_list(llm, spec, requests, verbose=tool_verbose),

        # Unit-level query agent
        Tool.from_function(
            func=unit_level_agent.run,
            name='agent_genesis_unit_level_query',
            description="Use for querying unit information or unit-level sensors of a specific unit, or to find unit id given the name"
        ),

        # TODO @Ashpak add sensor level agent tool here

        *additional_tools
    ]

    return initialize_agent(
        genesis_tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION if is_chat_model(llm) else AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=agent_verbose,
        memory=memory,
        agent_kwargs=get_agent_prompt(llm,
            prefix_prompt=GENESIS_AGENT_PROMPT_PREFIX
        )
    )
