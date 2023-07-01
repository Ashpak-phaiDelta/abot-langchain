
from pathlib import Path

from langchain.agents.agent_types import AgentType
from langchain.agents.tools import Tool, BaseTool
from langchain.requests import Requests

from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


# Trying out the ready-made way. It isn't as powerful, prompt is fixed

# from langchain.requests import RequestsWrapper
# from langchain.agents.agent_toolkits.openapi.planner import create_openapi_agent
# from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
# from langchain.agents.agent_toolkits import NLAToolkit

# Just a wrapper I made to ease in creating an Agent with memory
from .agent import make_agent

# All tools
from .genesis_api import *

from pydantic import BaseSettings, AnyUrl, FilePath
from typing import Union
from functools import lru_cache


class GenesisSettings(BaseSettings):
    auth_token: str
    agent_is_verbose: bool = False
    tool_is_verbose: bool = False
    openapi_file: Union[AnyUrl, FilePath, str] = 'genesis_openapi.yaml'

    class Config:
        env_file = '.env'
        env_prefix = 'genesis_'

# Prompts

# Genesis is an IoT platform with various levels of information. First level is Location level where warehouses are located across the globe. For example, VER_W1 is one such warehouse. Next level is Warehouse level which consists of Units (basically buildings) which may also contain sensors. A level inside it is the Unit level which have the sensors (temperature, energy, smoke(VESDA), etc.). Each level can have a summary. Every level (warehouse, unit, sensor) has an ID (number).
# The assistant is designed to be able to assist with a wide range of analytical and sensor-based tasks, from answering simple questions of getting values to providing a summary on a wide range of sensor data. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. Assistant aims to be accurate and provide responses only from the data available.


GENESIS_AGENT_PROMPT_PREFIX = """You are a very powerful IoT and Analytics Assistant for an application called Genesis made by the company phAIdelta. You are able to make use of various tools available as a means of answering questions.
You are also able to use tools in sequence to answer the question and to get context. Eg: asking summary of location VER_W1, you need to fetch the integer warehouse_id of VER_W1 first, then get the summary.
The ID (integer) must be given to whichever tool that requires it. For example, warehouse summary needs warehouse_id (number), but user may say "VER_W1" which is the name. You must fetch the ID first using another tool, then pass the ID to the appropriate tool. DO NOT pass name like VER_W1 to a tool that requires an integer ID.
Do NOT make up the ID of warehouse_id, unit_id or sensor_id, but use tool designed to fetch relevant IDs first. Eg. Use list of locations to get warehouse_id, and list of units to get unit_id. Do NOT assume the ID, always use tool to get this. You can run a sequence of tools to get to the final answer.
Make sure to display the information in the Final Answer when information is requested, after that give the answer. eg. List of sensors are also to be shown one per line as list items along with the text you will say.

Example:
Human: How many sensors are there in Cipla at VER_W1?
AI: use tool to find warehouse_id of VER_W1, then use tool to find unit_id of Cipla inside VER_W1, finally use tool to list and count sensors in the unit_id.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist. If you are unable to answer a question, ask user to provide the needed information or say `I don't know`

TOOLS:
------

Assistant has access to the following tools: """


GENESIS_TOOL_DESCRIPTION = "An IoT platform, Genesis is the product that can help users gather information about Sensors, locations such as warehouses and buildings, analytics, and status of the Sensors and locations themselves. Use this tool when any request to sensors, units/locations or warehouses is asked. Also used when followup questions or actions are asked. If an error occurs or sensor is not found, inform user about the error. Users are not aware of sensor IDs and mostly use names/aliases"


# Settings fetchers

@lru_cache()
def get_auth_token():
    return GenesisSettings().auth_token

@lru_cache()
def get_agent_is_verbose():
    return GenesisSettings().agent_is_verbose

@lru_cache()
def get_tool_is_verbose():
    return GenesisSettings().tool_is_verbose

@lru_cache()
def get_openapi_file():
    return GenesisSettings().openapi_file


def fetch_genesis_spec() -> OpenAPISpec:
    spec_file = get_openapi_file()

    if isinstance(spec_file, AnyUrl):
        return OpenAPISpec.from_url(spec_file)
    elif Path(spec_file).exists():
        return OpenAPISpec.from_file(spec_file)

    raise ValueError("You must set the setting `openapi_file` or `GENESIS_OPENAPI_FILE` environment to a path that exists.\nIt was set to '%s'" % str(spec_file))


def get_genesis_api_agent(llm, *additional_tools):
    # Requests with auth token
    requests = Requests(headers={"Authorization": "Bearer %s" % get_auth_token()})

    # Genesis API specifications (OpenAPI)
    spec = fetch_genesis_spec()

    agent_verbose = get_agent_is_verbose()
    tool_verbose = get_tool_is_verbose()

    # List of tools available to the agent
    genesis_tools = [
        # _get_tool_genesis_sensor_status(llm, spec, requests),
        # get_tool_genesis_sensor_list(llm, spec, requests, verbose=tool_verbose),
        get_tool_genesis_location_list(llm, spec, requests, verbose=tool_verbose),
        get_tool_genesis_location_summary(llm, spec, requests, verbose=tool_verbose),
        get_tool_genesis_warehouse_summary(llm, spec, requests, verbose=tool_verbose),
        get_tool_genesis_warehouse_unit_summary(llm, spec, requests, verbose=tool_verbose),
        get_tool_genesis_unit_sensor_list(llm, spec, requests, verbose=tool_verbose),
        *additional_tools
    ]
    

    return make_agent(
        llm,
        genesis_tools,
        # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=agent_verbose,
        agent_kwargs=dict(
            prefix=GENESIS_AGENT_PROMPT_PREFIX,
            # format_instructions=GENESIS_AGENT_PROMPT_FORMAT_INSTRUCTIONS
        )
    )


def make_genesis_tool(llm) -> BaseTool:
    genesis_agent = get_genesis_api_agent(llm)
    return Tool(
        name="Genesis",
        description=GENESIS_TOOL_DESCRIPTION,
        func=genesis_agent.run
    )
