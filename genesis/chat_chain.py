
from langchain.agents import load_tools

from genesis.genesis_agent import get_genesis_api_agent


cli_agent_chain = lambda llm, *additional_tools, input_func=input, llm_for_tool = None, **kwargs: get_genesis_api_agent(
    llm,
    *load_tools(
        [
            "human" # Human-input
        ],
        llm=llm_for_tool or llm,
        input_func=input_func
    ),
    *additional_tools,
    llm_for_tool=llm_for_tool,
    **kwargs
)

agent_chain = get_genesis_api_agent
