
from langchain.agents import AgentType, AgentExecutor, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.base_language import BaseLanguageModel
from langchain.agents.tools import BaseTool

from typing import List


def make_agent(
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        agent: AgentType = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        **kwargs) -> AgentExecutor:
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=1
    )

    return initialize_agent(
        tools,
        llm,
        agent,
        memory=memory,
        **kwargs
    )
