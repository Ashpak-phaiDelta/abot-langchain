
from langchain.agents import AgentType, AgentExecutor, initialize_agent
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory, ConversationSummaryBufferMemory
from langchain.base_language import BaseLanguageModel
from langchain.agents.tools import BaseTool

from typing import List


def make_agent(
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        agent: AgentType = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        **kwargs) -> AgentExecutor:

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )

    return initialize_agent(
        tools,
        llm,
        agent,
        memory=memory,
        **kwargs
    )
