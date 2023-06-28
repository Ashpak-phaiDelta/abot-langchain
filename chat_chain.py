
from langchain.llms import OpenAI

from genesis.genesis_agent import get_genesis_api_agent


llm = OpenAI(
    temperature=0,
    max_tokens=512
)

agent_chain = get_genesis_api_agent(llm)
