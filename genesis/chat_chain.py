
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from genesis.genesis_agent import get_genesis_api_agent


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=512,
    # verbose=True
)

llm_tool = OpenAI(
    temperature=0.1,
    max_tokens=512,
    # verbose=True
)


agent_chain = get_genesis_api_agent(llm, llm_for_tool=llm_tool)
