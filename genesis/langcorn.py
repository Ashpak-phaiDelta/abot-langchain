"""The Genesis Langchain generated for use with langcorn"""

from .chat_chain import agent_chain

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


chat_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=512,
    # verbose=True
)

completion_llm = OpenAI(
    temperature=0.1,
    max_tokens=512,
    # verbose=True
)

chain = agent_chain(chat_llm, llm_for_tool=completion_llm)
