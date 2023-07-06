
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


# Make sure to set environment variables before importing this


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
