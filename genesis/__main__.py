
from .chat_chain import cli_agent_chain
from .utils import read_input

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()


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


if __name__ == '__main__':
    agent = cli_agent_chain(llm, input_func=read_input, llm_for_tool=llm_tool)

    try:
        # REPL
        while True:
            print(agent.run(read_input()))
    except (KeyboardInterrupt, EOFError):
        # Quit gracefully
        pass
