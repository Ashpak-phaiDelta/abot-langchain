
from dotenv import load_dotenv

from .chat_chain import cli_agent_chain
from .utils import read_input

load_dotenv()

from .basic_llms import llm, llm_tool


if __name__ == '__main__':
    agent = cli_agent_chain(llm, input_func=read_input, llm_for_tool=llm_tool)

    try:
        # REPL
        while True:
            print(agent.run(read_input()))
    except (KeyboardInterrupt, EOFError):
        # Quit gracefully
        pass
