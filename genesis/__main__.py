
import traceback
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools

from .agent import make_agent
from .genesis_agent import make_genesis_tool, get_genesis_api_agent

from dotenv import load_dotenv

load_dotenv()


llm = OpenAI(
    temperature=0.1,
    max_tokens=512
)


tools = [
    # Genesis integration
    # make_genesis_tool(llm)
]

tools.extend(
    load_tools(
        [
            "human" # Human-input
        ],
        llm=llm,
        input_func=input
    )
)

def read_input():
    while True:
        inpt = input('> ')
        if len(inpt) > 0:
            return inpt

if __name__ == '__main__':
    agent = get_genesis_api_agent(llm, *tools)#make_agent(llm, tools)

    try:
        # REPL
        while True:
            print(agent.run(read_input()))
    except (KeyboardInterrupt, EOFError):
        # Quit gracefully
        pass
    except Exception:
        print("There was an error executing that action.")
        traceback.print_exc()
