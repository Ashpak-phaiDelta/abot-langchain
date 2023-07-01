
import traceback
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory, ConversationSummaryBufferMemory

from .agent import make_agent
from .genesis_agent import make_genesis_tool, get_genesis_api_agent

from dotenv import load_dotenv

load_dotenv()


def read_input():
    '''Simple prompt for user to enter input for asking'''
    while True:
        inpt = input('> ')
        if len(inpt) > 0:
            return inpt


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=512,
    # verbose=True
)

llm_tool = OpenAI(
    temperature=0.1,
    max_tokens=512,
    # verbose=True
)


tools = [
    # Genesis integration
    # make_genesis_tool(llm)
]

# tools.extend(
#     load_tools(
#         [
#             "human" # Human-input
#         ],
#         llm=llm,
#         input_func=read_input
#     )
# )

if __name__ == '__main__':
    agent = get_genesis_api_agent(llm, *tools, llm_for_tool=llm_tool)#make_agent(llm, tools)

    try:
        # REPL
        while True:
            print(agent.run(read_input()))
    except (KeyboardInterrupt, EOFError):
        # Quit gracefully
        pass
    # except Exception:
    #     print("There was an error executing that action.")
    #     traceback.print_exc()
