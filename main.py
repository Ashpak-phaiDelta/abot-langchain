
import traceback

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.tools import Tool

from dotenv import load_dotenv

load_dotenv()



llm = OpenAI(
    # openai_api_base="http://localhost:8000/v1",
    temperature=0.7,
    max_tokens=1024
)

agent_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5
)

html_build_templ = """You are a professional web page builder that can make a single page HTML. Given the following description, build a page with HTML and CSS styling.

Description: {query}

HTML:
"""


build_html_prompt = PromptTemplate(template=html_build_templ, input_variables=['query'])

html_prompt = LLMChain(
    llm=llm,
    prompt=build_html_prompt
)


# define a function to calculate nth fibonacci number
def fib(n: int) -> int:
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)


def fib_many(n_list: str, **kwargs) -> str:
    return ','.join(map(str, [fib(int(x)) for x in n_list]))

def sort_string(string, **kwargs):
    return ''.join(sorted(string))
    
def encrypt(word, **kwargs):
    encrypted_word = ""
    for letter in word:
        encrypted_word += chr(ord(letter) + 1)
    return encrypted_word

def decrypt(word, **kwargs):
    decrypted_word = ""
    for letter in word:
        decrypted_word += chr(ord(letter) - 1)
    return decrypted_word

import subprocess
def exec_program(cmd, **kwargs) -> dict:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc.wait()
    return {
        'output': proc.stdout.read().decode(errors='replace')
    }

def get_sensor_id(text: str, **kwargs) -> str:
    if 'temperature' in text:
        return '1'
    if 'humidity' in text:
        return '2'
    return 'Not found'

def get_sensor_status(text: str, **kwargs) -> str:
    if text == '1':
        return 'NORMAL'
    elif text == '2':
        return 'OUT_OF_RANGE'
    return 'Unknown'

tools = [
    Tool(
        name="sensor_id_get",
        func=get_sensor_id,
        description="used to find sensor id from type of sensor"
    ),
    Tool(
        name="sensor_status",
        func=get_sensor_status,
        description="used to find sensor's status after finding sensor ID using tool sensor_id_get. If sensor ID is not provided, request user or determine from sensor type using a tool."
    ),
    # Tool(
    #     name = "Fibonacci",
    #     func=fib_many,
    #     description="use when you want to find out the nth fibonacci number, given as comma separated list of positive integers. Can't be used for math expressions.",
    #     # return_direct=True
    # ),
    Tool(
        name="web",
        func=html_prompt.run,
        description="use to create web pages and output html from a site description",
        # return_direct=True
    ),
    # Tool(
    #     name="genesis",
    #     func=lambda cmd: "Status: OUT_OF_RANGE",
    #     description="use for anything to do with Genesis which has Sensors and Units in a Warehouse. Sensors such as temperature and humidity have a name and location, and Units are their location within the warehouse. An example would be 'status of Cipla unit'.",
    #     # return_direct=True
    # ),
    Tool(
        name="register_data",
        func=lambda cmd: "Done",
        description="use for saving preferences of user regarding Genesis sensors or units such as preferred sensor or unit that is regularly used.",
        # return_direct=True
    ),
    Tool(
        name = "Sort String",
        func=sort_string,
        description="use when you want to sort a string alphabetically",
        # return_direct=True
    ),
    Tool(
        name="exec",
        func=exec_program,
        description="useful in running or executing programs, for example notepad.exe, calc.exe, python, etc. with arguments being space separated. Program names typically have a .exe suffix, but user may not specify this. Program's stdout will be returned on completion. Observation is just the stdout.",
        # return_direct=True
    )
]


# Load built-in tools as well
tools.extend(
    load_tools(
        ["llm-math"],
        llm=llm
    )
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=agent_memory,
    verbose=True # verbose=True to see the agent's thought process
)

try:
    while True:
        print(
            agent_chain(dict(
                input=input('> ')
            ))['output']
        )
except (KeyboardInterrupt, EOFError):
    pass
except Exception:
    print("There was an error executing that action.")
    traceback.print_exc()
