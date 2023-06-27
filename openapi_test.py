
import traceback

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.tools import Tool
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain
from langchain.requests import Requests


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


spec = OpenAPISpec.from_url(
    "http://uat.phaidelta.com:8079/openapi.json")

operation = APIOperation.from_openapi_spec(spec, "/genesis/query/sensor", "get")

chain = OpenAPIEndpointChain.from_api_operation(
    operation,
    llm,
    requests=Requests(),
    verbose=True,
    return_intermediate_steps=True,  # Return request and response text
)

# Tool(func=chain.run, name="sensor_status_query")

try:
    print("API query.")
    while True:
        print(chain(input('> ')
        ))
        # table_names_to_use parameter
except (KeyboardInterrupt, EOFError):
    pass
except Exception:
    print("There was an error executing that action.")
    traceback.print_exc()
