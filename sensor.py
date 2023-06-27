
import traceback

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.tools import Tool
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

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


db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:ComplicatedPassword%40123@uat.phaidelta.com:65432/abot_test")

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)


try:
    print("Database query.")
    while True:
        print(db_chain.run(input('> ')))
        # table_names_to_use parameter
except (KeyboardInterrupt, EOFError):
    pass
except Exception:
    print("There was an error executing that action.")
    traceback.print_exc()
