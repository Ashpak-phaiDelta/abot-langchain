from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    temperature=0,
    model_name="text-davinci-003"

)