
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

VANILLA_PROMPT = PromptTemplate.from_template(
    template="{query}"
)

simple = lambda llm, **kwargs: LLMChain(llm=llm, prompt=VANILLA_PROMPT)
