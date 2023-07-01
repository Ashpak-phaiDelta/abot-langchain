
from langchain.chains import RetrievalQAWithSourcesChain

from langchain.llms.openai import OpenAI

from doc_db import db


TARGET_SOURCE_CHUNKS = 10

llm = OpenAI()

ask_doc_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
)
