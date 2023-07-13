
from functools import partial

from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

from vectorstores.doc_chroma import chromadb
from vectorstores.genesis_pg import genesisdb


combine_prompt_template = """Given the following extracted parts of a long document of json format and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer. Provide concise and direct answers if possible. Provide output as markdown text, such as lists.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: What is the unit status?
===
Content: {{"unit_name": "B1 Lupin Material", "unit_alias": "B1 Lupin", "unit_sensors_out_count": 0, "unit_health_state": "NORMAL", "unit_location_at": {{"location_name": "VER_W1", "location_alias": "Verna"}}}}
Source: Unit data for Verna
===
FINAL ANSWER: NORMAL
SOURCES: Unit data for Verna

QUESTION: How is the Verna warehouse faring?
===
Content: {{"location_name": "VER_W1", "location_alias": "Verna", "location_coords": {{"latitude": 15.3629, "longitude": 73.9489}}, "location_health_state": "OUT_OF_RANGE", "location_summary": {{"metrics": {{"value": 6.0, "state": "OUT_OF_RANGE"}}, "power": {{"value": 16279.0, "state": "INACTIVE", "unit": "KWH"}}, "attendance": {{"value": 0.0, "state": "NORMAL"}}, "emergencies": {{"value": 0.0, "state": "NORMAL"}}}}}}
Source: Genesis Warehouse Location
===
FINAL ANSWER: It is out of range, no emergencies, metric out of range and power is inactive (at 16279.0 KWH).
SOURCES: Genesis Warehouse Location

QUESTION: {question}
===
{summaries}
===
FINAL ANSWER:"""
COMBINE_PROMPT_GENESIS = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)


ask_doc_chain = partial(RetrievalQAWithSourcesChain.from_chain_type,
    chain_type="stuff",
    # combine_prompt="",
)

ask_genesis_chain = partial(RetrievalQAWithSourcesChain.from_chain_type,
    chain_type="stuff",
    chain_type_kwargs=dict(prompt=COMBINE_PROMPT_GENESIS)
)


docs_vectorstore_info = VectorStoreInfo(
    name="uploaded_docs",
    description="Documents uploaded by the user",
    vectorstore=chromadb,
)

genesis_vectorstore_info = VectorStoreInfo(
    name="genesis",
    description="Data of the Genesis server of warehouses, units, sensors and reports",
    vectorstore=genesisdb,
)

def vectorstore_agent(llm):
    toolkit = VectorStoreToolkit(
        vectorstore_info=genesis_vectorstore_info,
        # llm=llm
    )
    return create_vectorstore_agent(
        llm,
        toolkit=toolkit,
        verbose=True
    )
