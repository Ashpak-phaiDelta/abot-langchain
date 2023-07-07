
from functools import partial

from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

from doc_db import db


TARGET_SOURCE_CHUNKS = 16


ask_doc_chain = partial(RetrievalQAWithSourcesChain.from_chain_type,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS}, search_type='similarity')
)


vectorstore_info = VectorStoreInfo(
    name="uploaded_docs",
    description="Documents uploaded by the user",
    vectorstore=db,
)

def vectorstore_agent(llm):
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return create_vectorstore_agent(
        llm,
        toolkit=toolkit,
        verbose=True
    )
