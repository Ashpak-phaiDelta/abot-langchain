
from functools import partial

from langchain.chains import RetrievalQAWithSourcesChain

from doc_db import db


TARGET_SOURCE_CHUNKS = 4


ask_doc_chain = partial(RetrievalQAWithSourcesChain.from_chain_type,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
)
