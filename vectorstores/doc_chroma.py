
from .hf_embedding import embeddings

from langchain.vectorstores import Chroma

from chromadb.config import Settings as ChromaSettings


PERSIST_DIRECTORY = './datastore/'

CHROMA_SETTINGS = ChromaSettings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

chromadb = Chroma(
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
    persist_directory=PERSIST_DIRECTORY
)
