
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from chromadb.config import Settings as ChromaSettings


EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = './datastore/'

CHROMA_SETTINGS = ChromaSettings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

db = Chroma(
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
    persist_directory=PERSIST_DIRECTORY
)
