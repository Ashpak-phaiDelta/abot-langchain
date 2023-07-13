

from langchain.vectorstores.pgvector import PGVector

from .hf_embedding import embeddings


GENESIS_COLLECTION_NAME = 'genesis'
GENESIS_HOST_URL = "https://genesis.phaidelta.com"


genesisdb = PGVector(
    connection_string="postgresql+psycopg2://postgres:Genesis%40123@uat.phaidelta.com:5432/abot_vectorstore",
    embedding_function=embeddings,
    collection_name=GENESIS_COLLECTION_NAME,
    collection_metadata={"host": GENESIS_HOST_URL}
)
