

from langchain.vectorstores.pgvector import PGVector
from pydantic import BaseSettings, PostgresDsn

from .hf_embedding import embeddings


class GenesisVectorStoreSettings(BaseSettings):
    collection_name: str = 'genesis'
    host_url: str
    db_connection_string: PostgresDsn

    class Config:
        env_file = '.env'
        env_prefix = 'genesis_'


_settings = GenesisVectorStoreSettings()


genesisdb = PGVector(
    connection_string=_settings.db_connection_string,
    embedding_function=embeddings,
    collection_name=_settings.collection_name,
    collection_metadata={"host": _settings.host_url}
)
