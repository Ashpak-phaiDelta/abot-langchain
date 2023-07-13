

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


SPLIT_CHUNK_SIZE = 500
SPLIT_CHUNK_OVERLAP = 30


EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

text_doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SPLIT_CHUNK_SIZE,
    chunk_overlap=SPLIT_CHUNK_OVERLAP
)
