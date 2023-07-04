
import os
from typing import IO
from tempfile import _TemporaryFileWrapper
from io import BufferedRandom

from doc_db import db

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredFileIOLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xlsx": (UnstructuredExcelLoader, {})
    # Add more mappings for other file extensions and loaders as needed
}

SPLIT_CHUNK_SIZE = 500
"""How many characters to split each chunk at"""
SPLIT_CHUNK_OVERLAP = 50
"""How many characters between two chunks are same"""


def dump_documents_to_db(documents: List[Document]) -> int:
    # Split document into chunks (with some overlapping content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_CHUNK_OVERLAP
    )
    texts: List[Document] = text_splitter.split_documents(documents)

    # db.delete_collection()
    db.add_documents(texts)
    db.persist()
    return len(texts)

def handle_document(file: IO) -> int:
    try:
        metadata_name = os.path.basename(file.name)
        loader = UnstructuredFileIOLoader(file, metadata_filename=metadata_name)
        return dump_documents_to_db(loader.load())
    finally:
        pass


def upload_files(*files: _TemporaryFileWrapper):
    for file in files:
        with open(file.name, 'rb') as f:
            print("Uploading file %s..." % file.name)
            num_document_chunks = handle_document(f)
            print("%s uploaded, %d chunks" % (file.name, num_document_chunks))
